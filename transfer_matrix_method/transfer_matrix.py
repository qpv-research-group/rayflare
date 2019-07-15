""" This module serves as interface between solcore structures (junctions, layers, materials...) and the
transfer matrix package developed by Steven Byrnes and included in the PyPi repository.

"""
import numpy as np
import solcore
from solcore.interpolate import interp1d
from solcore.structure import ToStructure
import transfer_matrix_method.tmm_core_vec as tmm
from angles import make_angle_vector, fold_phi
from config import results_path
import os
import xarray as xr
from sparse import COO, save_npz, load_npz, stack

degree = np.pi / 180


class OptiStack(object):
    """ Class that contains an optical structure: a sequence of layers with a thickness and a complex refractive index.

    It serves as an intermediate step between solcore layers and materials and the stack of thicknesses and
    and n and k values necessary to run calculations involving TMM. When creating an OptiStack object, the thicknesses
    of all the layers forming the Solcore structure and the optical data of the materials of the layers are extracted
    and arranged in such a way they can be easily and fastly read by the TMM functions.

    In addition to a solcore structure with Layers, it can also take a list where each element represent a layer
    written as a list and contains the layer thickness and the dielectrical model, the raw n and k data as a function
    of wavelengths, or a whole Device structure as the type used in the PDD model.

    In summary, this class accepts:

        - A Solcore structure with layers
        - A list where each element is [thickness, DielectricModel]
        - A list where each element is [thickness, wavelength, n, k]
        - A list mixing the above:
            [ [thickness, DielectricModel],
              [thickness, wavelength, n, k],
              solcore.Layer,
              solcore.Layer ]

    This allows for maximum flexibility when creating the optical model, allowing to construct the stack with
    experimental data, modelled data and known material properties from the database.

    Yet anther way of defining the layers mixes experimental data with a DielectricModel within the same layer but in
    spectrally distinct regions. The syntaxis for the layer is:

    layer = [thickness, wavelength, n, k, DielectricModel, mixing]

    where mixing is a list containing three elements: [the mixing point (nm), the mixing width (nm),  zero or one]
    depending if the mixing function should be increasing with the wavelength or decreasing. If increasing (zero), the
    Dielectric model will be used at long wavelengths and the experimental data at short wavelengths. If decreasing
    (one) the oposite is done. The mixing point and mixing width control how smooth is the transition between one and
    the other type of data.

    Extra layers such as he semi-infinite, air-like first and last medium, and a back highly absorbing layer are
    included at runtime to fulfill the requirements of the TMM solver or to solve some of its limitations.
    """

    def __init__(self, structure=(), no_back_reflection=False, substrate=None, incidence=None):
        """ Class constructor. It takes a Solcore structure and extract the thickness and optical data from the
        Layers and the materials. Option is given to indicate if the reflexion from the back of the structure must be
        supressed, usefull for ellipsometry calculations. This is done by creating an artificial highly absorbing but
        not reflecting layer just at the back.

        Alternativelly, it can take simply a list of [thickness, DielectricModel] or [thickness, wavelength, n, k] for
        each layer accounting for the refractive index of the layers. The three options can be mixed for maximum
        flexibility.

        :param structure: A list with one or more layers.
        :param no_back_reflection: If reflexion from the back must be suppressed. Default=False.
        :param substrate: a semi-infinite transmission medium. Note that if no_back_reflection is set to True,
        adding a substrate won't make any difference.
        :param incidence: a semi-infinite incidence medium.
        """

        self.widths = []
        self.n_data = []
        self.k_data = []
        self.models = []
        self.layers = []
        self.substrate = substrate
        self.incidence = incidence

        self.num_layers = 0
        self.add_layers(structure)

        self.no_back_reflection = no_back_reflection

    def get_indices(self, wl):
        """ Returns the complex refractive index of the stack.

        :param wl: Wavelength of the light in nm.
        :return: A list with the complex refractive index of each layer, including the semi-infinite front and back
        layers and, opionally, the back absorbing layer used to suppress back surface relfexion.
        """

        out = []
        wl_m = solcore.si(wl, 'nm')

        if hasattr(self, 'n_sub'):
            n1 = self.n_sub(wl_m) + self.k_sub(wl_m)*1.0j
        else:
            if hasattr(wl, 'size'):
                n1 = np.ones_like(wl, dtype=complex)
            else:
                n1 = 1

        # incidence medium!
        if hasattr(self, 'n_sup'):
            n0 = self.n_sup(wl_m) #+ self.k_sup(wl_m)*1.0j ignore complex part for now to avoid errors
        else:
            if hasattr(wl, 'size'):
                n0 = np.ones_like(wl, dtype=complex)
            else:
                n0 = 1

        for i in range(self.num_layers):
            out.append(self.n_data[i](wl_m) + self.k_data[i](wl_m) * 1.0j)

        # substrate irrelevant if no_back_reflection = True
        if self.no_back_reflection:
            return [n0] + out + [self.n_data[-1](wl_m) + self._k_absorbing(wl_m) * 1.0j, n1] # look at last entry in stack,
            # make high;y absorbing layer based on it.

        else:
            return [n0] + out + [n1]


    def get_widths(self):
        """ Returns the widths of the layers of the stack.

        :return: A list with the widths each layer, including the semi-infinite front and back layers and, opionally,
        the back absorbing layer used to suppress back surface relfexion, defined as 1 mm thick.
        """

        if self.no_back_reflection:
            return [np.inf] + self.widths + [1e6, np.inf]
        else:
            return [np.inf] + self.widths + [np.inf]

    def _k_absorbing(self, wl):
        """ k value of the back highly absorbing layer. It is the maximum between the bottom layer of the stack or a
        finite, small value that will absorb all light within the absorbing layer thickness.

        :param wl: Wavelength of the light in nm.
        :return: The k value at each wavelength.
        """
        return np.maximum(wl / 1e-3, self.k_data[-1](wl))

    @staticmethod
    def _k_dummy(wl):
        """ Dummy k value to be used with the dielectric model, which produces the refractive index as a complex
        number.

        :param wl: Wavelength of the light in nm.
        :return: The k value at each wavelength... set to zero.
        """
        return 0.

    def add_layers(self, layers):
        """ Generic function to add layers to the OptiStack. Internally, it calls add_solcore_layer,
        add_modelled_layer or add_raw_nk_layer.

        :param layers: A list with the layers to add (even if it is just one layer) It can be one or more and it can
        mixed, Solcore-based and modelled layers.
        :return: None
        """
        try:
            # If the input is a whole device structure, we get just the layers information
            if type(layers) is dict:
                layers = ToStructure(layers)

            if self.substrate is not None:
                self.n_sub = self.substrate.n
                self.k_sub = self.substrate.k

            if self.incidence is not None:
                self.n_sup = self.incidence.n
                self.k_sup = self.incidence.k

            for layer in layers:
                self.layers.append(layer)
                if 'Layer' in str(type(layer)):
                    self._add_solcore_layer(layer)
                elif 'DielectricConstantModel' in str(type(layer[1])):
                    self._add_modelled_layer(layer)
                else:
                    self._add_raw_nk_layer(layer)

                self.num_layers += 1

        except Exception as err:
            print('Error when adding a new layer to the OptiStack. {}'.format(err))

    def remove_layer(self, idx):
        """ Removes layer with index idx from the OptiStack

        :param idx: Index of the layer to remove
        :return: None
        """
        if idx >= self.num_layers:
            print('Error when removing layers. idx must be: 0 <= idx <= {}.'.format(self.num_layers - 1))
            return

        self.widths.pop(idx)
        self.models.pop(idx)
        self.n_data.pop(idx)
        self.k_data.pop(idx)
        self.num_layers -= 1

    def swap_layers(self, idx1, idx2):
        """ Swaps two layers in the OptiStack.

        :param idx1: The index of one of the layers.
        :param idx2: The index of the other.
        :return: None
        """
        if idx1 >= self.num_layers or idx2 >= self.num_layers:
            print('Error when removing layers. idx must be: 0 <= idx1, idx2 <= {}.'.format(self.num_layers - 1))
            return

        self.widths[idx1], self.widths[idx2] = self.widths[idx2], self.widths[idx1]
        self.models[idx1], self.models[idx2] = self.models[idx2], self.models[idx1]
        self.n_data[idx1], self.n_data[idx2] = self.n_data[idx2], self.n_data[idx1]
        self.k_data[idx1], self.k_data[idx2] = self.k_data[idx2], self.k_data[idx1]

    def _add_solcore_layer(self, layer):
        """ Adds a Solcore layer to the end (bottom) of the stack, extracting its thickness and n and k data.

        :param layer: The Solcore layer
        :return: None
        """
        self.widths.append(solcore.asUnit(layer.width, 'nm'))
        self.models.append([])
        self.n_data.append(layer.material.n)
        self.k_data.append(layer.material.k)

    def _add_modelled_layer(self, layer):
        """ Adds a layer to the end (bottom) of the stack. The layer must be defined as a list containing the layer
        thickness in nm and a dielectric model.

        :param layer: The new layer to add as [thickness, DielectricModel]
        :return: None
        """
        self.widths.append(layer[0])
        self.models.append(layer[1])
        self.n_data.append(self.models[-1].n_and_k)
        self.k_data.append(self._k_dummy)

    def _add_raw_nk_layer(self, layer):
        """ Adds a layer to the end (bottom) of the stack. The layer must be defined as a list containing the layer
        thickness in nm, the wavelength, the n and the k data as array-like objects.

        :param layer: The new layer to add as [thickness, wavelength, n, k]
        :return: None
        """
        # We make sure that the wavelengths are increasing, revering the arrays otherwise.
        if layer[1][0] > layer[1][-1]:
            layer[1] = layer[1][::-1]
            layer[2] = layer[2][::-1]
            layer[3] = layer[3][::-1]

        self.widths.append(layer[0])

        if len(layer) >= 5:
            self.models.append(layer[4])
            c = solcore.si(layer[5][0], 'nm')
            w = solcore.si(layer[5][1], 'nm')
            d = layer[5][2]  # = 0 for increasing, =1 for decreasing

            def mix(x):

                out = 1 + np.exp(-(x - c) / w)
                out = d + (-1) ** d * 1 / out

                return out

            n_data = lambda x: self.models[-1].n_and_k(x) * mix(x) + (1 - mix(x)) * interp1d(
                x=solcore.si(layer[1], 'nm'), y=layer[2], fill_value=layer[2][-1])(x)
            k_data = lambda x: interp1d(x=solcore.si(layer[1], 'nm'), y=layer[3], fill_value=layer[3][-1])(x)

            self.n_data.append(n_data)
            self.k_data.append(k_data)

        else:
            self.models.append([])
            self.n_data.append(interp1d(x=solcore.si(layer[1], 'nm'), y=layer[2], fill_value=layer[2][-1]))
            self.k_data.append(interp1d(x=solcore.si(layer[1], 'nm'), y=layer[3], fill_value=layer[3][-1]))


def tmm_matrix(layers, transmission, incidence, surf_name, options,
               coherent=True, coherency_list=None, prof_layers=[], front_or_rear='front'):
    """Function which takes a layer stack and creates an angular redistribution matrix.

        :param layers: A list with one or more layers.
        :param transmission: transmission medium
        :param incidence: incidence medium
        :param surf_name: name of the surface (to save the matrices generated.
        :param options: a list of options
        :param coherent: whether or not the layer stack is coherent. If None, it is assumed to be fully coherent
        :param coherency: a list with the same number of entries as the layers, either 'c' for a coherent layer or
        'i' for an incoherent layer
        :param prof_layers: layers for which the absorption profile should be calculated
        (if None, do not calculate absorption profile at all)
        :param front_or_rear: a string, either 'front' or 'rear'; front incidence on the stack, from the incidence
        medium, or rear incidence on the stack, from the transmission medium.
        :return full_mat: R and T redistribution matrix
        :return A_mat: matrix describing absorption per layer
        """

    def make_matrix_wl(wl):
        RT_mat = np.zeros((len(theta_bins_in)*2, len(theta_bins_in)))
        A_mat = np.zeros((n_layers, len(theta_bins_in)))

        for i1 in range(len(theta_bins_in)):

            theta = angle_vector[i1, 1]

            data = allres.loc[dict(angle=theta, wl=wl)]
            R_prob = np.real(data['R'].data.item(0))
            T_prob = np.real(data['T'].data.item(0))

            Alayer_prob = np.real(data['Alayer'].data)
            phi_out = phis_out[i1]

            # reflection
            phi_int = phi_intv[theta_bins_in[i1]]
            phi_ind = np.digitize(phi_out, phi_int, right=True) - 1
            bin_out_r = np.argmin(abs(angle_vector[:, 0] - theta_bins_in[i1])) + phi_ind

            RT_mat[bin_out_r, i1] = R_prob

            # transmission
            theta_t = np.pi-np.arcsin((inc.n(wl * 1e-9) / trns.n(wl * 1e-9)) * np.sin(theta))
            if ~np.isnan(theta_t):

                theta_out_bin = np.digitize(theta_t, theta_intv, right=True) - 1
                phi_int = phi_intv[theta_out_bin]

                phi_ind = np.digitize(phi_out, phi_int, right=True) - 1
                bin_out_t = np.argmin(abs(angle_vector[:, 0] - theta_out_bin)) + phi_ind
                RT_mat[bin_out_t, i1] = T_prob

            # absorption
            A_mat[:, i1] = Alayer_prob

        fullmat = COO(RT_mat)
        A_mat = COO(A_mat)
        return fullmat, A_mat

    structpath = os.path.join(results_path, options['project_name'])
    if not os.path.isdir(structpath):
        os.mkdir(structpath)

    savepath_RT = os.path.join(structpath, surf_name + front_or_rear + 'RT.npz')
    savepath_A = os.path.join(structpath, surf_name + front_or_rear + 'A.npz')
    prof_mat_path = os.path.join(results_path, options['project_name'],
                                 surf_name + front_or_rear + 'profmat.nc')

    if os.path.isfile(savepath_RT):
        print('Existing angular redistribution matrices found')
        fullmat = load_npz(savepath_RT)
        A_mat = load_npz(savepath_A)

        if len(prof_layers) > 0:
            profile = xr.load_dataarray(prof_mat_path)
            return fullmat, A_mat, profile

    else:

        wavelengths = options['wavelengths']*1e9 # convert to nm

        theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'], options['c_azimuth'])
        angles_in = angle_vector[:int(len(angle_vector) / 2), :]
        thetas = np.unique(angles_in[:, 1])

        n_angles = len(thetas)

        n_layers = len(layers)

        if front_or_rear == 'front':
            optlayers = OptiStack(layers, substrate=transmission, incidence=incidence)
            trns = transmission
            inc = incidence

        else:
            optlayers = OptiStack(layers[::-1], substrate=incidence, incidence=transmission)
            trns = incidence
            inc = transmission


        if len(prof_layers) > 0:
            profile = True
            z_limit = np.sum(np.array(optlayers.widths))
            full_dist = np.arange(0, z_limit, options['nm_spacing'])
            layer_start = np.insert(np.cumsum(np.insert(optlayers.widths, 0, 0)), 0, 0)
            layer_end = np.cumsum(np.insert(optlayers.widths, 0, 0))

            dist = []

            for l in prof_layers:
                dist = np.hstack((dist, full_dist[np.all((full_dist >= layer_start[l], full_dist < layer_end[l]), 0)]))

        else:
            profile = False

        if options['pol'] == 'u':
            pols = ['s', 'p']
        else:
            pols = [options['pol']]


        R = xr.DataArray(np.empty((len(pols), len(wavelengths), n_angles)),
                         dims=['pol', 'wl', 'angle'],
                         coords={'pol': pols, 'wl': wavelengths, 'angle': thetas},
                         name='R')
        T = xr.DataArray(np.empty((len(pols), len(wavelengths), n_angles)),
                         dims=['pol', 'wl', 'angle'],
                         coords={'pol': pols, 'wl': wavelengths, 'angle': thetas},
                         name='T')
        Alayer = xr.DataArray(np.empty((len(pols), n_angles, len(wavelengths), n_layers)),
                              dims=['pol', 'angle', 'wl', 'layer'],
                              coords={'pol': pols,
                                      'wl': wavelengths,
                                      'angle': thetas,
                                      'layer': range(1, n_layers + 1)}, name='Alayer')

        theta_t = xr.DataArray(np.empty((len(pols), len(wavelengths), n_angles)),
                               dims=['pol', 'wl', 'angle'],
                               coords={'pol': pols, 'wl': wavelengths, 'angle': thetas},
                               name='theta_t')

        if profile:
            Aprof = xr.DataArray(np.empty((len(pols), n_angles, len(wavelengths), len(dist))),
                                 dims=['pol', 'angle', 'wl', 'z'],
                                 coords={'pol': pols,
                                         'wl': wavelengths,
                                         'angle': thetas,
                                         'z': dist}, name='Aprof')

        R_loop = np.empty((len(wavelengths), n_angles))
        T_loop = np.empty((len(wavelengths), n_angles))
        Alayer_loop = np.empty((n_angles, len(wavelengths), n_layers), dtype=np.complex_)
        th_t_loop = np.empty((len(wavelengths), n_angles))

        if front_or_rear == 'rear':
            Alayer_loop = np.flip(Alayer_loop, axis=2)

        if profile:
            Aprof_loop = np.empty((n_angles, len(wavelengths), len(dist)))

        for i2, pol in enumerate(pols):

            for i3, theta in enumerate(thetas):

                res = calculate_rat(optlayers, wavelengths, angle=theta, pol=pol,
                                    coherent=coherent, coherency_list=coherency_list, profile=profile,
                                    layers=prof_layers, nm_spacing=options['nm_spacing'])
                R_loop[:, i3] = np.real(res['R'])
                T_loop[:, i3] = np.real(res['T'])
                Alayer_loop[i3, :, :] = np.real(res['A_per_layer'].T)

                if profile:
                    Aprof_loop[i3, :, :] = res['profile']

            # sometimes get very small negative values (like -1e-20)
            R_loop[R_loop < 0] = 0
            T_loop[T_loop < 0] = 0
            Alayer_loop[Alayer_loop < 0] = 0


            R.loc[dict(pol=pol)] = R_loop
            T.loc[dict(pol=pol)] = T_loop
            Alayer.loc[dict(pol=pol)] = Alayer_loop
            theta_t.loc[dict(pol=pol)] = th_t_loop

            if profile:
                Aprof.loc[dict(pol=pol)] = Aprof_loop
                Aprof.transpose('pol', 'wl', 'angle', 'z')

        Alayer = Alayer.transpose('pol', 'wl', 'angle', 'layer')

        if profile:
            allres = xr.merge([R, T, Alayer, Aprof])
        else:
            allres = xr.merge([R, T, Alayer])

        if options['pol'] == 'u':
            allres = allres.reduce(np.mean, 'pol').assign_coords(pol='u').expand_dims('pol')

        # populate matrices
        angle_vector_th = angle_vector[:int(len(angle_vector)/2),1]
        angle_vector_phi = angle_vector[:int(len(angle_vector)/2),2]

        phis_out = fold_phi(angle_vector_phi + np.pi, options['phi_symmetry'])
        phis_out[phis_out == 0] = 1e-10

        theta_bins_in = np.digitize(angle_vector_th, theta_intv, right=True) -1


        mats = [make_matrix_wl(wl) for wl in wavelengths]

        fullmat = stack([item[0] for item in mats])
        A_mat = stack([item[1] for item in mats])

        save_npz(savepath_RT, fullmat)
        save_npz(savepath_A, A_mat)

    return fullmat, A_mat



def calculate_rat(stack, wavelength, angle=0, pol='u',
                  coherent=True, coherency_list=None, profile=False, layers=None, nm_spacing = 1):
    """ Calculates the reflected, absorbed and transmitted intensity of the structure for the wavelengths and angles
    defined.

    :param stack: an OptiStack object.
    :param wavelength: Wavelengths (in nm) in which calculate the data. An array.
    :param angle: Angle (in radians) of the incident light. Default: 0 (normal incidence).
    :param pol: Polarisation of the light: 's', 'p' or 'u'. Default: 'u' (unpolarised).
    :param coherent: If the light is coherent or not. If not, a coherency list must be added.
    :param coherency_list: A list indicating in which layers light should be treated as coeherent ('c') and in which
    incoherent ('i'). It needs as many elements as layers in the structure.
    :param profile: whether or not to calculate the absorption profile
    :param layers: indices of the layers in which to calculate the absorption profile. Layer 0 is the incidence medium.
    :return: A dictionary with the R, A and T at the specified wavelengths and angle.
    """
    num_wl = len(wavelength)
    if not coherent:
        if coherency_list is not None:
            assert len(coherency_list) == stack.num_layers, \
                'Error: The coherency list must have as many elements (now {}) as the ' \
                'number of layers (now {}).'.format(len(coherency_list), stack.num_layers)
            
            if stack.no_back_reflection:
                coherency_list = ['i'] + coherency_list + ['i', 'i']
            else:
                coherency_list = ['i'] + coherency_list + ['i']

        else:
            raise Exception('Error: For incoherent or partly incoherent calculations you must supply the '
                            'coherency_list parameter with as many elements as the number of layers in the '
                            'structure')

    output = {'R': np.zeros(num_wl), 'A': np.zeros(num_wl), 'T': np.zeros(num_wl), 'all_p': [], 'all_s': []}

    if pol in 'sp':
        if coherent:
            out = tmm.coh_tmm(pol, stack.get_indices(wavelength), stack.get_widths(), angle, wavelength)
            A_per_layer =  tmm.absorp_in_each_layer(out)
            output['R'] = out['R']
            output['A'] = 1 - out['R'] - out['T']
            output['T'] = out['T']
            output['A_per_layer'] = A_per_layer[1:-1]
        else:
            out = tmm.inc_tmm(pol, stack.get_indices(wavelength), stack.get_widths(), coherency_list, angle, wavelength)
            A_per_layer = np.array(tmm.inc_absorp_in_each_layer(out))
            output['R'] = out['R']
            output['A'] = 1 - out['R'] - out['T']
            output['T'] = out['T']
            output['A_per_layer'] = A_per_layer[1:-1]
    else:
        if coherent:
            out_p = tmm.coh_tmm('p', stack.get_indices(wavelength), stack.get_widths(), angle, wavelength)
            out_s = tmm.coh_tmm('s', stack.get_indices(wavelength), stack.get_widths(), angle, wavelength)
            A_per_layer_p = tmm.absorp_in_each_layer(out_p)
            A_per_layer_s = tmm.absorp_in_each_layer(out_s)
            output['R'] = 0.5 * (out_p['R'] + out_s['R'])
            output['T'] = 0.5 * (out_p['T'] + out_s['T'])
            output['A'] = 1 - output['R'] - output['T']
            output['A_per_layer'] = 0.5*(A_per_layer_p[1:-1] + A_per_layer_s[1:-1])

        else:
            out_p = tmm.inc_tmm('p', stack.get_indices(wavelength), stack.get_widths(), coherency_list, angle, wavelength)
            out_s = tmm.inc_tmm('s', stack.get_indices(wavelength), stack.get_widths(), coherency_list, angle, wavelength)

            A_per_layer_p = np.array(tmm.inc_absorp_in_each_layer(out_p))
            A_per_layer_s = np.array(tmm.inc_absorp_in_each_layer(out_s))

            output['R'] = 0.5 * (out_p['R'] + out_s['R'])
            output['T'] = 0.5 * (out_p['T'] + out_s['T'])
            output['A'] = 1 - output['R'] - output['T']
            output['all_p'] = out_p['power_entering_list']
            output['all_s'] = out_s['power_entering_list']
            output['A_per_layer'] = 0.5*(A_per_layer_p[1:-1] + A_per_layer_s[1:-1])


    # if requested, calculate absorption profile as well

    # layer indices: 0 is incidence, n is transmission medium
    if profile:

        z_limit = np.sum(np.array(stack.widths))
        full_dist = np.arange(0, z_limit, nm_spacing)
        layer_start = np.insert(np.cumsum(np.insert(stack.widths, 0, 0)), 0, 0)
        layer_end = np.cumsum(np.insert(stack.widths, 0, 0))

        dist = []

        for l in layers:
            dist = np.hstack((dist, full_dist[np.all((full_dist >= layer_start[l], full_dist < layer_end[l]), 0)]))

        if pol in 'sp':

            if coherent:
                fn = tmm.absorp_analytic_fn().fill_in(out, layers)
                layer, d_in_layer = tmm.find_in_structure_with_inf(stack.get_widths(), dist)
                data = tmm.position_resolved(layer, d_in_layer, out)
                output['profile'] = data['absor']

            else:
                fraction_reaching = 1 - np.cumsum(A_per_layer, axis=0)
                fn = tmm.absorp_analytic_fn()
                fn.a1, fn.a3, fn.A1, fn.A2, fn.A3 = np.empty((0, num_wl)), np.empty((0, num_wl)), np.empty((0, num_wl)), \
                                                    np.empty((0, num_wl)), np.empty((0, num_wl))

                layer, d_in_layer = tmm.find_in_structure_with_inf(stack.get_widths(), dist)
                data = tmm.inc_position_resolved(layer, d_in_layer, out, coherency_list,
                                                 4 * np.pi * np.imag(stack.get_indices(wavelength)) / wavelength)
                output['profile'] = data


                for i1, l in enumerate(layers):

                    if coherency_list[l] == 'c':
                        fn_l = tmm.inc_find_absorp_analytic_fn(l, out)
                        fn.a1 = np.vstack((fn.a1, fn_l.a1))
                        fn.a3 = np.vstack((fn.a3, fn_l.a3))
                        fn.A1 = np.vstack((fn.A1, fn_l.A1))
                        fn.A2 = np.vstack((fn.A2, fn_l.A2))
                        fn.A3 = np.vstack((fn.A3, fn_l.A3))

                    else:
                        # DO NOT KNOW IF UNITS ARE CORRECT
                        alpha = np.imag(stack.get_indices(wavelength)[l])*4*np.pi/wavelength
                        fn.a1 = np.vstack((fn.a1, alpha))
                        fn.A2 = np.vstack((fn.A2, alpha*fraction_reaching[l-1]))
                        fn.a3 = np.vstack((fn.a3, np.zeros((1, num_wl))))
                        fn.A1 = np.vstack((fn.A1, np.zeros((1, num_wl))))
                        fn.A3 = np.vstack((fn.A3, np.zeros((1, num_wl))))

        else:
            if coherent:
                fn_s = tmm.absorp_analytic_fn().fill_in(out_s, layers)
                fn_p = tmm.absorp_analytic_fn().fill_in(out_p, layers)
                fn = fn_s.add(fn_p).scale(0.5)

                layer, d_in_layer = tmm.find_in_structure_with_inf(stack.get_widths(), dist)
                data_s = tmm.position_resolved(layer, d_in_layer, out_s)
                data_p = tmm.position_resolved(layer, d_in_layer, out_p)

                output['profile'] = 0.5 * (data_s['absor'] + data_p['absor'])

            else:
                fraction_reaching_s = 1 - np.cumsum(A_per_layer_s, axis=0)
                fraction_reaching_p = 1 - np.cumsum(A_per_layer_s, axis=0)
                fraction_reaching = 0.5*(fraction_reaching_s + fraction_reaching_p)
                fn = tmm.absorp_analytic_fn()
                fn.a1, fn.a3, fn.A1, fn.A2, fn.A3 = np.empty((0, num_wl)), np.empty((0, num_wl)), np.empty((0, num_wl)), \
                                                    np.empty((0, num_wl)), np.empty((0, num_wl))

                layer, d_in_layer = tmm.find_in_structure_with_inf(stack.get_widths(), dist)
                data_s = tmm.inc_position_resolved(layer, d_in_layer, out_s, coherency_list,
                                                   4 * np.pi * np.imag(stack.get_indices(wavelength)) / wavelength)
                data_p = tmm.inc_position_resolved(layer, d_in_layer, out_p, coherency_list,
                                                   4 * np.pi * np.imag(stack.get_indices(wavelength)) / wavelength)

                output['profile'] = 0.5 * (data_s + data_p)

                for i1, l in enumerate(layers):
                    if coherency_list[l] == 'c':
                        fn_s = tmm.inc_find_absorp_analytic_fn(l, out_s)
                        fn_p = tmm.inc_find_absorp_analytic_fn(l, out_s)
                        fn_l = fn_s.add(fn_p).scale(0.5)
                        fn.a1 = np.vstack((fn.a1, fn_l.a1))
                        fn.a3 = np.vstack((fn.a3, fn_l.a3))
                        fn.A1 = np.vstack((fn.A1, fn_l.A1))
                        fn.A2 = np.vstack((fn.A2, fn_l.A2))
                        fn.A3 = np.vstack((fn.A3, fn_l.A3))


                    else:
                        # DO NOT KNOW IF UNITS ARE CORRECT
                        alpha = np.imag(stack.get_indices(wavelength)[l]) * 4 * np.pi / wavelength
                        fn.a1 = np.vstack((fn.a1, alpha))
                        fn.A2 = np.vstack((fn.A2, alpha * fraction_reaching[l - 1]))
                        fn.a3 = np.vstack((fn.a3, np.zeros((1, num_wl))))
                        fn.A1 = np.vstack((fn.A1, np.zeros((1, num_wl))))
                        fn.A3 = np.vstack((fn.A3, np.zeros((1, num_wl))))
        output['profile_coeff'] = np.stack((fn.A1, fn.A2, np.real(fn.A3), np.imag(fn.A3), fn.a1, fn.a3)) # shape is (5, n_layers, num_wl)

    return output



