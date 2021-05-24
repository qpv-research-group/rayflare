import numpy as np
import xarray as xr
from sparse import COO, save_npz, stack

from solcore.absorption_calculator import tmm_core_vec as tmm
from solcore.absorption_calculator import OptiStack

from rayflare.angles import make_angle_vector, fold_phi
from rayflare.utilities import get_matrices_or_paths


def TMM(layers, incidence, transmission, surf_name, options, structpath,
               coherent=True, coherency_list=None, prof_layers=None, front_or_rear='front', save=True):
    """
    Function which takes a layer stack and creates an angular redistribution matrix.

    :param layers: A list with one or more layers.
    :param incidence: incidence medium
    :param transmission: transmission medium
    :param surf_name: name of the surface (to save/load the matrices generated).
    :param options: a list of options
    :param structpath: file path where matrices will be stored or loaded from
    :param coherent: whether or not the layer stack is coherent. If None, it is assumed to be fully coherent
    :param coherency_list: a list with the same number of entries as the layers, either 'c' for a coherent layer or
            'i' for an incoherent layer
    :param prof_layers: layers for which the absorption profile should be calculated
            (if None, do not calculate absorption profile at all)
    :param front_or_rear: a string, either 'front' or 'rear'; front incidence on the stack, from the incidence
            medium, or rear incidence on the stack, from the transmission medium.
    :param save:

    :return: Number of returns depends on whether absorption profiles are being calculated; the first two items are
             always returned, the final one only if a profile is being calcualted.

                - fullmat: the R/T redistribution matrix at each wavelength, indexed as (wavelength, angle_bin_out, angle_bin_in)
                - A_mat: the absorption redistribution matrix (total absorption per layer), indexed as (wavelength, layer_out, angle_bin_in)
                - allres: xarray dataset storing the absorption profile data
    """

    def make_matrix_wl(wl):
        # binning into matrix, including phi
        RT_mat = np.zeros((len(theta_bins_in)*2, len(theta_bins_in)))
        A_mat = np.zeros((n_layers, len(theta_bins_in)))

        for i1, cur_theta in enumerate(theta_bins_in):

            theta = theta_lookup[i1] # angle_vector[i1, 1]

            data = allres.loc[dict(angle=theta, wl=wl)]

            R_prob = np.real(data['R'].data.item(0))
            T_prob = np.real(data['T'].data.item(0))

            Alayer_prob = np.real(data['Alayer'].data)
            phi_out = phis_out[i1]

            # reflection
            phi_int = phi_intv[cur_theta]
            phi_ind = np.digitize(phi_out, phi_int, right=True) - 1
            bin_out_r = np.argmin(abs(angle_vector[:, 0] - cur_theta)) + phi_ind

            RT_mat[bin_out_r, i1] = R_prob

            # transmission
            with np.errstate(divide='ignore', invalid='ignore'):
                theta_t = np.abs(-np.arcsin((inc.n(wl) / trns.n(wl)) * np.sin(theta_lookup[i1])) + quadrant)

            if np.isnan(theta_t) and T_prob > 1e-8:
                # bodge, but when transmitting into an absorbing medium, can't get total internal reflection even though
                # it is not possible to calculate the transmission angle through the method above.
                theta_t = np.abs(np.pi/2 - 1e-5 - quadrant)

            # theta switches half-plane (th < 90 -> th >90
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

    def make_prof_matrix_wl(wl):

        prof_wl = xr.DataArray(np.empty((len(dist), len(theta_bins_in))),
                               dims=['z', 'global_index'],
                               coords={'z': dist, 'global_index': np.arange(0, len(theta_bins_in))})

        for i1 in range(len(theta_bins_in)):

            theta = theta_lookup[i1]

            data = allres.loc[dict(angle=theta, wl=wl)]

            prof_depth = np.real(data['Aprof'].data[0])

            prof_wl[:, i1] = prof_depth

        return prof_wl

    existing_mats, path_or_mats = get_matrices_or_paths(structpath, surf_name, front_or_rear, prof_layers)

    if existing_mats:
        return path_or_mats

    else:

        wavelengths = options['wavelengths']

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
            if prof_layers is not None:
                prof_layers = np.sort(len(layers) - np.array(prof_layers) + 1).tolist()

        if prof_layers is not None:
            profile = True
            z_limit = np.sum(np.array(optlayers.widths))
            full_dist = np.arange(0, z_limit, options['depth_spacing']*1e9)
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
        Alayer_loop = np.empty((n_angles, len(wavelengths), n_layers))
        th_t_loop = np.empty((len(wavelengths), n_angles))

        if profile:
            Aprof_loop = np.empty((n_angles, len(wavelengths), len(dist)))

        tmm_struct = tmm_structure(optlayers, incidence, transmission, False)

        pass_options = {}
        pass_options['coherent'] = coherent
        pass_options['coherency_list'] = coherency_list
        pass_options['wavelengths'] = wavelengths
        pass_options['depth_spacing'] =  options['depth_spacing']

        for pol in pols:

            for i3, theta in enumerate(thetas):

                pass_options['pol'] = pol
                pass_options['theta_in'] = theta

                res = tmm_struct.calculate(pass_options, profile=profile, layers=prof_layers)

                R_loop[:, i3] = np.real(res['R'])
                T_loop[:, i3] = np.real(res['T'])
                Alayer_loop[i3, :, :] = np.real(res['A_per_layer'])

                if profile:
                    Aprof_loop[i3, :, :] = res['profile']

            # sometimes get very small negative values (like -1e-20)
            R_loop[R_loop < 0] = 0
            T_loop[T_loop < 0] = 0
            Alayer_loop[Alayer_loop < 0] = 0

            if profile:
                Aprof_loop[Aprof_loop < 0] = 0

            if front_or_rear == 'rear':
                Alayer_loop = np.flip(Alayer_loop, axis=2)

                if profile:
                    Aprof_loop = np.flip(Aprof_loop, axis=2)

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
        if front_or_rear == "front":

            angle_vector_th = angle_vector[:int(len(angle_vector)/2),1]
            angle_vector_phi = angle_vector[:int(len(angle_vector)/2),2]

            phis_out = fold_phi(angle_vector_phi + np.pi, options['phi_symmetry'])
            theta_lookup = angles_in[:,1]
            quadrant = np.pi


        else:
            angle_vector_th = angle_vector[int(len(angle_vector) / 2):, 1]
            angle_vector_phi = angle_vector[int(len(angle_vector) / 2):, 2]

            phis_out = fold_phi(angle_vector_phi + np.pi, options['phi_symmetry'])
            theta_lookup = angles_in[:,1][::-1]
            quadrant = 0

        phis_out[phis_out == 0] = 1e-10

        theta_bins_in = np.digitize(angle_vector_th, theta_intv, right=True) -1

        mats = [make_matrix_wl(wl) for wl in wavelengths]

        fullmat = stack([item[0] for item in mats])
        A_mat = stack([item[1] for item in mats])

        if save:
            save_npz(path_or_mats[0], fullmat)
            save_npz(path_or_mats[1], A_mat)

        if profile:
            prof_mat = [make_prof_matrix_wl(wl) for wl in wavelengths]

            profile = xr.concat(prof_mat, 'wl')
            intgr = xr.DataArray(np.sum(A_mat.todense(), 1), dims=['wl', 'global_index'],
                                 coords={'wl': wavelengths, 'global_index': np.arange(0, len(theta_bins_in))})
            intgr.name = 'intgr'
            profile.name = 'profile'
            allres = xr.merge([intgr, profile])

            if save:
                allres.to_netcdf(path_or_mats[2])

            return fullmat, A_mat, allres


        return fullmat, A_mat


class tmm_structure:
    """ Set up structure for TMM calculations.

    :param stack: an OptiStack or SolarCell object.
    :param incidence: incidence medium (Solcore material)
    :param transmission: transmission medium/substrate (Solcore material)
    :param no_back_reflection: whether to suppress reflections at the interface between the final material
            in the stack and the substrate (default False)
    """

    def __init__(self, stack, incidence=None, transmission=None, no_back_reflection=False):

        if 'OptiStack' in str(type(stack)):
            stack.no_back_reflection = no_back_reflection
        else:
            stack = OptiStack(stack, no_back_reflection=no_back_reflection,
                              substrate=transmission, incidence=incidence)


        self.stack = stack
        self.no_back_reflection = no_back_reflection


    def calculate(self, options, profile=False, layers=None):
        """ Calculates the reflected, absorbed and transmitted intensity of the structure for the wavelengths and angles
        defined.

        :param options: options for the calculation. The key entries are:

            - wavelength: Wavelengths (in m) in which calculate the data. An array.
            - theta_in: Angle (in radians) of the incident light.
            - pol: Polarisation of the light: 's', 'p' or 'u'.
            - coherent: If the light is coherent or not. If not, a coherency list must be added.
            - coherency_list: A list indicating in which layers light should be treated as coherent ('c') and in which \
                incoherent ('i'). It needs as many elements as layers in the structure.

        :param profile: whether or not to calculate the absorption profile
        :param layers: indices of the layers in which to calculate the absorption profile.
            Layer 0 is the incidence medium.

        :return: A dictionary with the R, A and T at the specified wavelengths and angle.
        """

        def calculate_profile(layers):
            # layer indices: 0 is incidence, n is transmission medium
            if layers is None:
                layers = np.arange(1, stack.num_layers + 1)

            depth_spacing = options['depth_spacing'] * 1e9  # convert from m to nm

            z_limit = np.sum(np.array(stack.widths))
            full_dist = np.arange(0, z_limit, depth_spacing)
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
                    fn.a1, fn.a3, fn.A1, fn.A2, fn.A3 = np.empty((0, num_wl)), np.empty((0, num_wl)), np.empty(
                        (0, num_wl)), \
                                                        np.empty((0, num_wl)), np.empty((0, num_wl))

                    layer, d_in_layer = tmm.find_in_structure_with_inf(stack.get_widths(), dist)
                    data = tmm.inc_position_resolved(layer, d_in_layer, out, coherency_list,
                                                     4 * np.pi * np.imag(stack.get_indices(wavelength)) / wavelength)
                    output['profile'] = data

                    for l in layers:

                        if coherency_list[l] == 'c':
                            fn_l = tmm.inc_find_absorp_analytic_fn(l, out)
                            fn.a1 = np.vstack((fn.a1, fn_l.a1))
                            fn.a3 = np.vstack((fn.a3, fn_l.a3))
                            fn.A1 = np.vstack((fn.A1, fn_l.A1))
                            fn.A2 = np.vstack((fn.A2, fn_l.A2))
                            fn.A3 = np.vstack((fn.A3, fn_l.A3))

                        else:
                            alpha = np.imag(stack.get_indices(wavelength)[l]) * 4 * np.pi / wavelength
                            fn.a1 = np.vstack((fn.a1, alpha))
                            fn.A2 = np.vstack((fn.A2, alpha * fraction_reaching[l - 1]))
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
                    fraction_reaching = 0.5 * (fraction_reaching_s + fraction_reaching_p)
                    fn = tmm.absorp_analytic_fn()
                    fn.a1, fn.a3, fn.A1, fn.A2, fn.A3 = np.empty((0, num_wl)), np.empty((0, num_wl)), np.empty(
                        (0, num_wl)), \
                                                        np.empty((0, num_wl)), np.empty((0, num_wl))

                    layer, d_in_layer = tmm.find_in_structure_with_inf(stack.get_widths(), dist)
                    data_s = tmm.inc_position_resolved(layer, d_in_layer, out_s, coherency_list,
                                                       4 * np.pi * np.imag(stack.get_indices(wavelength)) / wavelength)
                    data_p = tmm.inc_position_resolved(layer, d_in_layer, out_p, coherency_list,
                                                       4 * np.pi * np.imag(stack.get_indices(wavelength)) / wavelength)

                    output['profile'] = 0.5 * (data_s + data_p)

                    for l in layers:
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
                            alpha = np.imag(stack.get_indices(wavelength)[l]) * 4 * np.pi / wavelength
                            fn.a1 = np.vstack((fn.a1, alpha))
                            fn.A2 = np.vstack((fn.A2, alpha * fraction_reaching[l - 1]))
                            fn.a3 = np.vstack((fn.a3, np.zeros((1, num_wl))))
                            fn.A1 = np.vstack((fn.A1, np.zeros((1, num_wl))))
                            fn.A3 = np.vstack((fn.A3, np.zeros((1, num_wl))))

            output['profile_coeff'] = np.stack(
                    (fn.A1, fn.A2, np.real(fn.A3), np.imag(fn.A3), fn.a1, fn.a3))  # shape is (5, n_layers, num_wl)

        wavelength = options['wavelengths']*1e9
        pol =  options['pol']
        angle = options['theta_in']

        coherent = options['coherent'] if 'coherent' in options.keys() else True

        stack = self.stack

        if not coherent:
            coherency_list = self.build_coh_list(options)

        num_wl = len(wavelength)
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

        output['A_per_layer'] = output['A_per_layer'].T

        if profile:
            calculate_profile(layers)

        return output


    def calculate_profile(self, options, layers=None):

        prof = self.calculate(options, profile=True, layers=layers)
        return prof


    def set_widths(self, new_widths):

        self.stack.set_widths(new_widths)


    def build_coh_list(self, options):

        coherency_list = options['coherency_list'] if 'coherency_list' in options.keys() else None
        if coherency_list is not None:
            assert len(coherency_list) == self.stack.num_layers, \
                'Error: The coherency list (passed in the options) must have as many elements (now {}) as the ' \
                'number of layers (now {}).'.format(len(coherency_list), stack.num_layers)

            if self.no_back_reflection:
                coherency_list = ['i'] + coherency_list + ['i', 'i']
            else:
                coherency_list = ['i'] + coherency_list + ['i']

            return coherency_list

        else:
            raise Exception('Error: For incoherent or partly incoherent calculations you must supply the '
                            'coherency_list parameter with as many elements as the number of layers in the '
                            'structure')

