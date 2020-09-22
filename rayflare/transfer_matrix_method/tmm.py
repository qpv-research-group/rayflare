import numpy as np
from solcore.absorption_calculator import tmm_core_vec as tmm
from rayflare.angles import make_angle_vector, fold_phi
from rayflare.config import results_path
import os
import xarray as xr
from sparse import COO, save_npz, load_npz, stack
from solcore.absorption_calculator import OptiStack

degree = np.pi / 180


def TMM(layers, incidence, transmission, surf_name, options,
               coherent=True, coherency_list=None, prof_layers=[], front_or_rear='front', save=True):
    """
    Function which takes a layer stack and creates an angular redistribution matrix.

    :param layers: A list with one or more layers.
    :param transmission: transmission medium
    :param incidence: incidence medium
    :param surf_name: name of the surface (to save the matrices generated.
    :param options: a list of options
    :param coherent: whether or not the layer stack is coherent. If None, it is assumed to be fully coherent
    :param coherency: a list with the same number of entries as the layers, either 'c' for a coherent layer or \
    'i' for an incoherent layer
    :param prof_layers: layers for which the absorption profile should be calculated \
    (if None, do not calculate absorption profile at all)
    :param front_or_rear: a string, either 'front' or 'rear'; front incidence on the stack, from the incidence \
    medium, or rear incidence on the stack, from the transmission medium.

    :return: R and T redistribution matrix fullmat, matrix describing absorption per layer
    """

    def make_matrix_wl(wl):
        # binning into matrix, including phi
        RT_mat = np.zeros((len(theta_bins_in)*2, len(theta_bins_in)))
        A_mat = np.zeros((n_layers, len(theta_bins_in)))

        for i1 in range(len(theta_bins_in)):

            theta = theta_lookup[i1]#angle_vector[i1, 1]

            data = allres.loc[dict(angle=theta, wl=wl)]

            R_prob = np.real(data['R'].data.item(0))
            T_prob = np.real(data['T'].data.item(0))

            Alayer_prob = np.real(data['Alayer'].data)
            phi_out = phis_out[i1]

            #print(R_prob, T_prob)

            # reflection
            phi_int = phi_intv[theta_bins_in[i1]]
            phi_ind = np.digitize(phi_out, phi_int, right=True) - 1
            bin_out_r = np.argmin(abs(angle_vector[:, 0] - theta_bins_in[i1])) + phi_ind

            #print(bin_out_r, i1+offset)

            RT_mat[bin_out_r, i1] = R_prob
            #print(R_prob)
            # transmission
            theta_t = np.abs(-np.arcsin((inc.n(wl * 1e-9) / trns.n(wl * 1e-9)) * np.sin(theta_lookup[i1])) + quadrant)

            #print('angle in, transmitted', angle_vector_th[i1], theta_t)
            # theta switches half-plane (th < 90 -> th >90
            if ~np.isnan(theta_t):

                theta_out_bin = np.digitize(theta_t, theta_intv, right=True) - 1
                phi_int = phi_intv[theta_out_bin]

                phi_ind = np.digitize(phi_out, phi_int, right=True) - 1
                bin_out_t = np.argmin(abs(angle_vector[:, 0] - theta_out_bin)) + phi_ind

                RT_mat[bin_out_t, i1] = T_prob
                #print(bin_out_t, i1+offset)

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

    if os.path.isfile(savepath_RT) and save:
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
            full_dist = np.arange(0, z_limit, options['depth_spacing'])
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

        if profile:
            Aprof_loop = np.empty((n_angles, len(wavelengths), len(dist)))

        tmm_struct = tmm_structure(optlayers, coherent=coherent, coherency_list=coherency_list, no_back_reflection=False)

        for i2, pol in enumerate(pols):

            for i3, theta in enumerate(thetas):

                res = tmm_struct.calculate(wavelengths, angle=theta, pol=pol, profile=profile, layers=prof_layers, depth_spacing = options['depth_spacing'])

                R_loop[:, i3] = np.real(res['R'])
                T_loop[:, i3] = np.real(res['T'])
                Alayer_loop[i3, :, :] = np.real(res['A_per_layer'])

                if profile:
                    Aprof_loop[i3, :, :] = res['profile']

            # sometimes get very small negative values (like -1e-20)
            R_loop[R_loop < 0] = 0
            T_loop[T_loop < 0] = 0
            Alayer_loop[Alayer_loop < 0] = 0

            if front_or_rear == 'rear':
                Alayer_loop = np.flip(Alayer_loop, axis=2)
                #print('flipping')

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

        #print(theta_bins_in)
        mats = [make_matrix_wl(wl) for wl in wavelengths]

        fullmat = stack([item[0] for item in mats])
        A_mat = stack([item[1] for item in mats])

        if save:
            save_npz(savepath_RT, fullmat)
            save_npz(savepath_A, A_mat)

    return fullmat, A_mat #, allres


class tmm_structure:

    def __init__(self, stack, coherent=True, coherency_list=None, no_back_reflection=False):

        """ Set up structure for TMM calculations

        :param stack: an OptiStack object.
        :param wavelength: Wavelengths (in nm) in which calculate the data. An array.
        :param angle: Angle (in radians) of the incident light. Default: 0 (normal incidence).
        :param pol: Polarisation of the light: 's', 'p' or 'u'. Default: 'u' (unpolarised).
        :param coherent: If the light is coherent or not. If not, a coherency list must be added.
        :param coherency_list: A list indicating in which layers light should be treated as coeherent ('c') and in which \
        incoherent ('i'). It needs as many elements as layers in the structure.
        :param profile: whether or not to calculate the absorption profile
        :param layers: indices of the layers in which to calculate the absorption profile. Layer 0 is the incidence medium.
        :return: A dictionary with the R, A and T at the specified wavelengths and angle.
        """

        if 'OptiStack' in str(type(stack)):
            stack.no_back_reflection = no_back_reflection
        else:
            if hasattr(stack, 'substrate'):
                substrate = stack.substrate
            else:
                substrate = None
            stack = OptiStack(stack, no_back_reflection=no_back_reflection,
                              substrate=substrate)

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

        self.stack = stack
        self.coherent = coherent
        self.coherency_list = coherency_list



    def calculate(self, wavelength, angle=0, pol='u', profile=False, layers=None, depth_spacing = 1):
        """ Calculates the reflected, absorbed and transmitted intensity of the structure for the wavelengths and angles
        defined.

        :param stack: an OptiStack object.
        :param wavelength: Wavelengths (in nm) in which calculate the data. An array.
        :param angle: Angle (in radians) of the incident light. Default: 0 (normal incidence).
        :param pol: Polarisation of the light: 's', 'p' or 'u'. Default: 'u' (unpolarised).
        :param coherent: If the light is coherent or not. If not, a coherency list must be added.
        :param coherency_list: A list indicating in which layers light should be treated as coeherent ('c') and in which \
        incoherent ('i'). It needs as many elements as layers in the structure.
        :param profile: whether or not to calculate the absorption profile
        :param layers: indices of the layers in which to calculate the absorption profile. Layer 0 is the incidence medium.

        :return: A dictionary with the R, A and T at the specified wavelengths and angle.
        """

        stack = self.stack
        coherency_list = self.coherency_list
        coherent = self.coherent

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


        # if requested, calculate absorption profile as well

        # layer indices: 0 is incidence, n is transmission medium
        if profile:

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

        output['A_per_layer'] = output['A_per_layer'].T
        return output


    def set_widths(self, new_widths):

        self.stack.set_widths(new_widths)



