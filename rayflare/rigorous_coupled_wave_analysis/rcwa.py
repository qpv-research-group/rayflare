import numpy as np
import tmm
import xarray as xr
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from solcore.absorption_calculator import OptiStack

from rayflare.angles import make_angle_vector, overall_bin
from rayflare.utilities import get_matrices_or_paths

from sparse import COO, save_npz, stack

try:
    import S4
except Exception as err:
    print('WARNING: The RCWA solver will not be available because an S4 installation has not been found.')


def RCWA(structure, size, orders, options, structpath, incidence, transmission, only_incidence_angle=False,
                       prof_layers=None, front_or_rear='front', surf_name='', detail_layer=False, save=True):

    """Calculates the reflection/transmission and absorption redistribution matrices for an interface using
    rigorous coupled-wave analysis.

    :param structure: list of Solcore Layer objects for the surface
    :param size: tuple with the vectors describing the unit cell: ((x1, y1), (x2, y2))
    :param orders: number of RCWA orders to be used for the calculations
    :param options: user options (dictionary or State object)
    :param structpath: file path where matrices will be stored or loaded from
    :param incidence: incidence medium
    :param transmission: transmission medium
    :param only_incidence_angle: if True, the calculations will only be performed for the incidence theta and phi \
         specified in the options (rest of the matrix will be zeros). CURRENTLY DOES NOT WORK CORRECTLY
    :param prof_layers: If no profile calculations are being done, None. Otherwise a list of layers in which the profile
          should be calculated (front non-incidence medium layer has index 1)
    :param front_or_rear: a string, either 'front' or 'rear'; front incidence on the stack, from the incidence
            medium, or rear incidence on the stack, from the transmission medium.
    :param surf_name: name of the surface (to save the matrices generated).
    :param detail_layer:
    :param save: whether to save the redistribution matrices (True/False)
    :return:
    """
    # TODO: when doing unpolarized, why not just set s=0.5 p=0.5 in S4? (Maybe needs to be normalised differently). Also don't know if this is faster,
    # or if internally it will still do s & p separately
    # TODO: if incidence angle is zero, s and p polarization are the same so no need to do both

    existing_mats, path_or_mats = get_matrices_or_paths(structpath, surf_name, front_or_rear, prof_layers)

    if existing_mats:
        return path_or_mats

    else:
        wavelengths = options['wavelengths']

        if front_or_rear == 'front':
            layers = structure
            trns = transmission
            inc = incidence

        else:
            layers = structure[::-1]
            trns = incidence
            inc = transmission
            if prof_layers is not None:
                prof_layers = np.sort(len(layers) - np.array(prof_layers) + 1).tolist()

        # write a separate function that makes the OptiStack structure into an S4 object, defined materials etc.
        geom_list = [layer.geometry for layer in structure]
        geom_list.insert(0, {})  # incidence medium
        geom_list.append({})  # transmission medium

        ## Materials for the shapes need to be defined before you can do .SetRegion
        shape_mats, geom_list_str = necessary_materials(geom_list)

        shapes_oc = np.zeros((len(wavelengths), len(shape_mats)), dtype=complex)

        for i1, x in enumerate(shape_mats):
            shapes_oc[:, i1] = (x.n(wavelengths) + 1j*x.k(wavelengths))**2

        stack_OS = OptiStack(layers, no_back_reflection=False)
        widths = stack_OS.get_widths()
        layers_oc = np.zeros((len(wavelengths), len(structure)+2), dtype=complex)

        if prof_layers is not None:

            z_limit = np.sum(np.array(widths[1:-1]))
            full_dist = np.arange(0, z_limit, options['depth_spacing'] * 1e9)
            layer_start = np.insert(np.cumsum(np.insert(widths[1:-1], 0, 0)), 0, 0)
            layer_end = np.cumsum(np.insert(widths[1:-1], 0, 0))

            dist = []

            for l in prof_layers:
                dist = np.hstack((dist, full_dist[np.all((full_dist >= layer_start[l], full_dist < layer_end[l]), 0)]))

        else:
            dist = None


        layers_oc[:, 0] = (inc.n(wavelengths))**2#+ 1j*inc.k(wavelengths))**2
        layers_oc[:, -1] = (trns.n(wavelengths) + 1j*trns.k(wavelengths))**2

        for i1, x in enumerate(layers):
            layers_oc[:, i1+1] = (x.material.n(wavelengths) + 1j*x.material.k(wavelengths))**2

        shapes_names = [str(x) for x in shape_mats]

        phi_sym = options['phi_symmetry']
        n_theta_bins = options['n_theta_bins']
        c_az = options['c_azimuth']
        pol = options['pol']

        # RCWA options
        S4_options = dict(LatticeTruncation='Circular',
                            DiscretizedEpsilon=False,
                            DiscretizationResolution=8,
                            PolarizationDecomposition=False,
                            PolarizationBasis='Default',
                            LanczosSmoothing=False,
                            SubpixelSmoothing=False,
                            ConserveMemory=False,
                            WeismannFormulation=False,
                            Verbosity=0)

        user_options = options['S4_options'] if 'S4_options' in options.keys() else {}
        S4_options.update(user_options)

        theta_intv, phi_intv, angle_vector = make_angle_vector(n_theta_bins, phi_sym, c_az)

        if only_incidence_angle:
            thetas_in = np.array([options['theta_in']])
            phis_in = np.array([options['phi_in']])
        else:
            angles_in = angle_vector[:int(len(angle_vector) / 2), :]
            thetas_in = angles_in[:, 1]
            phis_in = angles_in[:, 2]

        # angle in degrees
        thetas_in = thetas_in*180/np.pi
        phis_in = phis_in*180/np.pi
        # initialise_S has to happen inside parallel job (get Pickle errors otherwise);
        # just pass relevant optical constants for each wavelength, like for RT

        angle_vector_0 = angle_vector[:, 0]

        if front_or_rear == "front":
            side = 1
        else:
            side = -1

        if options['parallel']:
            allres = Parallel(n_jobs=options['n_jobs'])(delayed(RCWA_wl)
                                                        (wavelengths[i1]*1e9, geom_list_str, layers_oc[i1], shapes_oc[i1], shapes_names,
                                                         pol, thetas_in, phis_in, widths, size,
                                                         orders, phi_sym, theta_intv, phi_intv,
                                                         angle_vector_0, S4_options, detail_layer, side, dist)
                                                        for i1 in range(len(wavelengths)))

        else:
            allres = [RCWA_wl(wavelengths[i1]*1e9, geom_list_str, layers_oc[i1], shapes_oc[i1], shapes_names,
                              pol, thetas_in, phis_in, widths, size, orders, phi_sym, theta_intv, phi_intv,
                              angle_vector_0, S4_options, detail_layer, side, dist)
                      for i1 in range(len(wavelengths))]


        A_mat = np.stack([item[2] for item in allres])
        full_mat = stack([item[3] for item in allres])

        A_mat = COO(A_mat)

        if save:
            save_npz(path_or_mats[0], full_mat)
            save_npz(path_or_mats[1], A_mat)

        if prof_layers is not None:
            prof_mat = np.stack([item[5] for item in allres])
            intgr_mat = np.stack([item[6] for item in allres])

            prof_mat[prof_mat < 0] = 0

            profile = xr.DataArray(prof_mat,
                                    dims=['wl', 'z', 'global_index'],
                                    coords={'wl': wavelengths, 'z': dist,
                                            'global_index': np.arange(0, prof_mat.shape[2])})

            profile.transpose('wl', 'global_index', 'z')

            intgr = xr.DataArray(intgr_mat,
                                 dims=['wl', 'global_index'],
                                 coords={'wl': wavelengths,
                                         'global_index': np.arange(0, prof_mat.shape[2])})

            intgr.name = 'intgr'
            profile.name = 'profile'
            prof_dataset = xr.merge([intgr, profile])

            if save:
                prof_dataset.to_netcdf(path_or_mats[2])


        if prof_layers is not None:
            return full_mat, A_mat, prof_dataset

        else:
            return full_mat, A_mat


def RCWA_wl(wl, geom_list, l_oc, s_oc, s_names, pol, theta, phi, widths, size, orders, phi_sym,
            theta_intv, phi_intv, angle_vector_0, S4_options, layer_details = False, side=1, dist=None):

    S = initialise_S(size, orders, geom_list, l_oc, s_oc, s_names, widths, S4_options)

    n_inc = np.real(np.sqrt(l_oc[0]))

    G_basis = np.array(S.GetBasisSet())

    f_mat = S.GetReciprocalLattice()

    fg_1x = f_mat[0][0]
    fg_1y = f_mat[0][1]
    fg_2x = f_mat[1][0]
    fg_2y = f_mat[1][1]

    R = np.zeros((len(theta)))
    T = np.zeros((len(theta)))
    A_layer = np.zeros((len(theta), len(widths)-2))

    if dist is not None:
        mat_prof = np.zeros((len(dist), int(len(angle_vector_0)/2)))
        mat_intgr = np.zeros(int(len(angle_vector_0)/2))

    mat_RT = np.zeros((len(angle_vector_0), int(len(angle_vector_0)/2)))
    mat_int = np.zeros((len(angle_vector_0), int(len(angle_vector_0)/2)))

    if side == 1:
        in_bin = np.arange(len(theta))

    else:
        binned_theta_in = np.digitize(np.pi-np.pi*theta/180, theta_intv, right=True) - 1
        phi_in = xr.DataArray(np.pi*phi/180,
                              coords={'theta_bin': (['angle_in'], binned_theta_in)},
                              dims=['angle_in'])

        in_bin = phi_in.groupby('theta_bin').map(overall_bin,
                                                   args=(phi_intv, angle_vector_0)).data - int(len(angle_vector_0)/2)

    print(wl)

    for i1, (th, ph) in enumerate(zip(theta, phi)):
        if pol in 'sp':
            if pol == 's':
                s = 1
                p = 0
            elif pol == 'p':
                s = 0
                p = 1

            S.SetExcitationPlanewave((th, ph), s, p, 0)
            S.SetFrequency(1 / wl)
            out, R_pfbo, T_pfbo, R_pfbo_int = rcwa_rat(S, len(widths), th, n_inc, layer_details)
            T[in_bin[i1]] = out['T']
            A_layer[in_bin[i1]] = rcwa_absorption_per_layer(S, len(widths), th, n_inc)

        else:

            S.SetFrequency(1 / wl)
            S.SetExcitationPlanewave((th, ph), 0, 1, 0)  # p-polarization
            out_p, R_pfbo_p, T_pfbo_p, R_pfbo_int_p = rcwa_rat(S, len(widths), th, n_inc, layer_details)
            Ap = rcwa_absorption_per_layer(S, len(widths), th, n_inc)

            S.SetExcitationPlanewave((th, ph), 1, 0, 0)  # s-polarization
            out_s, R_pfbo_s, T_pfbo_s, R_pfbo_int_s = rcwa_rat(S, len(widths), th, n_inc, layer_details)
            As = rcwa_absorption_per_layer(S, len(widths), th, n_inc)

            R_pfbo = R_pfbo_s + R_pfbo_p # this will not be normalized correctly! fix this outside if/else
            T_pfbo = 0.5*(T_pfbo_s + T_pfbo_p)

            T[in_bin[i1]] = 0.5 * (out_p['T'] + out_s['T'])
            A_layer[in_bin[i1]] = 0.5*(Ap+As)


        if dist is not None:

            A = np.sum(A_layer[in_bin[i1]])
            if len(pol) == 2:

                S.SetExcitationPlanewave((th, ph), pol[0], pol[1], 0)
                S.SetFrequency(1 / wl)

                for j, d in enumerate(dist):
                    layer, d_in_layer = tmm.find_in_structure_with_inf(widths,
                                                                       d)  # don't need to change this
                    layer_name = 'layer_' + str(layer + 1)  # layer_1 is air above so need to add 1
                    data = rcwa_position_resolved(S, layer_name, d_in_layer, A, th, np.sqrt(l_oc[0]))
                    mat_prof[j, in_bin[i1]] = data

                mat_intgr[in_bin[i1]] =  A


            else:
                if pol in 'sp':
                    if pol == 's':
                        s = 1
                        p = 0
                    elif pol == 'p':
                        s = 0
                        p = 1

                    S.SetExcitationPlanewave((th, ph), s, p, 0)

                    S.SetFrequency(1 / wl)

                    for j, d in enumerate(dist):
                        layer, d_in_layer = tmm.find_in_structure_with_inf(widths,
                                                                           d)  # don't need to change this
                        layer_name = 'layer_' + str(layer + 1)  # layer_1 is air above so need to add 1
                        data = rcwa_position_resolved(S, layer_name, d_in_layer, A, th, np.sqrt(l_oc[0]))
                        mat_prof[j, in_bin[i1]] = data

                    mat_intgr[in_bin[i1]] = A

                else:

                    S.SetFrequency(1 / wl)

                    for j, d in enumerate(dist):
                        layer, d_in_layer = tmm.find_in_structure_with_inf(widths, d)
                        layer_name = 'layer_' + str(layer + 1)  # layer_1 is air above so need to add 1
                        S.SetExcitationPlanewave((th, ph), 0, 1, 0)  # p-polarization
                        data_p = rcwa_position_resolved(S, layer_name, d_in_layer, np.sum(Ap), th, np.sqrt(l_oc[0]))
                        S.SetExcitationPlanewave((th, ph), 1, 0, 0)  # p-polarization
                        data_s = rcwa_position_resolved(S, layer_name, d_in_layer, np.sum(As), th, np.sqrt(l_oc[0]))
                        mat_prof[j, in_bin[i1]] = 0.5 * (data_s + data_p)

                    mat_intgr[in_bin[i1]] = 0.5 * (np.sum(Ap) + np.sum(As))


        R[in_bin[i1]] = 1 - T[in_bin[i1]] - np.sum(A_layer[in_bin[i1]])
        R_pfbo = np.real(R_pfbo)
        T_pfbo = np.real(T_pfbo)

        fi_x = np.real((np.real(np.sqrt(l_oc[0])) / wl) * np.sin(th * np.pi / 180) *
                       np.sin(ph * np.pi / 180))
        fi_y = np.real((np.real(np.sqrt(l_oc[0])) / wl) * np.sin(th * np.pi / 180) *
                       np.cos(ph * np.pi / 180))

        fr_x = fi_x + G_basis[:,0]*fg_1x + G_basis[:,1]*fg_2x
        fr_y = fi_y + G_basis[:,0]*fg_1y + G_basis[:,1]*fg_2y

        fr_z = np.sqrt((l_oc[0]/(wl**2))-fr_x**2 - fr_y**2)
        ft_z = np.sqrt((l_oc[-1]/(wl**2))-fr_x**2 - fr_y**2)

        phi_rt = np.nan_to_num(np.arctan(fr_x/fr_y))
        phi_rt = fold_phi(phi_rt, phi_sym)

        theta_r = np.real(np.arccos(fr_z/np.sqrt(fr_x**2 + fr_y**2 + fr_z**2)))
        theta_t = np.pi-np.real(np.arccos(ft_z/np.sqrt(fr_x**2 + fr_y**2 + ft_z**2)))

        np_r = theta_r == np.pi/2 # non-propagating reflected orders
        np_t = theta_t == np.pi/2 # non-propagating transmitted orders

        if side == -1:
            theta_r = np.pi - theta_r
            theta_t = np.pi - theta_t

        R_pfbo[np_r] = 0
        T_pfbo[np_t] = 0

        R_pfbo[np.abs(R_pfbo < 1e-16)] = 0 # sometimes get very small negative valyes
        T_pfbo[np.abs(T_pfbo < 1e-16)] = 0

        Rsum = np.sum(R_pfbo)
        R_pfbo = (R[in_bin[i1]]/Rsum)*R_pfbo

        theta_r[theta_r == 0] = 1e-10
        theta_t[theta_t == 0] = 1e-10
        phi_rt[phi_rt == 0] = 1e-10

        theta_r_bin = np.digitize(theta_r, theta_intv, right=True) - 1
        theta_t_bin = np.digitize(theta_t, theta_intv, right=True) - 1

        for i2 in np.nonzero(R_pfbo)[0]:
            phi_ind = np.digitize(phi_rt[i2], phi_intv[theta_r_bin[i2]], right=True) - 1
            bini = np.argmin(abs(angle_vector_0 -theta_r_bin[i2])) + phi_ind
            mat_RT[bini, in_bin[i1]] = mat_RT[bini, i1] + R_pfbo[i2]

        for i2 in np.nonzero(T_pfbo)[0]:
            phi_ind = np.digitize(phi_rt[i2], phi_intv[theta_t_bin[i2]], right=True) - 1
            bini = np.argmin(abs(angle_vector_0 -theta_t_bin[i2])) + phi_ind
            mat_RT[bini, in_bin[i1]] = mat_RT[bini, i1] + T_pfbo[i2]

        if layer_details:

            if pol not in 'sp':
                R_pfbo_int = (R_pfbo_int_p + R_pfbo_int_s)/2
            f_z = np.sqrt((l_oc[layer_details] / (wl ** 2)) - fr_x ** 2 - fr_y ** 2)

            theta_l = np.real(np.arccos(f_z / np.sqrt(fr_x ** 2 + fr_y ** 2 + f_z ** 2)))

            theta_l[theta_l == 0] = 1e-10

            np_l = theta_l == np.pi / 2  # non-propagating reflected orders

            R_pfbo_int[np_l] = 0

            R_pfbo_int[np.abs(R_pfbo_int < 1e-16)] = 0  # sometimes get very small negative valyes
            theta_l_bin = np.digitize(theta_l, theta_intv, right=True) - 1
            for i2 in np.nonzero(R_pfbo_int)[0]:
                phi_ind = np.digitize(phi_rt[i2], phi_intv[theta_l_bin[i2]], right=True) - 1
                bini = np.argmin(abs(angle_vector_0 - theta_l_bin[i2])) + phi_ind
                mat_int[bini, i1] = mat_int[bini, i1] + R_pfbo_int[i2]


    mat_RT = COO(mat_RT)
    mat_int = COO(mat_int)

    # want to output R, T, A_layer (in case doing single angle of incidence)
    # also want to output transmission and reflection efficiency/power flux per order and the angles (theta and phi)
    # relating to that order.
    # Theta depends on the medium and so is different for transmisson and reflection. Phi is the same.

    if side == -1:
        A_layer = np.flip(A_layer, 1)

        if dist is not None:
            mat_prof = np.flip(mat_prof, 0)

    if dist is not None:
        return R, T, A_layer.T, mat_RT, mat_int, mat_prof, mat_intgr

    else:
        return R, T, A_layer.T, mat_RT, mat_int#, mat_T


def fold_phi(phis, phi_sym):
    return (abs(phis//np.pi)*2*np.pi + phis) % phi_sym


def rcwa_rat(S, n_layers, theta, n_inc, det_l=False):
    # TODO: check normalization of pfbo values. Also, theta correction should happen inside here rather than externally
    below = 'layer_' + str(n_layers)  # identify which layer is the transmission medium

    # power flux by order backwards in layer_1: if incidence medium has n=1, this gives a (real part of) sum equal to R

    # transmission power flux by order always sums to T, regardless of optical constants of transmission/incidence medium
    R = -n_inc*np.real(S.GetPowerFlux('layer_1')[1])/np.cos(theta*np.pi/180)  # GetPowerFlux gives forward & backward Poynting vector, so sum to get power flux

    # this should be correct answer in 'far field': anythng that doesn't go into the surface must be reflected. but if n_incidence != 1
    # can get odd effects.

    R_pfbo = -np.array(S.GetPowerFluxByOrder('layer_1'))[:,1] # real part of backwards power flow. Not normalised correctly!

    if det_l:
        layer_name = 'layer_' + str(det_l + 2)
        R_pfbo_int = -np.array(S.GetPowerFluxByOrder(layer_name))[:, 1]

    else:
        R_pfbo_int = 0

    T = n_inc*sum(S.GetPowerFlux(below))/np.cos(theta*np.pi/180)

    T_pfbo = n_inc*np.real(np.sum(np.array(S.GetPowerFluxByOrder(below)), 1))/np.cos(theta*np.pi/180)
    return {'R': np.real(R), 'T': np.real(T)}, R_pfbo, T_pfbo, R_pfbo_int


def initialise_S(size, orders, geom_list, mats_oc, shapes_oc, shape_mats, widths, options):

    S = S4.New(size, orders)

    S.SetOptions(
        LatticeTruncation = options['LatticeTruncation'],
        DiscretizedEpsilon = options['DiscretizedEpsilon'],
        DiscretizationResolution = options['DiscretizationResolution'],
        PolarizationDecomposition = options['PolarizationDecomposition'],
        PolarizationBasis = options['PolarizationBasis'],
        LanczosSmoothing = options['LanczosSmoothing'],
        SubpixelSmoothing = options['SubpixelSmoothing'],
        ConserveMemory = options['ConserveMemory'],
        WeismannFormulation = options['WeismannFormulation'],
        Verbosity = options['Verbosity']
    )


    for i1, sh in enumerate(shapes_oc):  # create the materials needed for all the shapes in S4
        S.SetMaterial('shape_mat_' + str(i1 + 1), sh)

    for i1, _ in enumerate(widths):  # create 'dummy' materials for base layers including incidence and transmission media
        S.SetMaterial('layer_' + str(i1 + 1), mats_oc[i1])  # This is not strictly necessary but it means S.SetExcitationPlanewave
        # can be done outside the wavelength loop in calculate_rat_rcwa

    for i1, wid in enumerate(widths):  # set base layers
        layer_name = 'layer_' + str(i1 + 1)

        if wid == float('Inf'):
            S.AddLayer(layer_name, 0, layer_name)  # Solcore4 has incidence and transmission media widths set to Inf;
            # in S4 they have zero width
        else:
            S.AddLayer(layer_name, wid, layer_name)

        geometry = geom_list[i1]

        if bool(geometry):

            for shape in geometry:
                mat_name = 'shape_mat_' + str(shape_mats.index(str(shape['mat'])) + 1)
                if shape['type'] == 'circle':
                    S.SetRegionCircle(layer_name, mat_name, shape['center'], shape['radius'])
                elif shape['type'] == 'ellipse':
                    S.SetRegionEllipse(layer_name, mat_name, shape['center'], shape['angle'], shape['halfwidths'])
                elif shape['type'] == 'rectangle':
                    S.SetRegionRectangle(layer_name, mat_name, shape['center'], shape['angle'], shape['halfwidths'])
                elif shape['type'] == 'polygon':
                    S.SetRegionPolygon(layer_name, mat_name, shape['center'], shape['angle'], shape['vertices'])

    return S


def necessary_materials(geom_list):
    shape_mats = []
    geom_list_str = [None] * len(geom_list)
    for i1, geom in enumerate(geom_list):
        if bool(geom):
            shape_mats.append([x['mat'] for x in geom])
            geom_list_str[i1] =[{} for _ in range(len(geom))]
            for i2, g in enumerate(geom):
                for item in g.keys():
                    if item != 'mat':
                        geom_list_str[i1][i2][item] = g[item]
                    else:
                        geom_list_str[i1][i2][item] = str(g[item])

    return list(set([val for sublist in shape_mats for val in sublist])), geom_list_str


def rcwa_position_resolved(S, layer, depth, A, theta, n_inc):
    if A > 0:
        delta = 1e-9
        power_difference = np.real(
            sum(S.GetPowerFlux(layer, depth - delta)) - sum(S.GetPowerFlux(layer, depth + delta)))
        return n_inc * power_difference / (2 * delta * np.cos(theta*np.pi/180))  # absorbed energy density normalised to total absorption
    else:
        return 0

def rcwa_absorption_per_layer(S, n_layers, theta, n_inc):
    # layer 1 is incidence medium, layer n is the transmission medium
    A = np.empty(n_layers-2)
    for i1, layer in enumerate(np.arange(n_layers-2)+2):
        A[i1] = np.real(sum(S.GetPowerFlux('layer_' + str(layer))) -
                        sum(S.GetPowerFlux('layer_' + str(layer+1))))

    A = n_inc*np.array([x if x > 0 else 0 for x in A])/np.cos(theta*np.pi/180)

    return A

def rcwa_absorption_per_layer_order(S, n_layers, theta, n_inc):
    # layer 1 is incidence medium, layer n is the transmission medium
    n_orders =  len(S.GetBasisSet())
    A_per_order = np.empty((n_layers-2, n_orders))
    for i1, layer in enumerate(np.arange(n_layers-2)+2):

        per_order_top = np.sum(np.array(S.GetPowerFluxByOrder('layer_' + str(layer))), 1)
        per_order_bottom = np.sum(np.array(S.GetPowerFluxByOrder('layer_' + str(layer+1))), 1)

        A_per_order[i1,:] = n_inc*np.real(per_order_top - per_order_bottom)/np.cos(theta*np.pi/180)

    return A_per_order


def get_reciprocal_lattice(size, orders):
    """
    Returns the reciprocal lattice as defined in S4 (note that this is missing a foctor of 2pi compared to the
    standard definition).
    :param size: lattice vectors in real space ((ax, ay), (bx, by))
    :type size:
    :param orders: number of Fourier orders to keep
    :type orders:
    :return: reciprocal lattice (tuple)
    :rtype:
    """

    S = S4.New(size, orders)
    f_mat = S.GetReciprocalLattice()

    return f_mat


class rcwa_structure:
    # TODO: make this accept an OptiStack, and check the substrate of the SolarCell object
    """ Calculates the reflected, absorbed and transmitted intensity of the structure for the wavelengths and angles
    defined using an RCWA method implemented using the S4 package.

    :param structure: A Solcore Structure/SolarCell object with layers and materials. Alternatively, you can supply
           a list which can contain any mixture of Solcore Layer objects and layers defined in one of the two following ways:
            - 1. A list of length 4, for materials with a constant refractive index. The list entries are:
                 [width of the layer in nm, real part of refractive index (n), imaginary part of refractive index (k),
                 geometry]
            - 2. A list of length 5, for materials with a wavelength-dependent refractive index. The list entries are:
                 [width of the layer in nm, wavelengths, n at these wavelengths, k at these wavelengths, geometry]
    :param size: tuple with the vectors describing the unit cell: ((x1, y1), (x2, y2))
    :param incidence: semi-infinite incidence medium
    :param transmission: semi-infinite transmission medium (substrate)
    """

    def __init__(self, structure, size, options, incidence, transmission):

        self.transmission = transmission
        self.incidence = incidence

        wavelengths = options['wavelengths']

        geom_list = []
        list_for_OS = []

        for layer in structure:
            if isinstance(layer, list):
                if len(layer) == 4:
                    geom_list.append(layer[3])
                    list_for_OS.append([layer[0], wavelengths, layer[1]*np.ones_like(wavelengths), layer[2]*np.ones_like(wavelengths)])

                if len(layer) == 5:
                    geom_list.append(layer[4])
                    list_for_OS.append(layer[:4])

            else:
                geom_list.append(layer.geometry)
                list_for_OS.append(layer)

        geom_list.insert(0, {})  # incidence medium
        geom_list.append({})  # transmission medium

        self.list_for_OS = list_for_OS

        ## Materials for the shapes need to be defined before you can do .SetRegion
        shape_mats, geom_list_str = necessary_materials(geom_list)

        self.shape_mats = shape_mats

        self.update_oc(wavelengths)

        shapes_names = [str(x) for x in shape_mats]

        # depth_spacing = options['depth_spacing']

        # RCWA options
        S4_options = dict(LatticeTruncation='Circular',
                            DiscretizedEpsilon=False,
                            DiscretizationResolution=8,
                            PolarizationDecomposition=False,
                            PolarizationBasis='Default',
                            LanczosSmoothing=False,
                            SubpixelSmoothing=False,
                            ConserveMemory=False,
                            WeismannFormulation=False,
                            Verbosity=0)

        user_options = options['S4_options'] if 'S4_options' in options.keys() else {}
        S4_options.update(user_options)

        self.geom_list = geom_list_str
        self.shapes_names = shapes_names
        self.size = size


    def update_oc(self, wavelengths):
        # wavelength in m
        shapes_oc = np.zeros((len(wavelengths), len(self.shape_mats)), dtype=complex)

        for i1, x in enumerate(self.shape_mats):
            if isinstance(x, list):
                if len(x) == 3:
                    shapes_oc[:, i1] = np.ones_like(wavelengths)*(x[1] + 1j*x[2])**2

                if len(x) == 4:
                    shapes_oc[:, i1] = (x[2] + 1j*x[3])**2

            else:
                shapes_oc[:, i1] = (x.n(wavelengths) + 1j * x.k(wavelengths)) ** 2

        # prepare to pass to OptiStack.

        stack_OS = OptiStack(self.list_for_OS, no_back_reflection=False,
                             substrate=self.transmission, incidence=self.incidence)

        layers_oc = (np.array(stack_OS.get_indices(wavelengths*1e9))**2).T
        widths = stack_OS.get_widths()

        self.widths = widths
        self.shapes_oc = shapes_oc
        self.layers_oc = layers_oc
        self.current_wavelengths = wavelengths


    def calculate(self, options):
        """ Calculates the reflected, absorbed and transmitted intensity of the structure for the wavelengths and angles
             defined.

             :param options: options for the calculation. The key entries are:

                    - wavelength: Wavelengths (in m) in which calculate the data. An array.
                    - theta_in: polar angle (in radians) of the incident light.
                    - phi_in: azimuthal angle (in radians) of the incident light.
                    - pol: Polarisation of the light: 's', 'p' or 'u'.
                    - orders: number of Fourier orders to retain in the RCWA calculation
                    - parallel: True of False, whether or not to run simulation on parallel
                    - n_jobs: if parallel, specifies how many cores are used. See joblib documentation
                    - A_per_order: whether or not to calculate the absorption per diffraction order
                    - S4_options: options passed to the S4 solver.

             :return: A dictionary with the R, A and T at the specified wavelengths and angle.
             """

        wl = options['wavelengths']*1e9

        if not np.all(options['wavelengths'] == self.current_wavelengths):
            # need to update list of optical constants for correct wavelengths
            self.update_oc(options['wavelengths'])

        if options['parallel']:
            allres = Parallel(n_jobs=options['n_jobs'])(delayed(RCWA_structure_wl)
                                                        (wl[i1], self.geom_list, self.layers_oc[i1], self.shapes_oc[i1],
                                                         self.shapes_names, options['pol'], options['theta_in']*180/np.pi, options['phi_in']*180/np.pi,
                                                         self.widths, self.size,
                                                         options['orders'], options['A_per_order'], options['S4_options'])
                                                        for i1 in range(len(wl)))

        else:
            allres = [
                RCWA_structure_wl(wl[i1], self.geom_list, self.layers_oc[i1], self.shapes_oc[i1],
                                                         self.shapes_names, options['pol'], options['theta_in']*180/np.pi, options['phi_in']*180/np.pi,
                                                         self.widths, self.size,
                                                         options['orders'], options['A_per_order'], options['S4_options'])
                for i1 in range(len(wl))]

        R = np.real(np.stack([item[0] for item in allres]))
        T = np.real(np.stack([item[1] for item in allres]))
        A_mat = np.real(np.stack([item[2] for item in allres]))

        self.rat_output_A = np.sum(A_mat, 1)  # used for profile calculation

        if options['A_per_order']:

            A_order = np.real(np.stack([item[3] for item in allres]))

            S_for_orders = initialise_S(self.size, options['orders'], self.geom_list, self.layers_oc[0],
                             self.shapes_oc[0], self.shapes_names, self.widths, options['S4_options'])

            basis_set = S_for_orders.GetBasisSet()
            f_mat = S_for_orders.GetReciprocalLattice()

            results = {'R': R, 'T': T, 'A_per_layer': A_mat, 'A_layer_order': A_order, 'basis_set': basis_set, 'reciprocal': f_mat}
            self.results = results
            return results

        else:
            results = {'R': R, 'T': T, 'A_per_layer': A_mat}
            self.results = results
            return results


    def calculate_profile(self, options):
        """ It calculates the absorbed energy density within the material.

        In principle this has units of [power]/[volume], but we can express it as a multiple of incoming light power
        density on the material, which has units [power]/[area], so that absorbed energy density has units of 1/[length].'

        """

        wl = options['wavelengths'] * 1e9

        if not np.all(options['wavelengths'] == self.current_wavelengths):
            self.update_oc(options['wavelengths'])
            # if total R, A, T have already been calculated, it was for the wrong wavelengths
            self.calculate(options)

        if not hasattr(self, 'rat_output_A'):
            # Need to calculate R, A, T first
            self.calculate(options)

        dist = options['z_points'] if 'z_points' in options.keys() else None
        z_limit = options['z_limit'] if 'z_limit' in options.keys() else None

        step_size = options['depth_spacing']*1e9

        if dist is None:
            if z_limit is None:
                z_limit = np.sum(self.widths[1:-1])
            dist = np.arange(0, z_limit, step_size)

        self.dist = dist

        if options['parallel']:
            allres = Parallel(n_jobs=options['n_jobs'])(delayed(RCWA_wl_prof)
                                                             (wl[i1], self.rat_output_A[i1],
                                                              dist,
                                                              self.geom_list,
                                                              self.layers_oc[i1], self.shapes_oc[i1],
                                                              self.shapes_names, options['pol'],
                                                              options['theta_in'] * 180 / np.pi,
                                                              options['phi_in'] * 180 / np.pi,
                                                              self.widths, self.size,
                                                              options['orders'], options['S4_options'])
                                                             for i1 in range(len(wl)))

        else:
            allres = [
                RCWA_wl_prof(wl[i1], self.rat_output_A[i1],
                                                              dist,
                                                              self.geom_list,
                                                              self.layers_oc[i1], self.shapes_oc[i1],
                                                              self.shapes_names, options['pol'],
                                                              options['theta_in'] * 180 / np.pi,
                                                              options['phi_in'] * 180 / np.pi,
                                                              self.widths, self.size,
                                                              options['orders'], options['S4_options'])
                for i1 in range(len(wl))]

        output = np.real(np.stack(allres))

        to_return = self.results
        to_return['profile'] = output

        return to_return

    def save_layer_postscript(self, layer_index, options, filename):
        # layer_index: layer 0 is the incidence medium
        S = initialise_S(self.size, 1, self.geom_list, self.layers_oc[0], self.shapes_oc[0], self.shapes_names, self.widths, options.S4_options)
        S.OutputLayerPatternPostscript(Layer='layer_' + str(layer_index + 1), Filename=filename + '.ps')


    def get_fourier_epsilon(self, layer_index, wavelength, options, extent=None, n_points=200, plot=True):
        """
        Get the Fourier-decomposed epsilon scanning across x-y points for some layer in the structure for the number
        of order specified in the options for the structure. Can also plot this automatically.

        :param layer_index: index of the layer in which to get epsilon. layer 0 is the incidence medium, layer 1 is the first layer in the stack, etc.
        :param wavelength: wavelength (in nm) at which to get epsilon
        :param extent: range of x/y values in format [[x_min, x_max], [y_min, y_max]]. Default is 'None', will choose a reasonable area based \
        on the unit cell size by default
        :param n_points: number of points to scan across in the x and y directions
        :param plot: plot the results (True or False, default True)

        :return: xs, ys, a_r, a_i. The x points, y points, and the real and imaginary parts of the dielectric function.
        """

        wl_ind = np.argmin(np.abs(self.current_wavelengths * 1e9 - wavelength))

        if extent is None:
            xdim = np.max(abs(np.array(self.size)[:, 0]))
            ydim = np.max(abs(np.array(self.size)[:, 1]))
            xs = np.linspace(-1.5 * xdim, 1.5 * xdim, n_points)
            ys = np.linspace(-1.5 * ydim, 1.5 * ydim, n_points)

        else:
            xs = np.linspace(extent[0][0], extent[0][1], n_points)
            ys = np.linspace(extent[1][0], extent[1][1], n_points)

        xys = np.meshgrid(xs, ys, indexing='ij')

        a_r = np.zeros(len(xs) * len(ys))
        a_i = np.zeros(len(xs) * len(ys))
        S = initialise_S(self.size, options.orders, self.geom_list, self.layers_oc[wl_ind], self.shapes_oc[wl_ind], self.shapes_names,
                         self.widths, options.S4_options)

        if layer_index > 0:
            depth = np.cumsum([0] + self.widths[1:-1] + [0])[layer_index - 1] + 1e-10

        else:
            depth = -1

        for i, (xi, yi) in enumerate(zip(xys[0].flatten(), xys[1].flatten())):
            calc = S.GetEpsilon(xi, yi, depth)
            a_r[i] = np.real(calc)
            a_i[i] = np.imag(calc)

        a_r = a_r.reshape((len(xs), len(ys)))
        a_i = a_i.reshape((len(xs), len(ys)))

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(7, 2.6))
            im1 = axs[0].pcolormesh(xs, ys, a_r.T, cmap='magma')
            fig.colorbar(im1, ax=axs[0])
            axs[0].set_xlabel('x (nm)')
            axs[0].set_ylabel('y (nm)')
            axs[0].set_aspect(aspect=1)

            im2 = axs[1].pcolormesh(xs, ys, a_i.T, cmap='magma')
            fig.colorbar(im2, ax=axs[1])
            axs[1].set_xlabel('x (nm)')
            axs[1].set_ylabel('y (nm)')
            axs[1].set_aspect(aspect=1)
            plt.show()

        return xs, ys, a_r, a_i

    def get_fields(self, layer_index, wavelength, options, extent=None, depth=1e-10, n_points=200, plot=True):
        """
        Get the components of the E and H fields at a specific depth in a layer, over a range of x/y points. Can also plot results
        automatically. Uses the S4 function GetFields().

        :param layer_index: index of the layer in which to get epsilon. layer 0 is the incidence medium, layer 1 is the first layer in the stack, etc.
        :param wavelength: wavelength (in nm) at which to get epsilon
        :param extent: range of x/y values in format [[x_min, x_max], [y_min, y_max]]. Default is 'None', will choose a reasonable area based \
        on the unit cell size by default
        :param depth: depth in the layer (from the top of the layer) in nm at which to calculate the fields
        :param n_points: number of points to scan across in the x and y directions
        :param plot: plot the results (True or False, default True)

        :return: xs, ys, E, H, E_mag, H_mag. x points, y points, the complex (x, y, z) components of the E field, the complex (x, y, z) components of the H field,
                    the magnitude of the E-field, the magnitude of the H-field. The magnitude is given by sqrt(abs(Ex^2 + Ey^2 + Ez^2))
        """

        def vs_pol(s, p):
            S.SetExcitationPlanewave((options['theta_in']*180/np.pi, options['phi_in']*180/np.pi), s, p, 0)
            S.SetFrequency(1 / wavelength)

        wl_ind = np.argmin(np.abs(self.current_wavelengths * 1e9 - wavelength))

        S = initialise_S(self.size, options.orders, self.geom_list, self.layers_oc[wl_ind],
                         self.shapes_oc[wl_ind],
                         self.shapes_names,
                         self.widths, options.S4_options)

        if len(options['pol']) == 2:

            vs_pol(options['pol'][0], options['pol'][1])

        else:
            if options['pol'] in 'sp':
                vs_pol(int(options['pol'] == "s"), int(options['pol'] == "p"))

        if layer_index > 0:
            depth = np.cumsum([0] + self.widths[1:-1] + [0])[layer_index - 1] + depth

        if extent is None:
            xdim = np.max(abs(np.array(self.size)[:, 0]))
            ydim = np.max(abs(np.array(self.size)[:, 1]))
            xs = np.linspace(-1.5 * xdim, 1.5 * xdim, n_points)
            ys = np.linspace(-1.5 * ydim, 1.5 * ydim, n_points)

        else:
            xs = np.linspace(extent[0][0], extent[0][1], n_points)
            ys = np.linspace(extent[1][0], extent[1][1], n_points)

        xys = np.meshgrid(xs, ys, indexing='ij')

        Nx = len(xs)
        Ny = len(ys)

        inds = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), indexing='ij')
        ind_0 = inds[0].flatten().astype(int)
        ind_1 = inds[1].flatten().astype(int)

        E = np.zeros((Nx, Ny, 3), dtype='complex')
        H = np.zeros((Nx, Ny, 3), dtype='complex')

        total_points = len(xys[0].flatten())

        for x_i, y_i in zip(range(0, total_points), range(0, total_points)):
            calc = S.GetFields(xys[0].flatten()[x_i], xys[1].flatten()[y_i], depth)
            E[ind_0[x_i], ind_1[y_i]] = calc[0]
            H[ind_0[x_i], ind_1[y_i]]  = calc[1]

        E_mag = np.real(np.sqrt(np.sum(np.abs(E) ** 2, 2)))
        H_mag = np.real(np.sqrt(np.sum(np.abs(H) ** 2, 2)))

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(7, 2.6))
            im1 = axs[0].pcolormesh(xs, ys, E_mag.T, cmap='magma', shading='auto')
            fig.colorbar(im1, ax=axs[0])
            axs[0].set_xlabel('x (nm)')
            axs[0].set_ylabel('y (nm)')
            axs[0].set_aspect(aspect=1)

            im2 = axs[1].pcolormesh(xs, ys, H_mag.T, cmap='magma', shading='auto')
            fig.colorbar(im2, ax=axs[1])
            axs[1].set_xlabel('x (nm)')
            axs[1].set_ylabel('y (nm)')
            axs[1].set_aspect(aspect=1)
            plt.show()

        return xs, ys, E, H, E_mag, H_mag

    def get_fields_unit_cell(self, layer_index, wavelength, options, depth=1e-10, n_points=200):
        """
        Get the components of the E and H fields at a specific depth in a layer, over a range of x/y points. Can also plot results
        automatically. Uses the S4 function GetFields().

        :param layer_index: index of the layer in which to get epsilon. layer 0 is the incidence medium, layer 1 is the first layer in the stack, etc.
        :param wavelength: wavelength (in nm) at which to get epsilon
        :param depth: depth in the layer (from the top of the layer) in nm at which to calculate the fields
        :param n_points: number of points to scan across in the x and y directions

        :return: xs, ys, E, H, E_mag, H_mag. x points, y points, the complex (x, y, z) components of the E field, the complex (x, y, z) components of the H field,
                    the magnitude of the E-field, the magnitude of the H-field. The magnitude is given by sqrt(abs(Ex^2 + Ey^2 + Ez^2))
        """

        def vs_pol(s, p):
            S.SetExcitationPlanewave((options['theta_in']*180/np.pi, options['phi_in']*180/np.pi), s, p, 0)
            S.SetFrequency(1 / wavelength)

        wl_ind = np.argmin(np.abs(self.current_wavelengths * 1e9 - wavelength))

        S = initialise_S(self.size, options.orders, self.geom_list, self.layers_oc[wl_ind],
                         self.shapes_oc[wl_ind],
                         self.shapes_names,
                         self.widths, options.S4_options)

        if len(options['pol']) == 2:

            vs_pol(options['pol'][0], options['pol'][1])

        else:
            if options['pol'] in 'sp':
                vs_pol(int(options['pol'] == "s"), int(options['pol'] == "p"))

        if layer_index > 0:
            depth = np.cumsum([0] + self.widths[1:-1] + [0])[layer_index - 1] + depth

        E, H = S.GetFieldsOnGrid(z=depth, NumSamples=(n_points, n_points), Format='Array')


        return E, H

    def get_fields_z_integral(self, layer_index, wavelength, options, extent=None, n_points=200, plot=True):
        """
        Get the magnitude of the E and H fields integrated over z in a layer, over a range of x/y points. Can also plot results
        automatically.

        :param layer_index: index of the layer in which to get epsilon. layer 0 is the incidence medium, layer 1 is the first layer in the stack, etc.
        :param wavelength: wavelength (in nm) at which to get epsilon
        :param pol: polarization of the incident light, 's', 'p' or a tuple
        :param extent: range of x/y values in format [[x_min, x_max], [y_min, y_max]]. Default is 'None', will choose a reasonable area based \
        on the unit cell size by default
        :param n_points: number of points to scan across in the x and y directions
        :param plot: plot the results (True or False, default True)

        :return: xs, ys, E, H, E_mag, H_mag. x points, y points, the (x, y, z) amplitudes squared of the E-field (|Ex|^2 etc.), \
        the (x, y, z) amplitudes squared of the H-field (|Ex|^2 etc.) the magnitude of the E-field, \
        the magnitude of the H-field. The magnitude is given by sqrt(abs(Ex^2 + Ey^2 + Ez^2))
        """

        def vs_pol(s, p):
            S.SetExcitationPlanewave((options['theta_in']*180/np.pi, options['phi_in']*180/np.pi), s, p, 0)
            S.SetFrequency(1 / wavelength)

        wl_ind = np.argmin(np.abs(self.current_wavelengths * 1e9 - wavelength))

        S = initialise_S(self.size, options.orders, self.geom_list, self.layers_oc[wl_ind],
                         self.shapes_oc[wl_ind],
                         self.shapes_names,
                         self.widths, options.S4_options)

        if len(options['pol']) == 2:

            vs_pol(options['pol'][0], options['pol'][1])

        else:
            if options['pol'] in 'sp':
                vs_pol(int(options['pol'] == "s"), int(options['pol'] == "p"))

        if extent is None:
            xdim = np.max(abs(np.array(self.size)[:, 0]))
            ydim = np.max(abs(np.array(self.size)[:, 1]))
            xs = np.linspace(-1.5 * xdim, 1.5 * xdim, n_points)
            ys = np.linspace(-1.5 * ydim, 1.5 * ydim, n_points)

        else:
            xs = np.linspace(extent[0][0], extent[0][1], n_points)
            ys = np.linspace(extent[1][0], extent[1][1], n_points)

        xys = np.meshgrid(xs, ys, indexing='ij')

        Nx = len(xs)
        Ny = len(ys)

        inds = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), indexing='ij')
        ind_0 = inds[0].flatten().astype(int)
        ind_1 = inds[1].flatten().astype(int)

        E = np.zeros((Nx, Ny, 3), dtype='complex')
        H = np.zeros((Nx, Ny, 3), dtype='complex')

        total_points = len(xys[0].flatten())

        for x_i, y_i in zip(range(0, total_points), range(0, total_points)):
            calc = S.GetLayerZIntegral(Layer='layer_' + str(layer_index),
                                       xy=(xys[0].flatten()[x_i], xys[1].flatten()[y_i]))
            E[ind_0[x_i], ind_1[y_i]] = calc[0]
            H[ind_0[x_i], ind_1[y_i]]  = calc[1]

        E_mag = np.real(np.sqrt(np.sum(np.abs(E) ** 2, 2)))
        H_mag = np.real(np.sqrt(np.sum(np.abs(H) ** 2, 2)))

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(7, 2.6))
            im1 = axs[0].pcolormesh(xs, ys, E_mag.T, cmap='magma')
            fig.colorbar(im1, ax=axs[0])
            axs[0].set_xlabel('x (nm)')
            axs[0].set_ylabel('y (nm)')
            axs[0].set_aspect(aspect=1)

            im2 = axs[1].pcolormesh(xs, ys, H_mag.T, cmap='magma')
            fig.colorbar(im2, ax=axs[1])
            axs[1].set_xlabel('x (nm)')
            axs[1].set_ylabel('y (nm)')
            axs[1].set_aspect(aspect=1)
            plt.show()

        return xs, ys, E, H, E_mag, H_mag

    def set_widths(self, new_widths):
        new_widths = np.append(np.insert(np.array(new_widths, dtype='f'), 0, np.inf), np.inf).tolist()
        self.widths = new_widths

    def set_size(self, new_size):
        self.size = new_size

    def edit_geom_list(self, layer_index, geom_index, geom_entry):

        self.geom_list[layer_index][geom_index].update(geom_entry)


def RCWA_structure_wl(wl, geom_list, layers_oc, shapes_oc, s_names, pol, theta, phi, widths, size, orders,
            A_per_order, S4_options):

    def vs_pol(s, p):
        S.SetExcitationPlanewave((theta, phi), s, p, 0)
        S.SetFrequency(1 / wl)
        out = rcwa_rat(S, len(widths), theta, np.sqrt(layers_oc[0]))
        R = out[0]['R']
        T = out[0]['T']
        A_layer = rcwa_absorption_per_layer(S, len(widths), theta, np.sqrt(layers_oc[0]))
        if A_per_order:
            A_per_layer_order = rcwa_absorption_per_layer_order(S, len(widths), theta, np.sqrt(layers_oc[0]))
            return R, T, A_layer, A_per_layer_order
        else:
            return R, T, A_layer

    S = initialise_S(size, orders, geom_list, layers_oc, shapes_oc, s_names, widths, S4_options)

    if len(pol) == 2:

        results = vs_pol(pol[0], pol[1])

    else:
        if pol in 'sp':
            results = vs_pol(int(pol == "s"), int(pol == "p"))

        else:

            res_s = vs_pol(1, 0)
            res_p = vs_pol(0, 1)
            R = (res_s[0] + res_p[0]) / 2
            T = (res_s[1] + res_p[1]) / 2
            A_layer = (res_s[2] + res_p[2]) / 2

            if A_per_order:
                A_per_layer_order = (res_s[3] + res_p[3]) / 2
                results = R, T, A_layer, A_per_layer_order

            else:
                results = R, T, A_layer


    return results


def RCWA_wl_prof(wl, rat_output_A, dist, geom_list, layers_oc, shapes_oc, s_names, pol, theta, phi, widths, size, orders, S4_options):

    S = initialise_S(size, orders, geom_list, layers_oc, shapes_oc, s_names, widths, S4_options)
    profile_data = np.zeros(len(dist))

    A = rat_output_A

    if len(pol) == 2:

        S.SetExcitationPlanewave((theta, phi), pol[0], pol[1], 0)
        S.SetFrequency(1 / wl)
        for j, d in enumerate(dist):
            layer, d_in_layer = tmm.find_in_structure_with_inf(widths,
                                                               d)  # don't need to change this
            layer_name = 'layer_' + str(layer + 1)  # layer_1 is air above so need to add 1
            data = rcwa_position_resolved(S, layer_name, d_in_layer, A, theta, np.sqrt(layers_oc[0]))
            profile_data[j] = data


    else:
        if pol in 'sp':
            if pol == 's':
                s = 1
                p = 0
            elif pol == 'p':
                s = 0
                p = 1

            S.SetExcitationPlanewave((theta, phi), s, p, 0)


            S.SetFrequency(1 / wl)

            for j, d in enumerate(dist):
                layer, d_in_layer = tmm.find_in_structure_with_inf(widths,
                                                                   d)  # don't need to change this
                layer_name = 'layer_' + str(layer + 1)  # layer_1 is air above so need to add 1
                data = rcwa_position_resolved(S, layer_name, d_in_layer, A, theta, np.sqrt(layers_oc[0]))
                profile_data[j] = data

        else:


            S.SetFrequency(1 / wl)
            A = rat_output_A

            for j, d in enumerate(dist):
                layer, d_in_layer = tmm.find_in_structure_with_inf(widths,
                                                                   d)  # don't need to change this
                layer_name = 'layer_' + str(layer + 1)  # layer_1 is air above so need to add 1
                S.SetExcitationPlanewave((theta, phi), 0, 1, 0)  # p-polarization
                data_p = rcwa_position_resolved(S, layer_name, d_in_layer, A, theta, np.sqrt(layers_oc[0]))
                S.SetExcitationPlanewave((theta, phi), 1, 0, 0)  # p-polarization
                data_s = rcwa_position_resolved(S, layer_name, d_in_layer, A, theta, np.sqrt(layers_oc[0]))
                profile_data[j] = 0.5*(data_s + data_p)

    return profile_data
