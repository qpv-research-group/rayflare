import numpy as np
import tmm
import xarray as xr
import matplotlib.pyplot as plt
from solcore.absorption_calculator import OptiStack
from joblib import Parallel, delayed
from rayflare.angles import make_angle_vector, overall_bin
import os
from sparse import COO, save_npz, load_npz, stack
from rayflare.config import results_path

try:
    import S4
except Exception as err:
    print('WARNING: The RCWA solver will not be available because an S4 installation has not been found.')


def RCWA(structure, size, orders, options, incidence, transmission, only_incidence_angle=False,
                       front_or_rear='front', surf_name='', detail_layer=False, save=True):
    """ Calculates the reflected, absorbed and transmitted intensity of the structure for the wavelengths and angles
    defined using an RCWA method implemented using the S4 package.

    :param structure: A solcore Structure object with layers and materials or a OptiStack object.
    :param size: list with 2 entries, size of the unit cell (right now, can only be rectangular
    :param orders: number of orders to retain in the RCWA calculations.
    :param wavelength: Wavelengths (in nm) in which calculate the data.
    :param theta: polar incidence angle (in degrees) of the incident light. Default: 0 (normal incidence)
    :param phi: azimuthal incidence angle in degrees. Default: 0
    :param pol: Polarisation of the light: 's', 'p' or 'u'. Default: 'u' (unpolarised).
    :param transmission: semi-infinite transmission medium

    :return: A dictionary with the R, A and T at the specified wavelengths and angle.
    """
    # TODO: when non-zero incidence angle, not binned correctly in matrix (just goes in theta = 0)
    # TODO: when doing unpolarized, why not just set s=0.5 p=0.5 in S4? (Maybe needs to be normalised differently). Also don't know if this is faster,
    # or if internally it will still do s & p separately
    # TODO: if incidence angle is zero, s and p polarization are the same so no need to do both

    structpath = os.path.join(results_path, options['project_name'])
    if not os.path.isdir(structpath):
        os.mkdir(structpath)

    savepath_RT = os.path.join(structpath, surf_name + front_or_rear + 'RT.npz')
    savepath_A = os.path.join(structpath, surf_name + front_or_rear + 'A.npz')
    # prof_mat_path = os.path.join(results_path, options['project_name'],
    #                              surf_name + front_or_rear + 'profmat.nc')

    if os.path.isfile(savepath_RT) and save:
        print('Existing angular redistribution matrices found')
        full_mat = load_npz(savepath_RT)
        A_mat = load_npz(savepath_A)


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

        layers_oc[:, 0] = (inc.n(wavelengths))**2#+ 1j*inc.k(wavelengths))**2
        layers_oc[:, -1] = (trns.n(wavelengths) + 1j*trns.k(wavelengths))**2

        for i1, x in enumerate(layers):
            layers_oc[:, i1+1] = (x.material.n(wavelengths) + 1j*x.material.k(wavelengths))**2

        shapes_names = [str(x) for x in shape_mats]

        #depth_spacing = options['depth_spacing']
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
                            WeismannFormulation=False)

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
                                                         angle_vector_0, S4_options, detail_layer, side)
                                                        for i1 in range(len(wavelengths)))

        else:
            allres = [RCWA_wl(wavelengths[i1]*1e9, geom_list_str, layers_oc[i1], shapes_oc[i1], shapes_names,
                              pol, thetas_in, phis_in, widths, size, orders, phi_sym, theta_intv, phi_intv,
                              angle_vector_0, S4_options, detail_layer, side)
                      for i1 in range(len(wavelengths))]

        #R = np.stack([item[0] for item in allres])
        #T = np.stack([item[1] for item in allres])
        A_mat = np.stack([item[2] for item in allres])
        full_mat = stack([item[3] for item in allres])
        #int_mat = stack([item[4] for item in allres])
        #T_mat = np.stack([item[4] for item in allres])

        #full_mat = np.hstack((R_mat, T_mat))
        #full_mat = COO(full_mat)
        A_mat = COO(A_mat)

        if save:
            save_npz(savepath_RT, full_mat)
            save_npz(savepath_A, A_mat)

        #R_pfbo = np.stack([item[3] for item in allres])
        #T_pfbo = np.stack([item[4] for item in allres])
        #phi_rt = np.stack([item[5] for item in allres])
        #theta_r = np.stack([item[6] for item in allres])
        #theta_t = np.stack([item[7] for item in allres])
        #R_pfbo_2 = np.stack([item[8] for item in allres])


    #return {'R': R, 'T':T, 'A_layer': A_mat, 'full_mat': full_mat, 'int_mat': int_mat}#'R_pfbo': R_pfbo, 'T_pfbo': T_pfbo, 'phi_rt': phi_rt, 'theta_r': theta_r, 'theta_t': theta_t}#, 'R_pfbo_2': R_pfbo_2}
    return full_mat, A_mat # , R, T


def RCWA_wl(wl, geom_list, l_oc, s_oc, s_names, pol, theta, phi, widths, size, orders, phi_sym,
            theta_intv, phi_intv, angle_vector_0, S4_options, layer_details = False, side=1):

    S = initialise_S(size, orders, geom_list, l_oc, s_oc, s_names, widths, S4_options)

    n_inc = np.real(np.sqrt(l_oc[0]))

    G_basis = np.array(S.GetBasisSet())

    #print(len(G_basis))
    #print('G_basis', G_basis)
    f_mat = S.GetReciprocalLattice()
    #print('f_mat', f_mat)
    fg_1x = f_mat[0][0]
    fg_1y = f_mat[0][1]
    fg_2x = f_mat[1][0]
    fg_2y = f_mat[1][1]

    #print('basis', f_mat)

    R = np.zeros((len(theta)))
    T = np.zeros((len(theta)))
    A_layer = np.zeros((len(theta), len(widths)-2))

    mat_RT = np.zeros((len(angle_vector_0), int(len(angle_vector_0)/2)))
    mat_int = np.zeros((len(angle_vector_0), int(len(angle_vector_0)/2)))

    if side == 1:
        in_bin = np.arange(len(theta))

    else:
        binned_theta_in = np.digitize(np.pi-np.pi*theta/180, theta_intv, right=True) - 1

        #print(theta, binned_theta_in)
        phi_in = xr.DataArray(np.pi*phi/180,
                              coords={'theta_bin': (['angle_in'], binned_theta_in)},
                              dims=['angle_in'])
        #print(len(phi_intv), len(angle_vector_0))
        in_bin = phi_in.groupby('theta_bin').map(overall_bin,
                                                   args=(phi_intv, angle_vector_0)).data - int(len(angle_vector_0)/2)

    #print(in_bin)

    #imag_e = np.imag(l_oc)
    print(wl)
    #freq = 1/wl

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
            out, R_pfbo, T_pfbo, R_pfbo_int = rcwa_rat(S, len(widths), layer_details)
            R_pfbo = R_pfbo # this will not be normalized correctly! fix this outside if/else
            T_pfbo = n_inc*T_pfbo/np.cos(th*np.pi/180)
            #R[in_bin[i1]] = out['R']/np.cos(th*np.pi/180)
            T[in_bin[i1]] = n_inc*out['T']/np.cos(th*np.pi/180)
            A_layer[in_bin[i1]] = n_inc*rcwa_absorption_per_layer(S, len(widths))/np.cos(th*np.pi/180)
            #A_layer[in_bin[i1]] = rcwa_absorption_per_layer_lossfunc(S, len(widths), freq, imag_e)

        else:

            #print(th)
            S.SetFrequency(1 / wl)
            S.SetExcitationPlanewave((th, ph), 0, 1, 0)  # p-polarization
            out_p, R_pfbo_p, T_pfbo_p, R_pfbo_int_p = rcwa_rat(S, len(widths), layer_details)
            Ap = rcwa_absorption_per_layer(S, len(widths))
            #Ap = rcwa_absorption_per_layer_lossfunc(S, len(widths), freq, imag_e)
            S.SetExcitationPlanewave((th, ph), 1, 0, 0)  # s-polarization
            out_s, R_pfbo_s, T_pfbo_s, R_pfbo_int_s = rcwa_rat(S, len(widths), layer_details)
            As = rcwa_absorption_per_layer(S, len(widths))
            #As = rcwa_absorption_per_layer_lossfunc(S, len(widths), freq, imag_e)

            # by definition, should have R = 1-T-A_total.

            R_pfbo = (R_pfbo_s + R_pfbo_p) # this will not be normalized correctly! fix this outside if/else
            T_pfbo = n_inc*(T_pfbo_s + T_pfbo_p)/(2*np.cos(th*np.pi/180))

                #R_pfbo_2 = (R_pfbo_2s + R_pfbo_2p)/2

            #R[in_bin[i1]] = 0.5 * (out_p['R'] + out_s['R'])/np.cos(th*np.pi/180)# average
            T[in_bin[i1]] = n_inc*0.5 * (out_p['T'] + out_s['T'])/np.cos(th*np.pi/180)
                # output['all_p'].append(out_p['power_entering_list'])
                # output['all_s'].append(out_s['power_entering_list'])
            #A_layer[in_bin[i1]] = rcwa_absorption_per_layer(S, len(widths))
            A_layer[in_bin[i1]] = n_inc*0.5*(Ap+As)/np.cos(th*np.pi/180)


        R[in_bin[i1]] = 1 - T[in_bin[i1]] - np.sum(A_layer[in_bin[i1]])

                #fi_z = (l_oc[0] / wl) * np.cos(th * np.pi / 180)
        fi_x = np.real((np.real(np.sqrt(l_oc[0])) / wl) * np.sin(th * np.pi / 180) *
                       np.sin(ph * np.pi / 180))
        fi_y = np.real((np.real(np.sqrt(l_oc[0])) / wl) * np.sin(th * np.pi / 180) *
                       np.cos(ph * np.pi / 180))

            #print('inc', fi_x, fi_y)

        fr_x = fi_x + G_basis[:,0]*fg_1x + G_basis[:,1]*fg_2x
        fr_y = fi_y + G_basis[:,0]*fg_1y + G_basis[:,1]*fg_2y

        #print('eps/lambda', l_oc[0]/(wl**2))
        fr_z = np.sqrt((l_oc[0]/(wl**2))-fr_x**2 - fr_y**2)

        ft_z = np.sqrt((l_oc[-1]/(wl**2))-fr_x**2 - fr_y**2)

            #print('ref', fr_x, fr_y, fr_z)

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

        # renormalize so that np.sum(R_pfbo) = R[in_bin[i1]]
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
            #print(R_pfbo[i2], phi_rt[i2], bini, i1, phi_ind, np.argmin(abs(angle_vector_0 -theta_r_bin[i2])))
            mat_RT[bini, in_bin[i1]] = mat_RT[bini, i1] + R_pfbo[i2]

        for i2 in np.nonzero(T_pfbo)[0]:
            phi_ind = np.digitize(phi_rt[i2], phi_intv[theta_t_bin[i2]], right=True) - 1
            bini = np.argmin(abs(angle_vector_0 -theta_t_bin[i2])) + phi_ind
            mat_RT[bini, in_bin[i1]] = mat_RT[bini, i1] + T_pfbo[i2]

        if layer_details:
            #print(R_pfbo_int_p)
            if pol not in 'sp':
                R_pfbo_int = (R_pfbo_int_p + R_pfbo_int_s)/2
            f_z = np.sqrt((l_oc[layer_details] / (wl ** 2)) - fr_x ** 2 - fr_y ** 2)

            #print('fz', wl, f_z)
            #print('fy', wl, fr_y)
            #print('fx', wl, fr_x)
            theta_l = np.real(np.arccos(f_z / np.sqrt(fr_x ** 2 + fr_y ** 2 + f_z ** 2)))
            #print(l_oc[layer_details])
            #print('theta_l',theta_l)
            theta_l[theta_l == 0] = 1e-10
            #print('R_pfbo', R_pfbo_int)

            np_l = theta_l == np.pi / 2  # non-propagating reflected orders

            R_pfbo_int[np_l] = 0

            R_pfbo_int[np.abs(R_pfbo_int < 1e-16)] = 0  # sometimes get very small negative valyes
            #print(theta_l)
            theta_l_bin = np.digitize(theta_l, theta_intv, right=True) - 1
            #print(theta_l_bin)
            for i2 in np.nonzero(R_pfbo_int)[0]:
                phi_ind = np.digitize(phi_rt[i2], phi_intv[theta_l_bin[i2]], right=True) - 1
                bini = np.argmin(abs(angle_vector_0 - theta_l_bin[i2])) + phi_ind
                #print(bini)
                #print(R_pfbo_int[i2], phi_rt[i2], bini, i1, phi_ind,
                #      np.argmin(abs(angle_vector_0 - theta_l_bin[i2])))
                mat_int[bini, i1] = mat_int[bini, i1] + R_pfbo_int[i2]


    mat_RT = COO(mat_RT)
    mat_int = COO(mat_int)
    #print(mat_int)
    # want to output R, T, A_layer (in case doing single angle of incidence)
    # also want to output transmission and reflection efficiency/power flux per order and the angles (theta and phi)
    # relating to that order.
    # Theta depends on the medium and so is different for transmisson and reflection. Phi is the same.

    if side == -1:
        A_layer = np.flip(A_layer, 1)

    return R, T, A_layer.T, mat_RT, mat_int#, mat_T

def fold_phi(phis, phi_sym):
    return (abs(phis//np.pi)*2*np.pi + phis) % phi_sym


def rcwa_rat(S, n_layers, det_l=False):
    below = 'layer_' + str(n_layers)  # identify which layer is the transmission medium

    #print('sum', np.sum(R_pfbo))

    # power flux by order backwards in layer_1: if incidence medium has n=1, this gives a (real part of) sum equal to R calculated through 1-sum(S.GetPowerFlux('layer_2')
    # not so if incidence medium has different n

    # transmission power flux by order always sums to T, regardless of optical constants of transmission/incidence medium
    R = 1 - sum(S.GetPowerFlux('layer_2'))  # GetPowerFlux gives forward & backward Poynting vector, so sum to get power flux
    # this should be correct answer in 'far field': anythng that doesn't go into the surface must be reflected. but if n_incidence != 1
    # can get odd effects.

    # what about backwards power flow into layer_1?
    # forward into layer 2 = S.GetPowerFlux('layer_2')[0]
    #print(R, S.GetPowerFlux('layer_1')[1])
    #R = -np.real(S.GetPowerFlux('layer_1')[1])/np.real(S.GetPowerFlux('layer_2'))[0]

    R_pfbo = -np.array(S.GetPowerFluxByOrder('layer_1'))[:,1] # real part of backwards power flow
    Nrm = np.real(np.sum(R_pfbo))
    R_pfbo = np.real((R/Nrm)*R_pfbo)

    #R_pfbo_2 = -np.array(S.GetPowerFluxByOrder('layer_1'))[:,1]
    #Nrm = np.real(np.sum(R_pfbo_2))
    #R_pfbo_2 = (R/Nrm)*R_pfbo_2

    if det_l:
        layer_name = 'layer_' + str(det_l + 2)
        R_pfbo_int = -np.array(S.GetPowerFluxByOrder(layer_name))[:, 1]
        #print('R_pfbo', R_pfbo_int)

    else:
        R_pfbo_int = 0
    #print('R', R)
    # layer_2 is the top layer of the structure (layer_1 is incidence medium)
    T = sum(S.GetPowerFlux(below))

    T_pfbo = np.real(np.sum(np.array(S.GetPowerFluxByOrder(below)), 1))
    return {'R': np.real(R), 'T': np.real(T)}, R_pfbo, T_pfbo, R_pfbo_int#, R_pfbo_2


def initialise_S(size, orders, geom_list, mats_oc, shapes_oc, shape_mats, widths, options):
    # pass widths
    #print(widths)
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
        WeismannFormulation = options['WeismannFormulation']
    )


    for i1 in range(len(shapes_oc)):  # create the materials needed for all the shapes in S4
        S.SetMaterial('shape_mat_' + str(i1 + 1), shapes_oc[i1])
        #print('shape_mat_' + str(i1+1), shapes_oc[i1])

    ## Make the layers
    #stack_OS = OptiStack(stack, bo_back_reflection=False, substrate=substrate)
    #widths = stack_OS.get_widths()

    for i1, _ in enumerate(widths):  # create 'dummy' materials for base layers including incidence and transmission media
        S.SetMaterial('layer_' + str(i1 + 1), mats_oc[i1])  # This is not strictly necessary but it means S.SetExcitationPlanewave
        # can be done outside the wavelength loop in calculate_rat_rcwa
        #print('layer_' + str(i1 + 1), mats_oc[i1])

    for i1, wid in enumerate(widths):  # set base layers
        layer_name = 'layer_' + str(i1 + 1)
        #print(layer_name)
        if wid == float('Inf'):
            #print('zero width')
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


def update_epsilon(S, stack_OS, shape_mats_OS, wl):
    for i1 in range(len(stack_OS.get_widths())):
        S.SetMaterial('layer_' + str(i1 + 1), stack_OS.get_indices(wl)[i1] ** 2)
    for i1 in range(len(shape_mats_OS.widths)):  # initialise the materials needed for all the shapes in S4
        S.SetMaterial('shape_mat_' + str(i1 + 1), shape_mats_OS.get_indices(wl)[i1 + 1] ** 2)

    return S


def rcwa_position_resolved(S, layer, depth, A):
    if A > 0:
        delta = 1e-9
        power_difference = np.real(
            sum(S.GetPowerFlux(layer, depth - delta)) - sum(S.GetPowerFlux(layer, depth + delta)))
        return power_difference / (2 * delta)  # absorbed energy density normalised to total absorption
    else:
        return 0

def rcwa_absorption_per_layer(S, n_layers):
    # layer 1 is incidence medium, layer n is the transmission medium
    A = np.empty(n_layers-2)
    for i1, layer in enumerate(np.arange(n_layers-2)+2):
        A[i1] = np.real(sum(S.GetPowerFlux('layer_' + str(layer))) -
                        sum(S.GetPowerFlux('layer_' + str(layer+1))))
    A = np.array([x if x > 0 else 0 for x in A])

    return A

def rcwa_absorption_per_layer_order(S, n_layers):
    # layer 1 is incidence medium, layer n is the transmission medium
    n_orders =  len(S.GetBasisSet())
    A_per_order = np.empty((n_layers-2, n_orders))
    for i1, layer in enumerate(np.arange(n_layers-2)+2):

        per_order_top = np.sum(np.array(S.GetPowerFluxByOrder('layer_' + str(layer))), 1)
        per_order_bottom = np.sum(np.array(S.GetPowerFluxByOrder('layer_' + str(layer+1))), 1)

        A_per_order[i1,:] = np.real(per_order_top - per_order_bottom)

    return A_per_order

def rcwa_absorption_per_layer_lossfunc(S, n_layers, freq, imag_e):
    # layer 1 is incidence medium, layer n is the transmission medium
    A = np.empty(n_layers-2)
    for i1, layer in enumerate(np.arange(n_layers-2)+2):
        # A[i1] = np.real(sum(S.GetPowerFlux('layer_' + str(layer))) - sum(S.GetPowerFlux('layer_' + str(layer+1))))
        I = np.real(S.GetLayerVolumeIntegral(Layer='layer_' + str(layer), Quantity='U'))
        A[i1] = 0.5*freq*I*imag_e[i1+1]
        # print('layer_' + str(layer), imag_e[i1+1], I)
        # print(A[i1])

    A = np.array([x if x > 0 else 0 for x in A])

    return A

def get_reciprocal_lattice(size, orders):

    S = S4.New(size, orders)


    f_mat = S.GetReciprocalLattice()

    return f_mat


class rcwa_structure:
    # TODO: make this accept an OptiStack, and check the substrate of the SolarCell object
    """ Calculates the reflected, absorbed and transmitted intensity of the structure for the wavelengths and angles
    defined using an RCWA method implemented using the S4 package.

    :param structure: A solcore Structure/SolarCell object with layers and materials or a OptiStack object.
    :param size: list with 2 entries, size of the unit cell (right now, can only be rectangular
    :param incidence: semi-infinite incidence medium
    :param transmission: semi-infinite transmission medium (substrate)
    """

    def __init__(self, structure, size, options, incidence, transmission):

        wavelengths = options['wavelengths']
        geom_list = []
        list_for_OS = []

        for i1, layer in enumerate(structure):
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

        ## Materials for the shapes need to be defined before you can do .SetRegion
        shape_mats, geom_list_str = necessary_materials(geom_list)

        shapes_oc = np.zeros((len(wavelengths), len(shape_mats)), dtype=complex)

        for i1, x in enumerate(shape_mats):
            if isinstance(x, list):
                if len(x) == 3:
                    shapes_oc[:, i1] = np.ones_like(wavelengths)*(x[1] + 1j*x[2])**2

                if len(x) == 4:
                    shapes_oc[:, i1] = (x[2] + 1j*x[3])**2

            else:
                shapes_oc[:, i1] = (x.n(wavelengths) + 1j * x.k(wavelengths)) ** 2

        # prepare to pass to OptiStack.

        stack_OS = OptiStack(list_for_OS, no_back_reflection=False, substrate=transmission, incidence=incidence)
        widths = stack_OS.get_widths()

        layers_oc = (np.array(stack_OS.get_indices(wavelengths*1e9))**2).T

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
                            WeismannFormulation=False)

        user_options = options['S4_options'] if 'S4_options' in options.keys() else {}
        S4_options.update(user_options)

        self.geom_list = geom_list_str
        self.shapes_oc = shapes_oc
        self.shapes_names = shapes_names
        self.widths = widths
        self.size = size
        self.layers_oc = layers_oc


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

        R = np.stack([item[0] for item in allres])
        T = np.stack([item[1] for item in allres])
        A_mat = np.stack([item[2] for item in allres])

        self.rat_output_A = np.sum(A_mat, 1)  # used for profile calculation

        if options['A_per_order']:

            A_order = np.stack([item[3] for item in allres])

            S_for_orders = initialise_S(self.size, options['orders'], self.geom_list, self.layers_oc[0],
                             self.shapes_oc[0], self.shapes_names, self.widths, options['S4_options'])

            basis_set = S_for_orders.GetBasisSet()
            f_mat = S_for_orders.GetReciprocalLattice()

            return {'R': R, 'T': T, 'A_per_layer': A_mat, 'A_layer_order': A_order, 'basis_set': basis_set, 'reciprocal': f_mat}

        else:

            return {'R': R, 'T': T, 'A_per_layer': A_mat}


    def calculate_profile(self, options):
        """ It calculates the absorbed energy density within the material.

        In principle this has units of [power]/[volume], but we can express it as a multiple of incoming light power
        density on the material, which has units [power]/[area], so that absorbed energy density has units of 1/[length].'

        """

        wl = options['wavelengths'] * 1e9
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

        output = np.stack(allres)

        return output

    def save_layer_postscript(self, layer_index, filename):
        # layer_index: layer 0 is the incidence medium
        S = initialise_S(self.size, 1, self.geom_list, self.layers_oc[0], self.shapes_oc[0], self.shapes_names, self.widths, self.S4_options)
        S.OutputLayerPatternPostscript(Layer='layer_' + str(layer_index + 1), Filename=filename + '.ps')


    def get_fourier_epsilon(self, layer_index, wavelength, extent=None, n_points=200, plot=True):
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

        wl_ind = np.argmin(np.abs(self.wavelengths * 1e9 - wavelength))

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
        S = initialise_S(self.size, self.orders, self.geom_list, self.layers_oc[wl_ind], self.shapes_oc[wl_ind], self.shapes_names,
                         self.widths, self.S4_options)

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

    def get_fields(self, layer_index, wavelength, pol='s', extent=None, depth=1e-10, n_points=200, plot=True):
        """
        Get the components of the E and H fields at a specific depth in a layer, over a range of x/y points. Can also plot results
        automatically. Uses the S4 function GetFields().

        :param layer_index: index of the layer in which to get epsilon. layer 0 is the incidence medium, layer 1 is the first layer in the stack, etc.
        :param wavelength: wavelength (in nm) at which to get epsilon
        :param pol: polarization of the incident light, 's', 'p' or a tuple
        :param extent: range of x/y values in format [[x_min, x_max], [y_min, y_max]]. Default is 'None', will choose a reasonable area based \
        on the unit cell size by default
        :param depth: depth in the layer (from the top of the layer) in nm at which to calculate the fields
        :param n_points: number of points to scan across in the x and y directions
        :param plot: plot the results (True or False, default True)

        :return: xs, ys, E, H, E_mag, H_mag. x points, y points, the complex (x, y, z) components of the E field, the complex (x, y, z) components of the H field,
                    the magnitude of the E-field, the magnitude of the H-field. The magnitude is given by sqrt(abs(Ex^2 + Ey^2 + Ez^2))
        """

        def vs_pol(s, p):
            S.SetExcitationPlanewave((self.options['theta_in']*180/np.pi, self.options['phi_in']*180/np.pi), s, p, 0)
            S.SetFrequency(1 / wavelength)

        wl_ind = np.argmin(np.abs(self.wavelengths * 1e9 - wavelength))

        S = initialise_S(self.size, self.orders, self.geom_list, self.layers_oc[wl_ind],
                         self.shapes_oc[wl_ind],
                         self.shapes_names,
                         self.widths, self.S4_options)

        if len(pol) == 2:

            vs_pol(pol[0], pol[1])

        else:
            if pol in 'sp':
                vs_pol(int(pol == "s"), int(pol == "p"))

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

        E_mag = np.sqrt(np.abs(np.sum(E ** 2, 2)))
        H_mag = np.sqrt(np.abs(np.sum(H ** 2, 2)))

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


    def get_fields_z_integral(self, layer_index, wavelength, pol='s', extent=None, n_points=200, plot=True):
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
            S.SetExcitationPlanewave((self.options['theta_in']*180/np.pi, self.options['phi_in']*180/np.pi), s, p, 0)
            S.SetFrequency(1 / wavelength)

        wl_ind = np.argmin(np.abs(self.wavelengths * 1e9 - wavelength))

        S = initialise_S(self.size, self.orders, self.geom_list, self.layers_oc[wl_ind],
                         self.shapes_oc[wl_ind],
                         self.shapes_names,
                         self.widths, self.S4_options)

        if len(pol) == 2:

            vs_pol(pol[0], pol[1])

        else:
            if pol in 'sp':
                vs_pol(int(pol == "s"), int(pol == "p"))

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

        E_mag = np.real(np.sqrt(np.sum(E, 2)))
        H_mag = np.real(np.sqrt(np.sum(H, 2)))

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
        out, R_pfbo, T_pfbo, R_pfbo_int = rcwa_rat(S, len(widths))
        R = out['R']
        T = out['T']
        A_layer = rcwa_absorption_per_layer(S, len(widths))/np.cos(theta*np.pi/180)
        if A_per_order:
            A_per_layer_order = rcwa_absorption_per_layer_order(S, len(widths))/np.cos(theta*np.pi/180)
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
            data = rcwa_position_resolved(S, layer_name, d_in_layer, A)/np.cos(theta*np.pi/180)
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
                data = rcwa_position_resolved(S, layer_name, d_in_layer, A)/np.cos(theta*np.pi/180)
                profile_data[j] = data

        else:


            S.SetFrequency(1 / wl)
            A = rat_output_A

            for j, d in enumerate(dist):
                layer, d_in_layer = tmm.find_in_structure_with_inf(widths,
                                                                   d)  # don't need to change this
                layer_name = 'layer_' + str(layer + 1)  # layer_1 is air above so need to add 1
                S.SetExcitationPlanewave((theta, phi), 0, 1, 0)  # p-polarization
                data_p = rcwa_position_resolved(S, layer_name, d_in_layer, A)/np.cos(theta*np.pi/180)
                S.SetExcitationPlanewave((theta, phi), 1, 0, 0)  # p-polarization
                data_s = rcwa_position_resolved(S, layer_name, d_in_layer, A)/np.cos(theta*np.pi/180)
                profile_data[j] = 0.5*(data_s + data_p)

    return profile_data
