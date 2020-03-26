import numpy as np
import tmm
from solcore.absorption_calculator import OptiStack
from joblib import Parallel, delayed
from angles import make_angle_vector
import os
from sparse import COO, save_npz, load_npz, stack
from config import results_path
from time import time

try:
    import S4
except Exception as err:
    print('WARNING: The RCWA solver will not be available because an S4 installation has not been found.')


def RCWA(structure, size, orders, options, incidence, substrate, only_incidence_angle=False,
                       front_or_rear='front', surf_name='', detail_layer=False):
    """ Calculates the reflected, absorbed and transmitted intensity of the structure for the wavelengths and angles
    defined using an RCWA method implemented using the S4 package.

    :param structure: A solcore Structure object with layers and materials or a OptiStack object.
    :param size: list with 2 entries, size of the unit cell (right now, can only be rectangular
    :param orders: number of orders to retain in the RCWA calculations.
    :param wavelength: Wavelengths (in nm) in which calculate the data.
    :param theta: polar incidence angle (in degrees) of the incident light. Default: 0 (normal incidence)
    :param phi: azimuthal incidence angle in degrees. Default: 0
    :param pol: Polarisation of the light: 's', 'p' or 'u'. Default: 'u' (unpolarised).
    :param substrate: semi-infinite transmission medium
    :return: A dictionary with the R, A and T at the specified wavelengths and angle.
    """
    # TODO: when non-zero incidence angle, not binned correctly in matrix (just goes in theta = 0)
    # TODO: when doing unpolarized, why not just set s=0.5 p=0.5 in S4? (Maybe needs to be normalised differently). Also don't know if this is faster,
    # or if internally it will still do s & p separately
    # TODO: if incidence angle is zero, s and p polarization are the same
    structpath = os.path.join(results_path, options['project_name'])
    if not os.path.isdir(structpath):
        os.mkdir(structpath)

    savepath_RT = os.path.join(structpath, surf_name + front_or_rear + 'RT.npz')
    savepath_A = os.path.join(structpath, surf_name + front_or_rear + 'A.npz')

    wavelengths = options['wavelengths']

    # write a separate function that makes the OptiStack structure into an S4 object, defined materials etc.
    geom_list = [layer.geometry for layer in structure]
    geom_list.insert(0, {})  # incidence medium
    geom_list.append({})  # transmission medium

    ## Materials for the shapes need to be defined before you can do .SetRegion
    shape_mats, geom_list_str = necessary_materials(geom_list)

    shapes_oc = np.zeros((len(wavelengths), len(shape_mats)), dtype=complex)

    for i1, x in enumerate(shape_mats):
        shapes_oc[:, i1] = (x.n(wavelengths) + 1j*x.k(wavelengths))**2

    stack_OS = OptiStack(structure, bo_back_reflection=False, substrate=substrate)
    widths = stack_OS.get_widths()
    layers_oc = np.zeros((len(wavelengths), len(structure)+2), dtype=complex)

    layers_oc[:, 0] = (incidence.n(wavelengths))**2#+ 1j*incidence.k(wavelengths))**2
    layers_oc[:, -1] = (substrate.n(wavelengths) + 1j*substrate.k(wavelengths))**2

    for i1, x in enumerate(structure):
        layers_oc[:, i1+1] = (x.material.n(wavelengths) + 1j*x.material.k(wavelengths))**2

    shapes_names = [str(x) for x in shape_mats]

    #nm_spacing = options['nm_spacing']
    phi_sym = options['phi_symmetry']
    n_theta_bins = options['n_theta_bins']
    c_az = options['c_azimuth']
    pol = options['pol']

    # RCWA options
    rcwa_options = dict(LatticeTruncation = 'Circular',
        DiscretizedEpsilon = False,
        DiscretizationResolution = 8,
        PolarizationDecomposition = False,
        PolarizationBasis = 'Default',
        LanczosSmoothing = False,
        SubpixelSmoothing = False,
        ConserveMemory = False,
        WeismannFormulation = False)

    user_options = options['rcwa_options'] if 'rcwa_options' in options.keys() else {}
    rcwa_options.update(user_options)
    print(rcwa_options)

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
    # initialise_S has to happen inside parallel job (get Pickle errors otherwise); just pass relevant optical constants for each wavelength, like for RT

    angle_vector_0 = angle_vector[:, 0]

    if options['parallel']:
        allres = Parallel(n_jobs=options['n_jobs'])(delayed(RCWA_wl)
                                                    (wavelengths[i1]*1e9, geom_list, layers_oc[i1], shapes_oc[i1], shapes_names, pol, thetas_in, phis_in, widths, size,
                                                     orders, phi_sym, theta_intv, phi_intv, angle_vector_0, rcwa_options, detail_layer)
                                                    for i1 in range(len(wavelengths)))

    else:
        allres = [RCWA_wl(wavelengths[i1]*1e9, geom_list, layers_oc[i1], shapes_oc[i1], shapes_names, pol, thetas_in, phis_in, widths, size, orders, phi_sym, theta_intv, phi_intv,
                          angle_vector_0, rcwa_options, detail_layer)
                  for i1 in range(len(wavelengths))]

    R = np.stack([item[0] for item in allres])
    T = np.stack([item[1] for item in allres])
    A_mat = np.stack([item[2] for item in allres])
    full_mat = stack([item[3] for item in allres])
    int_mat = stack([item[4] for item in allres])
    #T_mat = np.stack([item[4] for item in allres])

    #full_mat = np.hstack((R_mat, T_mat))
    #full_mat = COO(full_mat)
    A_mat = COO(A_mat)

    save_npz(savepath_RT, full_mat)
    save_npz(savepath_A, A_mat)

    #R_pfbo = np.stack([item[3] for item in allres])
    #T_pfbo = np.stack([item[4] for item in allres])
    #phi_rt = np.stack([item[5] for item in allres])
    #theta_r = np.stack([item[6] for item in allres])
    #theta_t = np.stack([item[7] for item in allres])
    #R_pfbo_2 = np.stack([item[8] for item in allres])


    return {'R': R, 'T':T, 'A_layer': A_mat, 'full_mat': full_mat, 'int_mat': int_mat}#'R_pfbo': R_pfbo, 'T_pfbo': T_pfbo, 'phi_rt': phi_rt, 'theta_r': theta_r, 'theta_t': theta_t}#, 'R_pfbo_2': R_pfbo_2}



def RCWA_wl(wl, geom_list, l_oc, s_oc, s_names, pol, theta, phi, widths, size, orders, phi_sym,
            theta_intv, phi_intv, angle_vector_0, rcwa_options, layer_details = False):
    #print(wl)
    S = initialise_S(size, orders, geom_list, l_oc, s_oc, s_names, widths, rcwa_options)

    #print(l_oc[0])

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
    #mat_T = np.zeros((len(angle_vector_0), len(angle_vector_0)))

    for i1 in range(len(theta)):
        if pol in 'sp':
            if pol == 's':
                s = 1
                p = 0
            elif pol == 'p':
                s = 0
                p = 1

            S.SetExcitationPlanewave((theta[i1], phi[i1]), s, p, 0)
            S.SetFrequency(1 / wl)
            out, R_pfbo, T_pfbo, R_pfbo_int = rcwa_rat(S, len(widths), layer_details)
            R[i1] = out['R']
            T[i1] = out['T']
            A_layer[i1] = rcwa_absorption_per_layer(S, len(widths))

        else:

            #print(theta[i1])
            S.SetFrequency(1 / wl)
            S.SetExcitationPlanewave((theta[i1], phi[i1]), 0, 1, 0)  # p-polarization
            out_p, R_pfbo_p, T_pfbo_p, R_pfbo_int_p = rcwa_rat(S, len(widths), layer_details)
            S.SetExcitationPlanewave((theta[i1], phi[i1]), 1, 0, 0)  # s-polarization
            out_s, R_pfbo_s, T_pfbo_s, R_pfbo_int_s = rcwa_rat(S, len(widths), layer_details)

            R_pfbo = (R_pfbo_s + R_pfbo_p)/2
            T_pfbo = (T_pfbo_s + T_pfbo_p)/2

                #R_pfbo_2 = (R_pfbo_2s + R_pfbo_2p)/2

            R[i1] = 0.5 * (out_p['R'] + out_s['R'])  # average
            T[i1] = 0.5 * (out_p['T'] + out_s['T'])
                # output['all_p'].append(out_p['power_entering_list'])
                # output['all_s'].append(out_s['power_entering_list'])
            A_layer[i1] = rcwa_absorption_per_layer(S, len(widths))

                #fi_z = (l_oc[0] / wl) * np.cos(theta[i1] * np.pi / 180)
        fi_x = np.real((np.real(np.sqrt(l_oc[0])) / wl) * np.sin(theta[i1] * np.pi / 180) *
                       np.sin(phi[i1] * np.pi / 180))
        fi_y = np.real((np.real(np.sqrt(l_oc[0])) / wl) * np.sin(theta[i1] * np.pi / 180) *
                       np.cos(phi[i1] * np.pi / 180))

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

        R_pfbo[np_r] = 0
        T_pfbo[np_t] = 0

        R_pfbo[np.abs(R_pfbo < 1e-16)] = 0 # sometimes get very small negative valyes
        T_pfbo[np.abs(T_pfbo < 1e-16)] = 0

        theta_r[theta_r == 0] = 1e-10
        theta_t[theta_t == 0] = 1e-10
        phi_rt[phi_rt == 0] = 1e-10


        theta_r_bin = np.digitize(theta_r, theta_intv, right=True) - 1
        theta_t_bin = np.digitize(theta_t, theta_intv, right=True) - 1
        #print('theta_r', theta_r)
        #print('PFBO', R_pfbo)
        #print('bin', theta_r_bin)
        #print(theta_t)

        for i2 in np.nonzero(R_pfbo)[0]:
            phi_ind = np.digitize(phi_rt[i2], phi_intv[theta_r_bin[i2]], right=True) - 1
            bin = np.argmin(abs(angle_vector_0 -theta_r_bin[i2])) + phi_ind
            #print(R_pfbo[i2], phi_rt[i2], bin, i1, phi_ind, np.argmin(abs(angle_vector_0 -theta_r_bin[i2])))
            mat_RT[bin, i1] = mat_RT[bin, i1] + R_pfbo[i2]

        for i2 in np.nonzero(T_pfbo)[0]:
            phi_ind = np.digitize(phi_rt[i2], phi_intv[theta_t_bin[i2]], right=True) - 1
            bin = np.argmin(abs(angle_vector_0 -theta_t_bin[i2])) + phi_ind
            mat_RT[bin, i1] = mat_RT[bin, i1] + T_pfbo[i2]

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
                bin = np.argmin(abs(angle_vector_0 - theta_l_bin[i2])) + phi_ind
                #print(bin)
                #print(R_pfbo_int[i2], phi_rt[i2], bin, i1, phi_ind,
                #      np.argmin(abs(angle_vector_0 - theta_l_bin[i2])))
                mat_int[bin, i1] = mat_int[bin, i1] + R_pfbo_int[i2]






    mat_RT = COO(mat_RT)
    mat_int = COO(mat_int)
    #print(mat_int)
    # want to output R, T, A_layer (in case doing single angle of incidence)
    # also want to output transmission and reflection efficiency/power flux per order and the angles (theta and phi)
    # relating to that order.
    # Theta depends on the medium and so is different for transmisson and reflection. Phi is the same.

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
    S.SetOptions(  # these are the default
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

    for i1 in range(len(widths)):  # create 'dummy' materials for base layers including incidence and transmission media
        S.SetMaterial('layer_' + str(i1 + 1), mats_oc[i1])  # This is not strictly necessary but it means S.SetExcitationPlanewave
        # can be done outside the wavelength loop in calculate_rat_rcwa
        #print('layer_' + str(i1 + 1), mats_oc[i1])

    for i1 in range(len(widths)):  # set base layers
        layer_name = 'layer_' + str(i1 + 1)
        #print(layer_name)
        if widths[i1] == float('Inf'):
            #print('zero width')
            S.AddLayer(layer_name, 0, layer_name)  # Solcore4 has incidence and transmission media widths set to Inf;
            # in S4 they have zero width
        else:
            S.AddLayer(layer_name, widths[i1], layer_name)

        geometry = geom_list[i1]

        if bool(geometry):
            for shape in geometry:
                mat_name = 'shape_mat_' + str(shape_mats.index(str(shape['mat'])) + 1)
                #print(str(shape['mat']), mat_name)
                if shape['type'] == 'circle':
                    S.SetRegionCircle(layer_name, mat_name, shape['center'], shape['radius'])
                elif shape['type'] == 'ellipse':
                    S.SetRegionEllipse(layer_name, mat_name, shape['center'], shape['angle'], shape['halfwidths'])
                elif shape['type'] == 'rectangle':
                    #print('rect')
                    S.SetRegionRectangle(layer_name, mat_name, shape['center'], shape['angle'], shape['halfwidths'])
                elif shape['type'] == 'polygon':
                    S.SetRegionPolygon(layer_name, mat_name, shape['center'], shape['angle'], shape['vertices'])

    return S


def necessary_materials(geom_list):
    shape_mats = []
    for i1, geom in enumerate(geom_list):
        if bool(geom):
            shape_mats.append([x['mat'] for x in geom])
            for i2, g in enumerate(geom):
                geom_list[i1][i2]['mat'] = str(g['mat'])
    return list(set([val for sublist in shape_mats for val in sublist])), geom_list


#def update_epsilon(S, stack_OS, shape_mats_OS, wl):
    #    for i1 in range(len(stack_OS.get_widths())):
    #    S.SetMaterial('layer_' + str(i1 + 1), stack_OS.get_indices(wl)[i1] ** 2)
    #for i1 in range(len(shape_mats_OS.widths)):  # initialise the materials needed for all the shapes in S4
    #   S.SetMaterial('shape_mat_' + str(i1 + 1), shape_mats_OS.get_indices(wl)[i1 + 1] ** 2)

    return S


def calculate_absorption_profile_rcwa(structure, size, orders, wavelength, rat_output,
                                      z_limit=None, steps_size=2, dist=None, theta=0, phi=0, pol='u', substrate=None):
    """ It calculates the absorbed energy density within the material. From the documentation:

    'In principle this has units of [power]/[volume], but we can express it as a multiple of incoming light power
    density on the material, which has units [power]/[area], so that absorbed energy density has units of 1/[length].'

    Integrating this absorption profile in the whole stack gives the same result that the absorption obtained with
    calculate_rat as long as the spacial mesh (controlled by steps_thinest_layer) is fine enough. If the structure is
    very thick and the mesh not thin enough, the calculation might diverege at short wavelengths.

    For now, it only works for normal incident, coherent light.

    :param structure: A solcore structure with layers and materials.
    :param size: list with 2 entries, size of the unit cell (right now, can only be rectangular
    :param orders: number of orders to retain in the RCWA calculations.
    :param wavelength: Wavelengths (in nm) in which calculate the data.
    :param rat_output: output from calculate_rat_rcwa
    :param z_limit: Maximum value in the z direction at which to calculate depth-dependent absorption (nm)
    :param steps_size: if the dist is not specified, the step size in nm to use in the depth-dependent calculation
    :param dist: the positions (in nm) at which to calculate depth-dependent absorption
    :param theta: polar incidence angle (in degrees) of the incident light. Default: 0 (normal incidence)
    :param phi: azimuthal incidence angle in degrees. Default: 0
    :param pol: Polarisation of the light: 's', 'p' or 'u'. Default: 'u' (unpolarised).
    :param substrate: semi-infinite transmission medium
    :return: A dictionary containing the positions (in nm) and a 2D array with the absorption in the structure as a
    function of the position and the wavelength.
    """

    num_wl = len(wavelength)

    if dist is None:
        if z_limit is None:
            stack = OptiStack(structure)
            z_limit = np.sum(np.array(stack.widths))
        dist = np.arange(0, z_limit, steps_size)

    output = {'position': dist, 'absorption': np.zeros((num_wl, len(dist)))}

    S, stack_OS, shape_mats_OS = initialise_S(structure, size, orders, substrate)

    if pol in 'sp':
        if pol == 's':
            s = 1
            p = 0
        elif pol == 'p':
            s = 0
            p = 1

        S.SetExcitationPlanewave((theta, phi), s, p, 0)
        for i, wl in enumerate(wavelength):
            update_epsilon(S, stack_OS, shape_mats_OS, wl)
            S.SetFrequency(1 / wl)
            A = rat_output['A'][i]
            for j, d in enumerate(dist):
                layer, d_in_layer = tmm.find_in_structure_with_inf(stack_OS.get_widths(),
                                                                   d)  # don't need to change this
                layer_name = 'layer_' + str(layer + 1)  # layer_1 is air above so need to add 1
                data = rcwa_position_resolved(S, layer_name, d_in_layer, A)
                output['absorption'][i, j] = data

    else:
        for i, wl in enumerate(wavelength):  # set the material values and indices in here
            #print(i)
            update_epsilon(S, stack_OS, shape_mats_OS, wl)
            S.SetFrequency(1 / wl)
            A = rat_output['A'][i]

            for j, d in enumerate(dist):
                layer, d_in_layer = tmm.find_in_structure_with_inf(stack_OS.get_widths(),
                                                                   d)  # don't need to change this
                layer_name = 'layer_' + str(layer + 1)  # layer_1 is air above so need to add 1
                S.SetExcitationPlanewave((theta, phi), 0, 1, 0)  # p-polarization
                data_p = rcwa_position_resolved(S, layer_name, d_in_layer, A)
                S.SetExcitationPlanewave((theta, phi), 1, 0, 0)  # p-polarization
                data_s = rcwa_position_resolved(S, layer_name, d_in_layer, A)
                output['absorption'][i, j] = 0.5 * (data_p + data_s)

    return output


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
        A[i1] = np.real(sum(S.GetPowerFlux('layer_' + str(layer))) - sum(S.GetPowerFlux('layer_' + str(layer+1))))

    A = [x if x > 0 else 0 for x in A]

    return A

def get_reciprocal_lattice(size, orders):

    S = S4.New(size, orders)


    f_mat = S.GetReciprocalLattice()

    return f_mat


class rcwa_structure:

    def __init__(self, structure, size, orders, options, incidence, substrate):
        """ Calculates the reflected, absorbed and transmitted intensity of the structure for the wavelengths and angles
        defined using an RCWA method implemented using the S4 package.

        :param structure: A solcore Structure object with layers and materials or a OptiStack object.
        :param size: list with 2 entries, size of the unit cell (right now, can only be rectangular
        :param orders: number of orders to retain in the RCWA calculations.
        :param wavelength: Wavelengths (in nm) in which calculate the data.
        :param theta: polar incidence angle (in degrees) of the incident light. Default: 0 (normal incidence)
        :param phi: azimuthal incidence angle in degrees. Default: 0
        :param pol: Polarisation of the light: 's', 'p' or 'u'. Default: 'u' (unpolarised).
        :param substrate: semi-infinite transmission medium
        :return: A dictionary with the R, A and T at the specified wavelengths and angle.
        """

        wavelengths = options['wavelengths']

        # write a separate function that makes the OptiStack structure into an S4 object, defined materials etc.
        geom_list = [layer.geometry for layer in structure]
        geom_list.insert(0, {})  # incidence medium
        geom_list.append({})  # transmission medium

        ## Materials for the shapes need to be defined before you can do .SetRegion
        shape_mats, geom_list_str = necessary_materials(geom_list)

        shapes_oc = np.zeros((len(wavelengths), len(shape_mats)), dtype=complex)

        for i1, x in enumerate(shape_mats):
            shapes_oc[:, i1] = (x.n(wavelengths) + 1j * x.k(wavelengths)) ** 2

        stack_OS = OptiStack(structure, bo_back_reflection=False, substrate=substrate)
        widths = stack_OS.get_widths()
        layers_oc = np.zeros((len(wavelengths), len(structure) + 2), dtype=complex)

        layers_oc[:, 0] = (incidence.n(wavelengths)) ** 2  # + 1j*incidence.k(wavelengths))**2
        layers_oc[:, -1] = (substrate.n(wavelengths) + 1j * substrate.k(wavelengths)) ** 2

        for i1, x in enumerate(structure):
            layers_oc[:, i1 + 1] = (x.material.n(wavelengths) + 1j * x.material.k(wavelengths)) ** 2

        shapes_names = [str(x) for x in shape_mats]

        # nm_spacing = options['nm_spacing']


        # RCWA options
        rcwa_options = dict(LatticeTruncation='Circular',
                            DiscretizedEpsilon=False,
                            DiscretizationResolution=8,
                            PolarizationDecomposition=False,
                            PolarizationBasis='Default',
                            LanczosSmoothing=False,
                            SubpixelSmoothing=False,
                            ConserveMemory=False,
                            WeismannFormulation=False)

        user_options = options['rcwa_options'] if 'rcwa_options' in options.keys() else {}
        rcwa_options.update(user_options)

        self.wavelengths = wavelengths
        self.rcwa_options = rcwa_options
        self.options = options
        self.geom_list = geom_list
        self.shapes_oc = shapes_oc
        self.shapes_names = shapes_names
        self.widths = widths
        self.orders = orders
        self.size = size
        self.layers_oc = layers_oc

    def set_widths(self, new_widths):
        new_widths = np.append(np.insert(np.array(new_widths, dtype='f'), 0, np.inf), np.inf).tolist()
        self.widths = new_widths

    def set_size(self, new_size):
        self.size = new_size

    def edit_geom_list(self, layer_index, geom_index, geom_entry):

        self.geom_list[layer_index][geom_index].update(geom_entry)


    def calculate(self):

        if self.options['parallel']:
            allres = Parallel(n_jobs=self.options['n_jobs'])(delayed(self.RCWA_wl)
                                                        (self.wavelengths[i1] * 1e9, self.geom_list, self.layers_oc[i1], self.shapes_oc[i1],
                                                         self.shapes_names, self.options['pol'], self.options['theta_in'], self.options['phi_in'],
                                                         self.widths, self.size,
                                                         self.orders, self.rcwa_options)
                                                        for i1 in range(len(self.wavelengths)))

        else:
            allres = [
                self.RCWA_wl(self.wavelengths[i1] * 1e9, self.geom_list, self.layers_oc[i1], self.shapes_oc[i1],
                                                         self.shapes_names, self.options['pol'], self.options['theta_in'], self.options['phi_in'],
                                                         self.widths, self.size,
                                                         self.orders, self.rcwa_options)
                for i1 in range(len(self.wavelengths))]

        R = np.stack([item[0] for item in allres])
        T = np.stack([item[1] for item in allres])
        A_mat = np.stack([item[2] for item in allres])


        return {'R': R, 'T': T, 'A_layer': A_mat}


    def RCWA_wl(self, wl, geom_list, l_oc, s_oc, s_names, pol, theta, phi, widths, size, orders, rcwa_options):

        S = initialise_S(size, orders, geom_list, l_oc, s_oc, s_names, widths, rcwa_options)


        if pol in 'sp':
            if pol == 's':
                s = 1
                p = 0
            elif pol == 'p':
                s = 0
                p = 1

            S.SetExcitationPlanewave((theta, phi), s, p, 0)
            S.SetFrequency(1 / wl)
            out, R_pfbo, T_pfbo, R_pfbo_int = rcwa_rat(S, len(widths))
            R = out['R']
            T = out['T']
            A_layer = rcwa_absorption_per_layer(S, len(widths))

        else:

            S.SetFrequency(1 / wl)
            S.SetExcitationPlanewave((theta, phi), 0, 1, 0)  # p-polarization
            out_p, R_pfbo_p, T_pfbo_p, R_pfbo_int_p = rcwa_rat(S, len(widths))
            S.SetExcitationPlanewave((theta, phi), 1, 0, 0)  # s-polarization
            out_s, R_pfbo_s, T_pfbo_s, R_pfbo_int_s = rcwa_rat(S, len(widths))


            R = 0.5 * (out_p['R'] + out_s['R'])  # average
            T = 0.5 * (out_p['T'] + out_s['T'])
            A_layer = rcwa_absorption_per_layer(S, len(widths))

        return R, T, A_layer