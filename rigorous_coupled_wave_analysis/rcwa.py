import numpy as np
import tmm
from solcore.structure import Layer
from solcore.absorption_calculator import OptiStack
from joblib import Parallel, delayed
from angles import make_angle_vector
import copy

try:
    import S4
except ModuleNotFoundError:
    raise


def calculate_rat_rcwa(structure, size, orders, options, incidence, substrate, only_incidence_angle=False,
                       front_or_rear='front'):
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
        shapes_oc[:, i1] = (x.n(wavelengths) + 1j*x.k(wavelengths))**2

    stack_OS = OptiStack(structure, no_back_reflexion=False, substrate=substrate)
    widths = stack_OS.get_widths()
    layers_oc = np.zeros((len(wavelengths), len(structure)+2), dtype=complex)

    layers_oc[:, 0] = (incidence.n(wavelengths) + 1j*incidence.k(wavelengths))**2
    layers_oc[:, -1] = (substrate.n(wavelengths) + 1j*substrate.k(wavelengths))**2

    for i1, x in enumerate(structure):
        layers_oc[:, i1+1] = (x.material.n(wavelengths) + 1j*x.material.k(wavelengths))**2

    shapes_names = [str(x) for x in shape_mats]

    #nm_spacing = options['nm_spacing']
    phi_sym = options['phi_symmetry']
    n_theta_bins = options['n_theta_bins']
    c_az = options['c_azimuth']
    pol = options['pol']

    theta_intv, phi_intv, angle_vector = make_angle_vector(n_theta_bins, phi_sym, c_az)

    if only_incidence_angle:
        thetas_in = np.array([options['theta_in']])
        phis_in = np.array([options['phi_in']])
    else:
        angles_in = angle_vector[:int(len(angle_vector) / 2), :]
        thetas_in = angles_in[:, 1]
        phis_in = angles_in[:, 2]

    # initialise_S has to happen inside parallel job (get Pickle errors otherwise); just pass relevant optical constants for each wavelength, like for RT

    if options['parallel']:
        allres = Parallel(n_jobs=options['n_jobs'])(delayed(RCWA_wl)
                                                    (wavelengths[i1]*1e9, geom_list, layers_oc[i1], shapes_oc[i1], shapes_names, pol, thetas_in, phis_in, widths, size, orders)
                                                    for i1 in range(len(wavelengths)))

    else:
        allres = [RCWA_wl(wavelengths[i1]*1e9, geom_list, layers_oc[i1], shapes_oc[i1], shapes_names, pol, theta, phi, widths, size, orders)
                  for i1 in range(len(wavelengths))]

    R = np.stack([item[0] for item in allres])
    T = np.stack([item[1] for item in allres])
    A_layer = np.stack([item[2] for item in allres])

    return {'R': R, 'T':T, 'A_layer': A_layer}



def RCWA_wl(wl, geom_list, l_oc, s_oc, s_names, pol, theta, phi, widths, size, orders):

    S = initialise_S(size, orders, geom_list, l_oc, s_oc, s_names, widths)
    R = np.zeros((len(theta)))
    T = np.zeros((len(theta)))
    A_layer = np.zeros((len(theta), len(widths)-2))
    if pol in 'sp':
        if pol == 's':
            s = 1
            p = 0
        elif pol == 'p':
            s = 0
            p = 1
        for i1 in range(len(theta)):
            S.SetExcitationPlanewave((theta[i1], phi[i1]), s, p, 0)
            S.SetFrequency(1 / wl)
            out = rcwa_rat(S, len(widths))
            R[i1] = out['R']
            T[i1] = out['T']
            A_layer[i1] = rcwa_absorption_per_layer(S, len(widths))

    else:
        for i1 in range(len(theta)):
            S.SetFrequency(1 / wl)
            S.SetExcitationPlanewave((theta[i1], phi[i1]), 0, 1, 0)  # p-polarization
            out_p = rcwa_rat(S, len(widths))
            S.SetExcitationPlanewave((theta[i1], phi[i1]), 1, 0, 0)  # s-polarization
            out_s = rcwa_rat(S, len(widths))

            R[i1] = 0.5 * (out_p['R'] + out_s['R'])  # average
            T[i1] = 0.5 * (out_p['T'] + out_s['T'])
            # output['all_p'].append(out_p['power_entering_list'])
            # output['all_s'].append(out_s['power_entering_list'])
            A_layer[i1] = rcwa_absorption_per_layer(S, len(widths))

    return R, T, A_layer


def rcwa_rat(S, n_layers):
    below = 'layer_' + str(n_layers)  # identify which layer is the transmission medium
    R = 1 - sum(
        S.GetPowerFlux('layer_2'))  # GetPowerFlux gives forward & backward Poynting vector, so sum to get power flux
    # layer_2 is the top layer of the structure (layer_1 is incidence medium)
    T = sum(S.GetPowerFlux(below))
    return {'R': np.real(R), 'T': np.real(T)}


def initialise_S(size, orders, geom_list, mats_oc, shapes_oc, shape_mats, widths):
    # pass widths
    print(widths)
    S = S4.New(size, orders)
    S.SetOptions(  # these are the default
        LatticeTruncation='Circular',
        PolarizationDecomposition=False,
        PolarizationBasis='Default',
        WeismannFormulation = False
    )


    for i1 in range(len(shapes_oc)):  # create the materials needed for all the shapes in S4
        S.SetMaterial('shape_mat_' + str(i1 + 1), shapes_oc[i1])
        print('shape_mat_' + str(i1+1), shapes_oc[i1])

    ## Make the layers
    #stack_OS = OptiStack(stack, no_back_reflexion=False, substrate=substrate)
    #widths = stack_OS.get_widths()

    for i1 in range(len(widths)):  # create 'dummy' materials for base layers including incidence and transmission media
        S.SetMaterial('layer_' + str(i1 + 1), mats_oc[i1])  # This is not strictly necessary but it means S.SetExcitationPlanewave
        # can be done outside the wavelength loop in calculate_rat_rcwa
        print('layer_' + str(i1 + 1), mats_oc[i1])

    for i1 in range(len(widths)):  # set base layers
        layer_name = 'layer_' + str(i1 + 1)
        print(layer_name)
        if widths[i1] == float('Inf'):
            print('zero width')
            S.AddLayer(layer_name, 0, layer_name)  # Solcore4 has incidence and transmission media widths set to Inf;
            # in S4 they have zero width
        else:
            S.AddLayer(layer_name, widths[i1], layer_name)

        geometry = geom_list[i1]

        if bool(geometry):
            for shape in geometry:
                mat_name = 'shape_mat_' + str(shape_mats.index(str(shape['mat'])) + 1)
                print(str(shape['mat']), mat_name)
                if shape['type'] == 'circle':
                    S.SetRegionCircle(layer_name, mat_name, shape['center'], shape['radius'])
                elif shape['type'] == 'ellipse':
                    S.SetRegionEllipse(layer_name, mat_name, shape['center'], shape['angle'], shape['halfwidths'])
                elif shape['type'] == 'rectangle':
                    print('rect')
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
            print(i)
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