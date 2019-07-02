import numpy as np
from transfer_matrix_method.lookup_table import make_TMM_lookuptable
from structure import Interface, RTgroup, BulkLayer
from ray_tracing.rt_lookup import RTSurface, RT
from matrix_formalism.multiply_matrices import matrix_multiplication
from rigorous_coupled_wave_analysis.rcwa import rcwa
from transfer_matrix_method.transfer_matrix import tmm_matrix


def process_structure(SC, options):

    for i1, struct in enumerate(SC):
        if type(struct) == Interface:
            # is this an interface type which requires a lookup table?
            if struct.method == 'RT_TMM':
                print('Making lookuptable for element ' + str(i1) + ' in structure')
                if i1 == 0:
                    incidence = SC.incidence
                else:
                    incidence = SC[i1-1].material # bulk material above

                if i1 == (len(SC) - 1):
                    substrate = SC.transmission
                else:
                    substrate = SC[i1+1].material # bulk material below

                coherent = struct.coherent
                if not coherent:
                    coherency_list = struct.coherency_list
                else:
                    coherency_list = None
                prof_layers = struct.prof_layers

                make_TMM_lookuptable(struct.layers, substrate, incidence, struct.name,
                                              options, coherent, coherency_list, prof_layers)


    # make matrices by ray tracing

    for i1, struct in enumerate(SC):
        if type(struct) == Interface:
            # is this an interface type which requires a lookup table?

            if struct.method == 'Mirror':
                from angles import make_angle_vector
                from sparse import stack, COO, concatenate, save_npz
                from config import results_path
                import os
                _, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                       options['c_azimuth'])
                diag = np.array([1] * int(len(angle_vector) / 2))
                Rsp = stack([COO(np.diag(diag)) for _ in range(len(options['wavelengths']))])
                Tsp = COO([], [], (len(options['wavelengths']), int(len(angle_vector) / 2), int(len(angle_vector) / 2)))
                RTsp = concatenate((Rsp, Tsp), axis=1)
                Asp = COO([], [], (len(options['wavelengths']), 0, int(len(angle_vector) / 2)))
                savepath_RT = os.path.join(results_path, options['project_name'], struct.name + 'front' + 'RT.npz')
                savepath_A = os.path.join(results_path, options['project_name'], struct.name + 'front' + 'A.npz')
                save_npz(savepath_RT, RTsp)
                save_npz(savepath_A, Asp)

            if struct.method == 'TMM':
                print('Making matrix for planar surface using TMM for element ' + str(i1) + ' in structure')
                if i1 == 0:
                    incidence = SC.incidence
                else:
                    incidence = SC[i1 - 1].material  # bulk material above

                if i1 == (len(SC) - 1):
                    substrate = SC.transmission
                    which_sides = ['front']
                else:
                    substrate = SC[i1 + 1].material  # bulk material below
                    which_sides = ['front', 'rear']

                coherent = struct.coherent
                if not coherent:
                    coherency_list = struct.coherency_list
                else:
                    coherency_list = None

                for side in which_sides:
                    # print(only_incidence_angle)
                    tmm_matrix(struct.layers, substrate, incidence, struct.name, options,
                               coherent=coherent, coherency_list=coherency_list, prof_layers=None, front_or_rear=side)


            if struct.method == 'RT_TMM':
                print('Ray tracing with TMM lookup table for element ' + str(i1) + ' in structure')
                if i1 == 0:
                    incidence = SC.incidence
                else:
                    incidence = SC[i1-1].material # bulk material above

                if i1 == (len(SC) - 1):
                    substrate = SC.transmission
                    which_sides = ['front']
                else:
                    substrate = SC[i1+1].material # bulk material below
                    which_sides = ['front', 'rear']

                if len(struct.prof_layers) > 0:
                    prof = True
                else:
                    prof = False

                n_abs_layers = len(struct.layers)

                group = RTgroup(textures=[struct.texture])
                for side in which_sides:
                    if side == 'front' and i1 == 0 and options['only_incidence_angle']:
                        only_incidence_angle = True
                    else:
                        only_incidence_angle = False
                    #print(only_incidence_angle)
                    RT(group, incidence, substrate, struct.name, options, 1, side,
                       n_abs_layers, prof, only_incidence_angle)

            if struct.method == 'RT_Fresnel':
                print('Ray tracing with Fresnel equations for element ' + str(i1) + ' in structure')
                if i1 == 0:
                    incidence = SC.incidence
                else:
                    incidence = SC[i1-1].material # bulk material above

                if i1 == (len(SC) - 1):
                    substrate = SC.transmission
                    which_sides = ['front']
                else:
                    substrate = SC[i1+1].material # bulk material below
                    which_sides = ['front', 'rear']

                group = RTgroup(textures=[struct.texture])
                for side in which_sides:
                    RT(group, incidence, substrate, struct.name, options, 0, side, 0, False)

            if struct.method == 'RCWA':
                print('RCWA calculation for element ' + str(i1) + ' in structure')
                if i1 == 0:
                    incidence = SC.incidence
                else:
                    incidence = SC[i1-1].material # bulk material above

                if i1 == (len(SC) - 1):
                    substrate = SC.transmission
                    which_sides = ['front']
                else:
                    substrate = SC[i1+1].material # bulk material below
                    which_sides = ['front', 'rear']
                for side in which_sides:
                    rcwa(struct.layers, struct.d_vectors, struct.rcwa_orders, options, incidence, substrate, only_incidence_angle=False,
                         front_or_rear=side, surf_name=struct.name)




def calculate_RAT(SC, options):
    bulk_mats = []
    bulk_widths = []
    layer_widths = []
    n_layers = []
    layer_names = []

    for i1, struct in enumerate(SC):
        if type(struct) == BulkLayer:
            bulk_mats.append(struct.material)
            bulk_widths.append(struct.width)
        if type(struct) == Interface:
            layer_names.append(struct.name)

            if options['calc_profile']:
                    n_layers.append(len(struct.layers))
                    layer_widths.append((np.array(struct.widths)*1e9).tolist())


    results = matrix_multiplication(bulk_mats, bulk_widths, options,
                                                               layer_widths, n_layers, layer_names)

    return results