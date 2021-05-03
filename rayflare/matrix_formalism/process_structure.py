import numpy as np
import os
from rayflare.transfer_matrix_method.lookup_table import make_TMM_lookuptable
from rayflare.structure import Interface, RTgroup, BulkLayer
from rayflare.ray_tracing.rt import RT
from rayflare.rigorous_coupled_wave_analysis.rcwa import RCWA
from rayflare.transfer_matrix_method.tmm import TMM
from rayflare.angles import make_angle_vector
from rayflare.matrix_formalism.ideal_cases import lambertian_matrix, mirror_matrix

def process_structure(SC, options, save_location='default'):
    """
    Function which takes a list of Interface and BulkLayer objects and carries out the
    necessary calculations to populate the redistribution matrices.

    :param SC: list of Interface and BulkLayer objects. Order is [Interface, BulkLayer, Interface]
    :param options: options for the matrix calculations
    """

    layer_widths = []

    structpath = get_savepath(save_location, options['project_name'])

    for i1, struct in enumerate(SC):
        if isinstance(struct, BulkLayer):
            layer_widths.append(struct.width*1e9) # convert m to nm
        if isinstance(struct, Interface):
            layer_widths.append((np.array(struct.widths)*1e9).tolist()) # convert m to nm

    for i1, struct in enumerate(SC):
        if isinstance(struct, Interface):
            # Check: is this an interface type which requires a lookup table?
            if struct.method == 'RT_TMM':
                print('Making lookuptable for element ' + str(i1) + ' in structure')
                if i1 == 0:  # top interface
                    incidence = SC.incidence
                else: # not top interface
                    incidence = SC[i1-1].material # bulk material above

                if i1 == (len(SC) - 1): # bottom interface
                    substrate = SC.transmission
                else: # not bottom interface
                    substrate = SC[i1+1].material # bulk material below

                coherent = struct.coherent
                if not coherent:
                    coherency_list = struct.coherency_list
                else:
                    coherency_list = None

                prof_layers = struct.prof_layers

                make_TMM_lookuptable(struct.layers, incidence, substrate, struct.name,
                                              options, structpath, coherent, coherency_list, prof_layers)


    for i1, struct in enumerate(SC):
        if isinstance(struct, Interface):
            # perfect mirror

            if struct.method == 'Mirror':
                theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                                                options['c_azimuth'])
                mirror_matrix(angle_vector, theta_intv, phi_intv, struct.name, options, structpath,
                              front_or_rear='front', save=True)

            if struct.method == 'Lambertian':
                theta_intv, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                                                options['c_azimuth'])

                # assuming this is a Lambertian reflector right now
                lambertian_matrix(angle_vector, theta_intv, struct.name, structpath,
                                  'front', save = True)


            if struct.method == 'TMM':
                print('Making matrix for planar surface using TMM for element ' + str(i1) + ' in structure')
                if i1 == 0:
                    incidence = SC.incidence
                else:
                    incidence = SC[i1 - 1].material  # bulk material above

                if i1 == (len(SC) - 1):
                    substrate = SC.transmission
                    which_sides = ['front'] # if this is the bottom interface, we don't need to matrices for rear incidence
                else:
                    substrate = SC[i1 + 1].material  # bulk material below
                    which_sides = ['front', 'rear']

                coherent = struct.coherent
                if not coherent:
                    coherency_list = struct.coherency_list
                else:
                    coherency_list = None

                prof_layers = struct.prof_layers

                for side in which_sides:
                    TMM(struct.layers, incidence, substrate, struct.name, options, structpath,
                               coherent=coherent, coherency_list=coherency_list, prof_layers=prof_layers, front_or_rear=side, save=True)


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


                prof = struct.prof_layers
                n_abs_layers = len(struct.layers)

                group = RTgroup(textures=[struct.texture])
                for side in which_sides:
                    if side == 'front' and i1 == 0 and options['only_incidence_angle']:
                        only_incidence_angle = True
                    else:
                        only_incidence_angle = False

                    RT(group, incidence, substrate, struct.name, options, structpath, 1, side,
                       n_abs_layers, prof, only_incidence_angle, layer_widths[i1], save=True)

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
                    RT(group, incidence, substrate, struct.name, options, structpath, 0, side, 0, False, save=True)

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
                    RCWA(struct.layers, struct.d_vectors, struct.rcwa_orders, options, structpath,
                         incidence, substrate, only_incidence_angle=False,
                         front_or_rear=side, surf_name=struct.name, save=True)


def get_savepath(save_location, project_name):
    if save_location == 'current':
        cwd = os.getcwd()
        structpath = os.path.join(cwd, project_name)

    elif save_location == 'default':
        home = os.path.expanduser("~")
        structpath = os.path.join(home, "RayFlare_results", project_name)
        if not os.path.isdir(os.path.join(home, "RayFlare_results")):
            os.mkdir(os.path.join(home, "RayFlare_results"))

    else:
        structpath = save_location

    if not os.path.isdir(structpath):
        os.mkdir(structpath)

    return structpath
