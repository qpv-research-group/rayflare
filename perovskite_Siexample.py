import numpy as np
from solcore.structure import Layer
from transfer_matrix_method.lookup_table import make_TMM_lookuptable
from solcore import material
from solcore.material_system import create_new_material

from structure import Interface, RTgroup, BulkLayer, Structure
from ray_tracing.rt_lookup import RTSurface, RT
import math
from matrix_formalism.multiply_matrices import matrix_multiplication
from time import time

import matplotlib.pyplot as plt

### structure ###
# Air
# ---- pyramids: layers are GaInP, GaAs
# Si
# ---- pyramids: layers are GaAs
# Air

# matrix multiplication
wavelengths = np.linspace(300, 1150, 100)*1e-9
options = {'nm_spacing': 0.5,
           'project_name': 'testing2',
           'calc_profile': True,
           'n_theta_bins': 25,
           'c_azimuth': 0.25,
           'pol': 'u',
           'wavelengths': wavelengths,
           'theta_in': 0, 'phi_in': 'all',
           'I_thresh': 1e-4,
           'coherent': True,
           'coherency_list': None,
           'lookuptable_angles': 500,
           'prof_layers': [1,2],
           'n_rays': 50000,
           'random_angles': False,
           'nx': 2, 'ny': 2,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2}

#create_new_material('Perovskite_CsBr', 'examples/CsBr10p_1to2_n.txt', 'examples/CsBr10p_1to2_k.txt')
#create_new_material('ITO_lowdoping', 'examples/model_back_ito_n.txt', 'examples/model_back_ito_k.txt')
#create_new_material('Ag_Jiang', 'examples/Ag_UNSW_n.txt', 'examples/Ag_UNSW_k.txt')
#create_new_material('aSi_i', 'examples/model_i_a_silicon_n.txt', 'examples/model_i_a_silicon_k.txt')
#create_new_material('aSi_p', 'examples/model_p_a_silicon_n.txt', 'examples/model_p_a_silicon_k.txt')
#create_new_material('aSi_n', 'examples/model_n_a_silicon_n.txt', 'examples/model_n_a_silicon_k.txt')
#create_new_material('MgF2_RdeM', 'examples/MgF2_RdeM_n.txt', 'examples/MgF2_RdeM_k.txt')

Si = material('Si')()
Air = material('Air')()
MgF2 = material('MgF2_RdeM')()
ITO_back = material('ITO_lowdoping')()
Perovskite = material('Perovskite_CsBr')()
Ag = material('Ag_Jiang')()
aSi_i = material('aSi_i')()
aSi_p = material('aSi_p')()
aSi_n = material('aSi_n')()

char_angle = math.radians(55)
Lx = 1
Ly = 1
h = Lx*math.tan(char_angle)/2
x = np.array([0, Lx/2, Lx, 0, Lx])
y = np.array([0, Ly/2, 0, Ly, Ly])
z = np.array([0, h, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf = RTSurface(Points)

Lx = 1
Ly = 1
h = Lx*math.tan(char_angle)/2
x = np.array([0, Lx/2, Lx, 0, Lx])
y = np.array([0, Ly/2, 0, Ly, Ly])
z = np.array([0, -h, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf_back = RTSurface(Points)

front_materials = [Layer(100e-9, MgF2), Layer(440e-9, Perovskite),
                   Layer(6.5e-9, aSi_p), Layer(6.5e-9, aSi_i)]
back_materials = [Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_n), Layer(240e-9, ITO_back)]

front_surf = Interface('RT_TMM', texture = surf, layers=front_materials, name = 'Perovskite_aSi',
                       coherent=True, prof_layers = [1,2,3,4])
back_surf = Interface('RT_TMM', texture = surf_back, layers=back_materials, name = 'aSi_ITO',
                      coherent=True, prof_layers = [1,2,3])

bulk_Si = BulkLayer(260e-6, Si, name = 'Si_bulk')

SC = Structure([front_surf, bulk_Si, back_surf], incidence = Air, transmission = Ag)

# making lookuptable

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

            allres = make_TMM_lookuptable(struct.layers, substrate, incidence, struct.name,
                                          options, coherent, coherency_list, prof_layers)

            for side in [1, -1]:
                plt.figure()
                plt.subplot(2, 2, 1)
                allres['R'].sel(side=side, pol='s').plot.imshow('wl', 'angle')
                plt.subplot(2, 2, 3)
                allres['T'].sel(side=side, pol='s').plot.imshow('wl', 'angle')

                plt.subplot(2, 2, 2)
                allres['R'].sel(side=side, pol='p').plot.imshow('wl', 'angle')
                plt.subplot(2, 2, 4)
                allres['T'].sel(side=side, pol='p').plot.imshow('wl', 'angle')
                plt.show()
            for side in [1, -1]:

                plt.figure()
                for i1 in range(1, len(prof_layers) + 1):
                    plt.subplot(len(prof_layers), 1, i1)
                    allres['Alayer'].sel(side=side, pol='u', layer=i1).plot.imshow('wl', 'angle')
                plt.show()


# make matrices by ray tracing

for i1, struct in enumerate(SC):
    if type(struct) == Interface:
        # is this an interface type which requires a lookup table?
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

            coherent = struct.coherent
            if not coherent:
                coherency_list = struct.coherency_list
            else:
                coherency_list = None

            if len(struct.prof_layers) > 0:
                prof = True
            else:
                prof = False

            n_abs_layers = len(struct.layers)

            group = RTgroup(textures=[struct.texture])
            for side in which_sides:
                RT(group, incidence, substrate, struct.name, options, 1, side, n_abs_layers, prof)

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


# matrix multiplication
bulk_mats = []
bulk_widths = []
layer_widths = []
n_layers = []
layer_names = []

for i1, struct in enumerate(SC):
    if type(struct) == BulkLayer:
        bulk_mats.append(struct.material)
        bulk_widths.append(struct.width)

if options['calc_profile']:
    for i1, struct in enumerate(SC):
        if type(struct) == Interface:
            layer_names.append(struct.name)
            n_layers.append(len(struct.layers))
            widths = []
            for l in struct.layers:
                widths.append(l.width*1e9)
            layer_widths.append(widths)


RAT, profile, results_per_pass = matrix_multiplication(bulk_mats, bulk_widths, options,
                                                           layer_widths, n_layers, layer_names)

plt.figure()
plt.plot(options['wavelengths'], RAT['R'].T)
plt.plot(options['wavelengths'], RAT['T'].T)
plt.plot(options['wavelengths'], RAT['A_interface'].T)
plt.plot(options['wavelengths'], RAT['A_bulk'].T)
#plt.plot(options['wavelengths'], R[0] + T[0] + A_interface[0] + A_interface[1] + A_bulk[0])
plt.legend(['R', 'T', 'front', 'back', 'bulk'])
plt.show()

plt.figure()
for i1, prof in enumerate(profile):
     z = np.arange(0, np.sum(layer_widths[i1]), options['nm_spacing'])
     plt.plot(prof[0,:])
plt.show()
#
#
# R_per_pass = np.sum(results_per_pass['r'][0][:,3,:], 1)
# T_per_pass = np.sum(results_per_pass['t'][0][:,3,:], 1)
# A_per_pass = results_per_pass['A'][0][:,3]
# front_A_per_pass = np.sum(results_per_pass['a'][0][:,3,:], 1)
# back_A_per_pass = results_per_pass['a'][1][:,3,0]
#
# plt.figure()
# plt.plot(R_per_pass[0:10])
# plt.plot(T_per_pass[0:10])
