import numpy as np
from solcore.structure import Layer
from transfer_matrix_method.lookup_table import make_TMM_lookuptable
from solcore import material
from solcore.material_system import create_new_material

from solcore.absorption_calculator.nk_db import search_db

from structure import Interface, RTgroup, BulkLayer, Structure
from ray_tracing.rt_lookup import RTSurface, RT
import math
from matrix_formalism.multiply_matrices import matrix_multiplication
from time import time
from sparse import stack, COO
from angles import make_angle_vector
import xarray as xr

import matplotlib.pyplot as plt

### structure ###
# Air
# ---- pyramids: layers are GaInP, GaAs
# Si
# ---- pyramids: layers are GaAs
# Air

# matrix multiplication
wavelengths = np.linspace(300, 1200, 112)*1e-9
options = {'nm_spacing': 0.5,
           'project_name': 'UC_PC',
           'calc_profile': False,
           'n_theta_bins': 100,
           'c_azimuth': 0.25,
           'pol': 'u',
           'wavelengths': wavelengths,
           'theta_in': 1e-6, 'phi_in': 1e-6,
           'I_thresh': 1e-4,
           'coherent': True,
           'coherency_list': None,
           'lookuptable_angles': 500,
           #'prof_layers': [1,2],
           'n_rays': 1000000,
           'random_angles': False,
           'nx': 5, 'ny': 5,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2,
           'only_incidence_angle': True
           }


#create_new_material('Perovskite_CsBr', 'examples/CsBr10p_1to2_n.txt', 'examples/CsBr10p_1to2_k.txt')
#create_new_material('ITO_lowdoping', 'examples/model_back_ito_n.txt', 'examples/model_back_ito_k.txt')
#create_new_material('Ag_Jiang', 'examples/Ag_UNSW_n.txt', 'examples/Ag_UNSW_k.txt')
#create_new_material('aSi_i', 'examples/model_i_a_silicon_n.txt', 'examples/model_i_a_silicon_k.txt')
#create_new_material('aSi_p', 'examples/model_p_a_silicon_n.txt', 'examples/model_p_a_silicon_k.txt')
#create_new_material('aSi_n', 'examples/model_n_a_silicon_n.txt', 'examples/model_n_a_silicon_k.txt')
#create_new_material('MgF2_RdeM', 'examples/MgF2_RdeM_n.txt', 'examples/MgF2_RdeM_k.txt')
#create_new_material('C60', 'examples/C60_Ren_n.txt', 'examples/C60_Ren_k.txt')
#create_new_material('IZO', 'examples/IZO_Ballif_rO2_10pcnt_n.txt', 'examples/IZO_Ballif_rO2_10pcnt_k.txt')


Si = material('Si')()
Air = material('Air')()
MgF2 = material('MgF2_RdeM')()
ITO_back = material('ITO_lowdoping')()
Perovskite = material('Perovskite_CsBr')()
Ag = material('Ag_Jiang')()
aSi_i = material('aSi_i')()
aSi_p = material('aSi_p')()
aSi_n = material('aSi_n')()
LiF = material('LiF')()
IZO = material('IZO')()
C60 = material('C60')()
Spiro = [12e-9, np.array([0,1]), np.array([1.65, 1.65]), np.array([0,0])]
SnO2 = [10e-9, np.array([0,1]), np.array([2, 2]), np.array([0,0])]

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

front_materials = [Layer(100e-9, MgF2), Layer(110e-9, IZO),
                   SnO2,
                   Layer(15e-9, C60),
                   Layer(1e-9, LiF),
                   Layer(440e-9, Perovskite),
                   Spiro,
                   Layer(6.5e-9, aSi_n), Layer(6.5e-9, aSi_i)]
back_materials = [Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_p), Layer(240e-9, ITO_back)]


front_surf = Interface('RT_TMM', texture = surf, layers=front_materials, name = 'Perovskite_aSi_1e6',
                       coherent=True, prof_layers = [1,2,3,4,5,6,7,8,9])
back_surf = Interface('RT_TMM', texture = surf_back, layers=back_materials, name = 'aSi_ITO_1e6',
                      coherent=True, prof_layers = [1,2,3])



bulk_Si = BulkLayer(260e-6, Si, name = 'Si_bulk')

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

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

            make_TMM_lookuptable(struct.layers, substrate, incidence, struct.name,
                                          options, coherent, coherency_list, prof_layers)

            # for side in [1, -1]:
            #     plt.figure()
            #     plt.subplot(2, 2, 1)
            #     allres['R'].sel(side=side, pol='s').plot.imshow('wl', 'angle')
            #     plt.subplot(2, 2, 3)
            #     allres['T'].sel(side=side, pol='s').plot.imshow('wl', 'angle')
            #
            #     plt.subplot(2, 2, 2)
            #     allres['R'].sel(side=side, pol='p').plot.imshow('wl', 'angle')
            #     plt.subplot(2, 2, 4)
            #     allres['T'].sel(side=side, pol='p').plot.imshow('wl', 'angle')
            #     plt.show()
            # for side in [1, -1]:
            #
            #     plt.figure()
            #     for i1 in range(1, len(prof_layers) + 1):
            #         plt.subplot(len(prof_layers), 1, i1)
            #         allres['Alayer'].sel(side=side, pol='u', layer=i1).plot.imshow('wl', 'angle')
            #     plt.show()


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
                if side == 'front' and i1 == 0 and options['only_incidence_angle']:
                    only_incidence_angle = True
                else:
                    only_incidence_angle = False
                print(only_incidence_angle)
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
            layer_widths.append((np.array(struct.widths)*1e9).tolist())


RAT, profile, results_per_pass, bpf = matrix_multiplication(bulk_mats, bulk_widths, options,
                                                           layer_widths, n_layers, layer_names)

profile_bulk = np.sum(bpf, (0,1))
zb = np.arange(0, 260e-6, 0.5e-9)
plt.figure()
plt.plot(zb, profile_bulk[3])

print(np.trapz(profile_bulk, zb))

R_per_pass = np.sum(results_per_pass['r'][0], 2)
R_0 = R_per_pass[0]
R_escape = np.sum(R_per_pass[1:, :], 0)
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)
results_per_layer_back = np.sum(results_per_pass['a'][1], 0)

allres = np.flip(np.hstack((RAT['R'].T, results_per_layer_front, RAT['A_bulk'].T,
                    results_per_layer_back, RAT['T'].T)),1)
# Plot
import seaborn as sns
pal = sns.cubehelix_palette(15, start = 0)

plt.figure()
plt.stackplot(options['wavelengths']*1e9, allres.T,
              labels=['T', 'ITO', 'aSi-n', 'aSi-i', 'c-Si (bulk)', 'aSi-i', 'aSi-p', 'Perovskite',
                      'Spiro','LiF','C60', 'SnO2','IZO',
            'MgF_2', 'R'], colors = pal)
plt.legend(loc='center left')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorption')
plt.xlim(300, 1150)
plt.ylim(0, 1)
plt.show()
#
# plt.figure()
# plt.plot(options['wavelengths'], RAT['R'].T)
# plt.plot(options['wavelengths'], results_per_layer_front)
# plt.plot(options['wavelengths'], RAT['A_bulk'].T)
# plt.plot(options['wavelengths'], results_per_layer_back)
# plt.plot(options['wavelengths'], RAT['T'].T)
# #plt.plot(options['wavelengths'], R[0] + T[0] + A_interface[0] + A_interface[1] + A_bulk[0])
# plt.legend(['R', 'MgF_2', 'Perovskite', 'aSi-p', 'aSi-i', 'c-Si (bulk)', 'aSi-i', 'aSi-n',
#             'ITO', 'T'])
# plt.show()
#
#plt.figure()
#for i1, prof in enumerate(profile):
#      z = np.arange(0, np.sum(layer_widths[i1]), options['nm_spacing'])
#      plt.plot(prof[0,:])
#plt.show()
offset = np.cumsum([0]+layer_widths[0])
prof_plot = profile[0]
prof_plot = prof_plot.assign_coords(z=np.arange(0, np.sum(layer_widths[0]), options['nm_spacing']))
material_labels = ['MgF$_2$', 'IZO', 'SnO$_2$', 'C$_{60}$', 'LiF', 'Perovskite', 'Spiro-TTB', 'aSi-n', 'aSi-i']
pal2 = sns.cubehelix_palette(5, start = 0)
plt.figure()
j1=0
for i1 in [0,1,2,3]:
      z = np.arange(0, np.sum(layer_widths[0]), options['nm_spacing'])
      plt.plot(z, prof_plot[i1,:], color=pal2[4-j1], label = str(round(options['wavelengths'][i1]*1e9,1)))
      j1+=1
plt.ylabel('Absorbed energy density (nm$^{-1}$)')
plt.legend(title='Wavelength (nm)')
plt.xlabel('Distance into surface (nm)')
plt.ylim(0,0.0165)
plt.xlim(0,750)
for i1 in [0,1,3,5,7]:
    plt.text(offset[i1], np.max(prof_plot[0].sel(z=offset[i1]))+0.0005, material_labels[i1], rotation=0)
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

# def make_D_abs(alphas, z, thetas):
#     """
#     Makes the bulk absorption vector for the bulk material
#     :param alphas: absorption coefficient (m^{-1})
#     :param thick: thickness of the slab in m
#     :param thetas: incident thetas in angle_vector (second column)
#     :return:
#     """
#     diag = np.exp(-alphas[:, :, None] * z[:,None,:] / np.cos(thetas[None, :]))
#     D_1 = stack([COO(np.diag(x)) for x in diag])
#     return D_1
#
# z_bulk = np.arange(0, 260*1e-6, 1e-6)
#
# theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'],
#                                                        np.pi / 2, options['c_azimuth'])
#
# v1 = np.random.rand(676)
#
# v1 = v1/np.sum(v1)
#
# alphas = Si.alpha(options['wavelengths'])
#
# alphas*z_bulk[:, None]
# thetas = angle_vector[:,1]
#
# pref = xr.DataArray(v1, dims=['theta'], coords={'theta': thetas})
# alphas = xr.DataArray(alphas, dims=['wl'], coords={'wl': options['wavelengths']})
# thetas = xr.DataArray(thetas, dims=['theta'], coords={'theta': thetas})
# z = xr.DataArray(z_bulk, dims=['z'], coords={'z': z_bulk})
#
# ans = pref*np.exp(-alphas*z/ np.cos(thetas))
#

