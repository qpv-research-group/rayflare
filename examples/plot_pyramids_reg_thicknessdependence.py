from ray_tracing.rt import rt_structure
from time import time
from textures.standard_rt_textures import regular_pyramids, planar_surface
from solcore import material
from solcore import si
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
pal = sns.cubehelix_palette()

cols = cycler('color', pal)

params = {'legend.fontsize': 'small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small',
          'axes.prop_cycle': cols}

plt.rcParams.update(params)

Air = material('Air')()
Si = material('566', nk_db=True)()
size = 3
nxy = 21
#
# calc = False
# options = {'wavelengths': np.array([1100e-9]), 'theta': 0 * np.pi / 180, 'phi': 0,
#            'I_thresh': 0.02, 'nx': nxy, 'ny': nxy,
#            'parallel': True, 'pol': 'u', 'n_rays': 4 * nxy ** 2, 'depth_spacing': si('1um'),
#            'random_ray_position': False}
#
#
# thickness1 = np.arange(280, 320, 0.25)
# res_reg1 = []
# res_rand1 = []
#
# for th in thickness1:
#
#     flat_surf = planar_surface()
#     triangle_surf = regular_pyramids(55, upright=True, size=size)
#     options['avoid_edges'] = False
#     options['randomize'] = False
#     rtstr1 = rt_structure(textures=[triangle_surf, flat_surf],
#                         materials = [Si],
#                         widths=[si(th, 'um')], incidence=Air, transmission=Air)
#
#     start = time()
#     result_new1 = rtstr1.calculate(options)
#     print(str(size), time()-start)
#     res_reg1.append(result_new1['A_per_layer'][:,0][0])
#
# for th in thickness1:
#     flat_surf = planar_surface()
#     triangle_surf = regular_pyramids(55, upright=True, size=size)
#     options['avoid_edges'] = False
#     options['randomize'] = True
#     rtstr2 = rt_structure(textures=[triangle_surf, flat_surf],
#                          materials=[Si],
#                          widths=[si(th, 'um')], incidence=Air, transmission=Air)
#
#     start = time()
#     result_new2 = rtstr2.calculate(options)
#     print(str(size), time()-start)
#     res_rand1.append(result_new2['A_per_layer'][:,0][0])
#






# size = 5
#
# calc = False
# options = {'wavelengths': np.array([1100e-9]), 'theta': 0 * np.pi / 180, 'phi': 0,
#            'I_thresh': 0.02, 'nx': nxy, 'ny': nxy,
#            'parallel': True, 'pol': 'u', 'n_rays': 4 * nxy ** 2, 'depth_spacing': si('1um'),
#            'random_ray_position': False}
#
#
# thickness = np.arange(280, 320, 0.25)
# res_reg = []
# res_rand = []
#
# for th in thickness:
#
#
#     flat_surf = planar_surface()
#     triangle_surf = regular_pyramids(55, upright=True, size=size)
#     options['avoid_edges'] = False
#     options['randomize'] = False
#     rtstr1 = rt_structure(textures=[triangle_surf, flat_surf],
#                         materials = [Si],
#                         widths=[si(th, 'um')], incidence=Air, transmission=Air)
#
#     start = time()
#     result_new1 = rtstr1.calculate(options)
#     print(str(size), time()-start)
#     res_reg.append(result_new1['A_per_layer'][:,0][0])
#
# for th in thickness:
#     flat_surf = planar_surface()
#     triangle_surf = regular_pyramids(55, upright=True, size=size)
#     options['avoid_edges'] = False
#     options['randomize'] = True
#     rtstr2 = rt_structure(textures=[triangle_surf, flat_surf],
#                          materials=[Si],
#                          widths=[si(th, 'um')], incidence=Air, transmission=Air)
#
#     start = time()
#     result_new2 = rtstr2.calculate(options)
#     print(str(size), time()-start)
#     res_rand.append(result_new2['A_per_layer'][:,0][0])



# plt.figure()
# plt.plot(thickness1, res_rand1, label='rand')
# plt.plot(thickness1, res_reg1, label='reg')
# plt.legend()
# plt.title('2')
# plt.show()

# plt.figure()
# plt.plot(thickness, res_rand, 'o-', label='rand')
# plt.plot(thickness, res_reg, 'o-',  label='reg')
# plt.legend()
# plt.title('5')
# plt.show()
#
#
# plt.figure()
# A_single_pass = 1 - np.exp(-200e-6 * Si.alpha(options['wavelengths']))
# # plt.plot(thickness1, np.array(res_rand1)/A_single_pass, label='2 rand')
# # plt.plot(thickness1, np.array(res_reg1)/A_single_pass, label='2 reg')
# plt.plot(thickness, np.array(res_rand)/A_single_pass, label='5 rand')
# plt.plot(thickness, np.array(res_reg)/A_single_pass, label='5 reg')
# plt.legend()
# plt.show()
#
# # save results
# np.savetxt('rand_5_1150nm.txt', np.vstack((thickness, res_rand)).T)
# np.savetxt('reg_5_1150nm.txt', np.vstack((thickness, res_reg)).T)
# np.savetxt('rand_2.txt', np.vstack((thickness, res_rand1)).T)
# np.savetxt('reg_2.txt', np.vstack((thickness, res_reg1)).T)

rand2 = np.loadtxt('rand_2.txt')
reg2= np.loadtxt('reg_2.txt')


#
#
#
# # might be interesting to try perpendicular slats?
#
# # W = (m*d/2)*np.tan(th1+th2)

# d = (5/2)*np.tan(np.arctan(1/np.sqrt(2)) + 70*np.pi/180)
# size = 5
# thickness_high = 302.5
# thickness_low = 293.5
# result = []
# for thickness in[thickness_high, thickness_low]:
#     print('THICKNESS', thickness)
#     flat_surf = planar_surface()
#     triangle_surf = regular_pyramids(55, upright=True, size=size)
#     options['avoid_edges'] = False
#     options['randomize'] = True
#     rtstr2 = rt_structure(textures=[triangle_surf, flat_surf],
#                           materials=[Si],
#                           widths=[si(thickness, 'um')], incidence=Air, transmission=Air)
#
#     start = time()
#     result_new2 = rtstr2.calculate(options)
#     result.append(result_new2)
#
# plt.figure()
# plt.hist(result[0]['thetas'][0,:])
# plt.show()
#
# plt.figure()
# plt.hist(result[1]['thetas'][0,:])
# plt.show()
#
# n_R0_1 = np.sum((result[0]['n_passes'][0,:] == 1) * (result[0]['thetas'][0,:] < np.pi/2))
# n_R0_2 = np.sum((result[1]['n_passes'][0,:] == 1) * (result[1]['thetas'][0,:] < np.pi/2))
#
#
# n_T0_1 = np.sum((result[0]['n_passes'][0,:] == 3) * (result[0]['thetas'][0,:] > np.pi/2))
# n_T0_2 = np.sum((result[1]['n_passes'][0,:] == 3) * (result[1]['thetas'][0,:] > np.pi/2))
#
# n_R1_1 = np.sum((result[0]['n_passes'][0,:] == 3) * (result[0]['thetas'][0,:] < np.pi/2))
# n_R1_2 = np.sum((result[1]['n_passes'][0,:] == 3) * (result[1]['thetas'][0,:] < np.pi/2))




from solcore.structure import Layer

from solcore import si
from structure import Interface, BulkLayer, Structure
from matrix_formalism.process_structure import process_structure
from matrix_formalism.multiply_matrices import calculate_RAT

from textures.standard_rt_textures import regular_pyramids


angle_degrees_in = 0

# matrix multiplication
wavelengths = np.array([1100e-9])
options = {'nm_spacing': 0.5,
           'project_name': 'optos_for_random',
           'calc_profile': False,
           'n_theta_bins': 30,
           'c_azimuth': 0.25,
           'pol': 'u',
           'wavelengths': wavelengths,
           'theta_in': angle_degrees_in*np.pi/180, 'phi_in': 1e-6,
           'I_thresh': 0.001,
           #'coherent': True,
           #'coherency_list': None,
           'lookuptable_angles': 200,
           #'prof_layers': [1,2],
           'n_rays': 1e5,
           'random_angles': False,
           'nx': nxy, 'ny': nxy,
           'random_ray_position': False,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2,
           'only_incidence_angle': True,
           'avoid_edges': False
           }



# whether pyramids are upright or inverted is relative to front incidence.
# so if the same etch is applied to both sides of a slab of silicon, one surface
# will have 'upright' pyramids and the other side will have 'not upright' (inverted)
# pyramids in the model
size = 2
surf = regular_pyramids(elevation_angle=55, upright=True, size = size)


front_surf = Interface('RT_TMM', texture = surf, layers=[Layer(si('0.01nm'), Air)],
                       name = 'inv_pyramids' + str(options['n_rays']) + str(size))
back_surf = Interface('TMM', layers=[], name = 'planar_back' + str(options['n_rays']))


bulk_Si = BulkLayer(201.8e-6, Si, name = 'Si_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Air)

process_structure(SC, options)

results = calculate_RAT(SC, options)

OPTOS_res = []
thicknesses = np.arange(180, 220, 1)

for bulk_thick in thicknesses:
    bulk_Si = BulkLayer(bulk_thick*1e-6, Si, name='Si_bulk')  # bulk thickness in m

    SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Air)

    process_structure(SC, options)
    results = calculate_RAT(SC, options)
    OPTOS_res.append(np.asscalar(results[0]['A_bulk'][0][0].data))


plt.figure()

plt.plot(thickness, res_rand, 'o-', label='rand')
plt.plot(thickness, res_reg,'o-',  label='reg')
plt.plot(thicknesses, OPTOS_res)
plt.legend()
plt.title('5')
plt.show()