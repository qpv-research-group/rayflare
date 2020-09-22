from rayflare.ray_tracing.rt import rt_structure
from time import time
from textures.standard_rt_textures import regular_pyramids, planar_surface
from solcore import material
from solcore import si
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

pal = sns.color_palette("husl", 3)

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
size = 2
nxy = 41

calc = False
options = {'wavelengths': np.array([1100e-9]), 'theta': 0 * np.pi / 180, 'phi': 0,
           'I_thresh': 0.02, 'nx': nxy, 'ny': nxy,
           'parallel': True, 'pol': 'u', 'n_rays': 1 * nxy ** 2, 'depth_spacing': si('1um'),
           'random_ray_position': False}


thickness1 = np.arange(190, 210, 0.1)
res_reg1 = []
res_rand1 = []

if calc:
    # for th in thickness1:
    #
    #     flat_surf = planar_surface(size=size)
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
    #     flat_surf = planar_surface(size=size)
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




    size = 5

    calc = False
    options = {'wavelengths': np.array([1100e-9]), 'theta': 0 * np.pi / 180, 'phi': 0,
               'I_thresh': 0.02, 'nx': nxy, 'ny': nxy,
               'parallel': True, 'pol': 'u', 'n_rays': 1 * nxy ** 2, 'depth_spacing': si('1um'),
               'random_ray_position': False}


    thickness = np.arange(190, 210, 0.1)
    res_reg = []
    res_rand = []

    for th in thickness:


        flat_surf = planar_surface(size=size)
        triangle_surf = regular_pyramids(55, upright=True, size=size)
        options['avoid_edges'] = False
        options['randomize'] = False
        rtstr1 = rt_structure(textures=[triangle_surf, flat_surf],
                            materials = [Si],
                            widths=[si(th, 'um')], incidence=Air, transmission=Air)

        start = time()
        result_new1 = rtstr1.calculate(options)
        print(str(size), time()-start)
        res_reg.append(result_new1['A_per_layer'][:,0][0])

    for th in thickness:
        flat_surf = planar_surface(size=size)
        triangle_surf = regular_pyramids(55, upright=True, size=size)
        options['avoid_edges'] = False
        options['randomize'] = True
        rtstr2 = rt_structure(textures=[triangle_surf, flat_surf],
                             materials=[Si],
                             widths=[si(th, 'um')], incidence=Air, transmission=Air)

        start = time()
        result_new2 = rtstr2.calculate(options)
        print(str(size), time()-start)
        res_rand.append(result_new2['A_per_layer'][:,0][0])



    #plt.figure()
    #plt.plot(thickness1, res_rand1, label='rand')
    #plt.plot(thickness1, res_reg1, label='reg')
    #plt.legend()
    #plt.title('2')
    #plt.show()

    plt.figure()
    plt.plot(thickness, res_rand, 'o-', label='rand')
    plt.plot(thickness, res_reg, 'o-',  label='reg')
    plt.legend()
    plt.title('5')
    plt.show()

    th1 = 35.26 * np.pi / 180
    th2 = np.arcsin(np.cos(th1) / Si.n(1100e-9))
    n = np.arange(200)
    max = (n * 5 / 2) * np.tan(th1 + th2)
    pp = np.ones_like(max) * 6
    plt.figure()
    A_single_pass = 1 - np.exp(-200e-6 * Si.alpha(options['wavelengths']))
    #plt.plot(thickness1, np.array(res_rand1)/A_single_pass, 'o-', label='2 rand')
    #plt.plot(thickness1, np.array(res_reg1)/A_single_pass, 'D-', label='2 reg')
    plt.plot(max+2, pp, 'o')
    plt.plot(thickness, np.array(res_rand)/A_single_pass,'o--', label='5 rand', fillstyle='none')
    plt.plot(thickness, np.array(res_reg)/A_single_pass, 'D--', label='5 reg', fillstyle='none')
    plt.xlim(190, 210)
    plt.legend()
    plt.show()

    avg_th_1 = np.mean(np.array(thickness1[:-1]).reshape(-1,3), axis=1)
    #avg_rand_1 = np.mean(np.array(res_rand1[:-1]).reshape(-1,3), axis=1)
    #avg_reg_1 = np.mean(np.array(res_reg1[:-1]).reshape(-1,3), axis=1)
    avg_th = np.mean(np.array(thickness[:-1]).reshape(-1,3), axis=1)
    avg_rand = np.mean(np.array(res_rand[:-1]).reshape(-1,3), axis=1)
    avg_reg = np.mean(np.array(res_reg[:-1]).reshape(-1,3), axis=1)


    plt.figure()
    A_single_pass = 1 - np.exp(-200e-6 * Si.alpha(options['wavelengths']))
    #plt.plot(avg_th_1, np.array(avg_rand_1)/A_single_pass, 'o-', label='2 rand')
    #plt.plot(avg_th_1, np.array(avg_reg_1)/A_single_pass, 'D-', label='2 reg')
    plt.plot(avg_th, np.array(avg_rand)/A_single_pass,'o-', label='Random', fillstyle='none')
    plt.plot(avg_th, np.array(avg_reg)/A_single_pass, 'D-', label='Regular', fillstyle='none')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Path length enhancement')
    plt.legend()
    plt.show()

    n=1
    th1 = 35.26*np.pi/180
    th2 = np.arcsin(np.cos(th1)/Si.n(1100e-9))
    (n*2/2)*np.tan(th1 + th2)
    #
    # # save results
    np.savetxt('rand_5_41.txt', np.vstack((thickness, res_rand)).T)
    np.savetxt('reg_5_41.txt', np.vstack((thickness, res_reg)).T)
    # np.savetxt('rand_2_41.txt', np.vstack((thickness1, res_rand1)).T)
    # np.savetxt('reg_2_41.txt', np.vstack((thickness1, res_reg1)).T)

else:
    res_rand =np.loadtxt('rand_5_41.txt')
    res_reg = np.loadtxt('reg_5_41.txt')

    thickness = res_rand[:,0]
    res_rand = res_rand[:,1]
    res_reg = res_reg[:,1]


reg = regular_pyramids(size=5)



#avg_th = np.mean(np.array(thickness[:-1]).reshape(-1, 3), axis=1)
#avg_rand = np.mean(np.array(res_rand[:-1]).reshape(-1, 3), axis=1)
#avg_reg = np.mean(np.array(res_reg[:-1]).reshape(-1, 3), axis=1)

A_single_pass = 1 - np.exp(-thickness*1e-6 * Si.alpha(options['wavelengths']))

th1 = 35.26 * np.pi / 180
th2 = np.arcsin(np.cos(th1) / Si.n(1100e-9))
n = np.arange(200)
max = (n * 5 / 2) * np.tan(th1 + th2)
pp = np.ones_like(max) * 6

# plt.plot(thickness1, np.array(res_rand1)/A_single_pass, 'o-', label='2 rand')
# plt.plot(thickness1, np.array(res_reg1)/A_single_pass, 'D-', label='2 reg')
#plt.plot(max-1.2, pp, 'o')

fig = plt.figure(figsize=(9.5, 3.7))
grid = plt.GridSpec(1, 3, wspace=0, hspace=0.0, width_ratios=[0.48, 0.04, 0.48])#, wspace=0.4, hspace=0.3)
ax = plt.subplot(grid[0,0], projection='3d')
ax.view_init(elev=30., azim=60)
#ax.set_aspect('equal')
ax.plot_trisurf(reg[0].Points[:,0], reg[0].Points[:,1], reg[0].Points[:,2],
                triangles=reg[0].simplices,  linewidth=1, color = (0.8, 0.8, 0.8, 0.8))
ax.text(7, 0, 5, 'a)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


ax2 = plt.subplot(grid[0,2])
ax2.plot(thickness, np.array(res_rand) / A_single_pass, 'o', label='Random', fillstyle='none', color=pal[0])
ax2.plot(thickness, np.array(res_reg) / A_single_pass, 'D', label='Regular', fillstyle='none', color=pal[2])
ax2.plot(thickness[1:-1], moving_average(np.array(res_rand) / A_single_pass), '-',
         fillstyle='none', color=pal[0])
ax2.plot(thickness[1:-1], moving_average(np.array(res_reg) / A_single_pass), '-',
         fillstyle='none', color=pal[2])
ax2.plot(1, 1, '-k', label='Moving average')
ax2.set_xlim(190, 210)
ax2.set_ylim(5.2, 8)
ax2.set_xlabel('Si thickness ($\mu$m)')
ax2.set_ylabel('Path length enhancement')
ax2.text(187.5, 8, 'b)')

ax2.legend()

fig.savefig('regularrandomcomp.pdf', bbox_inches='tight', format='pdf')
plt.show()
#
# rand2 = np.loadtxt('rand_2.txt')
# reg2= np.loadtxt('reg_2.txt')
#
#
# #
# #
# #
# # # might be interesting to try perpendicular slats?
# #
# # # W = (m*d/2)*np.tan(th1+th2)
#
# # d = (5/2)*np.tan(np.arctan(1/np.sqrt(2)) + 70*np.pi/180)
# # size = 5
# # thickness_high = 302.5
# # thickness_low = 293.5
# # result = []
# # for thickness in[thickness_high, thickness_low]:
# #     print('THICKNESS', thickness)
# #     flat_surf = planar_surface()
# #     triangle_surf = regular_pyramids(55, upright=True, size=size)
# #     options['avoid_edges'] = False
# #     options['randomize'] = True
# #     rtstr2 = rt_structure(textures=[triangle_surf, flat_surf],
# #                           materials=[Si],
# #                           widths=[si(thickness, 'um')], incidence=Air, transmission=Air)
# #
# #     start = time()
# #     result_new2 = rtstr2.calculate(options)
# #     result.append(result_new2)
# #
# # plt.figure()
# # plt.hist(result[0]['thetas'][0,:])
# # plt.show()
# #
# # plt.figure()
# # plt.hist(result[1]['thetas'][0,:])
# # plt.show()
# #
# # n_R0_1 = np.sum((result[0]['n_passes'][0,:] == 1) * (result[0]['thetas'][0,:] < np.pi/2))
# # n_R0_2 = np.sum((result[1]['n_passes'][0,:] == 1) * (result[1]['thetas'][0,:] < np.pi/2))
# #
# #
# # n_T0_1 = np.sum((result[0]['n_passes'][0,:] == 3) * (result[0]['thetas'][0,:] > np.pi/2))
# # n_T0_2 = np.sum((result[1]['n_passes'][0,:] == 3) * (result[1]['thetas'][0,:] > np.pi/2))
# #
# # n_R1_1 = np.sum((result[0]['n_passes'][0,:] == 3) * (result[0]['thetas'][0,:] < np.pi/2))
# # n_R1_2 = np.sum((result[1]['n_passes'][0,:] == 3) * (result[1]['thetas'][0,:] < np.pi/2))
#
#
#
#
# from solcore.structure import Layer
#
# from solcore import si
# from structure import Interface, BulkLayer, Structure
# from matrix_formalism.process_structure import process_structure
# from matrix_formalism.multiply_matrices import calculate_RAT
#
# from textures.standard_rt_textures import regular_pyramids
#
#
# angle_degrees_in = 0
#
# # matrix multiplication
# wavelengths = np.array([1100e-9])
# options = {'depth_spacing': 0.5,
#            'project_name': 'optos_for_random',
#            'calc_profile': False,
#            'n_theta_bins': 30,
#            'c_azimuth': 0.25,
#            'pol': 'u',
#            'wavelengths': wavelengths,
#            'theta_in': angle_degrees_in*np.pi/180, 'phi_in': 1e-6,
#            'I_thresh': 0.001,
#            #'coherent': True,
#            #'coherency_list': None,
#            'lookuptable_angles': 200,
#            #'prof_layers': [1,2],
#            'n_rays': 1e5,
#            'random_angles': False,
#            'nx': nxy, 'ny': nxy,
#            'random_ray_position': False,
#            'parallel': True, 'n_jobs': -1,
#            'phi_symmetry': np.pi/2,
#            'only_incidence_angle': True,
#            'avoid_edges': False
#            }
#
#
#
# # whether pyramids are upright or inverted is relative to front incidence.
# # so if the same etch is applied to both sides of a slab of silicon, one surface
# # will have 'upright' pyramids and the other side will have 'not upright' (inverted)
# # pyramids in the model
# size = 2
# surf = regular_pyramids(elevation_angle=55, upright=True, size = size)
#
#
# front_surf = Interface('RT_TMM', texture = surf, layers=[Layer(si('0.01nm'), Air)],
#                        name = 'inv_pyramids' + str(options['n_rays']) + str(size))
# back_surf = Interface('TMM', layers=[], name = 'planar_back' + str(options['n_rays']))
#
#
# bulk_Si = BulkLayer(201.8e-6, Si, name = 'Si_bulk') # bulk thickness in m
#
# SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Air)
#
# process_structure(SC, options)
#
# results = calculate_RAT(SC, options)
#
# OPTOS_res = []
# thicknesses = np.arange(180, 220, 1)
#
# for bulk_thick in thicknesses:
#     bulk_Si = BulkLayer(bulk_thick*1e-6, Si, name='Si_bulk')  # bulk thickness in m
#
#     SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Air)
#
#     process_structure(SC, options)
#     results = calculate_RAT(SC, options)
#     OPTOS_res.append(np.asscalar(results[0]['A_bulk'][0][0].data))
#
#
# plt.figure()
#
# plt.plot(thickness, res_rand, 'o-', label='rand')
# plt.plot(thickness, res_reg,'o-',  label='reg')
# plt.plot(thicknesses, OPTOS_res)
# plt.legend()
# plt.title('5')
# plt.show()