from ray_tracing.rt import rt_structure
from time import time
from textures.standard_rt_textures import regular_pyramids, planar_surface
from solcore import material
from solcore import si
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

pal = sns.cubehelix_palette(7, start=.5, rot=-.9)

cols = cycler('color', pal)

plt.rcParams['axes.prop_cycle'] = cols

Air = material('Air')()
Si = material('566', nk_db=True)()

size=2
#for size in [4]:
n_passes = []
for nxy in [20]:

    flat_surf = planar_surface()
    triangle_surf = regular_pyramids(55, upright=False, size=size)

    #options = {'wavelengths': np.linspace(700, 1700, 100)*1e-9, 'theta': 45*np.pi/180, 'phi': 0,
    #           'I_thresh': 1e-4, 'nx': 5, 'ny': 5,
    #           'parallel': True, 'pol': 'p', 'n_rays': 2000, 'depth_spacing': 1, 'n_jobs': -1}

    options = {'wavelengths': np.linspace(1050, 1200, 10)*1e-9, 'theta': 0*np.pi/180, 'phi': 0,
               'I_thresh': 1e-2, 'nx': nxy, 'ny': nxy,
               'parallel': True, 'pol': 'u', 'n_rays': 2*nxy**2, 'depth_spacing': si('1um'), 'n_jobs': -1,
               'random_ray_position': True}
    #structure = RTgroup(textures=[flat_surf, flat_surf, flat_surf, flat_surf], materials = [GaAs, Si, Ge],
    #                    widths=[si('100um'), si('70um'), si('50um')], depth_spacing=1)

    rtstr = rt_structure(textures=[triangle_surf, flat_surf],
                        materials = [Si],
                        widths=[si('300um')], incidence=Air, transmission=Air)
    start = time()
    result_new = rtstr.calculate(options)
    print(str(size), time()-start)
    result = result_new


    PVlighthouse = np.loadtxt('data/RAT_data.csv', delimiter=',', skiprows=1)
    sim = np.loadtxt('data/optos_fig7_sim.csv', delimiter=',')
    #meas = np.loadtxt('data/optos_fig7_data.csv', delimiter=',')
    plt.figure()
    plt.plot(options['wavelengths']*1e9, result['R'], 'k-o')
    #plt.plot(options['wavelengths']*1e9, result['R0'], 'y-o')
    plt.plot(options['wavelengths']*1e9, result['T'], 'r-o')
    plt.plot(options['wavelengths']*1e9, result['A_per_layer'], 'g-o')
    plt.plot(PVlighthouse[:,0], PVlighthouse[:,2], 'k--o')
    plt.plot(PVlighthouse[:,0], PVlighthouse[:,9], 'r--o')
    plt.plot(PVlighthouse[:,0], PVlighthouse[:,3], 'y--o')
    plt.plot(PVlighthouse[:,0], PVlighthouse[:,5], 'g--o')
    #plt.plot(sim[:,0], sim[:,1])
    #plt.ylim(0,1)
    plt.plot(options['wavelengths']*1e9, result['R']+result['T']+np.sum(result['A_per_layer'],1), 'g')
    plt.plot(PVlighthouse[:,0], PVlighthouse[:,2] + PVlighthouse[:,9] + PVlighthouse[:,5], 'g')
    #plt.legend(['R', 'T', 'L1', 'L2', 'L3'])
    plt.title(str(size))
    plt.show()

    #plt.figure()
    #plt.plot(result['profile'][15].T)
    #plt.show()

    A_single_pass = 1 - np.exp(-200e-6*Si.alpha(options['wavelengths']))
    A_single_pass_PVL = 1 - np.exp(-200e-6*Si.alpha(PVlighthouse[:,0]/1e9))

    plt.figure()
    #plt.plot(PVlighthouse[:,0], PVlighthouse[:,10], 'r--o')
    plt.plot(PVlighthouse[:,0], PVlighthouse[:,5]/A_single_pass_PVL)
    plt.plot(options['wavelengths']*1e9, result['A_per_layer'][:,0]/A_single_pass)
    plt.title(str(size))
    plt.show()

    print('passes', np.mean(result['n_passes']))
    n_passes.append(np.mean(result['n_passes']))