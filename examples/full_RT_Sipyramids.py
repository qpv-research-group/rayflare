from rayflare.ray_tracing.rt import rt_structure
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

size = 2
#for size in [4]:
n_passes = []
nxy = 30

calc = False

options = {'wavelengths': np.arange(300, 1201, 20) * 1e-9, 'theta': 0 * np.pi / 180, 'phi': 0,
           'I_thresh': 1e-2, 'nx': nxy, 'ny': nxy,
           'parallel': True, 'pol': 'u', 'n_rays': 10 * nxy ** 2, 'depth_spacing': si('1um'),
           'random_ray_position': False, 'avoid_edges': False, 'randomize': False}

if calc:


    flat_surf = planar_surface(size=size)
    triangle_surf = regular_pyramids(55, upright=False, size=size)



    rtstr = rt_structure(textures=[triangle_surf, flat_surf],
                        materials = [Si],
                        widths=[si('300um')], incidence=Air, transmission=Air)
    start = time()
    result_new = rtstr.calculate(options)
    print(str(size), time()-start)
    result = result_new

    pal = sns.diverging_palette(150, 275, s=80, l=55, n=9, center='dark')
    PVlighthouse = np.loadtxt('data/RAT_data_300um_2um_55.csv', delimiter=',', skiprows=1)
    sim = np.loadtxt('data/optos_fig7_sim.csv', delimiter=',')
    #meas = np.loadtxt('data/optos_fig7_data.csv', delimiter=',')
    plt.figure()
    plt.plot(options['wavelengths']*1e9, result['R'], '-o', color=pal[0])
    plt.plot(options['wavelengths']*1e9, result['R0'], '-o', color=pal[1])
    plt.plot(options['wavelengths']*1e9, result['T'], '-o', color=pal[2])
    plt.plot(options['wavelengths']*1e9, result['A_per_layer'], '-o', color=pal[3])
    plt.plot(PVlighthouse[:,0], PVlighthouse[:,2], '--', color=pal[0])
    plt.plot(PVlighthouse[:,0], PVlighthouse[:,9], '--', color=pal[2])
    plt.plot(PVlighthouse[:,0], PVlighthouse[:,3], '--', color=pal[1])
    plt.plot(PVlighthouse[:,0], PVlighthouse[:,5], '--', color=pal[3])
    #plt.plot(sim[:,0], sim[:,1])
    plt.ylim(0,1)
    #plt.plot(options['wavelengths']*1e9, result['R']+result['T']+np.sum(result['A_per_layer'],1), 'g')
    #plt.plot(PVlighthouse[:,0], PVlighthouse[:,2] + PVlighthouse[:,9] + PVlighthouse[:,5], 'g')
    #plt.legend(['R', 'T', 'L1', 'L2', 'L3'])
    plt.title(str(size))
    plt.show()

    #plt.figure()
    #plt.plot(result['profile'][15].T)
    #plt.show()

    to_save = np.vstack((options['wavelengths']*1e9, result['R'], result['R0'], result['T'], result['A_per_layer'][:,0])).T
    np.savetxt('rayflare_fullrt_300um_2umpyramids_300_1200nm_3', to_save)

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


else:
    result = np.loadtxt('rayflare_fullrt_300um_2umpyramids_300_1200nm_3')
    pal =sns.color_palette("hls", 4)
    PVlighthouse = np.loadtxt('data/RAT_data_300um_2um_55.csv', delimiter=',', skiprows=1)
    sim = np.loadtxt('data/optos_fig7_sim.csv', delimiter=',')
    # meas = np.loadtxt('data/optos_fig7_data.csv', delimiter=',')
    fig=plt.figure(figsize=(9,3.7))
    plt.subplot(1,2,1)
    plt.plot(result[:,0], result[:,1], '-o', color=pal[0], label=r'R$_{total}$', fillstyle='none')
    plt.plot(result[:,0], result[:,2], '-o', color=pal[1], label=r'R$_0$', fillstyle='none')
    plt.plot(result[:,0], result[:,3], '-o', color=pal[2], label=r'T', fillstyle='none')
    plt.plot(result[:,0], result[:,4], '-o', color=pal[3], label=r'A', fillstyle='none')
    plt.plot(PVlighthouse[:, 0], PVlighthouse[:, 2], '--', color=pal[0])
    plt.plot(PVlighthouse[:, 0], PVlighthouse[:, 9], '--', color=pal[2])
    plt.plot(PVlighthouse[:, 0], PVlighthouse[:, 3], '--', color=pal[1])
    plt.plot(PVlighthouse[:, 0], PVlighthouse[:, 5], '--', color=pal[3])
    plt.text(150, 1, 'a)')
    plt.plot(-1, -1, '-ok', label='RayFlare')
    plt.plot(-1, -1, '--k', label='PVLighthouse')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('R / A / T')
    plt.ylim(0, 1)
    plt.xlim(300, 1200)

    plt.legend()
    #plt.show()

    A_single_pass = 1 - np.exp(-200e-6 * Si.alpha(options['wavelengths']))
    A_single_pass_PVL = 1 - np.exp(-200e-6 * Si.alpha(PVlighthouse[:, 0] / 1e9))
    lambertian = 4*Si.n(options['wavelengths'])**2

    #lambertian = A_single_pass*4*Si.n(options['wavelengths'])**2

    #plt.figure()
    plt.subplot(1,2,2)
    # plt.plot(PVlighthouse[:,0], PVlighthouse[:,10], 'r--o')
    plt.plot(result[:,0], result[:,4] / A_single_pass, '-k',  label='RayFlare raytracer')
    #plt.plot(options['wavelengths']*1e9, lambertian, label='Lambertian')
    plt.plot(PVlighthouse[:, 0], PVlighthouse[:, 5] / A_single_pass_PVL, '--b', label='PVLighthouse')
    #plt.plot(options['wavelengths']*1e9, lambertian)
    plt.legend(loc='upper left')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Path length enhancement')
    plt.xlim(300, 1200)
    plt.text(150, 36, 'b)')

    fig.savefig('PVLighthousecomp.pdf', bbox_inches='tight', format='pdf')
    plt.show()

