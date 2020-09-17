from rayflare.ray_tracing.rt import rt_structure
from time import time
from textures.standard_rt_textures import regular_pyramids, planar_surface, V_grooves
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

pal = sns.color_palette("husl", 8)
pal = sns.color_palette('bright')

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
Ag = material('Ag_Jiang')()

#for size in [4]:
n_passes = []
nxy = 41

calc = False

options = {'wavelengths': np.arange(800, 1201, 5) * 1e-9, 'theta': 0 * np.pi / 180, 'phi': 0,
           'I_thresh': 0.01, 'nx': nxy, 'ny': nxy,
           'parallel': True, 'pol': 'u', 'n_rays':  10*nxy ** 2, 'depth_spacing': si('1um'),
           'random_ray_position': False, 'avoid_edges': False, 'randomize': False}

if calc:


    Vgroove_x1 = V_grooves(width=5, direction='x')
    Vgroove_x2 = V_grooves(width=5, direction='x')

    rtstr = rt_structure(textures=[Vgroove_x1, Vgroove_x2],
                        materials = [Si],
                        widths=[si('200um')], incidence=Air, transmission=Air)
    start = time()
    result_new = rtstr.calculate(options)
    print(time()-start)
    result_same = result_new

    n_passes_Vsame = np.mean(result_same['n_passes'],1)
    n_interactions_Vsame = np.mean(result_same['n_interactions'],1)
    to_save_same = np.vstack((options['wavelengths']*1e9, result_same['R'], result_same['R0'], result_same['T'], result_same['A_per_layer'][:,0],
                             n_passes_Vsame, n_interactions_Vsame)).T
    np.savetxt('rayflare_fullrt_100um_Vgrooves_samedir', to_save_same)


    Vgroove_x = V_grooves(width=5, direction='x')
    Vgroove_y = V_grooves(width=5, direction='y')

    rtstr = rt_structure(textures=[Vgroove_x, Vgroove_y],
                        materials = [Si],
                        widths=[si('200um')], incidence=Air, transmission=Air)
    start = time()
    result_new = rtstr.calculate(options)
    print(time()-start)
    result_opposite = result_new


    n_passes_Vopp = np.mean(result_opposite['n_passes'], 1)
    n_interactions_Vopp = np.mean(result_opposite['n_interactions'], 1)
    to_save_opp = np.vstack((options['wavelengths']*1e9, result_opposite['R'], result_opposite['R0'], result_opposite['T'], result_opposite['A_per_layer'][:,0],
                             n_passes_Vopp, n_interactions_Vopp)).T
    np.savetxt('rayflare_fullrt_100um_Vgrooves_oppositedir', to_save_opp)

    options['randomize'] = False

    pyramids = regular_pyramids(size=5, upright=True)
    planar = planar_surface(size=5)

    rtstr = rt_structure(textures=[pyramids, planar],
                         materials=[Si],
                         widths=[si('200um')], incidence=Air, transmission=Air)
    start = time()
    result_new = rtstr.calculate(options)
    print(time() - start)
    result_pyrfront = result_new

    n_passes_pyrreg = np.mean(result_pyrfront['n_passes'], 1)
    n_interactions_pyrreg = np.mean(result_pyrfront['n_interactions'], 1)
    to_save_pyrfront = np.vstack((options['wavelengths'] * 1e9, result_pyrfront['R'], result_pyrfront['R0'], result_pyrfront['T'],
                              result_pyrfront['A_per_layer'][:, 0], n_passes_pyrreg, n_interactions_pyrreg)).T
    np.savetxt('rayflare_fullrt_100um_regpyramid_frontonly', to_save_pyrfront)

    options['randomize'] = True

    pyramids = regular_pyramids(size=5, upright=True)
    planar = planar_surface(size=5)
    #pyramids_back = regular_pyramids(size=5, upright=False)

    rtstr = rt_structure(textures=[pyramids, planar],
                         materials=[Si],
                         widths=[si('200um')], incidence=Air, transmission=Air)
    start = time()
    result_new = rtstr.calculate(options)
    print(time() - start)
    result_pyrrandom = result_new

    n_passes_pyrrand = np.mean(result_pyrrandom['n_passes'], 1)
    n_interactions_pyrrand = np.mean(result_pyrrandom['n_interactions'], 1)
    to_save_pyrrandom = np.vstack(
        (options['wavelengths'] * 1e9, result_pyrrandom['R'], result_pyrrandom['R0'], result_pyrrandom['T'],
         result_pyrrandom['A_per_layer'][:, 0], n_passes_pyrrand, n_interactions_pyrrand)).T
    np.savetxt('rayflare_fullrt_100um_randompyramid_frontonly', to_save_pyrrandom)





else:
    to_save_same = np.loadtxt('rayflare_fullrt_100um_Vgrooves_samedir')

    to_save_opp = np.loadtxt('rayflare_fullrt_100um_Vgrooves_oppositedir')

    to_save_pyrfront = np.loadtxt('rayflare_fullrt_100um_regpyramid_frontonly')

    to_save_pyrrandom = np.loadtxt('rayflare_fullrt_100um_randompyramid_frontonly')


mirror = np.loadtxt('mirror_rayflare.txt')
lambertian = np.loadtxt('lambertian_rayflare.txt')
mirror_pyr = np.loadtxt('mirrorpyramids_rayflare.txt')
lambertian_pyr = np.loadtxt('lambertianpyramids_rayflare.txt')

wl = wavelengths = np.arange(800, 1200, 5)

fig = plt.figure(figsize=(6.5,4.5))
plt.plot(to_save_same[:,0], to_save_same[:,4], label='V-grooves both sides, same direction')
plt.plot(to_save_opp[:,0], to_save_opp[:,4], label='V-grooves both sides, opposite direction')
plt.plot(moving_average(wl, 2), moving_average(mirror[:,0], 2), '--', label='V-grooves front + perfect mirror')
plt.plot(moving_average(wl, 2), moving_average(lambertian[:,0], 2), '-.', label='V-grooves front + Lambertian rear')
plt.plot(to_save_same[:,0], to_save_pyrfront[:,4], label='Regular front pyramids + planar rear')
plt.plot(to_save_opp[:,0], to_save_pyrrandom[:,4],  label='Random front pyramids + planar rear')

plt.plot(moving_average(wl, 2), moving_average(mirror_pyr[:,0], 2), '--', label='Random pyramids front + perfect mirror')
plt.plot(moving_average(wl, 2), moving_average(lambertian_pyr[:,0], 2), '-.', label = 'Random pyramids front + Lambertian rear')

plt.plot(to_save_same[:,0], to_save_same[:,2])
plt.plot(to_save_opp[:,0], to_save_opp[:,2])
plt.plot(moving_average(wl, 2), moving_average(mirror[:,1], 2), '--')
plt.plot(moving_average(wl, 2), moving_average(lambertian[:,1], 2), '-.')
plt.plot(to_save_same[:,0], to_save_pyrfront[:,2])
plt.plot(to_save_opp[:,0], to_save_pyrrandom[:,2])

plt.plot(moving_average(wl, 2), moving_average(mirror_pyr[:,1], 2), '--')
plt.plot(moving_average(wl, 2), moving_average(lambertian_pyr[:,1], 2), '-.')
plt.text(810, 0.13, 'R$_0$')
plt.text(1120, 0.75, 'A')


plt.legend()
#plt.ylabel('R / A/ T')
plt.ylabel('A / R$_0$')
plt.xlabel('Wavelength (nm)')
plt.xlim(min(options['wavelengths']*1e9), max(options['wavelengths']*1e9))
plt.ylim(0,1)

fig.savefig('V_pyr_lambertian_mirror_comp.pdf', bbox_inches='tight', format='pdf')
plt.show()



plt.figure()
plt.plot(to_save_same[:,0], to_save_same[:,5], label='V-grooves, same direction')
plt.plot(to_save_opp[:,0], to_save_opp[:,5], label='V-grooves, opposite direction')
plt.plot(to_save_same[:,0], to_save_pyrfront[:,5], label='Regular front pyramids')
plt.plot(to_save_opp[:,0], to_save_pyrrandom[:,5],  label='Random front pyramids')

plt.plot(to_save_same[:,0], to_save_same[:,6], '--')
plt.plot(to_save_opp[:,0], to_save_opp[:,6], '--')
plt.plot(to_save_same[:,0], to_save_pyrfront[:,6], '--')
plt.plot(to_save_opp[:,0], to_save_pyrrandom[:,6], '--',)


plt.plot(-1, -1, '-k', label='n$_{passes}$')
plt.plot(-1, -1, '--k', label='n$_{interactions}$')
plt.legend()
plt.ylabel('n')
plt.xlabel('Wavelength (nm)')
plt.xlim(min(options['wavelengths']*1e9), max(options['wavelengths']*1e9))
plt.ylim(0,20)
plt.show()


plt.figure()
plt.plot(to_save_same[:,0], to_save_same[:,2], label='V-grooves both sides, same direction')
plt.plot(to_save_opp[:,0], to_save_opp[:,2], label='V-grooves both sides, opposite direction')
plt.plot(moving_average(wl, 2), moving_average(mirror[:,1], 2), '--', label='V-grooves front + perfect mirror')
plt.plot(moving_average(wl, 2), moving_average(lambertian[:,1], 2), '-.', label='V-grooves front + Lambertian rear')
plt.plot(to_save_same[:,0], to_save_pyrfront[:,2], label='Regular front pyramids + planar rear')
plt.plot(to_save_opp[:,0], to_save_pyrrandom[:,2],  label='Random front pyramids + planar rear')

plt.plot(moving_average(wl, 2), moving_average(mirror_pyr[:,1], 2), '--', label='Random pyramids front + perfect mirror')
plt.plot(moving_average(wl, 2), moving_average(lambertian_pyr[:,1], 2), '-.', label = 'Random pyramids front + Lambertian rear')

# plt.plot(to_save_same[:,0], to_save_same[:,1]-to_save_same[:,2], '--')
# plt.plot(to_save_opp[:,0], to_save_opp[:,1]- to_save_opp[:,2], '--')
# plt.plot(to_save_same[:,0], to_save_pyrfront[:,1]-to_save_pyrfront[:,2], '--')
# plt.plot(to_save_opp[:,0], to_save_pyrrandom[:,1]-to_save_pyrrandom[:,2], '--',)
#
# plt.plot(to_save_same[:,0], to_save_same[:,3], '-.')
# plt.plot(to_save_opp[:,0], to_save_opp[:,3], '-.')
# plt.plot(to_save_same[:,0], to_save_pyrfront[:,3], '-.')
# plt.plot(to_save_opp[:,0], to_save_pyrrandom[:,3], '-.')

# plt.plot(-1, -1, '-k', label='A')
# plt.plot(-1, -1, '--k', label='R$_{escape}$')
# plt.plot(-1, -1, '-.k', label='T')
plt.legend()
#plt.ylabel('R / A/ T')
plt.ylabel('A')
plt.xlabel('Wavelength (nm)')
plt.xlim(min(options['wavelengths']*1e9), max(options['wavelengths']*1e9))
plt.ylim(0,1)
plt.show()