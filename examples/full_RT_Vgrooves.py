from ray_tracing.rt import rt_structure
from time import time
from textures.standard_rt_textures import regular_pyramids, planar_surface, V_grooves
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

options = {'wavelengths': np.arange(900, 1201, 20) * 1e-9, 'theta': 0 * np.pi / 180, 'phi': 0,
           'I_thresh': 1e-2, 'nx': nxy, 'ny': nxy,
           'parallel': True, 'pol': 'u', 'n_rays': 10 * nxy ** 2, 'depth_spacing': si('1um'),
           'random_ray_position': False, 'avoid_edges': False, 'randomize': False}

if calc:


    Vgroove_x1 = V_grooves(width=5, direction='x')
    Vgroove_x2 = V_grooves(width=5, direction='x')

    rtstr = rt_structure(textures=[Vgroove_x1, Vgroove_x2],
                        materials = [Si],
                        widths=[si('200um')], incidence=Air, transmission=Air)
    start = time()
    result_new = rtstr.calculate(options)
    print(str(size), time()-start)
    result_same = result_new

    to_save_same = np.vstack((options['wavelengths']*1e9, result_same['R'], result_same['R0'], result_same['T'], result_same['A_per_layer'][:,0])).T
    np.savetxt('rayflare_fullrt_200um_Vgrooves_samedir', to_save_same)


    Vgroove_x = V_grooves(width=5, direction='x')
    Vgroove_y = V_grooves(width=5, direction='y')

    rtstr = rt_structure(textures=[Vgroove_x, Vgroove_y],
                        materials = [Si],
                        widths=[si('200um')], incidence=Air, transmission=Air)
    start = time()
    result_new = rtstr.calculate(options)
    print(str(size), time()-start)
    result_opposite = result_new


    to_save_opp = np.vstack((options['wavelengths']*1e9, result_opposite['R'], result_opposite['R0'], result_opposite['T'], result_opposite['A_per_layer'][:,0])).T
    np.savetxt('rayflare_fullrt_200um_Vgrooves_oppositedir', to_save_opp)


plt.figure()
plt.plot(to_save_same[:,0], to_save_same[:,4], label='Same direction')
plt.plot(to_save_opp[:,0], to_save_opp[:,4], label='Opposite direction')
plt.show()