from textures.standard_rt_textures import regular_pyramids, random_pyramids
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
palhf.reverse()
seamap = mpl.colors.ListedColormap(palhf)

reg = regular_pyramids(size=5)

rand = random_pyramids()


fig = plt.figure(figsize=(8,6))
grid = plt.GridSpec(7, 7, wspace=0, hspace=0)#, wspace=0.4, hspace=0.3)
ax = plt.subplot(grid[1,0], projection='3d')
ax.view_init(elev=30., azim=60)
#ax.set_aspect('equal')
ax.plot_trisurf(reg[0].Points[:,0], reg[0].Points[:,1], reg[0].Points[:,2],
                triangles=reg[0].simplices,  linewidth=1, color = (0.8, 0.8, 0.8, 0.8))
ax.text(7, 0, 9, 'a)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plt.show()

# fig = plt.figure()
ax2 = ax = plt.subplot(grid[0:3,1:], projection='3d')
#ax.set_aspect('equal')
ax2.view_init(elev=40., azim=60)
ax2.plot_trisurf(rand[0].Points[:,0], rand[0].Points[:,1], rand[0].Points[:,2],
                triangles=rand[0].simplices,  linewidth=1, color = (0.5, 0.5, 0.5, 0.5), cmap=seamap)
ax2.text(20, 0, 6, 'b)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
# plt.show()


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

size = 5
nxy = 21

calc = False
options = {'wavelengths': np.arange(300, 1201, 20) * 1e-9, 'theta': 0 * np.pi / 180, 'phi': 0,
           'I_thresh': 1e-2, 'nx': nxy, 'ny': nxy,
           'parallel': True, 'pol': 'u', 'n_rays': 2 * nxy ** 2, 'depth_spacing': si('1um'),
           'random_ray_position': False}

if calc:


    flat_surf = planar_surface()
    triangle_surf = regular_pyramids(55, upright=True, size=size)
    options['avoid_edges'] = False
    rtstr = rt_structure(textures=[triangle_surf, flat_surf],
                        materials = [Si],
                        widths=[si('200um')], incidence=Air, transmission=Air)

    start = time()
    result_new = rtstr.calculate(options)
    print(str(size), time()-start)

    result_reg = result_new

    to_save = np.vstack((options['wavelengths']*1e9, result_reg['R'], result_reg['R0'], result_reg['T'], result_reg['A_per_layer'][:,0])).T
    np.savetxt('rayflare_regular_upright', to_save)

    options['avoid_edges'] = False

    flat_surf = planar_surface()
    triangle_surf = random_pyramids()

    rtstr = rt_structure(textures=[triangle_surf, flat_surf],
                        materials = [Si],
                        widths=[si('200um')], incidence=Air, transmission=Air)

    start = time()
    result_new = rtstr.calculate(options)
    print(str(size), time()-start)
    result_rand = result_new

    to_save = np.vstack((options['wavelengths']*1e9, result_rand['R'], result_rand['R0'], result_rand['T'], result_rand['A_per_layer'][:,0])).T
    np.savetxt('rayflare_random_upright', to_save)


result_reg = np.loadtxt('rayflare_regular_upright')
result_rand = np.loadtxt('rayflare_random_upright')
pal =sns.color_palette("hls", 4)

# meas = np.loadtxt('data/optos_fig7_data.csv', delimiter=',')
# fig=plt.figure(figsize=(9,3.7))
# plt.subplot(1,2,1)

ax = plt.subplot(grid[4:,0:3])
plt.plot(result_reg[:,0], result_reg[:,1], '-', color=pal[0], label=r'R$_{total}$', fillstyle='none')
plt.plot(result_reg[:,0], result_reg[:,2], '-', color=pal[1], label=r'R$_0$', fillstyle='none')
plt.plot(result_reg[:,0], result_reg[:,3], '-', color=pal[2], label=r'T', fillstyle='none')
plt.plot(result_reg[:,0], result_reg[:,4], '-', color=pal[3], label=r'A')#, fillstyle='none')

plt.plot(result_rand[:,0], result_rand[:,1], '--', color=pal[0], fillstyle='none')
plt.plot(result_rand[:,0], result_rand[:,2], '--', color=pal[1], fillstyle='none')
plt.plot(result_rand[:,0], result_rand[:,3], '--', color=pal[2], fillstyle='none')
plt.plot(result_rand[:,0], result_rand[:,4], '--', color=pal[3])#, fillstyle='none')

plt.text(150, 1, 'c)')
plt.plot(-1, -1, '-k', label='Regular')
plt.plot(-1, -1, '--k', label='Random')
plt.xlabel('Wavelength (nm)')
plt.ylabel('R / A / T')
plt.ylim(0, 1)
plt.xlim(300, 1200)

plt.legend()
#plt.show()

A_single_pass = 1 - np.exp(-200e-6 * Si.alpha(options['wavelengths']))

# plt.figure()
# plt.subplot(1,2,2)
# plt.plot(PVlighthouse[:,0], PVlighthouse[:,10], 'r--o')
ax = plt.subplot(grid[4:,4:])

plt.plot(result_reg[:,0], result_reg[:,4] / A_single_pass, '-k',  label='Regular')
plt.plot(result_rand[:, 0], result_rand[:, 4] / A_single_pass, '--b', label='Random')
plt.legend(loc='upper left')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Path length enhancement')
plt.xlim(300, 1200)
plt.text(150, 125, 'd)')

fig.savefig('regularrandomcomp.pdf', bbox_inches='tight', format='pdf')
plt.show()

