import numpy as np

from rayflare.ray_tracing import rt_structure
from rayflare.textures import regular_pyramids, planar_surface
from rayflare.options import default_options

from solcore import material
from solcore import si

# imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# setting up some colours for plotting
pal = sns.color_palette("husl", 4)


# setting up Solcore materials
Air = material('Air')()
Si = material('566', nk_db=True)()

# number of x and y points to scan across
nxy = 25

calc = True

# setting options
options = default_options()
options.wavelengths = np.linspace(300, 1201, 50) * 1e-9
options.nx = nxy
options.ny = nxy
options.n_rays = 2 * nxy ** 2
options.depth_spacing = si('1um')
options.parallel = True

PVlighthouse = np.loadtxt('data/RAT_data_300um_2um_55.csv', delimiter=',', skiprows=1)

if calc:

    flat_surf = planar_surface(size=2) # pyramid size in microns
    triangle_surf = regular_pyramids(55, upright=False, size=2)

    # set up ray-tracing options
    rtstr = rt_structure(textures=[triangle_surf, flat_surf],
                        materials = [Si],
                        widths=[si('300um')], incidence=Air, transmission=Air)
    result_new = rtstr.calculate(options)
    result = result_new


    result = np.vstack((options['wavelengths']*1e9, result['R'], result['R0'], result['T'], result['A_per_layer'][:,0])).T
    np.savetxt('data/rayflare_fullrt_300um_2umpyramids_300_1200nm_3', result)

else:
    result = np.loadtxt('data/rayflare_fullrt_300um_2umpyramids_300_1200nm_3')
    PVlighthouse = np.loadtxt('data/RAT_data_300um_2um_55.csv', delimiter=',', skiprows=1)


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
plt.title('a)', loc='left')
plt.plot(-1, -1, '-ok', label='RayFlare')
plt.plot(-1, -1, '--k', label='PVLighthouse')
plt.xlabel('Wavelength (nm)')
plt.ylabel('R / A / T')
plt.ylim(0, 1)
plt.xlim(300, 1200)

plt.legend()

A_single_pass = 1 - np.exp(-200e-6 * Si.alpha(options['wavelengths']))
A_single_pass_PVL = 1 - np.exp(-200e-6 * Si.alpha(PVlighthouse[:, 0] / 1e9))
lambertian = 4*Si.n(options['wavelengths'])**2

plt.subplot(1,2,2)
plt.plot(result[:,0], result[:,4] / A_single_pass, '-k',  label='RayFlare raytracer')
plt.plot(PVlighthouse[:, 0], PVlighthouse[:, 5] / A_single_pass_PVL, '--b', label='PVLighthouse')
plt.legend(loc='upper left')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Path length enhancement')
plt.xlim(300, 1200)
plt.title('b)', loc='left')

#fig.savefig('PVLighthousecomp.pdf', bbox_inches='tight', format='pdf')
plt.show()

