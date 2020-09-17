from rayflare.ray_tracing.rt import rt_structure
from time import time
from textures.standard_rt_textures import regular_pyramids, planar_surface
from solcore import material
from solcore import si
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
pal = sns.cubehelix_palette(15)

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
#for size in [4]:
n_passes = []
nxy = [5, 10, 20, 35, 50]
reps = [1,2,4,10]

calc = True

options = {'wavelengths': np.arange(300, 1201, 20) * 1e-9, 'theta': 0 * np.pi / 180, 'phi': 0,
           'I_thresh': 1e-2,
           'parallel': True, 'pol': 'u', 'depth_spacing': si('1um'), 'nx': 0,
           'ny': 0, 'n_rays': 0,
           'random_ray_position': False, 'avoid_edges': False, 'randomize': False}

# if calc:
#     for n in nxy:
#         for nr in reps:
#             print('\n \n nxy/reps', n, nr)
#             options['nx'] = n
#             options['ny'] = n
#
#             options['n_rays'] = nr*(n**2)
#
#             flat_surf = planar_surface(size=size)
#             triangle_surf = regular_pyramids(55, upright=False, size=size)
#
#             rtstr = rt_structure(textures=[triangle_surf, flat_surf],
#                                 materials = [Si],
#                                 widths=[si('200um')], incidence=Air, transmission=Air)
#
#             start = time()
#             result_new = rtstr.calculate(options)
#             print(str(size), time()-start)
#             result = result_new
#             to_save = np.vstack(
#                 (options['wavelengths'] * 1e9, result['R'], result['R0'], result['T'], result['A_per_layer'][:, 0])).T
#             np.savetxt('RT_convergence_nxy_' + str(n) +'_nreps_' + str(nr) + '.txt', to_save)
#
#
# else:
#
#
#     for n in nxy:
#
#         plt.figure()
#
#         for nr in reps:
#
#             result = np.loadtxt('RT_convergence_nxy_' + str(n) + '_nreps_' + str(nr) + '.txt')
#             plt.plot(result[:,0], result[:,4], label = str(n**2) + ' points, ' + str(nr*n**2) + ' total rays')
#
#         plt.title(str(n))
#         plt.legend()
#         plt.show()

size = 5
# for size in [4]:
n_passes = []
#nxy = [5, 10, 20, 35, 50]
#reps = [1, 2, 4, 10]

calc = True
options = {'wavelengths': np.arange(300, 1201, 20) * 1e-9, 'theta': 0 * np.pi / 180, 'phi': 0,
           'I_thresh': 1e-2,
           'parallel': True, 'pol': 'u', 'depth_spacing': si('1um'), 'nx': 0,
           'ny': 0, 'n_rays': 0,
           'random_ray_position': False, 'avoid_edges': False, 'randomize': False}


#n_rays = [2500, 5000, 10000, 50000]

nxy = np.ceil(np.linspace(1, 50, 15)).astype(int)[:]
n_rays = np.ceil(np.linspace(2500, 50000, 15)).astype(int)
if calc:
    for n in nxy:
        for nr in n_rays:
            print('\n \n nxy/reps', n, nr)
            options['nx'] = n
            options['ny'] = n

            options['n_rays'] = nr

            flat_surf = planar_surface(size=size)
            triangle_surf = regular_pyramids(55, upright=False, size=size)

            rtstr = rt_structure(textures=[triangle_surf, flat_surf],
                                 materials=[Si],
                                 widths=[si('200um')], incidence=Air, transmission=Air)

            start = time()
            result_new = rtstr.calculate(options)
            print(str(size), time() - start)
            result = result_new
            to_save = np.vstack(
                (options['wavelengths'] * 1e9, result['R'], result['R0'], result['T'], result['A_per_layer'][:, 0])).T
            np.savetxt('RT_convergence_nxy_' + str(n) + '_nrays_' + str(nr) + '.txt', to_save)


else:

    for n in nxy:

        plt.figure()

        for nr in n_rays:
            result = np.loadtxt('RT_convergence_nxy_' + str(n) + '_nrays_' + str(nr) + '.txt')
            plt.plot(result[:, 0], result[:, 4], label=str(n ** 2) + ' points, ' + str(nr) + ' total rays')

        plt.title(str(n))
        plt.legend()
        plt.show()



nxy = np.ceil(np.linspace(1, 50, 15)).astype(int)[1:]
n_rays = np.ceil(np.linspace(2500, 50000, 15)).astype(int)

# take 50,000 rays with 50 x 50 points as 'correct'
n_ref = 50
nrays_ref = 50000
ref_res = np.loadtxt('RT_convergence_nxy_' + str(n_ref) + '_nrays_' + str(nrays_ref) + '.txt')

fig = plt.figure(figsize=(12, 4/3*3.7))
plt.subplot(1, 2, 1)
for n in nxy:
    rms = []
    for nr in n_rays:
        result = np.loadtxt('RT_convergence_nxy_' + str(n) + '_nrays_' + str(nr) + '.txt')
        rms.append(np.sum(np.sqrt((ref_res[:, 4] - result[:,4])**2)))


    plt.plot(n_rays, rms, label = str(n) + ' x/y points')

plt.xlabel('Number of rays traced')
plt.ylabel('RMS difference')
plt.text(-5000, 0.5, 'a)')
plt.legend()
#plt.show()

# plt.figure()
# for nr in n_rays:
#     rms = []
#     for n in nxy:
#         result = np.loadtxt('RT_convergence_nxy_' + str(n) + '_nrays_' + str(nr) + '.txt')
#         rms.append(np.sum(np.sqrt((ref_res[:, 4] - result[:, 4]) ** 2)))
#
#     plt.plot(nxy, rms, label=str(nr) + ' rays')
#
# plt.xlabel('x/y points on surface')
# plt.ylabel('RMS difference')
# plt.legend()
# plt.show()


alpha_Si = Si.alpha(options['wavelengths'])
alpha_Si = alpha_Si/np.max(alpha_Si)
alpha_Si = 1/alpha_Si


plt.subplot(1, 2, 2)
for n in nxy:
    rms = []
    for nr in n_rays:
        result = np.loadtxt('RT_convergence_nxy_' + str(n) + '_nrays_' + str(nr) + '.txt')
        rms.append(np.mean(result[:, 4] / alpha_Si))

    plt.plot(n_rays, rms, label=str(n) + ' x/y points')

plt.xlabel('Number of rays traced')
plt.ylabel('Weighted A')
#plt.legend(loc='upper right')
plt.text(-5000, 0.0537, 'b)')
fig.savefig('RT_convergence.pdf', bbox_inches='tight', format='pdf')
plt.show()

# plt.figure()
# for nr in n_rays:
#     rms = []
#     for n in nxy:
#         result = np.loadtxt('RT_convergence_nxy_' + str(n) + '_nrays_' + str(nr) + '.txt')
#         rms.append(np.mean(result[:, 4] / alpha_Si))
#
#     plt.plot(nxy, rms, label=str(nr) + ' rays')
#
# plt.xlabel('x/y points on surface')
# plt.ylabel('Weighted A')
# plt.legend()
# plt.show()

pal = sns.cubehelix_palette(4, reverse=False)

nxy = np.ceil(np.linspace(1, 50, 15)).astype(int)[[0, 5,10, 14]]
n_rays = np.ceil(np.linspace(2500, 50000, 15)).astype(int)[[0,5, 10, 14]]
fig = plt.figure(figsize=(6, 4/3*3.7))

linet = ['--', '-', '-.', ':']
for i1, nr in enumerate(n_rays):

    for j1, n in enumerate(nxy):
        result = np.loadtxt('RT_convergence_nxy_' + str(n) + '_nrays_' + str(nr) + '.txt')
        plt.plot(result[:,0], result[:,4], color=pal[j1], linestyle=linet[i1])

for i1, nr in enumerate(n_rays):
    plt.plot(0, -1, 'k', linestyle=linet[i1], label = str(nr) + ' rays')

for j1, n in enumerate(nxy):
    plt.plot(0, -1, color=pal[j1], label=str(n) + ' x/y points')
plt.xlim([950, 1200])
plt.ylim([0, 1])
plt.legend()
fig.savefig('RT_convergence_ex.pdf', bbox_inches='tight', format='pdf')
plt.show()