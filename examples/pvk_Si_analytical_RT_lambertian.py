from solcore.light_source import LightSource
import numpy as np
from rayflare.textures import regular_pyramids, planar_surface
import matplotlib.pyplot as plt

from solcore import material
from solcore.structure import Layer
from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure
from time import time
import seaborn as sns

SiN = material("Si3N4")()
Si = material("Si")()
Air = material("Air")()
Ag = material("Ag")()
MgF2 = material("MgF2")()
Ge = material("Ge")()

coverglass_ARC = material("coverglass_AR_JJ")()
coverglass = material("coverglass_JJ")()

GaAs = material("GaAs")()

Pvk = material("Pvk_Ox_165")()
epoxy = material("BK7")()

n_rays = 800

d = 100e-6

wavelengths = np.linspace(300, 1180, 50) * 1e-9

AM15G = LightSource(source_type='standard', version='AM1.5g', x=wavelengths,
                    output_units='photon_flux_per_m')

lambert_approx = 30
options = default_options()

options.analytical_ray_tracing = 0
options.wavelength = wavelengths
options.project_name = 'integration_testing'
options.nx = 10
options.ny = 10
options.theta_in = 0.1
options.parallel = True
# options.n_jobs = -3
options.randomize_surface = True
options.I_thresh = 5e-3
options.depth_spacing_bulk = 1e-8
options.n_rays = n_rays
options.lambertian_approximation = lambert_approx

# n_bounces = np.arange(1, 11, dtype=int)
# n_bounces = [1, 2, 3, 5, 10, 15]  # 10, 15, 20, 30, 70]

# front_text = regular_pyramids(10, True,
#                               interface_layers=[Layer(100e-9, coverglass_ARC)],)

front_text = planar_surface(
                              interface_layers=[Layer(100e-9, coverglass_ARC)],
)

front_text_2 = regular_pyramids(52, True, 1,
                                interface_layers=[Layer(100e-9, MgF2), Layer(400e-9, Pvk)]
                                                             )
rear_text = regular_pyramids(52, False, 1)

rt_str = rt_structure(textures=[front_text, front_text_2, rear_text], materials=[coverglass, Si],
                      widths=[10e-6, d], incidence=Air, transmission=Ag,
                      options=options, use_TMM=True, save_location='current',
                      overwrite=True)

# options.n_rays = 10
# options.analytical_ray_tracing = 0
# result = rt_str.calculate(options)

options.n_rays = n_rays
options.lambertian_approximation = 0
# #
start = time()
result_1 = rt_str.calculate_profile(options)
normal_time = time() - start
print('Elapsed time: ', normal_time)

A_layer = result_1['A_per_layer']
A_per_interface = result_1['A_per_interface']
total_1 = result_1['R'] + result_1['T'] + np.sum(A_layer, axis=1) + A_per_interface[1][:,1]
#
# plt.figure()
# plt.plot(wavelengths*1e9, result_1['R'], label='R')
# plt.plot(wavelengths*1e9, result_1['T'], label='T')
# plt.plot(wavelengths*1e9, A_layer, label='A')
# plt.plot(wavelengths*1e9, A_per_interface[1][:,1], label='GaAs')
# # plt.plot(wavelengths*1e9, A_per_interface[2][:,0], label='Ge_back')
# plt.plot(wavelengths*1e9, total, 'k-', label='total')
# plt.axhline(1)
# plt.legend()
# plt.show()

n_bounces = [3, 5, 10, 20, 30, 40, 60, 80, 100, 120, 140, 200, 0]
options.analytical_ray_tracing = 2


timing = np.zeros(len(n_bounces))
result_list = []

for i1, n_b in enumerate(n_bounces):

    options.lambertian_approximation = n_b
    start = time()
    result = rt_str.calculate_profile(options)
    result_list.append(result)
    timing[i1] = time() - start
    print(f'{n_b} passes elapsed time: ', timing[i1])

n_bounces_plot = n_bounces[:]
n_bounces_plot[-1] = max(n_bounces) + 10

plt.figure()
alphas = np.linspace(0.2, 1, len(n_bounces))

for i1, result in enumerate(result_list):
    A_layer_2 = result['A_per_layer']
    A_per_interface_2 = result['A_per_interface']
    # import matplotlib.pyplot as plt
    plt.plot(wavelengths*1e9, result['R'], 'k-', alpha=alphas[i1])
    plt.plot(wavelengths*1e9, A_layer_2[:,0], 'r-', alpha=alphas[i1])
    plt.plot(wavelengths*1e9, A_layer_2[:,1], 'g-', alpha=alphas[i1])
    plt.plot(wavelengths*1e9, A_per_interface_2[1][:,1], 'y-', alpha=alphas[i1])

plt.show()

for i1, result in enumerate(result_list[:-1]):
    ratio = result['A_per_layer'][:,1]/result_list[-1]['A_per_layer'][:,1]
    plt.plot(wavelengths*1e9, ratio, 'k-', alpha=alphas[i1], label=f'{n_bounces[i1]} passes')

plt.legend()
plt.show()

max_diff = np.zeros(len(result_list)-1)
for i1, result in enumerate(result_list[:-1]):
    difference = result['A_per_layer'][:,1] - result_list[-1]['A_per_layer'][:,1]
    plt.plot(wavelengths*1e9, difference, 'k-', alpha=alphas[i1], label=f'{n_bounces[i1]} passes')
    max_diff[i1] = np.max(np.abs(difference))
plt.legend()
plt.show()

plt.figure()
plt.plot(n_bounces_plot, timing, 'o', mfc='none')
plt.plot(max(n_bounces) + 20, normal_time, 'o', mfc='none')
plt.xlabel("Allowed passes")
plt.ylabel("Time (s)")
ax2 = plt.gca().twinx()
ax2.plot(n_bounces_plot[:-1], max_diff, 'o', mfc='none', color='r')
ax2.set_ylabel("Max difference in A")
plt.show()

result_i = 3

int_A = 1e9*np.trapz(result_1['profile'], dx=options.depth_spacing_bulk)
int_A_lamb = 1e9*np.trapz(result_list[result_i]['profile'], dx=options.depth_spacing_bulk)

total_2 = (result_list[result_i]['R'] + result_list[result_i]['T'] +
           np.sum(result_list[result_i]['A_per_layer'], axis=1) +
           result_list[result_i]['A_per_interface'][1][:,1])

plt.figure()
plt.plot(wavelengths*1e9, int_A, label='full RT')
plt.plot(wavelengths*1e9, result_1['A_per_layer'][:,1])
plt.plot(wavelengths*1e9, total_1)
plt.plot(wavelengths*1e9, result_1['R'])

plt.plot(wavelengths*1e9, int_A_lamb, '-r', label='lambertian')
plt.plot(wavelengths*1e9, result_list[result_i]['A_per_layer'][:,1], '--k')
plt.plot(wavelengths*1e9, total_2, '--')
plt.plot(wavelengths*1e9, result_list[result_i]['R'], '--')
plt.legend()
plt.show()

cols = sns.color_palette('viridis', len(result_list))

# check if integrated A makes sense (bulk)
for i1, result in enumerate(result_list):
    int_A = 1e9*np.trapz(result['profile'], dx=options.depth_spacing_bulk)
    plt.plot(wavelengths*1e9, int_A, label=f'{n_bounces[i1]} passes', color=cols[i1])
    plt.plot(wavelengths*1e9, np.sum(result['A_per_layer'], axis=1), '--', color=cols[i1])

plt.xlim(500, 1180)
plt.legend()
plt.show()





plt.figure()
plt.semilogy(result_1['profile'][-2])
plt.semilogy(result_list[0]['profile'][-2])
plt.show()

#
# A_layer_2 = result['A_per_layer']
# A_per_interface_2 = result['A_per_interface']
# total = result['R'] + result['T'] + np.sum(A_layer_2, axis=1) + A_per_interface_2[1][:,1]
# # import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(wavelengths*1e9, result['R'], 'k-', label='R')
# plt.plot(wavelengths*1e9, result['T'], 'r-', label='T')
# plt.plot(wavelengths*1e9, A_layer_2, 'g-', label='A')
# plt.plot(wavelengths*1e9, A_per_interface_2[1][:,1], 'y-', label='GaAs')
# plt.plot(wavelengths*1e9, total, 'k-', label='total', alpha=0.5)
# plt.axhline(1)
# plt.legend()
# plt.show()
#
# plt.figure()
# plt.plot(wavelengths*1e9, result['R'], 'k-', label='R')
# plt.plot(wavelengths*1e9, result['T'], 'r-', label='T')
# plt.plot(wavelengths*1e9, A_layer_2[:,0], 'b-', label='glass')
# plt.plot(wavelengths*1e9, A_per_interface_2[1][:,1], 'y-', label='Pvk')
# plt.plot(wavelengths*1e9, A_layer_2[:,1], 'g-', label='Si')
#
# plt.plot(wavelengths*1e9, total, 'k-', label='total', alpha=0.5)
#
# plt.plot(wavelengths*1e9, result_1['R'], 'k--')
# plt.plot(wavelengths*1e9, result_1['T'], 'r--')
# plt.plot(wavelengths*1e9, A_layer[:,1], 'g--')
# plt.plot(wavelengths*1e9, A_layer[:,0], 'b--', label='glass')
# plt.plot(wavelengths*1e9, A_per_interface[1][:,1], 'y--')
# # plt.plot(wavelengths*1e9, A_per_interface[2][:,0], label='Ge')
# plt.plot(wavelengths*1e9, total_1, 'k--', label='total', alpha=0.5)
# # plt.axhline(1)
# plt.legend()
# plt.show()
#
# # number of passes:
# plt.figure()
# plt.plot(wavelengths*1e9, np.mean(result_1['n_passes'],axis=1), 'k--')
# plt.plot(wavelengths*1e9, np.mean(result['n_passes'], axis=1), 'k')
#
# plt.plot(wavelengths*1e9, np.max(result_1['n_passes'],axis=1), 'r--')
# plt.plot(wavelengths*1e9, np.max(result['n_passes'], axis=1), 'r')
# plt.show()
#
# # result_1 is just RT, result is analytical RT
#
# fig, axes = plt.subplots(5, 2, figsize=(8, 15))
# for i, ax in enumerate(axes.flatten()):
#     ax.hist(result['n_interactions'][i], color='k', bins=51, alpha=0.5, label='Analytical RT', histtype='step', range=[0,50])
#     ax.hist(result_1['n_interactions'][i], color='r', bins=51, alpha=0.5, label='RT', histtype='step', range=[0, 50])
# plt.title('n_interactions')
# plt.show()
#
# fig, axes = plt.subplots(5, 2, figsize=(8, 15))
# for i, ax in enumerate(axes.flatten()):
#     ax.hist(result['n_passes'][i], color='k', bins=11, alpha=0.5, label='Analytical RT', histtype='step', range=[0,10])
#     ax.hist(result_1['n_passes'][i], color='r', bins=11, alpha=0.5, label='RT', histtype='step', range=[0,10])
# plt.title('n_passes')
# plt.show()
#
#
# fig, axes = plt.subplots(5, 2, figsize=(8, 15))
# for i, ax in enumerate(axes.flatten()):
#     ax.hist(result['thetas'][i], color='k', bins=30, alpha=0.5, label='Analytical RT', histtype='step', range=[0, np.pi])
#     ax.hist(result_1['thetas'][i], color='r', bins=30, alpha=0.5, label='RT', histtype='step', range=[0, np.pi])
# plt.title('thetas')
# plt.show()
#
# fig, axes = plt.subplots(5, 2, figsize=(8, 15))
# for i, ax in enumerate(axes.flatten()):
#     ax.hist(result['phis'][i], color='k', bins=30, alpha=0.5, label='Analytical RT', histtype='step', range=[0, 2*np.pi])
#     ax.hist(result_1['phis'][i], color='r', bins=30, alpha=0.5, label='RT', histtype='step', range=[0, 2*np.pi])
# plt.title('phis')
# plt.show()