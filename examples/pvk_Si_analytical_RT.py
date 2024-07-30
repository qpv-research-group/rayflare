from rayflare.ray_tracing.analytical_rt import lambertian_scattering

from solcore.light_source import LightSource
from solcore.constants import q
import numpy as np
from rayflare.textures import regular_pyramids, planar_surface
import matplotlib.pyplot as plt

from solcore import material
from solcore.structure import Layer
from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure
from rayflare.ray_tracing.analytical_rt import analytical_start
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

n_rays = 2000

d = 100e-6

wavelengths = np.linspace(300, 1200, 50) * 1e-9

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
options.randomize_surface = True
options.I_thresh = 1e-3
options.depth_spacing_bulk = 1e-8
options.n_rays = n_rays
options.lambertian_approximation = lambert_approx

# n_bounces = np.arange(1, 11, dtype=int)
# n_bounces = [1, 2, 3, 5, 10, 15]  # 10, 15, 20, 30, 70]

# front_text = regular_pyramids(10, True,
#                               interface_layers=[Layer(100e-9, coverglass_ARC)],)

front_text = planar_surface(
                              # interface_layers=[Layer(100e-9, coverglass_ARC)],
)

front_text_2 = regular_pyramids(52, True, 1,
                                interface_layers=[Layer(100e-9, MgF2), Layer(1000e-9, Pvk)]
                                                             )
rear_text = regular_pyramids(52, False, 1)

rt_str = rt_structure(textures=[front_text, front_text_2, rear_text], materials=[coverglass, Si],
                      widths=[1000e-6, d], incidence=Air, transmission=Ag,
                      options=options, use_TMM=True, save_location='current',
                      overwrite=True)

# options.n_rays = 10
# options.analytical_ray_tracing = 0
# result = rt_str.calculate(options)

options.n_rays = n_rays
options.lambertian_approximation = 0
# #
start = time()
result_1 = rt_str.calculate(options)
print('Elapsed time: ', time() - start)

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


options.analytical_ray_tracing = 2
options.lambertian_approximation = lambert_approx

start = time()
result = rt_str.calculate(options)
print('Elapsed time: ', time() - start)

A_layer_2 = result['A_per_layer']
A_per_interface_2 = result['A_per_interface']
total = result['R'] + result['T'] + np.sum(A_layer_2, axis=1) + A_per_interface_2[1][:,1]
# import matplotlib.pyplot as plt
plt.figure()
plt.plot(wavelengths*1e9, result['R'], 'k-', label='R')
plt.plot(wavelengths*1e9, result['T'], 'r-', label='T')
plt.plot(wavelengths*1e9, A_layer_2, 'g-', label='A')
plt.plot(wavelengths*1e9, A_per_interface_2[1][:,1], 'y-', label='GaAs')
plt.plot(wavelengths*1e9, total, 'k-', label='total', alpha=0.5)
plt.axhline(1)
plt.legend()
plt.show()

plt.figure()
plt.plot(wavelengths*1e9, result['R'], 'k-', label='R')
plt.plot(wavelengths*1e9, result['T'], 'r-', label='T')
plt.plot(wavelengths*1e9, A_layer_2[:,0], 'b-', label='glass')
plt.plot(wavelengths*1e9, A_per_interface_2[1][:,1], 'y-', label='Pvk')
plt.plot(wavelengths*1e9, A_layer_2[:,1], 'g-', label='Si')

plt.plot(wavelengths*1e9, total, 'k-', label='total', alpha=0.5)

plt.plot(wavelengths*1e9, result_1['R'], 'k--')
plt.plot(wavelengths*1e9, result_1['T'], 'r--')
plt.plot(wavelengths*1e9, A_layer[:,1], 'g--')
plt.plot(wavelengths*1e9, A_layer[:,0], 'b--', label='glass')
plt.plot(wavelengths*1e9, A_per_interface[1][:,1], 'y--')
# plt.plot(wavelengths*1e9, A_per_interface[2][:,0], label='Ge')
plt.plot(wavelengths*1e9, total_1, 'k--', label='total', alpha=0.5)
# plt.axhline(1)
plt.legend()
plt.show()

# number of passes:
plt.figure()
plt.plot(wavelengths*1e9, np.mean(result_1['n_passes'],axis=1), 'k--')
plt.plot(wavelengths*1e9, np.mean(result['n_passes'], axis=1), 'k')

plt.plot(wavelengths*1e9, np.max(result_1['n_passes'],axis=1), 'r--')
plt.plot(wavelengths*1e9, np.max(result['n_passes'], axis=1), 'r')
plt.show()

# result_1 is just RT, result is analytical RT

fig, axes = plt.subplots(5, 2, figsize=(8, 15))
for i, ax in enumerate(axes.flatten()):
    ax.hist(result['n_interactions'][i], color='k', bins=51, alpha=0.5, label='Analytical RT', histtype='step', range=[0,50])
    ax.hist(result_1['n_interactions'][i], color='r', bins=51, alpha=0.5, label='RT', histtype='step', range=[0, 50])
plt.title('n_interactions')
plt.show()

fig, axes = plt.subplots(5, 2, figsize=(8, 15))
for i, ax in enumerate(axes.flatten()):
    ax.hist(result['n_passes'][i], color='k', bins=11, alpha=0.5, label='Analytical RT', histtype='step', range=[0,10])
    ax.hist(result_1['n_passes'][i], color='r', bins=11, alpha=0.5, label='RT', histtype='step', range=[0,10])
plt.title('n_passes')
plt.show()


fig, axes = plt.subplots(5, 2, figsize=(8, 15))
for i, ax in enumerate(axes.flatten()):
    ax.hist(result['thetas'][i], color='k', bins=30, alpha=0.5, label='Analytical RT', histtype='step', range=[0, np.pi])
    ax.hist(result_1['thetas'][i], color='r', bins=30, alpha=0.5, label='RT', histtype='step', range=[0, np.pi])
plt.title('thetas')
plt.show()

fig, axes = plt.subplots(5, 2, figsize=(8, 15))
for i, ax in enumerate(axes.flatten()):
    ax.hist(result['phis'][i], color='k', bins=30, alpha=0.5, label='Analytical RT', histtype='step', range=[0, 2*np.pi])
    ax.hist(result_1['phis'][i], color='r', bins=30, alpha=0.5, label='RT', histtype='step', range=[0, 2*np.pi])
plt.title('phis')
plt.show()