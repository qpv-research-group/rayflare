from solcore.light_source import LightSource
import numpy as np

# the ray-tracer (from v2.0.0) will use numba's just-in-time (jit) compilation to make
# heavily-used functions faster. However, the initial compilation also takes time, and thus if
# these functions are not called many times, it is faster to disable the jit compilation
# completely. The two lines below do this; this has to be done before importing any rayflare
# functions or classes.
from numba import config
config.DISABLE_JIT = True

from rayflare.textures import regular_pyramids
import matplotlib.pyplot as plt

from solcore import material
from solcore.structure import Layer
from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure
from time import time

SiN = material("Si3N4")()
Si = material("Si")()
Air = material("Air")()
Ag = material("Ag")()
MgF2 = material("MgF2")()
Ge = material("Ge")()

GaAs = material("GaAs")()

Pvk = material("Perovskite_CsBr_1p6eV")()

n_rays = 2000

d = 100e-6

wavelengths = np.linspace(300, 1000, 50) * 1e-9

AM15G = LightSource(source_type='standard', version='AM1.5g', x=wavelengths,
                    output_units='photon_flux_per_m')

options = default_options()

options.wavelength = wavelengths
options.project_name = 'pvk_Si_analytical_RT'
options.nx = 20
options.ny = 20
options.theta_in = 0.1
options.parallel = False # for this example, with this choice of wavelengths, initializing the parallel
# threads takes more time than it saves on executing the ray-tracing. For a more transparent structure
# (at longer wavelengths in this case), this would no longer be true.
options.randomize_surface = True
options.I_thresh = 1e-3
options.depth_spacing_bulk = 1e-8
options.n_rays = n_rays
options.maximum_passes = 100
options.n_rays = n_rays
options.pol = 'u'
options.use_numba = False

front_text = regular_pyramids(52, True, 1,
                                interface_layers=[Layer(100e-9, MgF2), Layer(1000e-9, Pvk)],
                                analytical=True)

# note: not setting analytical = True for the last surface; rays will no longer be travelling in a
# single direction after interacting with front_text_2, so this surface will not be treated analytically
# anyway, even if we set analytical = True.

rear_text = regular_pyramids(52, False, 1)

rt_str = rt_structure(textures=[front_text, rear_text], materials=[Si],
                      widths=[d], incidence=Air, transmission=Ag,
                      options=options, use_TMM=True, save_location='current',
                      overwrite=True)

start = time()
result = rt_str.calculate(options)
print('Elapsed time: ', time() - start)

A_layer = result['A_per_layer']
A_per_interface = result['A_per_interface']
total = result['R'] + result['T'] + np.sum(A_layer, axis=1) + A_per_interface[0][:, 1]

plt.figure()

plt.plot(wavelengths * 1e9, result['R'], 'k-', label='R')
plt.plot(wavelengths * 1e9, result['R0'], 'k-', alpha=0.5, label='R0')
plt.plot(wavelengths * 1e9, result['T'], 'r-', label='T')
plt.plot(wavelengths*1e9, A_layer[:,0], 'b-', label='Si')
plt.plot(wavelengths*1e9, A_per_interface[0][:,1], 'y-', label='Pvk')
plt.plot(wavelengths * 1e9, total, 'k--', label='total', alpha=0.5)
plt.xlabel("Wavelength (nm)")
plt.ylabel("R/A/T")
plt.legend()
plt.show()

# number of passes:
plt.figure()
plt.plot(wavelengths * 1e9, np.mean(result.ray_data.n_passes, axis=1), 'k-', label='Mean number of passes')
plt.plot(wavelengths * 1e9, np.max(result.ray_data.n_passes, axis=1), 'r-', label='Max. number of passes')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Number of passes through a bulk material")
plt.legend()
plt.show()

frac_reaching_max = [np.sum(x == options.maximum_passes) for x in result.ray_data.n_passes]

print("Maximum % of rays which reaches maximum_passes:", 100*np.max(frac_reaching_max)/options.n_rays)

# plot distribution of number of interactions 10 different wavelengths
wavelengths_long = np.where(wavelengths*1e9 > 600)[0]
wavelength_inds = np.linspace(np.min(wavelengths_long), np.max(wavelengths_long), 10).astype(int)

fig, axes = plt.subplots(5, 2, figsize=(8, 15))
axes = axes.flatten()

for i, wl_ind in enumerate(wavelength_inds):
    axes[i].hist(result.ray_data.n_interactions[wl_ind], color='r', bins=10, alpha=0.5,
                 label='n_interactions', histtype='step')
    axes[i].hist(result.ray_data.n_passes[wl_ind], color='k', bins=11, alpha=0.5,
                 label='n_passes', histtype='step')
    axes[i].set_title(f'{int(wavelengths[wl_ind]*1e9)} nm')
    if i > 7:
        axes[i].set_xlabel('Number of interactions/passes')

    if i % 2 == 0:
        axes[i].set_ylabel('Number of rays')

axes[0].legend()
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(5, 2, figsize=(8, 15))
axes = axes.flatten()

for i, wl_ind in enumerate(wavelength_inds):
    axes[i].hist(result.ray_data.thetas[wl_ind], color='r', bins=10, alpha=0.5,
                 label='theta', histtype='step')
    axes[i].hist(result.ray_data.phis[wl_ind], color='k', bins=11, alpha=0.5,
                 label='phi', histtype='step')
    axes[i].set_title(f'{int(wavelengths[wl_ind]*1e9)} nm')
    if i > 7:
        axes[i].set_xlabel('Number of interactions/passes')

    if i % 2 == 0:
        axes[i].set_ylabel('Number of rays')

axes[0].legend()
plt.tight_layout()
plt.show()