from rayflare.ray_tracing.analytical_approximation import lambertian_scattering

from solcore.light_source import LightSource
from solcore.constants import q
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

d = 100e-6

n_wl = 2
n_rays = 500
wavelengths =  np.linspace(280, 1000, n_wl) * 1e-9


AM15G = LightSource(source_type='standard', version='AM1.5g', x=wavelengths, output_units='photon_flux_per_m')

options = default_options()

options.analytical_ray_tracing = 0
options.wavelength = wavelengths
options.project_name = 'integration_testing'
options.nx = 20
options.ny = 20
options.n_rays = n_rays
options.parallel = False
options.randomize_surface = True
options.I_thresh = 1e-3

n_bounces = np.arange(1, 11, dtype=int)
n_bounces = [2, 5, 10, 15] #, 20, 30, 70]

front_text = regular_pyramids(52, True, 1,
                              interface_layers=[Layer(80e-9, SiN)],
                                                )

# front_text = planar_surface()
rear_text = regular_pyramids(52, False, 1, 
                             interface_layers=[Layer(80e-9, SiN)],
                                                )
#
# rear_text = planar_surface(
#                              interface_layers=[Layer(80e-9, SiN)
#                                                 ])

options.lambertian_approximation = 0
options.analytical_ray_tracing = 0

rt_str = rt_structure(textures=[front_text, rear_text], materials=[Si],
                      widths=[d], incidence=Air, transmission=Air,
                      options=options, use_TMM=True, save_location='current',
                      overwrite=True)

rt_str.calculate(options)
#
# options.n_rays = n_rays
#
# times = np.empty(len(n_bounces))
#
# results = []
#
# for i1, n_lamb in enumerate(n_bounces):
#
#     options.lambertian_approximation = n_lamb
#
#     rt_str = rt_structure(textures=[front_text, rear_text], materials=[Si],
#                           widths=[d], incidence=Air, transmission=Air,
#                           options=options, use_TMM=True, save_location='current',
#                           overwrite=True)
#
#     start = time()
#     results.append(rt_str.calculate(options))
#     times[i1] = time() - start
#
# options.lambertian_approximation = 0
# options.analytical_ray_tracing = 0
#
# start = time()
# result_rt = rt_str.calculate(options)
# time_rt = time() - start
#
# current_rt = q*np.trapz(AM15G.spectrum()[1] * result_rt['A_per_layer'][:,0], wavelengths)/10
# currents = np.empty(len(n_bounces))
# cols = sns.cubehelix_palette(len(n_bounces), gamma=0.5)
#
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
#
# for i1, res in enumerate(results):
#     ax1.plot(wavelengths*1e9, res['A_per_layer'][:,0], label=f'{n_bounces[i1]} passes',
#              color=cols[i1])
#     ax1.plot(wavelengths * 1e9, res['R'], '--', color=cols[i1])
#     ax1.plot(wavelengths * 1e9, res['T'], '-.', color=cols[i1])
#
#     currents[i1] = q*np.trapz(AM15G.spectrum()[1] * res['A_per_layer'][:,0], wavelengths)/10
#
#     # ax2.plot(wavelengths*1e9, res['A_per_layer'][:,0]/result_rt['A_per_layer'][:,0], color=cols[i1],
#     #          label=f'{n_bounces[i1]} passes')
#
# ax1.text(600, 0.9, '$A_{Si}$')
# ax1.text(300, 0.2, '$R$')
# ax1.text(1100, 0.08, '$Air$')
# ax1.plot(wavelengths*1e9, result_rt['A_per_layer'][:,0], '--', color='k', label='No approximations')
# ax1.plot(wavelengths*1e9, result_rt['R'], '--', color='k')
# ax1.plot(wavelengths*1e9, result_rt['T'], '-.', color='k')
# # ax1.plot(wavelengths*1e9, AM15G.spectrum()[1]/np.max(AM15G.spectrum()[1]), '-k', label='AM1.5G')
# ax1.legend()
# ax1.set_xlabel("Wavelength (nm)")
# ax1.set_ylabel("R / A / T")
#
# ax4.plot(wavelengths*1e9, np.percentile(result_rt['n_passes'],50, axis=1), '-k', label='Median')
# ax4.plot(wavelengths*1e9, np.percentile(result_rt['n_passes'], 25, 1), '--r', label='25th percentile')
# ax4.plot(wavelengths*1e9, np.percentile(result_rt['n_passes'], 75,1), '--b', label='75th percentile')
# ax4.legend()
# ax4.set_xlabel("Wavelength (nm)")
# ax4.set_ylabel("Number of passes")
#
# # ax5 = ax4.twinx()
# # ax5.set_ylim(0.99*np.min(currents/current_rt), 1)
# ax3.plot(n_bounces, times/n_wl, 'o-k')
# # ax3.set_ylim(0, 1.02)
# ax3.axhline(time_rt/n_wl, color='k', linestyle='--')
# ax2.axhline(1, color='r', linestyle='--')
# ax3.set_xlabel('Number of passes')
# ax3.set_ylabel('Time / Time RT')
#
# ax2.plot(n_bounces, currents/current_rt, 'o-r')
# ax2.set_xlabel('Number of passes')
# ax2.set_ylabel(r'$J_{max}$ / $J_{max, exact}$')
#
# ax1.set_title('Absorption, reflection and transmission')
# ax2.set_title(r'$J_{max}$ with # of passes before Lambertian scattering')
# ax3.set_title('Time to calculate with approximations / time without')
# ax4.set_title('# of passes with wavelength, without approximations')
#
# plt.tight_layout()
# plt.show()
#
#
# # print(time_lamb, time_rt)
# # P_escape_front_down, P_escape_back_down, P_absorb_down, P_front_surf_down, P_rear_surf_down, \
# #                             P_escape_front_up, P_escape_back_up, P_absorb_up, P_front_surf_up, P_rear_surf_up = result_lambertian[0]
#
