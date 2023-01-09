import numpy as np
import os

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
#rc('text', usetex=True)

from solcore.structure import Layer
from solcore import material, si
from solcore.light_source import LightSource
from solcore.constants import q

from rayflare.textures import regular_pyramids, planar_surface
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism import calculate_RAT, process_structure
from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure
from rayflare.angles import make_angle_vector
import matplotlib.pyplot as plt
import seaborn as sns

from cycler import cycler

wavelengths = np.linspace(300, 1200, 100) * 1e-9
#
# options = default_options()
# options.wavelengths = wavelengths
# options.n_theta_bins = 100
# options.nx = 20
# options.ny = options.nx
# options.n_rays = 10*options.nx**2
# options.depth_spacing = 1e-9
# options.pol = "u"
# options.I_thresh = 1e-3
# options.randomize_surface = True
# options.n_jobs = -2
#
# Si = material("Si")()
# Air = material("Air")()
# MgF2 = material("MgF2_RdeM")()
# ITO_back = material("ITO_lowdoping")()
# Perovskite = material("Perovskite_CsBr_1p6eV")()
# Ag = material("Ag_Jiang")()
# aSi_i = material("aSi_i")()
# aSi_p = material("aSi_p")()
# aSi_n = material("aSi_n")()
# LiF = material("LiF")()
# IZO = material("IZO")()
# C60 = material("C60")()
#
# # stack based on doi:10.1038/s41563-018-0115-4
# front_pero_materials = [
#     Layer(100e-9, MgF2),
#     Layer(110e-9, IZO),
#     Layer(15e-9, C60),
#     Layer(1e-9, LiF)
#     ]
#
# front_Si_materials = [
#     Layer(6.5e-9, aSi_n),
#     Layer(6.5e-9, aSi_i),
# ]
#
# back_materials = [
#     Layer(6.5e-9, aSi_i),
#     Layer(6.5e-9, aSi_p),
#     Layer(240e-9, ITO_back)
# ]
#
# options.project_name = "perovskite_Si_RT_poster"
#
# planar_front = planar_surface(size=1,
#                               interface_layers=front_pero_materials,
#                               name="pero_front")
#
# triangle_surf = regular_pyramids(elevation_angle=55, upright=True, size=1,
#                                  interface_layers=front_Si_materials,
#                                  name="Si_front",
#                                  coherent=False, coherency_list=["i"]*len(front_Si_materials)
#                                  )
#
# triangle_surf_back = regular_pyramids(elevation_angle=55, upright=False, size=1,
#                                       interface_layers=back_materials,
#                                       name="Si_back",
#                                       #coherent=False, coherency_list=["i"]*len(back_materials)
#                                       )
#
# rtstr = rt_structure(textures=[planar_front, triangle_surf, triangle_surf_back],
#                      materials= [Perovskite, Si],
#                     widths=[440e-9, 260e-6],
#                      incidence=Air, transmission=Ag,
#                      use_TMM=True, options=options, save_location="current"
#                          )
#
# result = rtstr.calculate(options)
#
pal = sns.color_palette("Paired", n_colors=4)

cols = cycler('color', pal)

params = {
    "axes.prop_cycle": cols,
}

plt.rcParams.update(params)


# np.save("perovskite_Si_RT_poster/result_coh_pero_Si_planar.npy", result)

result = np.load("perovskite_Si_RT_poster/result_coh_pero_Si_planar.npy",allow_pickle=True)[()]

text_res = np.load("perovskite_Si_RT_poster/result_coh_pero_Si.npy",allow_pickle=True)[()]

photon_flux_Si = LightSource(source_type="standard", version="AM1.5g",
                          wl=wavelengths, output_units="photon_flux_per_m").spectrum(wavelengths)[1]

def calc_Si_current(wl, abs, flux):
    return q * np.trapz(abs * flux, wl)/10

J_p1 = calc_Si_current(wavelengths, text_res['A_per_interface'][0][:,4], photon_flux_Si)
J_S1 = calc_Si_current(wavelengths, text_res['A_per_layer'][:,0], photon_flux_Si)
J_p2 = calc_Si_current(wavelengths, result['A_per_layer'][:,0], photon_flux_Si)
J_S2 = calc_Si_current(wavelengths, result['A_per_layer'][:,1], photon_flux_Si)

fig=plt.figure(figsize=(6,3))
plt.subplot(1,1,1)

plt.plot(wavelengths*1e9, text_res['R'], '--k+', mfc="none", label="Conformal pero (2)",
         markersize=6)
plt.plot(wavelengths*1e9, text_res['A_per_interface'][0][:,4], '--+', mfc="none",
         markersize=6, color=pal[3])
plt.plot(wavelengths*1e9, text_res['A_per_layer'][:,0], '--+', mfc="none",
         markersize=6, color=pal[1])

plt.plot(wavelengths*1e9, result['R'], '-ko', mfc="none", label="Planar front (4)", markersize=4)
plt.plot(wavelengths*1e9, result['A_per_layer'][:,0], '--o', mfc="none",
         markersize=4, color=pal[3], alpha=0.5)
plt.plot(wavelengths*1e9, result['A_per_layer'][:,1], '--o', mfc="none",
         markersize=4, color=pal[1], alpha=0.5)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflection / Absorption')
plt.ylim(0, 1)
plt.xlim(300, 1200)
plt.text(600, 0.83, np.round(J_p1,1), color=pal[3])
plt.text(600, 0.66, np.round(J_p2,1), color=pal[3], alpha=0.65)
plt.text(1070, 0.8, np.round(J_S1,1), color=pal[1])
plt.text(1020, 0.6, np.round(J_S2,1), color=pal[1], alpha=0.65)
plt.legend(loc = [0.52, 0.2])
plt.tight_layout()
fig.savefig("perovskite_Si.pdf")
plt.show()

#
# from rayflare.transfer_matrix_method import tmm_structure
#
#
# tmm_struc = tmm_structure([Layer(100e-9, MgF2),
#     Layer(110e-9, IZO),
#     Layer(15e-9, C60),
#     Layer(1e-9, LiF), Layer(440e-9, Perovskite)], incidence=Air, transmission=Si)
#
# tmm_res = tmm_struc.calculate(options)
#
# plt.figure()
# plt.plot(wavelengths*1e9, tmm_res["R"])
# plt.show()
#
#
#
#
#
#
