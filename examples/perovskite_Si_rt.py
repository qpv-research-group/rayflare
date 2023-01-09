import numpy as np
import os

from solcore.structure import Layer
from solcore import material

from rayflare.textures import regular_pyramids
from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure
import matplotlib.pyplot as plt
import seaborn as sns

from cycler import cycler

wavelengths = np.linspace(300, 1200, 100) * 1e-9

options = default_options()
options.wavelengths = wavelengths
options.n_theta_bins = 100
options.nx = 20
options.ny = options.nx
options.n_rays = 10*options.nx**2
options.depth_spacing = 1e-9
options.pol = "u"
options.I_thresh = 1e-3
options.randomize_surface = True
options.n_jobs = -1

Si = material("Si")()
Air = material("Air")()
MgF2 = material("MgF2_RdeM")()
ITO_back = material("ITO_lowdoping")()
Perovskite = material("Perovskite_CsBr_1p6eV")()
Ag = material("Ag_Jiang")()
aSi_i = material("aSi_i")()
aSi_p = material("aSi_p")()
aSi_n = material("aSi_n")()
LiF = material("LiF")()
IZO = material("IZO")()
C60 = material("C60")()

# stack based on doi:10.1038/s41563-018-0115-4
front_materials = [
    Layer(100e-9, MgF2),
    Layer(110e-9, IZO),
    Layer(15e-9, C60),
    Layer(1e-9, LiF),
    Layer(440e-9, Perovskite),
    Layer(6.5e-9, aSi_n),
    Layer(6.5e-9, aSi_i),
]

back_materials = [
    Layer(6.5e-9, aSi_i),
    Layer(6.5e-9, aSi_p),
    Layer(240e-9, ITO_back)
]

options.project_name = "perovskite_Si_RT_poster"

triangle_surf = regular_pyramids(elevation_angle=55, upright=True, size=1,
                                 interface_layers=front_materials,
                                 name="coh_front"
                                 )

triangle_surf_back = regular_pyramids(elevation_angle=55, upright=False, size=1,
                                      interface_layers=back_materials,
                                      name="Si_back",
                                      #coherent=False, coherency_list=["i"]*len(back_materials)
                                      )

rtstr_coh = rt_structure(textures=[triangle_surf, triangle_surf_back],
                     materials= [Si],
                    widths=[260e-6],
                     incidence=Air, transmission=Ag,
                     use_TMM=True, options=options, save_location="current"
                         )

result_coh = rtstr_coh.calculate(options)

# triangle_surf = regular_pyramids(elevation_angle=55, upright=True, size=1,
#                                  interface_layers=front_materials,
#                                  coherent=False,
#                                  coherency_list=["i"] * len(front_materials),
#                                  name="inc_front"
#                                  )
#
# triangle_surf_back = regular_pyramids(elevation_angle=55, upright=False, size=1,
#                                       interface_layers=back_materials,
#                                       coherent=False,
#                                       coherency_list=["i"]*len(back_materials),
#                                       name="inc_back"
#                                       )
#
# rtstr_inc = rt_structure(textures=[triangle_surf, triangle_surf_back],
#                      materials= [Si],
#                     widths=[260e-6],
#                      incidence=Air, transmission=Ag,
#                      use_TMM=True, options=options, save_location="current"
#                          )
#
# result_inc = rtstr_inc.calculate(options)

pal = sns.color_palette("husl", n_colors=len(front_materials) + len(back_materials) + 2)

cols = cycler('color', pal)

params = {
    "axes.prop_cycle": cols,
}

plt.rcParams.update(params)




fig=plt.figure(figsize=(9,3.7))
plt.subplot(1,1,1)
plt.plot(wavelengths*1e9, result_coh['R'], '-ko')
plt.plot(wavelengths*1e9, result_coh['T'], mfc='none')
plt.plot(wavelengths*1e9, result_coh['A_per_layer'][:,0], '--o')
plt.plot(wavelengths*1e9, result_coh['A_per_interface'][0], '--o')
plt.plot(wavelengths*1e9, result_coh['A_per_interface'][1], '--o')

# plt.plot(wavelengths*1e9, result_inc['R'], '--ko', mfc='none')
# plt.plot(wavelengths*1e9, result_inc['T'], mfc='none')
# plt.plot(wavelengths*1e9, result_inc['A_per_layer'][:,0], '--o', mfc='none')
# plt.plot(wavelengths*1e9, result_inc['A_per_interface'][0], '--o', mfc='none')
# plt.plot(wavelengths*1e9, result_inc['A_per_interface'][1], '--o', mfc='none')

plt.xlabel('Wavelength (nm)')
plt.ylabel('R / A / T')
plt.ylim(0, 1)
plt.xlim(300, 1200)

plt.show()

np.save("perovskite_Si_RT_poster/result_coh_pero_Si.npy", result_coh)
















