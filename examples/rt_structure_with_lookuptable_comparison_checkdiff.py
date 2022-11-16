import numpy as np
import os

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

wavelengths = np.linspace(1100, 1150, 1) * 1e-9

options = default_options()
options.wavelengths = wavelengths
options.project_name = "perovskite_Si_diff_linear"
options.n_theta_bins = 100
options.nx = 1
options.ny = 1
options.n_rays = 1300*options.nx**2
options.depth_spacing = 1e-9
options.pol = "s"
options.I_thresh = 1e-3
options.theta_spacing = "sin"

th_intv, _, av = make_angle_vector(options.n_theta_bins, options.phi_symmetry, options.c_azimuth, options.theta_spacing)

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

# whether pyramids are upright or inverted is relative to front incidence.
# so if the same etch is applied to both sides of a slab of silicon, one surface
# will have 'upright' pyramids and the other side will have 'not upright' (inverted)
# pyramids in the model

surf = regular_pyramids(elevation_angle=55, upright=True)
surf_back = regular_pyramids(elevation_angle=55, upright=False)

front_surf = Interface(
    "RT_TMM",
    texture=surf,
    layers=front_materials,
    name="Perovskite_aSi_coherent",
    coherent=True,
    # coherency_list = ["i"]*len(front_materials)
)

back_surf = Interface(
    "RT_TMM", texture=surf_back, layers=back_materials, name="aSi_ITO_coherent", coherent=True
)


bulk_Si = BulkLayer(260e-6, Si, name="Si_bulk")  # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options, save_location="current")

result = calculate_RAT(SC, options, save_location="current")
result_ARM = result[0]
result_ARM_perpass = result[1]

options.n_rays = options.nx**2

# ray-tracing WITHOUT angular redistribution method

triangle_surf = regular_pyramids(elevation_angle=55, upright=True, size=1,
                                 interface_layers=front_materials,
                                 name="coh_front"
                                 )

triangle_surf_back = regular_pyramids(elevation_angle=55, upright=False, size=1,
                                      interface_layers=back_materials,
                                      name="coh_back"
                                      )

rtstr_inc = rt_structure(textures=[triangle_surf, triangle_surf_back],
                     materials= [Si],
                    widths=[260e-6],
                     incidence=Air, transmission=Ag,
                     use_TMM=True, options=options, save_location="current"
                         )
options.parallel = False
result_RT = rtstr_inc.calculate(options)

results_per_layer_front = np.sum(result_ARM_perpass["a"][0], 0)
results_per_layer_back = np.sum(result_ARM_perpass["a"][1], 0)

pal = sns.color_palette("husl", n_colors=len(front_materials) + len(back_materials) + 1)

cols = cycler('color', pal)

params = {
    "axes.prop_cycle": cols,
}

plt.rcParams.update(params)

sum_ARM = result_ARM['R'][0] + result_ARM['T'][0] + result_ARM['A_bulk'][0] + \
          np.sum(results_per_layer_front, 1) + np.sum(results_per_layer_back,1)

sum_RT = result_RT['R'] + result_RT['T'] + result_RT['A_per_layer'].T + \
          np.sum(result_RT['A_per_interface'][0], 1) + np.sum(result_RT['A_per_interface'][1], 1)

total_abs = np.sum(np.isnan(result_RT["thetas"]), 1)/options.n_rays

plt.figure()
plt.plot(wavelengths*1e9, total_abs)
plt.plot(wavelengths*1e9, result_RT['A_per_layer'][:,0] + \
          np.sum(result_RT['A_per_interface'][0], 1) +
         np.sum(result_RT['A_per_interface'][1], 1),
         '--r'
)
plt.show()

plt.figure()
plt.plot(wavelengths*1e9, sum_ARM, '-or')
plt.plot(wavelengths*1e9, sum_RT.T, '-ob')
plt.show()

fig=plt.figure(figsize=(9,3.7))
plt.subplot(1,1,1)
plt.plot(wavelengths*1e9, result_ARM['R'][0], '-ko')
plt.plot(wavelengths*1e9, result_ARM['T'][0], '-ro')
plt.plot(wavelengths*1e9, result_ARM['A_bulk'][0], '-o')
plt.plot(wavelengths*1e9, results_per_layer_front, '-o')
plt.plot(wavelengths*1e9, results_per_layer_back, '-o')

plt.plot(wavelengths*1e9, result_RT['R'], '--ko', mfc='none')
plt.plot(wavelengths*1e9, result_RT['T'], '--ro', mfc='none')
plt.plot(wavelengths*1e9, result_RT['A_per_layer'][:,0], '--o', mfc='none')
plt.plot(wavelengths*1e9, result_RT['A_per_interface'][0], '--o', mfc='none')
plt.plot(wavelengths*1e9, result_RT['A_per_interface'][1], '--o', mfc='none')

plt.xlabel('Wavelength (nm)')
plt.ylabel('R / A / T')
plt.ylim(0, 1)
plt.xlim(300, 1200)

plt.show()


fig=plt.figure(figsize=(9,3.7))
plt.subplot(1,1,1)
plt.plot(wavelengths*1e9, result_ARM['R'][0], 'k-o')
plt.plot(wavelengths*1e9, result_ARM['T'][0], 'r-o')

plt.plot(wavelengths*1e9, result_RT['R'], 'k--o', mfc='none')
plt.plot(wavelengths*1e9, result_RT['T'], 'r--o', mfc='none')

plt.xlabel('Wavelength (nm)')
plt.ylabel('R / A / T')
plt.xlim(300, 1200)

plt.show()

