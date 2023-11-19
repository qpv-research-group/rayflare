"""
Look at reflection, absorption, transmission and absorption profiles in a
GaInP/GaAs/Si triple-junction cell with an epoxy bonding layer in between
the III-V and Si cells. The Si cell has a pyramidal surface texture while
the front surface is planar. We will use ray-tracing with integrated TMM
as outlined in the schematic below.

---------
ARC
---------
GaAs
---------
GaAs
---------
Epoxy/spacer/bonding layer
/\/\/\/\/\/\/\/\
Si with pyramidal surface
\/\/\/\/\/\/\/\/
Al back mirror

Incidence medium = Air
Interface 1 = ARC/GaInP/GaAs
Bulk 1 = Epoxy/bonding layer
Interface 2 = Si cell front layers
Bulk 2 = Si
Interface 3 = Si cell back layers
Transmission medium = Al

"""

import numpy as np
import matplotlib.pyplot as plt

from rayflare.textures import regular_pyramids, planar_surface
from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure

from solcore.structure import Layer
from solcore import material
from solcore.absorption_calculator import search_db
from solcore.constants import q
from solcore.light_source import LightSource

import os

wavelengths = np.linspace(300, 1200, 150) * 1e-9

options = default_options()
options.n_rays = 2000
options.randomize_surface = True
options.project_name = "GaAs_GaAs_Si_spacer_tmm_rt"
options.wavelength = wavelengths

# Set up materials
# from solcore.absorption_calculator import download_db
# download_db() # uncomment to download the database if you haven't already
epoxy_result = search_db("NOA-61")[0]  # Norland optical adhesive 61. n = 1.5, typical for materials used in this way

MgF2_pageid = str(search_db(os.path.join("MgF2", "Rodriguez-de Marcos"))[0][0])
Ta2O5_pageid = str(search_db(os.path.join("Ta2O5", "Rodriguez-de Marcos"))[0][0])
MgF2 = material(MgF2_pageid, nk_db=True)()
Ta2O5 = material(Ta2O5_pageid, nk_db=True)()

GaAs = material("GaAs")()
Si = material("Si")()
Al = material("Al")()
Air = material("Air")()
NOA = material(str(epoxy_result[0]), nk_db=True)()
NOA = material("Si3N4")()
Al2O3 = material("Al2O3")()
Ag = material("Ag_Jiang")()
aSi_i = material("aSi_i")()
aSi_p = material("aSi_p")()
aSi_n = material("aSi_n")()
ITO_back = material("ITO_back")()
InAlP = material("AlInP")(Al=0.53)
GaInP = material("GaInP")(In=0.49)
AlGaAs = material("AlGaAs")(Al=0.8)

# from solcore.material_system import create_new_material
# cur_path = os.path.dirname(os.path.abspath(__file__))
# create_new_material('aSi_i', os.path.join(cur_path, 'data/model_i_a_silicon_n.txt'),os.path.join(cur_path, 'data/model_i_a_silicon_k.txt'))
# create_new_material('aSi_p', os.path.join(cur_path, 'data/model_p_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_p_a_silicon_k.txt'))
# create_new_material('aSi_n', os.path.join(cur_path, 'data/model_n_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_n_a_silicon_k.txt'))
# create_new_material('ITO_front', os.path.join(cur_path, 'data/front_ITO_n.txt'), os.path.join(cur_path, 'data/front_ITO_k.txt'))
# create_new_material('ITO_back', os.path.join(cur_path, 'data/back_ITO_n.txt'), os.path.join(cur_path, 'data/back_ITO_k.txt'))

ITO_front = material("ITO_front")()
ITO_back = material("ITO_back")()
aSi_i = material("aSi_i")()
aSi_p = material("aSi_p")()
aSi_n = material("aSi_n")()

plt.figure()
plt.plot(wavelengths * 1e9, GaAs.n(wavelengths), label="GaAs")
plt.plot(wavelengths * 1e9, GaInP.n(wavelengths), label="GaInP")
plt.plot(wavelengths * 1e9, NOA.n(wavelengths), label="epoxy")
plt.plot(wavelengths * 1e9, Si.n(wavelengths), label="Si")
plt.show()

# x = np.linspace(0, 1, 6)
#
# for Al in x:
#     AlGaAs = material("AlGaAs")(Al=Al)
#     plt.plot(wavelengths*1e9, AlGaAs.n(wavelengths), label=Al)
# plt.legend()
# plt.show()

front_layers = [
    Layer(50e-9, MgF2),  # ARC 1
    Layer(40e-9, Ta2O5),  # ARC 2
    Layer(30e-9, InAlP),  # tunnel junction/cell 1 window layer
    Layer(150e-9, GaAs),  # cell 1
    Layer(120e-9, GaInP),  # tunnel junction/cell 2 window layer
    Layer(3000e-9, GaAs),  # cell 2
    Layer(120e-9, GaInP),  # cell 2 BSF
    # Layer(20e-6, AlGaAs), # spacer to reduce interference losses
]

front_labels = [
    r"MgF$_2$",
    r"Ta$_2$O$_5$",
    "InAlP (window)",
    "GaAs (cell 1)",
    "GaInP (tunnel/window)",
    "GaAs (cell 2)",
    "GaInP (BSF)",
]

cell_layer_ind = [4, 6]


Si_front_layers = [Layer(30e-9, ITO_front), Layer(6.5e-9, aSi_p), Layer(6.5e-9, aSi_i)]

Si_front_labels = ["ITO (front)", "a-Si p", "a-Si i"]

Si_back_layers = [Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_n), Layer(240e-9, ITO_back)]

Si_back_labels = ["a-Si i", "a-Si n", "ITO (back)"]

front_surface = planar_surface(interface_layers=front_layers, prof_layers=cell_layer_ind)
Si_surface = regular_pyramids(upright=True, interface_layers=Si_front_layers)
back_surface = regular_pyramids(upright=False, interface_layers=Si_back_layers)

rt_str = rt_structure(
    textures=[front_surface, Si_surface, back_surface],
    materials=[NOA, Si],
    widths=[200e-6, 250e-6],
    incidence=Air,
    transmission=Al,
    use_TMM=True,
    options=options,
    overwrite=True,
)


result = rt_str.calculate(options)

front_plot_res = []
front_plot_labels = []
for j1, layer_res in enumerate(result["A_per_interface"][0].T):
    if np.any(layer_res > 0.01):
        front_plot_res.append(layer_res)
        front_plot_labels.append(front_labels[j1])

# check which layers in rear stack are absorbing more than 1% of light at any wavelength
Si_front_plot_res = []
Si_front_plot_labels = []
for j1, layer_res in enumerate(result["A_per_interface"][1].T):
    if np.any(layer_res > 0.01):
        Si_front_plot_res.append(layer_res)
        Si_front_plot_labels.append(Si_front_labels[j1])

Si_back_plot_res = []
Si_back_plot_labels = []
for j1, layer_res in enumerate(result["A_per_interface"][2].T):
    if np.any(layer_res > 0.01):
        Si_back_plot_res.append(layer_res)
        Si_back_plot_labels.append(Si_back_labels[j1])

plt.figure(figsize=(8, 4))
plt.plot(options.wavelength * 1e9, result["T"], label="T (Al)")
plt.plot(options.wavelength * 1e9, result["R"], "--", label="R")

if len(front_plot_res) > 0:
    plt.plot(options.wavelength * 1e9, np.array(front_plot_res).T, label=front_plot_labels)

if len(Si_front_plot_res) > 0:
    plt.plot(options.wavelength * 1e9, np.array(Si_front_plot_res).T, label=Si_front_plot_labels)

plt.plot(options.wavelength * 1e9, result["A_per_layer"][:, 1], label="Si")

if len(Si_back_plot_res) > 0:
    plt.plot(options.wavelength * 1e9, np.array(Si_back_plot_res).T, label=Si_back_plot_labels)

plt.legend(bbox_to_anchor=(1.05, 1))
plt.xlabel("Wavelength (nm)")
plt.ylabel("R/A/T")
plt.tight_layout()
plt.show()

# limiting currents:

light_source = LightSource(source_type="standard", version="AM1.5g", x=wavelengths, output_units="photon_flux_per_m")

photon_flux = light_source.spectrum(wavelengths)[1]

J_GaAs_1 = q * np.trapz(result["A_per_interface"][0][:, cell_layer_ind[0] - 1] * photon_flux, wavelengths) / 10
J_GaAs_2 = q * np.trapz(result["A_per_interface"][0][:, cell_layer_ind[1] - 1] * photon_flux, wavelengths) / 10
J_Si = q * np.trapz(result["A_per_layer"][:, 1] * photon_flux, wavelengths) / 10

print(J_GaAs_1, J_GaAs_2, J_Si)
