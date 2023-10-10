import numpy as np

from solcore.structure import Layer
from solcore import material

from rayflare.textures import regular_pyramids
from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure
import matplotlib.pyplot as plt
import seaborn as sns
import os

from cycler import cycler

wavelengths = np.linspace(300, 1200, 40) * 1e-9

options = default_options()
options.wavelengths = wavelengths
options.nx = 5
options.ny = options.nx
options.n_rays = 4 * options.nx**2
options.depth_spacing = 1e-9
options.pol = "u"
options.I_thresh = 1e-3
options.randomize_surface = True
# mimic random pyramids; do not want correlation between incident position on
# front and rear pyramids

options.n_jobs = -1  # use all cores; to use all but one, change to -2 etc.

# You only need to run this once to add materials to the database!
from solcore.material_system import create_new_material

cur_path = cur_path = os.path.dirname(os.path.abspath(__file__))
create_new_material(
    "Perovskite_CsBr_1p6eV",
    os.path.join(cur_path, "../../solcore-education/solcore-workshop/notebooks/data/CsBr10p_1to2_n_shifted.txt"),
    os.path.join(cur_path, "../../solcore-education/solcore-workshop/notebooks/data/CsBr10p_1to2_k_shifted.txt"),
)
create_new_material(
    "ITO_lowdoping",
    os.path.join(cur_path, "data/model_back_ito_n.txt"),
    os.path.join(cur_path, "data/model_back_ito_k.txt"),
)
create_new_material(
    "Ag_Jiang",
    os.path.join(cur_path, "data/Ag_UNSW_n.txt"),
    os.path.join(cur_path, "data/Ag_UNSW_k.txt"),
)
create_new_material(
    "aSi_i",
    os.path.join(cur_path, "data/model_i_a_silicon_n.txt"),
    os.path.join(cur_path, "data/model_i_a_silicon_k.txt"),
)
create_new_material(
    "aSi_p",
    os.path.join(cur_path, "data/model_p_a_silicon_n.txt"),
    os.path.join(cur_path, "data/model_p_a_silicon_k.txt"),
)
create_new_material(
    "aSi_n",
    os.path.join(cur_path, "data/model_n_a_silicon_n.txt"),
    os.path.join(cur_path, "data/model_n_a_silicon_k.txt"),
)
create_new_material(
    "MgF2_RdeM",
    os.path.join(cur_path, "data/MgF2_RdeM_n.txt"),
    os.path.join(cur_path, "data/MgF2_RdeM_k.txt"),
)
create_new_material(
    "C60",
    os.path.join(cur_path, "data/C60_Ren_n.txt"),
    os.path.join(cur_path, "data/C60_Ren_k.txt"),
)
create_new_material(
    "IZO",
    os.path.join(cur_path, "data/IZO_Ballif_rO2_10pcnt_n.txt"),
    os.path.join(cur_path, "data/IZO_Ballif_rO2_10pcnt_k.txt"),
)
# Only run once until here, then can comment out

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

back_materials = [Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_p), Layer(240e-9, ITO_back)]

options.project_name = "perovskite_Si_rt"

triangle_surf = regular_pyramids(
    elevation_angle=55,
    upright=True,
    size=1,
    interface_layers=front_materials,
    name="coh_front",
    prof_layers=[5],
)

triangle_surf_back = regular_pyramids(
    elevation_angle=55,
    upright=False,
    size=1,
    interface_layers=back_materials,
    name="Si_back",
)

rtstr_coh = rt_structure(
    textures=[triangle_surf, triangle_surf_back],
    materials=[Si],
    widths=[260e-6],
    incidence=Air,
    transmission=Ag,
    use_TMM=True,
    options=options,
    save_location="current",
)

result_coh = rtstr_coh.calculate(options)

triangle_surf = regular_pyramids(
    elevation_angle=55,
    upright=True,
    size=1,
    interface_layers=front_materials,
    coherency_list=["i"] * len(front_materials),
    name="inc_front",
    prof_layers=[5],
)

triangle_surf_back = regular_pyramids(
    elevation_angle=55,
    upright=False,
    size=1,
    interface_layers=back_materials,
    coherency_list=["i"] * len(back_materials),
    name="inc_back",
)

rtstr_inc = rt_structure(
    textures=[triangle_surf, triangle_surf_back],
    materials=[Si],
    widths=[260e-6],
    incidence=Air,
    transmission=Ag,
    use_TMM=True,
    options=options,
    save_location="current",
)

result_inc = rtstr_inc.calculate(options)

pal = sns.color_palette("husl", n_colors=len(front_materials) + len(back_materials) + 2)

cols = cycler("color", pal)

params = {
    "axes.prop_cycle": cols,
}

plt.rcParams.update(params)


fig = plt.figure(figsize=(8, 3.7))
plt.subplot(1, 1, 1)
plt.plot(wavelengths * 1e9, result_coh["R"], "-ko", label="R")
plt.plot(wavelengths * 1e9, result_coh["T"], mfc="none", label="T")
plt.plot(wavelengths * 1e9, result_coh["A_per_layer"][:, 0], "-o", label="Si")
plt.plot(
    wavelengths * 1e9,
    result_coh["A_per_interface"][0],
    "-o",
    label=[None, "IZO", "C60", None, "Perovskite", None, None],
)
plt.plot(
    wavelengths * 1e9, result_coh["A_per_interface"][1], "-o", label=[None, None, "ITO"]
)

plt.plot(wavelengths * 1e9, result_inc["R"], "--ko", mfc="none")
plt.plot(wavelengths * 1e9, result_inc["T"], mfc="none")
plt.plot(wavelengths * 1e9, result_inc["A_per_layer"][:, 0], "--o", mfc="none")
plt.plot(wavelengths * 1e9, result_inc["A_per_interface"][0], "--o", mfc="none")
plt.plot(wavelengths * 1e9, result_inc["A_per_interface"][1], "--o", mfc="none")

plt.plot([300, 301], [0, 0], "-k", label="coherent")
plt.plot([300, 301], [0, 0], "--k", label="incoherent")
plt.xlabel("Wavelength (nm)")
plt.ylabel("R / A / T")
plt.ylim(0, 1)
plt.xlim(300, 1200)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

wl_Eg = wavelengths < 800e-9

pal = sns.cubehelix_palette(sum(wl_Eg), reverse=True)
cols = cycler("color", pal)
params = {
    "axes.prop_cycle": cols,
}
plt.rcParams.update(params)

pos = np.arange(0, 440, options.depth_spacing * 1e9)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(pos, result_coh["interface_profiles"][0][wl_Eg].T)
ax1.set_ylim(0, 0.02)
ax1.set_xlabel("z (nm)")
ax1.set_ylabel("a(z)")
ax2.plot(pos, result_inc["interface_profiles"][0][wl_Eg].T)
ax2.set_ylim(0, 0.02)
ax2.yaxis.set_ticklabels([])
ax2.set_xlabel("z (nm)")
plt.show()


pos_bulk = pos = np.arange(0, 260, options.depth_spacing_bulk * 1e6)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.semilogy(pos, result_coh["profile"][~wl_Eg].T)
ax1.set_ylim(1e-8, 0.00015)
ax1.set_xlabel("z (um)")
ax1.set_ylabel("a(z)")
ax2.semilogy(pos, result_inc["profile"][~wl_Eg].T)
ax2.set_ylim(1e-8, 0.00015)
ax2.yaxis.set_ticklabels([])
ax2.set_xlabel("z (um)")
plt.show()
