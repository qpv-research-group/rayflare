import numpy as np

from solcore.structure import Layer
from solcore import material

from rayflare.textures import regular_pyramids, planar_surface
from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure
import matplotlib.pyplot as plt
import seaborn as sns

from time import time

from cycler import cycler

wavelengths = np.linspace(300, 1180, 40) * 1e-9

options = default_options()
options.wavelength = wavelengths
options.nx = 30
options.ny = options.nx
options.n_rays = 4 * options.nx**2
options.depth_spacing = 1e-9
options.pol = "u"
options.I_thresh = 1e-3
options.randomize_surface = True
options.pol = 'u'
options.parallel
# mimic random pyramids; do not want correlation between incident position on
# front and rear pyramids

options.n_jobs = -1  # use all cores; to use all but one, change to -2 etc.

# same materials used in other perovskite examples (see code there to add materials to
# database if necessary)

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
glass = material("BK7")()

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

options.project_name = "perovskite_Si_coverglass_rt"

# glass surface (no ARC)
front_surf = planar_surface(
    analytical=True
)

triangle_surf = regular_pyramids(
    elevation_angle=52, upright=True, size=1, interface_layers=front_materials,
    name="coh_front",
    analytical=True,
)

triangle_surf_back = regular_pyramids(
    elevation_angle=52, upright=False, size=1, interface_layers=back_materials,
    name="Si_back"
)

rtstr_coh = rt_structure(
    textures=[front_surf, triangle_surf, triangle_surf_back],
    materials=[glass, Si],
    widths=[1e-6, 260e-6],
    incidence=Air,
    transmission=Ag,
    use_TMM=True,
    options=options,
    save_location="current",
    overwrite=True, # recalculate TMM lookuptables every time
)

start = time()
result_coh = rtstr_coh.calculate(options)
print("Time taken (coherent): ", time() - start)

pal = sns.color_palette("husl", n_colors=len(front_materials) + len(back_materials) + 2)

cols = cycler("color", pal)

params = {"axes.prop_cycle": cols}

plt.rcParams.update(params)

fig = plt.figure(figsize=(8, 3.7))
plt.subplot(1, 1, 1)
plt.plot(wavelengths * 1e9, result_coh["R"], "-ko", label="R")
plt.plot(wavelengths * 1e9, result_coh["T"], mfc="none", label="T")
plt.plot(wavelengths * 1e9, result_coh["A_per_layer"][:, 1], "-o", label="Si")
plt.plot(
    wavelengths * 1e9,
    result_coh["A_per_interface"][1],
    "-o",
    label=[None, "IZO", "C60", None, "Perovskite", None, None],
)
plt.plot(wavelengths * 1e9, result_coh["A_per_interface"][2], "-o", label=[None, None, "ITO"])

plt.xlabel("Wavelength (nm)")
plt.ylabel("R / A / T")
plt.ylim(0, 1)
plt.xlim(300, 1200)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
