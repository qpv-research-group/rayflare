from rayflare.textures import rough_pyramids, regular_pyramids
import numpy as np

from solcore.structure import Layer
from solcore import material

from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure
import matplotlib.pyplot as plt
import seaborn as sns

from time import time

from cycler import cycler

elevation_angle = 55
size = 1
noise_height = 0.1

wavelengths = np.linspace(300, 1200, 40) * 1e-9

options = default_options()
options.wavelength = wavelengths
options.n_theta_bins = 100
options.nx = 30
options.ny = options.nx
options.n_rays = 5 * options.nx**2
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
front_materials = [Layer(100e-9, MgF2), Layer(110e-9, IZO), Layer(15e-9, C60), Layer(1e-9, LiF)]
# Layer(440e-9, Perovskite),

Si_materials = [Layer(6.5e-9, aSi_n), Layer(6.5e-9, aSi_i)]

back_materials = [Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_p), Layer(240e-9, ITO_back)]

options.project_name = "perovskite_Si_RT_rough"


triangle_surf_smooth = regular_pyramids(
    elevation_angle=55, upright=True, size=1, interface_layers=Si_materials, name="Si_front"
)

triangle_surf_back = regular_pyramids(
    elevation_angle=55, upright=False, size=1, interface_layers=back_materials, name="Si_back"
)
pal = sns.color_palette("husl", n_colors=len(front_materials) + len(back_materials) + 3)

cols = cycler("color", pal)

params = {"axes.prop_cycle": cols}

plt.rcParams.update(params)

noise_heights = np.linspace(0, 0.3, 5)

results = []

for j1, nh in enumerate(noise_heights):
    start = time()

    triangle_surf = rough_pyramids(
        elevation_angle=55,
        upright=True,
        size=1,
        noise_fraction=nh,
        interface_layers=front_materials,
        name="coh_front_{}".format(noise_height),
    )

    # fig = plt.figure()
    # ax = plt.subplot(projection="3d")
    # ax.view_init(elev=20.0, azim=60)
    # ax.plot_trisurf(
    #     triangle_surf[0].Points[:, 0],
    #     triangle_surf[0].Points[:, 1],
    #     triangle_surf[0].Points[:, 2],
    #     triangles=triangle_surf[0].simplices,
    #     linewidth=1,
    #     color=(0.8, 0.8, 0.8, 0.8),
    # )
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    #
    # plt.show()
    #

    rtstr_coh = rt_structure(
        textures=[triangle_surf, triangle_surf_smooth, triangle_surf_back],
        materials=[Perovskite, Si],
        widths=[440e-9, 260e-6],
        incidence=Air,
        transmission=Ag,
        use_TMM=True,
        options=options,
        save_location="current",
    )

    result_coh = rtstr_coh.calculate(options)

    results.append(result_coh)

    print("TIME:", time() - start)

    fig = plt.figure(figsize=(9, 3.7))
    plt.subplot(1, 1, 1)
    plt.plot(wavelengths * 1e9, result_coh["R"], "-ko", label="R")
    plt.plot(wavelengths * 1e9, result_coh["T"], mfc="none", label="T")
    plt.plot(wavelengths * 1e9, result_coh["A_per_layer"][:, 0], "--o", label="Pero")
    plt.plot(wavelengths * 1e9, result_coh["A_per_layer"][:, 1], "--o", label="Si")
    plt.plot(wavelengths * 1e9, result_coh["A_per_interface"][0], "--o")
    plt.plot(wavelengths * 1e9, result_coh["A_per_interface"][1], "--o")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("R / A / T")
    plt.ylim(0, 1)
    plt.xlim(300, 1200)
    plt.title(str(nh))
    plt.legend()
    plt.show()


fig = plt.figure()
alpha = np.linspace(1, 0.3, len(results))

for i1, res in enumerate(results):
    plt.plot(wavelengths * 1e9, res["R"], alpha=alpha[i1], color=pal[0])
    plt.plot(wavelengths * 1e9, res["T"], alpha=alpha[i1], color=pal[1])
    plt.plot(wavelengths * 1e9, res["A_per_layer"][:, 0], alpha=alpha[i1], color=pal[2])
    plt.plot(wavelengths * 1e9, res["A_per_layer"][:, 1], alpha=alpha[i1], color=pal[3])
    plt.plot(wavelengths * 1e9, np.sum(res["A_per_interface"][0], axis=1), alpha=alpha[i1], color=pal[4])
    plt.plot(wavelengths * 1e9, np.sum(res["A_per_interface"][2], axis=1), alpha=alpha[i1], color=pal[5])

plt.legend(["R", r"T$_{Ag}$", "Pero", "Si", "front interface", "back interface"], loc=(0.55, 0.35))
plt.xlim(300, 1200)
plt.ylim(0, 1)

plt.show()
