# test height_distribution

import numpy as np

from rayflare.ray_tracing import rt_structure
from rayflare.transfer_matrix_method import tmm_structure
from rayflare.textures import regular_pyramids, planar_surface
from rayflare.options import default_options
from solcore.structure import Layer

from solcore import material
from solcore import si

# imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

Si = material("Si")()
Air = material("Air")()

# number of x and y points to scan across
nxy = 25

calc = True

# setting options
options = default_options()
options.wavelength = np.linspace(300, 1100, 7) * 1e-9
options.nx = nxy
options.ny = nxy
options.n_rays = 4 * nxy**2
options.depth_spacing = si("1um")
options.parallel = True
options.n_jobs = -1
options.coherent = False
options.coherency_list = ['i']

planar_ref = tmm_structure(
    layer_stack = [Layer(width=si("300um"), material=Si)],
    incidence = Air,
    transmission = Air
)

planar_result = planar_ref.calculate(options)

flat_surf = planar_surface(size=2)  # pyramid size in microns
triangle_surf = regular_pyramids(52, upright=True, size=1)

# for pyramids with a base of 1 micron, height h is:
# tan(angle) = h/0.5
h = 0.5*np.tan(np.deg2rad(52))
# h = 0.64 for an opening angle of 52 degrees (triangle_surf defined above).

# set up ray-tracing options
rtstr = rt_structure(
    textures=[triangle_surf, flat_surf], materials=[Si], widths=[si("300um")], incidence=Air, transmission=Air
)
result= rtstr.calculate(options)


# Define some more surfaces with steeper & less steep pyramids. Generate a simple
# distribution around some height

mean_h_list = [0.2, 0.4, 0.6, 0.8]
prob = np.array([0.2]*5)

pal = sns.color_palette("husl", len(mean_h_list)+1)

# figure with 3 subplots, R, R0 and A in Si:

fig, ax = plt.subplots(1, 3, figsize=(10, 3))

# R = total reflectance including escaping light,
# R0 = initial reflectance (interaction with first interface) only
ax[0].plot(options.wavelength * 1e9, result["R"], "--k", label='h=0.64um (fixed)')
ax[1].plot(options.wavelength * 1e9, result["R0"], "--k")
ax[2].plot(options.wavelength * 1e9, result["A_per_layer"][:, 0], '--k')

for i1, mean_h in enumerate(mean_h_list):

    h_dist = dict(h=np.linspace(mean_h-0.2, mean_h+0.2, 5), p=prob)
    triangle_surf_hd = regular_pyramids(52, upright=True, size=1,
                                        height_distribution=h_dist, analytical=False)
    result_hd = rt_structure(
        textures=[triangle_surf_hd, flat_surf], materials=[Si], widths=[si("300um")], incidence=Air, transmission=Air
    ).calculate(options)

    ax[0].plot(options.wavelength*1e9, result_hd["R"], color=pal[i1], label=f'h_mean={mean_h:.2f} um')
    ax[1].plot(options.wavelength*1e9, result_hd["R0"], color=pal[i1])
    ax[2].plot(options.wavelength*1e9, result_hd["A_per_layer"][:,0], color=pal[i1])

ax[0].set_title("R")
ax[1].set_title("R0")
ax[2].set_title("A in Si")

ax[0].plot(options.wavelength*1e9, planar_result["R"], "--", color=pal[i1+1], label="Planar")
ax[2].plot(options.wavelength*1e9, planar_result["A_per_layer"][:,0], "--", color=pal[i1+1])

for a in ax:
    a.set_xlabel("Wavelength (nm)")
    a.set_ylim(0, 1)
    a.set_xlim(300, 1100)

ax[0].legend()
plt.tight_layout()
plt.show()