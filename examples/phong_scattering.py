import numpy as np

from rayflare.ray_tracing.rt import rt_structure
from rayflare.textures import regular_pyramids, planar_surface
from rayflare.options import default_options

from solcore import material
from solcore import si

# imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# setting up some colours for plotting
pal = sns.color_palette("husl", 4)


# setting up Solcore materials
Air = material("Air")()

from solcore.absorption_calculator import download_db, search_db

Si = material("Si")()

# number of x and y points to scan across
nxy = 30

# setting options
options = default_options()
options.wavelength = np.linspace(900, 1180, 20) * 1e-9
options.nx = nxy
options.ny = nxy
options.n_rays = 2 * nxy**2
options.depth_spacing = si("0.1um")
options.parallel = True
options.analytical_ray_tracing = 2
options.I_thresh = 0.005

triangle_surf = regular_pyramids(52, upright=True)
flat_surf = planar_surface(phong=False)  # pyramid size in microns

# set up ray-tracing options
rtstr_planar = rt_structure(
    textures=[triangle_surf, flat_surf],
    materials=[Si],
    widths=[si("100um")],
    incidence=Air,
    transmission=Air,
)

result_planar = rtstr_planar.calculate(options)

triangle_phong_surf = regular_pyramids(52, upright=True, phong=True)

phong_surf = planar_surface(phong=True)  # pyramid size in microns

# set up ray-tracing options
rtstr_phong = rt_structure(
    textures=[triangle_phong_surf, phong_surf],
    materials=[Si],
    widths=[si("100um")],
    incidence=Air,
    transmission=Air,
)

result_phong = rtstr_phong.calculate(options)

# plot the results
plt.figure()
plt.plot(options.wavelength * 1e9, result_planar["R"], '-k',  label="R")
plt.plot(options.wavelength * 1e9, result_phong["R"], '--k')

plt.plot(options.wavelength * 1e9, result_planar["A_per_layer"], '-r', label="A")
plt.plot(options.wavelength * 1e9, result_phong["A_per_layer"], '--r')

plt.plot(options.wavelength * 1e9, result_planar["T"], '-b', label="T")
plt.plot(options.wavelength * 1e9, result_phong["T"], '--b')
plt.ylim(0, 1)
plt.legend()
plt.show()


# plot the results at 1100 nm; distribution of theta and phi
wl_ind = np.argmin(np.abs(options.wavelength - 1100e-9))

plt.figure()
plt.hist(result_planar["thetas"][wl_ind], color=pal[0], alpha=0.5, label="Planar",
         bins=50, density=True)
plt.hist(result_phong["thetas"][wl_ind], color=pal[1], alpha=0.5, label="Phong",
         bins=50, density=True)

plt.xlabel("Theta (rad)")
plt.ylabel("Number of rays")
plt.legend()
plt.show()