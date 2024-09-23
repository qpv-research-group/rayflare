# NOTE: this example requires RayFlare v2.0.0 or later

import numpy as np

from rayflare.ray_tracing import rt_structure
from rayflare.textures import regular_pyramids, planar_surface
from rayflare.options import default_options

from solcore import material
from solcore import si

# imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# setting up some colours for plotting
pal = sns.color_palette("husl", 4)

# setting up Solcore materials
Air = material("Air")()

Si = material("Si")()

d_Si = 60e-6

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
options.I_thresh = 0.005

# start with s-polarized light:
options.pol = 's'

triangle_surf = regular_pyramids(52, upright=True,
                                 analytical=True)
# use analytical ray-tracing (normal incidence)

flat_surf = planar_surface(phong=False)  # pyramid size in microns

# set up ray-tracing options
rtstr_nophong = rt_structure(
    textures=[triangle_surf, flat_surf],
    materials=[Si],
    widths=[d_Si],
    incidence=Air,
    transmission=Air,
)

start = time()

result_nophong = rtstr_nophong.calculate(options)

print("non phong calc:", time() - start)

# Perfectly Lambertian scattering:

# options: [1, True] means Lambertian scattering (alpha = 1) and randomization of the polarization
# after Phong scattering is applied (True).

triangle_phong_surf = regular_pyramids(52, upright=True, phong=True,
                                       phong_options=[1, True],
                                       analytical=True)

phong_surf = planar_surface(phong=True, phong_options=[1, True])

# set up ray-tracing options
rtstr_phong = rt_structure(
    textures=[triangle_phong_surf, phong_surf],
    materials=[Si],
    widths=[d_Si],
    incidence=Air,
    transmission=Air,
)

start = time()

result_phong = rtstr_phong.calculate(options)

print("phong calc:", time() - start)

# plot the results
plt.figure()
plt.plot(options.wavelength * 1e9, result_nophong["R"], '-k', label="R")
plt.plot(options.wavelength * 1e9, result_phong["R"], '--k')

plt.plot(options.wavelength * 1e9, result_nophong["A_per_layer"], '-r', label="A")
plt.plot(options.wavelength * 1e9, result_phong["A_per_layer"], '--r')

plt.plot(options.wavelength * 1e9, result_nophong["T"], '-b', label="T")
plt.plot(options.wavelength * 1e9, result_phong["T"], '--b')

plt.plot(0, 0, '-k', label="Without Phong")
plt.plot(0, 0, '--k', label="With Phong")

plt.xlabel("Wavelength (nm)")
plt.ylabel("RAT")
plt.ylim(0, 1)
plt.xlim(np.min(options.wavelength * 1e9), np.max(options.wavelength * 1e9))
plt.legend()
plt.show()


# plot the results at 1000 nm; distribution of theta and phi

# find index for wavelength closest to 1000 nm
wl_ind = np.argmin(np.abs(options.wavelength - 1000e-9))

theta_nophong = result_nophong.ray_data.thetas[wl_ind]
theta_phong = result_phong.ray_data.thetas[wl_ind]

# plot distribution of transmitted angles:
plt.figure()
plt.hist(theta_nophong[theta_nophong > np.pi/2], color=pal[0], alpha=0.5, label="No Phong",
         range=(np.pi/2, np.pi), bins=30, density=True)
plt.hist(theta_phong[theta_phong > np.pi/2], color=pal[1], alpha=0.5, label="Phong",
         range=(np.pi/2, np.pi), bins=30, density=True)

plt.xlabel("Theta (rad)")
plt.ylabel("Probability")
plt.legend()
plt.title("Distribution of transmitted polar angles at 1000 nm")
plt.show()

phi_nophong = result_nophong.ray_data.phis[wl_ind]
phi_phong = result_phong.ray_data.phis[wl_ind]

# plot distribution of transmitted angles:
plt.figure()
plt.hist(phi_nophong[theta_nophong > np.pi/2], color=pal[0], alpha=0.5, label="No Phong",
         range=(np.pi/2, np.pi), bins=30, density=True)
plt.hist(phi_phong[theta_phong > np.pi/2], color=pal[1], alpha=0.5, label="Phong",
         range=(np.pi/2, np.pi), bins=30, density=True)

plt.xlabel("Phi (rad)")
plt.ylabel("Probability")
plt.legend()
plt.title("Distribution of transmitted azimuthal angles at 1000 nm")
plt.show()


# s-component of the polarization of final rays at 1000 nm:

plt.figure()
plt.hist(result_nophong.ray_data.pol[wl_ind, theta_nophong > np.pi/2, 0], color=pal[0], alpha=0.5, label="No Phong",
         bins=50, density=True)
plt.hist(result_phong.ray_data.pol[wl_ind, theta_nophong > np.pi/2, 0], alpha=0.5, label="Phong",
         bins=50, density=True)

plt.xlabel("s-component of final transmitted ray")
plt.ylabel("Probability")
plt.legend()
plt.show()

