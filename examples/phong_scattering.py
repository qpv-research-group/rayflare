import numpy as np

from rayflare.ray_tracing.rt import rt_structure
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
options.analytical_ray_tracing = 0
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

start = time()

result_planar = rtstr_planar.calculate(options)

print("non phong calc:", time() - start)

triangle_phong_surf = regular_pyramids(52, upright=True, phong=True,
                                       phong_options=[1, True])

phong_surf = planar_surface(phong=True, phong_options=[1, True])  # pyramid size in microns

# set up ray-tracing options
rtstr_phong = rt_structure(
    textures=[triangle_phong_surf, phong_surf],
    materials=[Si],
    widths=[si("100um")],
    incidence=Air,
    transmission=Air,
)

start = time()

result_phong = rtstr_phong.calculate(options)

print("phong calc:", time() - start)
options.analytical_ray_tracing = 0

result_phong_full = rtstr_phong.calculate(options)

# plot the results
plt.figure()
plt.plot(options.wavelength * 1e9, result_planar["R"], '-k',  label="R")
plt.plot(options.wavelength * 1e9, result_phong["R"], '--k')
plt.plot(options.wavelength*1e9, result_phong_full["R"], '-.k')

plt.plot(options.wavelength * 1e9, result_planar["A_per_layer"], '-r', label="A")
plt.plot(options.wavelength * 1e9, result_phong["A_per_layer"], '--r')
plt.plot(options.wavelength*1e9, result_phong_full["A_per_layer"], '-.r')

plt.plot(options.wavelength * 1e9, result_planar["T"], '-b', label="T")
plt.plot(options.wavelength * 1e9, result_phong["T"], '--b')
plt.plot(options.wavelength*1e9, result_phong_full["T"], '-.b')

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

# planar phong scattering surface with 100 % transmission:

options.wavelength= np.array([300e-9])
options.analytical_ray_tracing = 0

alpha = 20 # alpha = 1 is Lambertian

flat_surf = planar_surface(phong=True, phong_options=[alpha, False])
rtstr_planar = rt_structure(
    textures=[flat_surf],
    materials=[],
    widths=[],
    incidence=Air,
    transmission=Air,
)

options.n_rays = 5e4
rat_scat = rtstr_planar.calculate(options)

x = np.linspace(0, np.pi/2)
cos_x = np.cos(x)
power_law = cos_x**(alpha)

thetas = np.pi-rat_scat["thetas"][0]

fig, ax1 = plt.subplots()
n, bins, _ = ax1.hist(np.pi-rat_scat["thetas"][0], color=pal[2], alpha=0.5, label="Phong 100 % transmission",
         bins=70, density=True)
plt.plot(x, power_law)
plt.show()

mean_theta_bin = np.mean([bins[0:-1], bins[1:]], 0)
scaled_intensity = n/np.sin(mean_theta_bin)

fig, ax1 = plt.subplots()
plt.plot(mean_theta_bin, scaled_intensity/np.max(scaled_intensity))
plt.plot(x, power_law)
plt.show()

fig, ax1 = plt.subplots()
n, _, _ = ax1.hist(thetas, color=pal[2], alpha=0.5, label="Phong 100 % transmission",
         bins=70, density=True)
plt.plot((x+1), power_law)
plt.show()

x2 = np.linspace(0, 1, 100)

plt.figure()
plt.plot(x2, x2**(1/(1+alpha)))
plt.show()

# find height of top bin:
n, _, _ = plt.hist((np.cos(np.pi-rat_scat["thetas"][0])), color=pal[2], alpha=0.5,
         label="Phong 100 % transmission", density=True,
         bins=70)

plt.plot(cos_x, np.max(n)*power_law)
plt.show()

# divide hemisphere into equally spaced theta bins:
n_bins = 100
theta_bins = np.linspace(0, np.pi/2, n_bins)
mean_thetas = np.mean([theta_bins[0:-1], theta_bins[1:]], 0)

apparent_brightness = np.zeros(n_bins-1)

for bin_i in range(n_bins-1):
    theta_min = theta_bins[bin_i]
    theta_max = theta_bins[bin_i+1]

    # what is the area of a strip along all phi between theta and delta theta?
    area_strip = np.sin(mean_thetas[bin_i])

    # find rays in this bin
    in_bin = (np.pi - rat_scat["thetas"][0] > theta_min) & (np.pi - rat_scat["thetas"][0] < theta_max)

    # find number of rays in this bin
    n_rays_in_bin = np.sum(in_bin)

    # what is the solid angle subtended:
    d_omega = np.cos(mean_thetas[bin_i])

    apparent_brightness[bin_i] = d_omega*n_rays_in_bin
    # find number of rays in this bin that are reflected

plt.plot(np.mean([theta_bins[0:-1], theta_bins[1:]], 0)*180/np.pi,
         apparent_brightness)
plt.show()


# generate 1 million random numbers:
n_points = 1e6

uniform_rand = np.random.rand(int(n_points))

theta_dist = np.arccos((uniform_rand)**(1/(1+alpha)))

plt.figure()
plt.plot(x, np.cos(x)**alpha)
plt.show()