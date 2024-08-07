import matplotlib.pyplot as plt
import numpy as np
from rayflare.textures import xyz_texture
import seaborn as sns
from solcore import material
from rayflare.ray_tracing import rt_structure
from rayflare.options import default_options
from rayflare.textures.standard_rt_textures import hyperhemisphere
import os
from joblib import Parallel, delayed
from time import time

# paper: https://doi.org/10.1021/acsphotonics.2c00223

# NOTE: variables exp_points, nxs and number of theta points changed to make calculation faster for
# the example. To get
# results as generated in the paper, change to values stated in comments.

d_bulk = 0

r = 0.8  # radius of hyperhemisphere
h = 0.242  # height shift of hyperhemisphere

# 50 thetas, exp_points = 13 and nxs = 30 takes 220 seconds on my MacBook Pro
# (32 GB RAM, M1 chip, 10 cores)

n_thetas = 50  # 100 used for paper data
exp_points = 13  # 2**exp_points on surface of WHOLE sphere. exp_points = 15 used for paper data
nxs = 70  # 70 points used for paper data
thetas = np.linspace(0, np.pi / 2 - 0.05, n_thetas)  # 100 angles used for paper data

# structure:

# Air above lens
# ----------- planar interface between air and GaAs
# |         |
# |        |
#  \_____/     hyperhemisphere pointing down
# Air below lens.

GaAs = material("GaAs")()
Air = material("Air")()

pal = sns.color_palette("rocket", 4)

options = default_options()

options.x_limits = [-0.05, 0.05]  # area of the diode
options.y_limits = [-0.05, 0.05]

options.initial_material = (
    1
    # the rays start in the GaAs (material index 1) rather than in the air above the cell (material index 0)
)
options.initial_direction = 1  # default initial direction, which is 1 (downwards)

options.periodic = 0
options.wavelength = np.array([6e-6])
options.parallel = False
options.n_rays = nxs ** 2

options.theta = 0.1
options.nx = nxs
options.ny = nxs
options.pol = "u"

[front, back] = hyperhemisphere(2 ** exp_points, r, h)

hyperhemi = [back, front]

# now want to make closed, flat top surface: find z == 0 points of hyperhemisphere surface

edge_points = back.Points[back.Points[:, 2] == 0]

edge_points = np.vstack([edge_points, [0, 0, 0]])  # add point at centre

flat_surf = xyz_texture(edge_points[:, 0], edge_points[:, 1], edge_points[:, 2],
                        coverage_height=0)

# plot the hyperhemisphere: 'front' and 'back'. In this case we are actually going to flip it and use the 'back' interface.

fig = plt.figure()

ax = plt.subplot(111, projection="3d")
ax.view_init(elev=30.0, azim=60)
ax.plot_trisurf(
    back.Points[:, 0],
    back.Points[:, 1],
    back.Points[:, 2],
    triangles=back.simplices,
    shade=False,
    cmap=plt.get_cmap("Blues"),
    edgecolors="none",
)
ax.plot_trisurf(
    flat_surf[0].Points[:, 0],
    flat_surf[0].Points[:, 1],
    flat_surf[0].Points[:, 2],
    triangles=flat_surf[0].simplices,
    shade=False,
    cmap=plt.get_cmap("Blues"),
    edgecolors="none",
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()


GaAs = material("GaAs")()
Air = material("Air")()

rtstr = rt_structure(
    textures=[flat_surf, hyperhemi], materials=[GaAs], widths=[0], incidence=Air,
    transmission=Air
)

def parallel_theta(options, th):
    options.theta_in = th
    result = rtstr.calculate(options)

    return result['T'][0], np.mean(result['n_interactions']), result['thetas'][0]

if os.path.isfile(
    "sphere_raytrace_totalT_2e" + str(exp_points) + "_" + str(nxs) + "_points_" + str(options.n_rays) + "_rays_2.txt"
):
    T_total = np.loadtxt(
        "sphere_raytrace_totalT_2e" + str(exp_points) + "_" + str(nxs) + "_points_" + str(options.n_rays) + "_rays.txt"
    )
    n_interactions = np.loadtxt(
        "sphere_raytrace_ninter_2e" + str(exp_points) + "_" + str(nxs) + "_points_" + str(options.n_rays) + "_rays.txt"
    )
    theta_distribution = np.loadtxt(
        "sphere_raytrace_thetas_2e" + str(exp_points) + "_" + str(nxs) + "_points_" + str(options.n_rays) + "_rays.txt"
    )

else:

    # run in parallel:
    start = time()
    all_results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(parallel_theta)(options, th) for th in thetas)
    print("Time taken: ", time() - start)

    T_total = np.array([x[0] for x in all_results])
    n_interactions = np.array([x[1] for x in all_results])
    theta_distribution = np.array([x[2] for x in all_results])

    np.savetxt(
        "sphere_raytrace_totalT_2e" + str(exp_points) + "_" + str(nxs) + "_points_" + str(options.n_rays) + "_rays.txt",
        T_total,
    )
    np.savetxt(
        "sphere_raytrace_ninter_2e" + str(exp_points) + "_" + str(nxs) + "_points_" + str(options.n_rays) + "_rays.txt",
        n_interactions,
    )
    np.savetxt(
        "sphere_raytrace_thetas_2e" + str(exp_points) + "_" + str(nxs) + "_points_" + str(options.n_rays) + "_rays.txt",
        theta_distribution,
    )

min_angle_1 = np.pi - 17.5 * np.pi / 180
min_angle_2 = np.pi - np.pi * 45 / 180

T_175 = np.array([np.sum(x > min_angle_1) / options.n_rays for x in theta_distribution])
T_45 = np.array([np.sum(x > min_angle_2) / options.n_rays for x in theta_distribution])

plt.figure()

plt.plot(thetas * 180 / np.pi, T_total, color=pal[0])
plt.scatter(thetas * 180 / np.pi, T_total, edgecolors=pal[0], facecolors="none")
plt.plot(thetas * 180 / np.pi, T_45, color=pal[1])
plt.scatter(thetas * 180 / np.pi, T_45, edgecolors=pal[1], facecolors="none")
plt.plot(thetas * 180 / np.pi, T_175, color=pal[2])
plt.scatter(thetas * 180 / np.pi, T_175, edgecolors=pal[2], facecolors="none")
plt.xlim(0, 90)
plt.xlabel(r"$\beta$ (rads)")
plt.ylabel("Transmission")

plt.show()