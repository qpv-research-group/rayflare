import matplotlib.pyplot as plt
import numpy as np
from rayflare.textures import xyz_texture, planar_surface
import seaborn as sns
from solcore import material
from rayflare.ray_tracing import rt_structure
from rayflare.transfer_matrix_method import tmm_structure
from rayflare.options import default_options
from rayflare.textures.standard_rt_textures import hyperhemisphere
from solcore.material_system import create_new_material
from solcore.structure import Layer
import os
import sys

# NOTE: variables exp_points, nxs and number of theta points changed to make calculation faster for the example. To get
# results as generated in the paper, change to values stated in comments.
n_thetas = 100  # 100 used for paper data
exp_points = (
    17  # 2**exp_points on surface of WHOLE sphere. exp_points = 15 used for paper data
)
nxs = 20  # 70 points used for paper data
thetas = np.linspace(0, np.pi / 2 - 0.05, n_thetas)  # 100 angles used for paper data

cols = sns.color_palette('husl', 4)
options = default_options()

options.project_name = 'lenses'
options.x_limits = [0, 0]  # area of the diode
options.y_limits = [0, 0]

options.initial_material = 1  # the rays start in the GaAs (material index 1) rather than in the air above the cell (material index 0)
options.initial_direction = 1  # default initial direction, which is 1 (downwards)


options.periodic = 0


pal = sns.color_palette("rocket", 4)


options.wavelengths = np.array([6e-6])
options.parallel = False
options.n_rays = nxs**2
options.theta_in = 0

GaAs = material("GaAs")()
Air = material("Air")()
Ag = material('Air')()



wl_list = np.array([6e-6, 7e-6])

ideal_n_arc = [1e9*options.wavelengths[0], wl_list, np.sqrt(GaAs.n(wl_list)), np.array([0, 0])]

np.savetxt('ideal_n_arc_n.txt', np.array([wl_list, np.sqrt(GaAs.n(wl_list))]).T)
np.savetxt('ideal_n_arc_k.txt', np.array([wl_list, np.sqrt(GaAs.k(wl_list))]).T)

# create_new_material('GaAs_IR_idealARC', 'ideal_n_arc_n.txt', 'ideal_n_arc_k.txt')
ARC_mat = material("GaAs_IR_idealARC")()


tmm_str = tmm_structure([Layer(options.wavelengths[0]/(4*ARC_mat.n(options.wavelengths)[0]), ARC_mat)], incidence=Air, transmission=GaAs)

RAT = tmm_str.calculate(options)

print(RAT["R"])


d_bulk = 0

r = 0.8  # radius of hyperhemisphere
h = 0.242  # height shift of hyperhemisphere

labels = ['Hemispherical', 'Hyperhemispherical', 'Slab']

plt.figure()

for i1, h in enumerate([0, 0.242]):

    if labels[i1] == 'Slab':

        flat_surf = planar_surface()
        d_bulk = 1e-6

        hyperhemi = planar_surface(interface_layers=[Layer(options.wavelengths[0] / (4 * ARC_mat.n(options.wavelengths)[0]), ARC_mat)])

    else:
        d_bulk = 0
        [front, back] = hyperhemisphere(2 ** exp_points, r, h, interface_layers=[
            Layer(options.wavelengths[0] / (4 * ARC_mat.n(options.wavelengths)[0]), ARC_mat)])

        hyperhemi = [back, front]

        edge_points = back.Points[back.Points[:, 2] == 0]

        edge_points = np.vstack([edge_points, [0, 0, 0]])  # add point at centre

        flat_surf = xyz_texture(
            edge_points[:, 0], edge_points[:, 1], edge_points[:, 2], coverage_height=0
        )

    rtstr = rt_structure(
        textures=[flat_surf, hyperhemi],
        materials=[GaAs],
        widths=[d_bulk],
        incidence=Air,
        transmission=Air,
        use_TMM=True,
        options=options,
    )

    # now want to make closed, flat top surface: find z == 0 points of hyperhemisphere surface

    options.theta = 0
    options.nx = nxs
    options.ny = nxs
    options.pol = "u"

    T_values = np.zeros(len(thetas))
    T_total = np.zeros(len(thetas))
    n_interactions = np.zeros(len(thetas))
    theta_distribution = np.zeros((len(thetas), options.n_rays))

    if os.path.isfile(
        "sphere_raytrace_totalT_2e"
        + str(exp_points)
        + "_"
        + str(nxs)
        + "_points_"
        + str(options.n_rays)
        + "_rays_2.txt"
    ):
        T_total = np.loadtxt(
            "sphere_raytrace_totalT_2e"
            + str(exp_points)
            + "_"
            + str(nxs)
            + "_points_"
            + str(options.n_rays)
            + "_rays.txt"
        )
        n_interactions = np.loadtxt(
            "sphere_raytrace_ninter_2e"
            + str(exp_points)
            + "_"
            + str(nxs)
            + "_points_"
            + str(options.n_rays)
            + "_rays.txt"
        )
        theta_distribution = np.loadtxt(
            "sphere_raytrace_thetas_2e"
            + str(exp_points)
            + "_"
            + str(nxs)
            + "_points_"
            + str(options.n_rays)
            + "_rays.txt"
        )

    else:
        # print('start')
        # filename = 'hyperhemisphere_tracking.txt'
        # sys.stdout = open(filename, 'w')

        for j1, th in enumerate(thetas):
            # print(j1, th)

            options.theta_in = th
            result = rtstr.calculate(options)
            T_total[j1] = result["T"]
            n_interactions[j1] = np.mean(result["n_interactions"])
            theta_distribution[j1] = result["thetas"]

        np.savetxt(
            "sphere_raytrace_totalT_2e"
            + str(exp_points)
            + "_"
            + str(nxs)
            + "_points_"
            + str(options.n_rays)
            + "_rays.txt",
            T_total,
        )
        np.savetxt(
            "sphere_raytrace_ninter_2e"
            + str(exp_points)
            + "_"
            + str(nxs)
            + "_points_"
            + str(options.n_rays)
            + "_rays.txt",
            n_interactions,
        )
        np.savetxt(
            "sphere_raytrace_thetas_2e"
            + str(exp_points)
            + "_"
            + str(nxs)
            + "_points_"
            + str(options.n_rays)
            + "_rays.txt",
            theta_distribution,
        )

    # sys.stdout.close()

    min_angle_1 = np.pi - 17.5 * np.pi / 180
    min_angle_2 = np.pi - np.pi * 45 / 180

    T_175 = np.array([np.sum(x > min_angle_1) / options.n_rays for x in theta_distribution])
    T_45 = np.array([np.sum(x > min_angle_2) / options.n_rays for x in theta_distribution])


    plt.plot(thetas, T_total, color=cols[i1], label=labels[i1])
    plt.scatter(thetas, T_total, edgecolors=cols[i1], facecolors="none")
    # plt.plot(thetas * 180 / np.pi, T_45, color=pal[1])
    # plt.scatter(thetas * 180 / np.pi, T_45, edgecolors=pal[1], facecolors="none")
    # plt.plot(thetas * 180 / np.pi, T_175, color=pal[2])
    # plt.scatter(thetas * 180 / np.pi, T_175, edgecolors=pal[2], facecolors="none")
    # plt.xlim(0, 90)
    # plt.axvline(np.arcsin(1/GaAs.n(options.wavelengths)))
    plt.xlabel(r"$\beta_{emission}$ (rads)")
    plt.ylabel("Transmission out of lens")

plt.legend()
plt.show()


plt.figure()
plt.plot(thetas, theta_distribution, 'o')
plt.show()


# process data
#
# th_index = 0
# back_points = [[]]*len(thetas)
#
# cols = sns.cubehelix_palette(len(thetas), start=0.5, rot=-0.9)
#
# plt.figure()
#
# with open(filename) as file:
#     for line in file:
#
#         if line.startswith('1wl'):
#             # new angle
#             ray_index = 0
#             th_index += 1
#
#         elif line.startswith('# ray'):
#             n_interaction = 0
#             # new ray at same angle
#
#         else:
#             if n_interaction > 0:
#                 # print(th_index)
#                 # back_points[th_index-1].append(np.fromstring(line, sep=' '))
#                 # print(line)
#                 loc = np.fromstring(line, sep=' ')
#                 if np.abs(loc[2]) < 1e-9:
#                     plt.plot(loc[0], loc[1], 'o', markerfacecolor='none',
#                     markeredgecolor=cols[th_index-1], markersize=3,
#                              # label=str(thetas[th_index-1])
#                              )
#
#             n_interaction += 1
#
# # plt.legend()
# plt.show()
#
#
# for i1, bp in enumerate(back_points):
#
#     bp_arr = np.array(bp)
#     bp_arr = bp_arr[np.abs(bp_arr[:, 2]) < 1e-9]
#
#     plt.plot(bp_arr[:,0], bp_arr[:,1], 'o', markerfacecolor='none')#),
#              #markeredgecolor=cols[i1], markersize=3)
#
# plt.show()
#
#     # which ones strike rear surface?
#
#
#
