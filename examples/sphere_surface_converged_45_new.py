import matplotlib.pyplot as plt
import numpy as np
from rayflare.textures import xyz_texture
from time import time
import seaborn as sns
from solcore import material
from rayflare.ray_tracing import rt_structure
from rayflare.options import default_options
from rayflare.textures.standard_rt_textures import hyperhemisphere
import os

d_bulk = 0

r = 0.8
h = 0.242
exp_points = 15


r_cross = np.sqrt(0.8**2 - 0.242**2)

n_points = 10001

Z_new = np.zeros(n_points)
phis = np.linspace(0, 2*np.pi, n_points)
X_new = r_cross*np.cos(phis)
Y_new = r_cross*np.sin(phis)

GaAs = material('GaAs')()
Air = material('Air')()

flat_surf_2 = xyz_texture(np.hstack([0,X_new]), np.hstack([0,Y_new]), np.hstack([0,Z_new]))

[front, back] = hyperhemisphere(2**exp_points, r, h)

hyperhemi = [back, front]

# X_norm = np.mean([front.P_0s[:, 0], front.P_1s[:, 0], front.P_2s[:, 0]], 0)
# Y_norm = np.mean([front.P_0s[:, 1], front.P_1s[:, 1], front.P_2s[:, 1]], 0)
# Z_norm = np.mean([front.P_0s[:, 2], front.P_1s[:, 2], front.P_2s[:, 2]], 0)
#
# fig = plt.figure()
#
#
# ax = plt.subplot(111, projection='3d')
# ax.view_init(elev=30., azim=60)
# ax.plot_trisurf(front.Points[:,0], front.Points[:,1], front.Points[:,2],
#                 triangles=front.simplices,  shade=False, cmap=plt.get_cmap('Blues'),# array=colors,
#         edgecolors='none')
# # ax.plot_trisurf(flat_surf_2[0].Points[:,0], flat_surf_2[0].Points[:,1], flat_surf_2[0].Points[:,2],
# #                 triangles=flat_surf_2[0].simplices)
#
# #
# ax.quiver(X_norm, Y_norm, Z_norm, front.crossP[:,0], front.crossP[:,1], front.crossP[:,2], length=0.1, normalize=True,
#           color='k')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

X_norm = np.mean([back.P_0s[:, 0], back.P_1s[:, 0], back.P_2s[:, 0]], 0)
Y_norm = np.mean([back.P_0s[:, 1], back.P_1s[:, 1], back.P_2s[:, 1]], 0)
Z_norm = np.mean([back.P_0s[:, 2], back.P_1s[:, 2], back.P_2s[:, 2]], 0)

fig = plt.figure()


ax = plt.subplot(111, projection='3d')
ax.view_init(elev=30., azim=60)
ax.plot_trisurf(back.Points[:,0], back.Points[:,1], back.Points[:,2],
                triangles=back.simplices,  shade=False, cmap=plt.get_cmap('Blues'),# array=colors,
        edgecolors='none')
# ax.plot_trisurf(flat_surf_2[0].Points[:,0], flat_surf_2[0].Points[:,1], flat_surf_2[0].Points[:,2],
#                 triangles=flat_surf_2[0].simplices)


ax.quiver(X_norm, Y_norm, Z_norm, back.crossP[:,0], back.crossP[:,1], back.crossP[:,2], length=0.1, normalize=True,
          color='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

rtstr = rt_structure(textures=[flat_surf_2, hyperhemi],
                    materials = [GaAs],
                    widths=[d_bulk], incidence=Air, transmission=Air)

options = default_options()

options.x_limits = [-0.05, 0.05]
options.y_limits = [-0.05, 0.05]

options.initial_material = 1
options.initial_direction = 1

options.periodic = 0

nxs = [70]

thetas = np.linspace(0, np.pi / 2 - 0.05, 100)

thetas_1 = thetas[0:25]
thetas_2 = thetas[25:50]
thetas_3 = thetas[50:75]
thetas_4 = thetas[75:100]

thetas = thetas_4

thetas_min = np.int(10*np.round(180*np.min(thetas)/np.pi, 1))
thetas_max = np.int(10*np.round(180*np.max(thetas)/np.pi, 1))

pal = sns.color_palette("rocket", len(nxs))

plt.figure()

for i1, nx in enumerate(nxs):

    options.wavelengths = np.array([6e-6])
    options.parallel = False
    options.n_rays = nx**2

    options.theta = 0.1
    options.nx = nx
    options.ny = nx
    options.pol = 'u'

    print(options.n_rays)

    minimum_angle = np.pi - np.pi*45/180

    T_values = np.zeros(len(thetas))
    T_total = np.zeros(len(thetas))
    n_interactions = np.zeros(len(thetas))
    theta_distribution = np.zeros((len(thetas), options.n_rays))

    if os.path.isfile('results/sphere_raytrace_2e12_' + str(options.n_rays) + 'rays_' + str(thetas_min) + str(thetas_max) + '45_2.txt'):

        T_values = np.loadtxt('results/sphere_raytrace_2e12_' + str(options.n_rays) + 'rays_' + str(thetas_min) + str(thetas_max) + '45_2.txt')
        T_total = np.loadtxt('results/sphere_raytrace_totalT_2e12_' + str(options.n_rays) + 'rays_' + str(thetas_min) + str(thetas_max) + '45_2.txt')
        n_interactions = np.loadtxt('results/sphere_raytrace_ninter_2e12_' + str(options.n_rays) + 'rays_' + str(thetas_min) + str(thetas_max) + '45_2.txt')
        theta_distribution = np.loadtxt('results/sphere_raytrace_thetas_2e12_' + str(options.n_rays) + 'rays_' + str(thetas_min) + str(thetas_max) + '45_2.txt')

    else:

        start = time()

        for j1, th in enumerate(thetas):
            print(j1, th)

            options.theta_in = th
            result = rtstr.calculate(options)
            T_values[j1] = np.sum(result['thetas'] > minimum_angle)/options.n_rays
            T_total[j1] = result['T']
            n_interactions[j1] = np.mean(result['n_interactions'])
            theta_distribution[j1] = result['thetas']

        print(time() - start)


        np.savetxt('results/sphere_raytrace_2e12_' + str(options.n_rays) + 'rays_' + str(thetas_min) + str(thetas_max) + '45_2.txt', T_values)
        np.savetxt('results/sphere_raytrace_totalT_2e12_' + str(options.n_rays) + 'rays_' + str(thetas_min) + str(thetas_max) + '45_2.txt', T_total)
        np.savetxt('results/sphere_raytrace_ninter_2e12_' + str(options.n_rays) + 'rays_' + str(thetas_min) + str(thetas_max) + '45_2.txt', n_interactions)
        np.savetxt('results/sphere_raytrace_thetas_2e12_' + str(options.n_rays) + 'rays_' + str(thetas_min) + str(thetas_max) + '45_2.txt', theta_distribution)

    min_angle_old = np.pi - 17.5*np.pi/180

    T_175 = np.array([np.sum(x > min_angle_old) / options.n_rays for x in theta_distribution])
    T_45 = np.array([np.sum(x > minimum_angle) / options.n_rays for x in theta_distribution])

    plt.plot(thetas*180/np.pi, T_values, '--', label=str(options.n_rays), color=pal[i1])
    plt.scatter(thetas*180/np.pi, T_values, edgecolors=pal[i1], facecolors='none')
    plt.plot(thetas*180/np.pi, T_total, color=pal[i1])
    plt.scatter(thetas*180/np.pi, T_total, edgecolors=pal[i1], facecolors='none')
    plt.plot(thetas*180/np.pi, T_175, color='grey')
    plt.scatter(thetas*180/np.pi, T_175, edgecolors='grey', facecolors='none')
    plt.legend(title="Number of rays")
    plt.xlim(0, 90)
    plt.xlabel(r'$\beta$ (rads)')
    plt.ylabel('Transmission')

#
# for ln in [17.5, 2*17.5, 3*17.5, 4*17.5]:
#     plt.axvline(x=ln)
plt.title(r'Convergence with number of rays ($N_{triangles} = 2^{12}$)')

plt.show()

    #
    # plt.figure()
    # plt.plot(thetas, n_interactions)
    # plt.show()