from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from scipy.spatial import ConvexHull
from rayflare.textures import xyz_texture
from time import time
import os
import seaborn as sns
from solcore import material
from rayflare.ray_tracing import rt_structure
from rayflare.options import default_options
from rayflare.textures import planar_surface
from copy import deepcopy

d_bulk = 1e-10
# going to get converted to um (multiplied by 1e6)

def points_on_sphere(N, radius, h):
    """ Generate N evenly distributed points on the unit sphere centered at
    the origin. Uses the 'Golden Spiral'.
    Code by Chris Colbert from the numpy-discussion list.
    """
    phi = (1 + np.sqrt(5)) / 2 # the golden ratio
    long_incr = 2*np.pi / phi # how much to increment the longitude

    dz = 2.0 / float(N) # a unit sphere has diameter 2
    bands = np.arange(N) # each band will have one point placed on it
    z = bands * dz - 1 + (dz/2) # the height z of each band/point
    r = np.sqrt(1 - z*z) # project onto xy-plane
    az = bands * long_incr # azimuthal angle of point modulo 2 pi
    x = r * np.cos(az)
    y = r * np.sin(az)

    x = radius * x
    y = radius * y
    z = radius * z + h
    return x, y, z

def average_g(triples):
    return np.mean([triple[2] for triple in triples])


GaAs = material('GaAs')()
Air = material('Air')()

flat_surf = planar_surface(size=0.8) # pyramid size in microns


# fig = plt.figure()
# ax = plt.subplot(121, projection='3d')
# ax.view_init(elev=30., azim=60)
# #ax.set_aspect('equal')
# ax.plot_trisurf(front.Points[:,0], front.Points[:,1], front.Points[:,2],
#                 triangles=front.simplices[~bottom_planar, :],  shade=False, cmap=plt.get_cmap('Blues'), array=colors,
#         edgecolors='none')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# ax = plt.subplot(111, projection='3d')
# ax.view_init(elev=30., azim=60)
#ax.set_aspect('equal')
# ax.plot_trisurf(back.Points[:,0], back.Points[:,1], back.Points[:,2],
#                 triangles=back.simplices[~bottom_planar, :],  shade=False, cmap=plt.get_cmap('Blues'), array=colors,
#         edgecolors='none')
#
# ax.plot_trisurf(flat_surf[0].Points[:,0], flat_surf[0].Points[:,1], flat_surf[0].Points[:,2],
#                 triangles=flat_surf[0].simplices)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

options = default_options()

n_spheres = np.arange(8, 16)
# n_spheres = [16, 17]

nx = 50

pal = sns.color_palette("rocket", len(n_spheres))

times = np.zeros(len(n_spheres))

minimum_angle = np.pi - np.pi * 17.5 / 180

thetas = np.linspace(0, np.pi / 2 - 0.1, 10)

# thetas = thetas[thetas > 0.95]

n_fudges = []

plt.figure()

for i1, ns in enumerate(n_spheres):

    X, Y, Z = points_on_sphere(2 ** ns, radius=0.8, h=0.242)

    X = X[Z >= 0]
    Y = Y[Z >= 0]
    Z = Z[Z >= 0]

    Triples = np.array(list(zip(X, Y, Z)))

    Triples_back = np.array(list(zip(X, Y, -Z - d_bulk)))

    hull = ConvexHull(Triples)
    hull_back = ConvexHull(Triples_back)
    triangles = hull.simplices
    triangles_back = hull_back.simplices

    colors = np.array([average_g([Triples[idx] for idx in triangle]) for
                       triangle in triangles_back])



    [front, back] = xyz_texture(X, Y, Z)

    front.simplices = triangles
    front.P_0s = front.Points[triangles[:, 0]]
    front.P_1s = front.Points[triangles[:, 1]]
    front.P_2s = front.Points[triangles[:, 2]]
    front.crossP = np.cross(front.P_1s - front.P_0s, front.P_2s - front.P_0s)
    front.size = front.P_0s.shape[0]
    front.zcov = 0

    back.simplices = triangles_back
    back.P_0s = back.Points[triangles_back[:, 0]]
    back.P_1s = back.Points[triangles_back[:, 1]]
    back.P_2s = back.Points[triangles_back[:, 2]]
    back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)
    back.size = back.P_0s.shape[0]
    back.zcov = 0

    cross_normalized = back.crossP / np.sqrt(np.sum(back.crossP ** 2, 1))[:, None]

    flat = np.abs(cross_normalized[:, 2]) > 0.9
    bottom_surface = np.all((back.P_0s[:, 2] > -0.1, back.P_1s[:, 2] > -0.1, back.P_2s[:, 2] > -0.1), axis=0)
    bottom_planar = np.all((flat, bottom_surface), axis=0)

    back.simplices = triangles_back[~bottom_planar]
    back.P_0s = back.P_0s[~bottom_planar]
    back.P_1s = back.P_1s[~bottom_planar]
    back.P_2s = back.P_2s[~bottom_planar]
    back.crossP = back.crossP[~bottom_planar]
    back.size = back.P_0s.shape[0]

    X_norm = np.mean([back.P_0s[:, 0], back.P_1s[:, 0], back.P_2s[:, 0]], 0)
    Y_norm = np.mean([back.P_0s[:, 1], back.P_1s[:, 1], back.P_2s[:, 1]], 0)
    Z_norm = np.mean([back.P_0s[:, 2], back.P_1s[:, 2], back.P_2s[:, 2]], 0)


    for index in np.arange(len(X_norm)):
        current_N = back.crossP[index, :]
        X_sign = -np.sign(X_norm[index])
        # N[2] should be < 0
        if np.sign(current_N[0]) != X_sign:
            P1_current = deepcopy(back.P_1s[index, :])
            P2_current = deepcopy(back.P_2s[index, :])
            back.P_1s[index, :] = P2_current
            back.P_2s[index, :] = P1_current

    back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)

    for index in np.arange(len(X_norm)):
        current_N = back.crossP[index, :]
        Y_sign = -np.sign(Y_norm[index])
        # N[2] should be < 0
        if np.sign(current_N[1]) != Y_sign:
            P1_current = deepcopy(back.P_1s[index, :])
            P2_current = deepcopy(back.P_2s[index, :])
            back.P_1s[index, :] = P2_current
            back.P_2s[index, :] = P1_current

    back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)

    above_middle = np.where(Z_norm > -0.2)[0]
    below_middle = np.where(Z_norm < -0.3)[0]

    for index in above_middle:
        current_N = back.crossP[index, :]

        # N[2] should be < 0
        if current_N[2] > 0:
            P1_current = deepcopy(back.P_1s[index, :])
            P2_current = deepcopy(back.P_2s[index, :])
            back.P_1s[index, :] = P2_current
            back.P_2s[index, :] = P1_current

    for index in below_middle:
        current_N = back.crossP[index, :]

        # N[2] should be > 0
        if current_N[2] < 0:
            P1_current = deepcopy(back.P_1s[index, :])
            P2_current = deepcopy(back.P_2s[index, :])
            back.P_1s[index, :] = P2_current
            back.P_2s[index, :] = P1_current

    back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)

    # fig = plt.figure()
    #
    # ax = plt.subplot(111, projection='3d')
    # ax.view_init(elev=30., azim=60)
    # ax.plot_trisurf(back.Points[:, 0], back.Points[:, 1], back.Points[:, 2],
    #                 triangles=back.simplices, shade=False, cmap=plt.get_cmap('Blues'), array=colors,
    #                 edgecolors='none')
    # # ax.plot_trisurf(flat_surf[0].Points[:,0], flat_surf[0].Points[:,1], flat_surf[0].Points[:,2],
    # #                 triangles=flat_surf[0].simplices)
    # #
    #
    # ax.quiver(X_norm, Y_norm, Z_norm, back.crossP[:, 0], back.crossP[:, 1], back.crossP[:, 2], length=0.1,
    #           normalize=True)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.title(ns)
    # plt.show()

    hyperhemi = [back, front]

    rtstr = rt_structure(textures=[flat_surf, hyperhemi],
                         materials=[GaAs],
                         widths=[d_bulk], incidence=Air, transmission=Air)

    options.wavelengths = np.array([6e-6])
    options.parallel = False
    options.n_rays = nx**2
    options.xlim = 0.05
    options.ylim = 0.05
    options.nx = nx
    options.ny = nx

    print(ns)


    T_values = np.zeros(len(thetas))
    T_total = np.zeros(len(thetas))
    n_interactions = np.zeros(len(thetas))
    theta_distribution = np.zeros((len(thetas), options.n_rays))

    if os.path.isfile('results/sphere_raytrace_2e' + str(ns) + '_' + str(nx**2) + 'rays2.txt'):

        T_values = np.loadtxt('results/sphere_raytrace_2e' + str(ns) + '_' + str(nx**2) + 'rays2.txt')
        T_total = np.loadtxt('results/sphere_raytrace_totalT_2e' + str(ns) + '_' + str(nx**2) + 'rays2.txt')
        n_interactions = np.loadtxt('results/sphere_raytrace_ninter_2e' + str(ns) + '_' + str(nx**2) + 'rays2.txt')
        theta_distribution = np.loadtxt('results/sphere_raytrace_thetas_2e' + str(ns) + '_' + str(nx**2) + 'rays2.txt')

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

        times[i1] = time() - start


        np.savetxt('results/sphere_raytrace_2e' + str(ns) + '_' + str(nx**2) + 'rays2.txt', T_values)
        np.savetxt('results/sphere_raytrace_totalT_2e' + str(ns) + '_' + str(nx**2) + 'rays2.txt', T_total)
        np.savetxt('results/sphere_raytrace_ninter_2e' + str(ns) + '_' + str(nx**2) + 'rays2.txt', n_interactions)
        np.savetxt('results/sphere_raytrace_thetas_2e' + str(ns) + '_' + str(nx**2) + 'rays2.txt', theta_distribution)

    n_fudges.append(np.sum(theta_distribution == 0))
    plt.plot(thetas*180/np.pi, T_values, '-', label=r'$2^{%i}$' %ns, color=pal[i1])
    plt.scatter(thetas*180/np.pi, T_values, edgecolors=pal[i1], facecolors='none')
    # plt.plot(thetas*180/np.pi, T_total, 'o--', color=pal[i1])


plt.xlim(0, 90)
plt.xlabel(r'$\beta$ (rads)')
plt.ylabel('Transmission')

plt.title("Convergence with number of surface triangles ($n_{rays} = 2500$)")
plt.legend(title="Number of triangles")
plt.show()

N = 200
bottom = 0

pal_bar = sns.color_palette("rocket", len(thetas))

plt.figure()
ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location("N")

theta_bins = np.linspace(0.0, np.pi, N, endpoint=False)
theta_centre = (theta_bins[1:] + theta_bins[:-1]) / 2
width = (np.pi) / N

for k1 in range(len(thetas)):

    radii = np.histogram(theta_distribution[k1], theta_bins)

    bars = ax.bar(theta_centre, radii[0], width=width, bottom=bottom, color=pal_bar[k1], label=str(np.round(thetas[k1]*180/np.pi, 1)))
    # ax.set_ylim(0, 100)

    # Use custom colors and opacity
    for bar in bars:
        bar.set_alpha(0.8)


plt.legend()
plt.show()

    #
    # plt.figure()
    # plt.plot(thetas, n_interactions)
    # plt.show()

# plt.figure()
# plt.plot(2**n_spheres, times, 'o-')
# plt.show()

from rayflare.ray_tracing.rt import calc_R

# thetas = np.linspace(0, np.pi/2, 200)
#
# plt.plot(thetas, calc_R(3.324, 1, thetas, 's'))
# plt.plot(thetas, calc_R(3.324, 1, thetas, 'p'))
# plt.show()

plt.figure()
plt.plot(2**n_spheres, n_fudges, 'o-')
plt.show()