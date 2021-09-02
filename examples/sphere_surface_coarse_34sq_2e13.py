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

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
X, Y, Z = points_on_sphere(2**13, radius=0.8, h=0.242)

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

# collec = ax.plot_trisurf(mtri.Triangulation(X, Y, triangles),
# #         Z, shade=False, cmap=plt.get_cmap('Blues'), array=colors,
# #         edgecolors='none')
# # collec.autoscale()
#
# plt.show()

[front, back] = xyz_texture(X, Y, Z)


front.simplices = triangles
front.P_0s = front.Points[triangles[:,0]]
front.P_1s = front.Points[triangles[:,1]]
front.P_2s = front.Points[triangles[:,2]]
front.crossP = np.cross(front.P_1s - front.P_0s, front.P_2s - front.P_0s)
front.size = front.P_0s.shape[0]
front.zcov = 0

back.simplices = triangles_back
back.P_0s = back.Points[triangles_back[:,0]]
back.P_1s = back.Points[triangles_back[:,1]]
back.P_2s = back.Points[triangles_back[:,2]]
back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)
back.size = back.P_0s.shape[0]
back.zcov = 0

cross_normalized = back.crossP/np.sqrt(np.sum(back.crossP**2,1))[:,None]

flat = np.abs(cross_normalized[:,2]) > 0.9
bottom_surface = np.all((back.P_0s[:,2] > -0.1, back.P_1s[:,2] > -0.1, back.P_2s[:,2] > -0.1), axis=0)
bottom_planar = np.all((flat, bottom_surface), axis=0)



from solcore import material
from rayflare.ray_tracing import rt_structure
from rayflare.options import default_options
from rayflare.textures import planar_surface

hyperhemi = [back, front]


GaAs = material('GaAs')()
Air = material('Air')()

flat_surf = planar_surface(size=0.8) # pyramid size in microns


fig = plt.figure()
# ax = plt.subplot(121, projection='3d')
# ax.view_init(elev=30., azim=60)
# #ax.set_aspect('equal')
# ax.plot_trisurf(front.Points[:,0], front.Points[:,1], front.Points[:,2],
#                 triangles=front.simplices[~bottom_planar, :],  shade=False, cmap=plt.get_cmap('Blues'), array=colors,
#         edgecolors='none')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

ax = plt.subplot(111, projection='3d')
ax.view_init(elev=30., azim=60)
#ax.set_aspect('equal')
ax.plot_trisurf(back.Points[:,0], back.Points[:,1], back.Points[:,2],
                triangles=back.simplices[~bottom_planar, :],  shade=False, cmap=plt.get_cmap('Blues'), array=colors,
        edgecolors='none')
#
# ax.plot_trisurf(flat_surf[0].Points[:,0], flat_surf[0].Points[:,1], flat_surf[0].Points[:,2],
#                 triangles=flat_surf[0].simplices)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

rtstr = rt_structure(textures=[flat_surf, hyperhemi],
                    materials = [GaAs],
                    widths=[d_bulk], incidence=Air, transmission=Air)

options = default_options()

nxs = [34]

pal = sns.color_palette("rocket", len(nxs))

plt.figure()

for i1, nx in enumerate(nxs):

    options.wavelengths = np.array([6e-6])
    options.parallel = False
    options.n_rays = nx**2
    options.xlim = 0.05
    options.ylim = 0.05
    options.theta = 0.1
    options.nx = nx
    options.ny = nx

    print(options.n_rays)

    minimum_angle = np.pi - np.pi*17.5/180

    thetas = np.linspace(0, np.pi/2-0.02, 75)

    T_values = np.zeros(len(thetas))
    T_total = np.zeros(len(thetas))
    n_interactions = np.zeros(len(thetas))

    if os.path.isfile('sphere_raytrace_2e13_' + str(options.n_rays) + 'rays.txt'):

        T_values = np.loadtxt('sphere_raytrace_2e13_' + str(options.n_rays) + 'rays.txt')
        T_total = np.loadtxt('sphere_raytrace_totalT_2e13_' + str(options.n_rays) + 'rays.txt')

    else:

        for j1, th in enumerate(thetas):
            print(j1, th)
            start = time()

            options.theta_in = th
            result = rtstr.calculate(options)
            T_values[j1] = np.sum(result['thetas'] > minimum_angle)/options.n_rays
            T_total[j1] = result['T']
            n_interactions[j1] = np.mean(result['n_interactions'])

            print(time() - start)


        np.savetxt('sphere_raytrace_2e13_' + str(options.n_rays) + 'rays.txt', T_values)
        np.savetxt('sphere_raytrace_totalT_2e13_' + str(options.n_rays) + 'rays.txt', T_total)



    plt.plot(thetas*180/np.pi, T_values, 'o-', label=str(options.n_rays), color=pal[i1])
    plt.plot(thetas*180/np.pi, T_total, 'o--', color=pal[i1])
    plt.legend()
    plt.xlim(0, 90)
    plt.xlabel(r'$\beta$ (rads)')
    plt.ylabel('Transmission')
    plt.title(str(options.n_rays))

# for ln in [17.5, 2*17.5, 3*17.5, 4*17.5]:
#     plt.axvline(x=ln)
#

plt.show()

    #
    # plt.figure()
    # plt.plot(thetas, n_interactions)
    # plt.show()