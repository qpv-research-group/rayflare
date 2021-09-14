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
X, Y, Z = points_on_sphere(2**15, radius=0.8, h=0.242)

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


back.simplices = triangles_back[~bottom_planar]
back.P_0s = back.P_0s[~bottom_planar]
back.P_1s = back.P_1s[~bottom_planar]
back.P_2s = back.P_2s[~bottom_planar]
back.crossP = back.crossP[~bottom_planar]
back.size = back.P_0s.shape[0]


from solcore import material
from rayflare.ray_tracing import rt_structure
from rayflare.options import default_options
from rayflare.textures import planar_surface, regular_pyramids

hyperhemi = [back, front]


GaAs = material('GaAs')()
Air = material('Air')()

flat_surf = planar_surface(size=0.8) # pyramid size in microns

reg_pyr = regular_pyramids()


X_norm = np.mean([back.P_0s[:,0], back.P_1s[:,0], back.P_2s[:,0]], 0)
Y_norm = np.mean([back.P_0s[:,1], back.P_1s[:,1], back.P_2s[:,1]], 0)
Z_norm = np.mean([back.P_0s[:,2], back.P_1s[:,2], back.P_2s[:,2]], 0)


# make all the normals points inwards/"upwards":
from copy import deepcopy

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



X_norm = np.mean([back.P_0s[:,0], back.P_1s[:,0], back.P_2s[:,0]], 0)
Y_norm = np.mean([back.P_0s[:,1], back.P_1s[:,1], back.P_2s[:,1]], 0)
Z_norm = np.mean([back.P_0s[:,2], back.P_1s[:,2], back.P_2s[:,2]], 0)


fig = plt.figure()


ax = plt.subplot(111, projection='3d')
ax.view_init(elev=30., azim=60)
ax.plot_trisurf(back.Points[:,0], back.Points[:,1], back.Points[:,2],
                triangles=back.simplices,  shade=False, cmap=plt.get_cmap('Blues'), array=colors,
        edgecolors='none')
# ax.plot_trisurf(flat_surf[0].Points[:,0], flat_surf[0].Points[:,1], flat_surf[0].Points[:,2],
#                 triangles=flat_surf[0].simplices)
#

ax.quiver(X_norm, Y_norm, Z_norm, back.crossP[:,0], back.crossP[:,1], back.crossP[:,2], length=0.1, normalize=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

rtstr = rt_structure(textures=[flat_surf, hyperhemi],
                    materials = [GaAs],
                    widths=[d_bulk], incidence=Air, transmission=Air)

options = default_options()

nx = 50

thetas = np.linspace(0, np.pi / 2 - 0.1, 100)

thetas_1 = thetas[0:20]
thetas_2 = thetas[20:40]
thetas_3 = thetas[40:60]
thetas_4 = thetas[60:80]
thetas_5 = thetas[80:100]

thetas_min = []
thetas_max = []

for theta_g in [thetas_1, thetas_2, thetas_3, thetas_4, thetas_5]:

    thetas_min.append(np.int(10*np.round(180*np.min(theta_g)/np.pi, 1)))
    thetas_max.append(np.int(10*np.round(180*np.max(theta_g)/np.pi, 1)))

T_v = []
T_t = []
n_int = []
theta_dist = []



for i1 in range(5):

    theta_min = thetas_min[i1]
    theta_max = thetas_max[i1]

    options.wavelengths = np.array([6e-6])
    options.parallel = False
    options.n_rays = nx**2
    options.xlim = 0.05
    options.ylim = 0.05
    
    options.theta = 0.1
    options.nx = nx
    options.ny = nx


    if os.path.isfile('results/sphere_raytrace_2e12_' + str(options.n_rays) + 'rays_' + str(theta_min) + str(theta_max) + '.txt'):

        T_values = np.loadtxt('results/sphere_raytrace_2e12_' + str(options.n_rays) + 'rays_' + str(theta_min) + str(theta_max) + '.txt')
        T_total = np.loadtxt('results/sphere_raytrace_totalT_2e12_' + str(options.n_rays) + 'rays_' + str(theta_min) + str(theta_max) + '.txt')
        n_interactions = np.loadtxt('results/sphere_raytrace_ninter_2e12_' + str(options.n_rays) + 'rays_' + str(theta_min) + str(theta_max) + '.txt')
        theta_distribution = np.loadtxt('results/sphere_raytrace_thetas_2e12_' + str(options.n_rays) + 'rays_' + str(theta_min) + str(theta_max) + '.txt')

        T_v.append(T_values)
        T_t.append(T_total)
        n_int.append(n_interactions)
        theta_dist.append(theta_distribution)


T_all = np.hstack(T_v)
T_tot_all = np.hstack(T_t)
n_int_all = np.hstack(n_int)
th_dist_all = np.vstack(theta_dist)

plt.figure()
plt.plot(thetas*180/np.pi, T_all, color='black')
plt.scatter(thetas*180/np.pi, T_all, edgecolors='black', facecolors='none')
plt.plot(thetas*180/np.pi, T_tot_all, color='black')
plt.scatter(thetas*180/np.pi, T_tot_all, edgecolors='black', facecolors='none')
# plt.legend(title="Number of rays")
plt.xlim(0, 90)
plt.xlabel(r'$\beta$ (rads)')
plt.ylabel('Transmission')


# plt.title(r'Convergence with number of rays ($N_{triangles} = 2^{12}$)')

plt.show()

np.savetxt('hyperhemi_T_withinangle_u.txt', np.vstack([thetas, T_all, T_tot_all]).T)
