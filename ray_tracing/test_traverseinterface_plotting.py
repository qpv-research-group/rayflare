import numpy as np
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ray_tracing.rt import single_ray, RTSurface, calc_R

from cmath import sin, cos, asin, sqrt, acos, atan
from math import atan2
from random import random

# issue with ray ending up outside box: wrong initial position (ValueError: bins must be monotonically increasing or decreasing)
# this seems to happen when there is no initial intersection
# incidence exactly on edge?

# simple pyramids:

char_angle = math.radians(55)
Lx = 1
Ly = 1
h = Lx*math.tan(char_angle)/2
x = np.array([0, Lx/2, Lx, 0, Lx])
y = np.array([0, Ly/2, 0, Ly, Ly])
z = np.array([0, -h, 0, 0, 0])
# large pyramid array
#Z = np.genfromtxt('pyramids.csv', delimiter=',')

#(X, Y) = np.meshgrid(np.linspace(0, 20, Z.shape[0]), np.linspace(0, 20, Z.shape[1]))
#x = X.flatten()
#y = Y.flatten()
#z = Z.flatten()


Points = np.vstack([x, y, z]).T

surf = RTSurface(Points)

char_angle = math.radians(55)
Lx = 1.7
Ly = 1.7
h = Lx*math.tan(char_angle)/2
x2 = np.array([0, Lx/2, Lx, 0, Lx])
y2 = np.array([0, Ly/2, 0, Ly, Ly])
z2 = np.array([0, 0, 0, 0, 0])

Points = np.vstack([x2, y2, z2-10]).T

surf2 = RTSurface(Points)


char_angle = math.radians(55)
Lx = 1.5
Ly = 1.5
h = Lx*math.tan(char_angle)/2
x3 = np.array([0, Lx/2, Lx, 0, Lx])
y3 = np.array([0, Ly/2, 0, Ly, Ly])
z3 = np.array([0, -h, 0, 0, 0])
# large pyramid array
#Z = np.genfromtxt('pyramids.csv', delimiter=',')

#(X, Y) = np.meshgrid(np.linspace(0, 20, Z.shape[0]), np.linspace(0, 20, Z.shape[1]))
#x = X.flatten()
#y = Y.flatten()
#z = Z.flatten()


Points = np.vstack([x3, y3, z3-11]).T

surf3 = RTSurface(Points)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, z, triangles=surf.simplices,  linewidth=1, color = (0.5, 0.5, 0.5, 0.5))
ax.plot_trisurf(x2, y2, z2-10, triangles=surf2.simplices,  linewidth=1, color = (0.5, 0.5, 0.5, 0.5))
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.set_zlim(-9, -10.5)
#ax.set_xlim(0, 1.5)

surfaces = [surf, surf2]
materials = {'nk': [1, 3.614+0.0021701*1j, 1], 'alpha': [0, 4*np.pi*0.0021701/0.9, 0], 'width': [0, 10, 0]}
#materials = {'nk': [1, 3+0.5*1*1j], 'alpha': [0, 0.05], 'width': [0, 0]}

I_thresh = 1e-6

n_points = [0, 1000, 0]


theta = 0
phi = 0

#x = 0.18
#y = 0.673
x = 0.179
y = 0.3

# need to translate ray back into unit cell of layer, so that initial point hits surface.
z_space = 1e-3
z_pos = np.arange(0, sum(materials['width']), z_space)

h = max(surfaces[0].Points[:, 2])
r = abs((h + 1) / cos(theta))
r_a_0 = np.real(np.array([r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]))

I, profile, A_per_layer, theta, ray_path = single_ray(x, y, r_a_0, theta, phi, surfaces, materials, z_pos, I_thresh)

n_ray = 1

for i1 in range(len(ray_path)-1):
    if ray_path[i1+1, 3] == 0:
        ax.plot([ray_path[i1, 0], ray_path[i1+1, 0]], [ray_path[i1, 1], ray_path[i1+1, 1]], [ray_path[i1, 2], ray_path[i1+1, 2]])
        ax.text((ray_path[i1, 0] +ray_path[i1+1, 0])/2, (ray_path[i1, 1] +ray_path[i1+1, 1])/2, (ray_path[i1, 2] +ray_path[i1+1, 2])/2, str(n_ray))
        n_ray = n_ray + 1

    else:
        ax.plot([ray_path[i1, 0], ray_path[i1 + 1, 0]], [ray_path[i1, 1], ray_path[i1 + 1, 1]],
                [ray_path[i1, 2], ray_path[i1 + 1, 2]], '--')

ax.pbaspect = [1,1,1]
#
# depths = []
# for i1, n in enumerate(n_points):  # don't care about absorption profile in incidence and transmission medium
#     depths.append(np.linspace(0, materials['width'][i1], n))
#
# print(np.trapz(profile[1], depths[1]))
# print(np.trapz(profile[2], depths[2]))
#
# #
# # ray_path=  np.array([[ 0.2,         0.3,         1.70617532],
# #  [ 0.2,         0.3,         0.28247013],
# #  [ 0.8281526,   0.3,        -0.49562004],
# #  [-0.79999937,  0.3,         0.28246935],
# #  [-0.17184677,  0.3,        -0.49562082],
# #  [-1.79999937,  0.3,         0.28246935],
# #  [ 0.50788244,  0.3,        -2.57629481],
# #  [ 1.13603504, -0.43391228, -2.83474612],
# #  [ 0.50788307,  1.29999927, -2.57629506],
# #  [ 1.13603567,  0.56608699, -2.83474637],
# #                      [0.507874,    0.3000332, - 2.57624791],
# #                      [1.1360266,- 0.43387907, - 2.83469922],
# #                      [0.50787463,  1.30003247, - 2.57624817],
# #                     [1.13602723,0.56612019, - 2.83469948],
# #                      [0.50781112,  0.30009794, - 2.57615648],
# #                      [1.13596372, - 0.43381433, - 2.83460779],
# #                      [0.50781175,  1.30009721, - 2.57615673],
# #                     [1.13596435,
# # 0.56618493, - 2.83460804]
# # ])
# #
# #
#
# char_angle = math.radians(54.7)
# Lx = 1
# Ly = 1
# h = Lx*math.tan(char_angle)/2
# x = np.array([0, Lx/2, Lx, 0, Lx])
# y = np.array([0-Ly, Ly/2-Ly, 0-Ly, Ly-Ly, Ly-Ly])
# z = np.array([0, h, 0, 0, 0])
# # large pyramid array
# #Z = np.genfromtxt('pyramids.csv', delimiter=',')
#
# #(X, Y) = np.meshgrid(np.linspace(0, 20, Z.shape[0]), np.linspace(0, 20, Z.shape[1]))
# #x = X.flatten()
# #y = Y.flatten()
# #z = Z.flatten()
#
#
# Points = np.vstack([x, y, z]).T
#
# surf3 = RTSurface(Points)
#
# from traverse_interface import calc_angle
#
# r_a = np.array([ 0.3699984,   0.29999931, -9.57629505])
# d= np.array([ 0.68106626, -0.69064818, -0.24321562])
#
#
# def determine_side(r_a, d, Lx, Ly):
#     # which side of the unit cell will the ray cross first?
#     n = np.array([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]])
#     p_0 = np.array([[0, Ly, 0], [Lx, 0, 0], [0, 0, 0], [0, 0, 0]])
#     t = np.sum((p_0 - r_a) * n, axis=1) / np.sum(d * n, axis=1)
#     which_intersect = t > 0
#     t[~which_intersect] = float('inf')
#     return r_a
#
#     side = np.argmin(t)
#
#
# p_0 = np.array([[0, Ly, 0], [Lx, 0, 0], [0, 0, 0], [0, 0, 0]])
# r_a = np.array([0.6, 0.6, 0])
#
# d = np.array([0.6,0.5,0])
# d = d/np.linalg.norm(d)
#
# n = np.array([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]])
#
# t = np.sum((p_0-r_a)*n, axis = 1)/np.sum(d*n, axis = 1)
# which_intersect = t > 0
# t[~which_intersect] = float('inf')
#
# side = np.argmin(t)
#
# dir2D = calc_angle(d[[0, 1]])
# # if sum(r_a[0:2] >= 1):
# #     print('translation')
# #     print(r_a, d)
# #     print(ray_path)
# #     print(r_b)
#
# # print(Lx, Ly, r_a)
# angles = [calc_angle(x) for x in
#           [[Lx - r_a[0], Ly - r_a[1]], [Lx - r_a[0], -r_a[1]], [-r_a[0], -r_a[1]], [-r_a[0], Ly - r_a[1]]]]
# angles = [2 * np.pi + x if x < 0 else x for x in np.append(dir2D, angles)]
# # print(angles)
# which_side = np.digitize([angles[0]], angles[1:])[0] % 4  # modulu division because want 4 -> 0
#
#

x_lim = 2
y_lim = 2

nx = 15
ny = 15

xc = 1
yc = 1

xs = np.linspace(x_lim / 100, x_lim - (x_lim / 100), nx)
ys = np.linspace(y_lim / 100, y_lim - (y_lim / 100), ny)
xv, yv = np.meshgrid(xs, ys)
xv = xv.flatten()
yv = yv.flatten()

quadr = np.all(np.array([np.arctan2(xv-xc, yv-yc) < np.pi/4, np.arctan2(xv-xc, yv-yc) > 0]), axis = 0)

xv[quadr]