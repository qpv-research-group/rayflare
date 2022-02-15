# Copyright (C) 2021 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU General Public License (GPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: p.pearce@unsw.edu.au

from rayflare.ray_tracing.rt import RTSurface
from rayflare.textures.define_textures import xyz_texture
from scipy.spatial import ConvexHull
from copy import deepcopy
import math
import numpy as np
import os

def regular_pyramids(elevation_angle=55, upright=True, size=1):
    """Defines RTSurface textures for ray-tracing of regular upright or inverted pyramids.

    :param elevation_angle: angle between the horizontal and a face of the pyramid, in degrees
    :param upright: if True, upright pyramids. If False, inverted pyramids
    :param size: size of the pyramids; the units are arbitrary, but should be kept consistent across
            different interfaces if you are not randomizing the ray positions.
    :return: a list of two RTSurface objects: [front_incidence, rear_incidence]
    """

    char_angle = math.radians(elevation_angle)
    Lx = size*1
    Ly = size*1
    h = Lx*math.tan(char_angle)/2
    x = np.array([0, Lx/2, Lx, 0, Lx])
    y = np.array([0, Ly/2, 0, Ly, Ly])

    if upright:
        z = np.array([0, h, 0, 0, 0])

    if not upright:
        z = np.array([0, -h, 0, 0, 0])

    Points = np.vstack([x, y, z]).T
    surf_fi = RTSurface(Points)

    Points_ri = np.vstack([x, y, -z]).T
    surf_ri = RTSurface(Points_ri)

    return [surf_fi, surf_ri]

def planar_surface(size=1):
    """Defines RTSurface textures for ray-tracing for a planar surface for ray-tracing.

    :param size: size of the unit cell (this should not affect the results, as the surface is planar).
    :return: a list of two RTSurface objects: [front_incidence, rear_incidence]
    """
    Lx = 1*size
    Ly = 1*size
    x = np.array([-Lx, Lx, Lx, -Lx])
    y = np.array([-Ly, Ly, -Ly, Ly])
    z = np.array([0, 0, 0, 0])

    Points = np.vstack([x, y, z]).T
    surf_fi = RTSurface(Points)
    surf_ri = RTSurface(Points)

    return [surf_fi, surf_ri]


def random_pyramids():
    """Defines RTSurface textures for ray-tracing for a surface of random pyramids (based on
    real surface scan data).

    :return: a list of two RTSurface objects: [front_incidence, rear_incidence]
    """
    cur_path = os.path.dirname(os.path.abspath(__file__))
    z_map = np.loadtxt(os.path.join(cur_path, 'pyramids.csv'), delimiter=',')
    x = np.linspace(0, 20, z_map.shape[0])
    x_map, y_map = np.meshgrid(x, x)

    x = x_map.flatten()
    y = y_map.flatten()
    z = z_map.flatten()

    Points = np.vstack([x, y, z]).T
    surf_fi = RTSurface(Points)

    Points_ri = np.vstack([x, y, -z]).T
    surf_ri = RTSurface(Points_ri)

    return [surf_fi, surf_ri]


def V_grooves(elevation_angle=55, width=1, direction='y'):
    """Defines RTSurface textures for ray-tracing for a surface of V-grooves.

    :param elevation_angle: angle between the horizontal and a face of the V-grooves, in degrres
    :param width: width of the V-grooves (units are arbitrary but should be kept consistent between surfaces
            if you are not randomizing the ray positions).
    :param direction: Whether the V-grooves lie along the 'x' or 'y' direction (string)
    :return: a list of two RTSurface objects: [front_incidence, rear_incidence]
    """
    char_angle = math.radians(elevation_angle)
    h = width*math.tan(char_angle)/2
    if direction == 'y':
        x = np.array([0, width, 0, width, width/2, width/2])
        y = np.array([0, 0, width, width, 0, 1])


    if direction == 'x':
        y = np.array([0, width, 0, width, width/2, width/2])
        x = np.array([0, 0, width, width, 0, 1])

    z = np.array([0, 0, 0, 0, h, h])

    Points = np.vstack([x, y, z]).T
    surf_fi = RTSurface(Points)

    Points_ri = np.vstack([x, y, -z]).T
    surf_ri = RTSurface(Points_ri)

    return [surf_fi, surf_ri]


def hyperhemisphere(N_points=2**15, radius=1, h=0):
    """ Generate N evenly distributed points on the unit sphere centered at
    the origin. Uses the 'Golden Spiral'.
    Code by Chris Colbert from the numpy-discussion list.
    """

    def switch_points(surface, index):
        P1_current = deepcopy(surface.P_1s[index, :])
        P2_current = deepcopy(surface.P_2s[index, :])
        surface.P_1s[index, :] = P2_current
        surface.P_2s[index, :] = P1_current

        s1_current = surface.simplices[index, 1]
        s2_current = surface.simplices[index, 2]
        surface.simplices[index, 1] = s2_current
        surface.simplices[index, 2] = s1_current

    def switch_xy(surface, X_or_Y, x_y_ind, direction):
        # direction = 1, point OUT of sphere
        # direction = -1, point IN to sphere
        for index in np.arange(len(X_norm_back)):
            current_N = surface.crossP[index, :]
            sign = direction*np.sign(X_or_Y[index])
            # N[2] should be < 0
            if np.sign(current_N[x_y_ind]) != sign:
                switch_points(surface, index)


    phi = (1 + np.sqrt(5)) / 2 # the golden ratio
    long_incr = 2*np.pi / phi # how much to increment the longitude

    dz = 2.0 / float(N_points) # a unit sphere has diameter 2
    bands = np.arange(N_points) # each band will have one point placed on it
    z = bands * dz - 1 + (dz/2) # the height z of each band/point
    r = np.sqrt(1 - z*z) # project onto xy-plane
    az = bands * long_incr # azimuthal angle of point modulo 2 pi
    x = r * np.cos(az)
    y = r * np.sin(az)

    X = radius * x
    Y = radius * y
    Z = radius * z + h

    X = X[Z >= 0]
    Y = Y[Z >= 0]
    Z = Z[Z >= 0]

    # add a line of points at Z = 0 to make sure there won't be a gap between sphere and planar surface

    r_cross = np.sqrt(radius ** 2 - h ** 2)

    n_points = int(np.sum(
        np.all([np.sqrt(X ** 2 + Y ** 2) < r_cross + 0.01, np.sqrt(X ** 2 + Y ** 2) > r_cross - 0.01], axis=0) / 2))

    # new points
    Z_new = np.zeros(n_points)
    phis = np.linspace(0, 2 * np.pi, n_points)
    X_new = r_cross * np.cos(phis)
    Y_new = r_cross * np.sin(phis)

    X = np.hstack([X_new, X])
    Y = np.hstack([Y_new, Y])
    Z = np.hstack([Z_new, Z])

    Triples = np.array(list(zip(X, Y, Z)))

    Triples_back = np.array(list(zip(X, Y, -Z)))

    hull = ConvexHull(Triples)
    hull_back = ConvexHull(Triples_back)
    triangles = hull.simplices
    triangles_back = hull_back.simplices

    [front, back] = xyz_texture(X, Y, Z) # just need to get an RTSurface object to then modify

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

    cross_normalized = front.crossP / np.sqrt(np.sum(front.crossP**2, 1))[:, None]
    cross_normalized_back = back.crossP / np.sqrt(np.sum(back.crossP ** 2, 1))[:, None]

    # remove triangles corresponding to flat face of triangle

    flat = np.abs(cross_normalized[:, 2]) > 0.9
    bottom_surface = np.all((front.P_0s[:, 2] < 0.1*radius, front.P_1s[:, 2] < 0.1*radius, front.P_2s[:, 2] < 0.1*radius), axis=0)
    bottom_planar = np.all((flat, bottom_surface), axis=0)

    flat_back = np.abs(cross_normalized_back[:, 2]) > 0.9
    top_surface = np.all((back.P_0s[:, 2] > -0.1*radius, back.P_1s[:, 2] > -0.1*radius, back.P_2s[:, 2] > -0.1*radius), axis=0)
    top_planar = np.all((flat_back, top_surface), axis=0)

    back.simplices = triangles_back[~top_planar]
    back.P_0s = back.P_0s[~top_planar]
    back.P_1s = back.P_1s[~top_planar]
    back.P_2s = back.P_2s[~top_planar]
    back.crossP = back.crossP[~top_planar]
    back.size = back.P_0s.shape[0]


    front.simplices = triangles[~bottom_planar]
    front.P_0s = front.P_0s[~bottom_planar]
    front.P_1s = front.P_1s[~bottom_planar]
    front.P_2s = front.P_2s[~bottom_planar]
    front.crossP = front.crossP[~bottom_planar]
    front.size = front.P_0s.shape[0]

    # find the middle of each triangle (X, Y and Z points):

    X_norm = np.mean([front.P_0s[:, 0], front.P_1s[:, 0], front.P_2s[:, 0]], 0)
    Y_norm = np.mean([front.P_0s[:, 1], front.P_1s[:, 1], front.P_2s[:, 1]], 0)
    Z_norm = np.mean([front.P_0s[:, 2], front.P_1s[:, 2], front.P_2s[:, 2]], 0)

    X_norm_back = np.mean([back.P_0s[:, 0], back.P_1s[:, 0], back.P_2s[:, 0]], 0)
    Y_norm_back = np.mean([back.P_0s[:, 1], back.P_1s[:, 1], back.P_2s[:, 1]], 0)
    Z_norm_back = np.mean([back.P_0s[:, 2], back.P_1s[:, 2], back.P_2s[:, 2]], 0)

    # make all the normals points inwards (for some reason this doesn't happen automatically
    # with ConvexHull! Four things to check:

    # 1. In each quadrant, if x of the middle of the triangle is +ve, sign of normal should be -ve and vice versa
    # 2. In each quadrant, if y of the middle of the triangle is +ve, sign of normal should be -ve and vice versa
    # 3. Above the centre of the sphere, the normal should be pointing down (N[2] < 0)
    # 4. Below the centre of the sphere, the normal should be pointing up (N[2] > 0)

    # If this isn't the case, simply need to switch two of the points of the triangle to flip the direction of the normal.
    # In this case, we flip P_1 and P_2.
    # Note that in principle it should be sufficient to just do the first two steps, but for triangles which lie
    # almost exactly on the x or y axis it won't work.

    # To keep things consistent, also need to switch the indices for the simplices if you are switching points around


    switch_xy(back, X_norm_back, 0, -1)
    switch_xy(front, X_norm, 0, 1)

    back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)
    front.crossP = np.cross(front.P_1s - front.P_0s, front.P_2s - front.P_0s)

    switch_xy(back, Y_norm_back, 1, -1)
    switch_xy(front, Y_norm, 1, 1)

    back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)
    front.crossP = np.cross(front.P_1s - front.P_0s, front.P_2s - front.P_0s)

    above_middle = np.where(Z_norm > 1.03*h)[0]
    below_middle = np.where(Z_norm < 0.97*h)[0]

    for index in above_middle:
        current_N = front.crossP[index, :]

        # N[2] should be < 0
        if current_N[2] < 0:
            switch_points(front, index)

    for index in below_middle:
        current_N = front.crossP[index, :]

        # N[2] should be > 0
        if current_N[2] > 0:
            switch_points(front, index)

    front.crossP = np.cross(front.P_1s - front.P_0s, front.P_2s - front.P_0s)

    above_middle_back = np.where(Z_norm_back > -0.97*h)[0]
    below_middle_back = np.where(Z_norm_back < -1.03*h)[0]

    for index in above_middle_back:
        current_N = back.crossP[index, :]

        # N[2] should be < 0
        if current_N[2] > 0:
            switch_points(back, index)

    for index in below_middle_back:
        current_N = back.crossP[index, :]

        # N[2] should be > 0
        if current_N[2] < 0:
            switch_points(back, index)

    back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)

    hyperhemi = [front, back]

    return hyperhemi