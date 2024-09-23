# Copyright (C) 2021-2024 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU Lesser General Public License (LGPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: p.pearce@unsw.edu.au

from rayflare.ray_tracing.rt_common import RTSurface
from rayflare.textures.define_textures import xyz_texture
from scipy.spatial import ConvexHull
from copy import deepcopy
import math
import numpy as np

def regular_pyramids(elevation_angle=55, upright=True, size=1, **kwargs):
    """Defines RTSurface textures for ray-tracing of regular upright or inverted regular square-base pyramids.

    :param elevation_angle: angle between the horizontal and a face of the pyramid, in degrees
    :param upright: if True, upright pyramids. If False, inverted pyramids. Whether the pyramids are upright or inverted
                    is always from the perspective of the front incidence surface, so for pyramids facing out on the
                    rear surface of a cell, you would set upright=False.
    :param size: size of the pyramids; the units are arbitrary, but should be kept consistent across
            different interfaces if you are not randomizing the ray positions.
    :param kwargs: additional keyword arguments to pass to the RTSurface object, these are all optional.
            The "height_distribution" argument can be set to as a dictionary with entries "h" and
            "p", which are lists of heights and probabilities for the heights of the pyramids. This is useful
            when simulating a random pyramid structure with a distribution of opening angles. The probabilities
            must sum to 1.

    :return: a list of two RTSurface objects: [front_incidence, rear_incidence]
    """

    if elevation_angle == 45:
        elevation_angle = 45.001 # this is to prevent rays travelling exactly
        # parallel to the surface, which causes numerical errors in the ray tracing

    char_angle = math.radians(elevation_angle)
    Lx = size * 1
    Ly = size * 1

    if "height_distribution" in kwargs:
        h = np.random.choice(kwargs["height_distribution"]["h"],
                                      p=kwargs["height_distribution"]["p"])

    else:
        h = Lx * math.tan(char_angle) / 2

    x = np.array([0, Lx / 2, Lx, 0, Lx])
    y = np.array([0, Ly / 2, 0, Ly, Ly])

    if upright:
        z = np.array([0, h, 0, 0, 0])

    if not upright:
        z = np.array([0, -h, 0, 0, 0])

    Points = np.vstack([x, y, z]).T
    surf_fi = RTSurface(Points, **kwargs)

    Points_ri = np.vstack([x, y, -z]).T
    surf_ri = RTSurface(Points_ri)
    surf_ri.name = surf_fi.name

    return [surf_fi, surf_ri]


def planar_surface(size=1, **kwargs):
    """Defines RTSurface textures for ray-tracing for a planar surface for ray-tracing.

    :param size: size of the unit cell (this should not affect the results, as the surface is planar).
    :return: a list of two RTSurface objects: [front_incidence, rear_incidence]
    """
    Lx = 1 * size
    Ly = 1 * size
    x = np.array([0, Lx, Lx, 0])
    y = np.array([0, Ly, 0, Ly])
    z = np.array([0, 0, 0, 0])

    Points = np.vstack([x, y, z]).T
    surf_fi = RTSurface(Points, **kwargs)
    surf_ri = RTSurface(Points)
    surf_ri.name = surf_fi.name

    return [surf_fi, surf_ri]


def V_grooves(elevation_angle=55, width=1, direction="y", **kwargs):
    """Defines RTSurface textures for ray-tracing for a surface of V-grooves.

    :param elevation_angle: angle between the horizontal and a face of the V-grooves, in degrres
    :param width: width of the V-grooves (units are arbitrary but should be kept consistent between surfaces
            if you are not randomizing the ray positions).
    :param direction: Whether the V-grooves lie along the 'x' or 'y' direction (string)
    :return: a list of two RTSurface objects: [front_incidence, rear_incidence]
    """

    if elevation_angle == 45:
        elevation_angle = 45.001 # this is to prevent rays travelling exactly
        # parallel to the surface, which causes numerical errors in the ray tracing

    char_angle = math.radians(elevation_angle)
    h = width * math.tan(char_angle) / 2
    if direction == "y":
        x = np.array([0, width, 0, width, width / 2, width / 2])
        y = np.array([0, 0, width, width, 0, 1])

    if direction == "x":
        y = np.array([0, width, 0, width, width / 2, width / 2])
        x = np.array([0, 0, width, width, 0, 1])

    z = np.array([0, 0, 0, 0, h, h])

    Points = np.vstack([x, y, z]).T
    surf_fi = RTSurface(Points, **kwargs)

    Points_ri = np.vstack([x, y, -z]).T
    surf_ri = RTSurface(Points_ri)
    surf_ri.name = surf_fi.name

    return [surf_fi, surf_ri]


def hyperhemisphere(N_points=2**15, radius=1, h=0, **kwargs):
    """Defines RTSurface textures for ray-tracing of a (hyper)hemisphere. Useful for hemispherical or hyperhemispherical
    lenses. Note that this is a surface of the hyperhemisphere only, and not a surrounding surface which reaches the
    unit cell edges. The sphere will be open at the bottom. The points for the sphere are generated using the 'Golden
    Spiral'.
    Code by Chris Colbert from the numpy-discussion list.

    :param N_points: Number of points on the sphere. The final surface may have fewer points depending on h.
    :param radius: radius of the sphere (units are arbitrary but should be kept consistent between surfaces)
    :param h: by how much the origin of the sphere is raised or lowered. With h=0, a hemisphere is returned with
            the origin at the center of the hemisphere. With h=radius, the whole sphere is returned; with negative
            h, the hemisphere is truncated.
    :return: a list of two RTSurface objects: [front_incidence, rear_incidence]
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
            sign = direction * np.sign(X_or_Y[index])
            # N[2] should be < 0
            if np.sign(current_N[x_y_ind]) != sign:
                switch_points(surface, index)

    phi = (1 + np.sqrt(5)) / 2  # the golden ratio
    long_incr = 2 * np.pi / phi  # how much to increment the longitude

    dz = 2.0 / float(N_points)  # a unit sphere has diameter 2
    bands = np.arange(N_points)  # each band will have one point placed on it
    z = bands * dz - 1 + (dz / 2)  # the height z of each band/point
    r = np.sqrt(1 - z * z)  # project onto xy-plane
    az = bands * long_incr  # azimuthal angle of point modulo 2 pi
    x = r * np.cos(az)
    y = r * np.sin(az)

    X = radius * x
    Y = radius * y
    Z = radius * z + h

    X = X[Z >= 0]
    Y = Y[Z >= 0]
    Z = Z[Z >= 0]

    # add a line of points at Z = 0 to make sure there won't be a gap between sphere and planar surface

    r_cross = np.sqrt(radius**2 - h**2)

    n_points = int(
        np.sum(
            np.all(
                [
                    np.sqrt(X**2 + Y**2) < r_cross + 0.01,
                    np.sqrt(X**2 + Y**2) > r_cross - 0.01,
                ],
                axis=0,
            )
            / 2
        )
    )

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

    [front, back] = xyz_texture(
        X, Y, Z, coverage_height=0, **kwargs
    )  # just need to get an RTSurface object to then modify

    front.simplices = triangles
    front.P_0s = front.Points[triangles[:, 0]]
    front.P_1s = front.Points[triangles[:, 1]]
    front.P_2s = front.Points[triangles[:, 2]]
    front.crossP = np.cross(front.P_1s - front.P_0s, front.P_2s - front.P_0s)
    front.size = front.P_0s.shape[0]
    # front.zcov = 0

    back.simplices = triangles_back
    back.P_0s = back.Points[triangles_back[:, 0]]
    back.P_1s = back.Points[triangles_back[:, 1]]
    back.P_2s = back.Points[triangles_back[:, 2]]
    back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)
    back.size = back.P_0s.shape[0]
    # back.zcov = 0

    cross_normalized = front.crossP / np.sqrt(np.sum(front.crossP**2, 1))[:, None]
    cross_normalized_back = back.crossP / np.sqrt(np.sum(back.crossP**2, 1))[:, None]

    # remove triangles corresponding to flat face of triangle

    flat = np.abs(cross_normalized[:, 2]) > 0.9
    bottom_surface = np.all(
        (
            front.P_0s[:, 2] < 0.1 * radius,
            front.P_1s[:, 2] < 0.1 * radius,
            front.P_2s[:, 2] < 0.1 * radius,
        ),
        axis=0,
    )
    bottom_planar = np.all((flat, bottom_surface), axis=0)

    flat_back = np.abs(cross_normalized_back[:, 2]) > 0.9
    top_surface = np.all(
        (
            back.P_0s[:, 2] > -0.1 * radius,
            back.P_1s[:, 2] > -0.1 * radius,
            back.P_2s[:, 2] > -0.1 * radius,
        ),
        axis=0,
    )
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

    # make all the normals points outwards (for some reason this doesn't happen automatically
    # with ConvexHull!) Four things to check:

    # 1. In each quadrant, if x of the middle of the triangle is +ve, sign of normal should be +ve and vice versa
    # 2. In each quadrant, if y of the middle of the triangle is +ve, sign of normal should be +ve and vice versa
    # 3. Above the centre of the sphere, the normal should be pointing up (N[2] > 0)
    # 4. Below the centre of the sphere, the normal should be pointing down (N[2] < 0)

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

    above_middle = np.where(Z_norm > 1.03 * h)[0]
    below_middle = np.where(Z_norm < 0.97 * h)[0]

    for index in above_middle:
        current_N = front.crossP[index, :]

        # N[2] should be > 0
        if current_N[2] < 0:
            switch_points(front, index)

    for index in below_middle:
        current_N = front.crossP[index, :]

        # N[2] should be < 0
        if current_N[2] > 0:
            switch_points(front, index)

    front.crossP = np.cross(front.P_1s - front.P_0s, front.P_2s - front.P_0s)

    above_middle_back = np.where(Z_norm_back > -0.97 * h)[0]
    below_middle_back = np.where(Z_norm_back < -1.03 * h)[0]

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

    front.N = front.crossP / np.linalg.norm(front.crossP, axis=1)[:, None]
    back.N = back.crossP / np.linalg.norm(back.crossP, axis=1)[:, None]

    back.name = front.name
    hyperhemi = [front, back]

    return hyperhemi


def rough_pyramids(
    elevation_angle=55,
    upright=True,
    size=1,
    noise_angle=0.1,
    n_points=100,
    regular_grid=False,
    **kwargs
):

    """Defines RTSurface textures for ray-tracing of regular upright or inverted regular square-base pyramids, with
    random noise added to the height of each point to simulate surface roughness.

    :param elevation_angle: angle between the horizontal and a face of the pyramid, in degrees
    :param upright: if True, upright pyramids. If False, inverted pyramids. Whether the pyramids are upright or inverted
                    is always from the perspective of the front incidence surface, so for pyramids facing out on the
                    rear surface of a cell, you would set upright=False.
    :param size: size of the pyramids; the units are arbitrary, but should be kept consistent across
            different interfaces if you are not randomizing the ray positions.
    :param noise_angle: the maximum opening angle/surface normal angle that will be used to generate the random noise.
                    This is used to keep the height of the roughness for some noise_angle consistent with diferent n_points
    :param n_points: number of points to use to define the surface (in total, not per side). Noise will be added
            to the height of each point so a large number means a texture with smaller features
    :param regular_grid: if True, the points will be placed on a regular grid. If False, the points will be placed
            randomly in the unit cell. n_points will be rounded up to the nearest square number if
            regular_grid=True.
    :return: a list of two RTSurface objects: [front_incidence, rear_incidence]
    """

    if elevation_angle == 45:
        elevation_angle = 45.001 # this is to prevent rays travelling exactly
        # parallel to the surface, which causes numerical errors in the ray tracing

    char_angle = math.radians(elevation_angle)

    if regular_grid:
        n_per_side = int(np.ceil(np.sqrt(n_points)))
        x = np.linspace(0, size, n_per_side)
        y = np.linspace(0, size, n_per_side)
        coords = np.meshgrid(x, y)
        coords = np.vstack((coords[0].flatten(), coords[1].flatten())).T
        ds = np.diff(x)[0]

    else:
        coords = size * np.random.random_sample(
            (n_points - 4, 2)
        )  # -4 because we will explicitly add corners later
        ds = size / np.sqrt(n_points)
        corners = np.array([[0, 0, 0], [0, size, 0], [size, size, 0], [size, 0, 0]])

    # CALCULATE NOISE_FRACTION!
    noise_height = ds * np.tan(noise_angle)
    # sort into quadrants
    top = coords[coords[:, 0] <= coords[:, 1]]
    bottom = coords[coords[:, 0] > coords[:, 1]]

    left = top[top[:, 1] <= -top[:, 0] + size]
    top = top[top[:, 1] > -top[:, 0] + size]
    right = bottom[bottom[:, 1] > -bottom[:, 0] + size]
    bottom = bottom[bottom[:, 1] <= -bottom[:, 0] + size]

    bottom = np.hstack((bottom, (bottom[:, 1] * np.tan(char_angle))[:, None]))
    top = np.hstack((top, np.tan(char_angle) * (size - top[:, 1])[:, None]))
    right = np.hstack((right, np.tan(char_angle) * (size - right[:, 0])[:, None]))
    left = np.hstack((left, (left[:, 0] * np.tan(char_angle))[:, None]))

    if upright:
        sign = 1

    else:
        sign = -1

    bottom[:, 2] = sign * (
        bottom[:, 2] + noise_height * np.random.random_sample(len(bottom))
    )
    top[:, 2] = sign * (top[:, 2] + noise_height * np.random.random_sample(len(top)))
    right[:, 2] = sign * (
        right[:, 2] + noise_height * np.random.random_sample(len(right))
    )
    left[:, 2] = sign * (left[:, 2] + noise_height * np.random.random_sample(len(left)))

    if regular_grid:
        all_points = np.vstack((bottom, top, right, left))

    else:
        all_points = np.vstack((bottom, top, right, left, corners))
    # all_points[:, 2] = sign * (all_points[:, 2] + noise_height * np.random.random_sample(len(all_points)))

    # set heights all around edges to zero, otherwise there will be holes in the periodic surface!

    all_points[:, 2][all_points[:, 0] == 0] = 0
    all_points[:, 2][all_points[:, 1] == 0] = 0

    all_points[:, 2][all_points[:, 0] == size] = 0
    all_points[:, 2][all_points[:, 1] == size] = 0

    surfs = xyz_texture(*all_points.T, **kwargs)

    return surfs


def rough_planar_surface(
    size=1, noise_angle=0.1, n_points=100, regular_grid=False, **kwargs
):

    """Defines RTSurface textures for ray-tracing of regular upright or inverted regular square-base pyramids, with
    random noise added to the height of each point to simulate surface roughness.

    :param size: size of the unit cell; the units are arbitrary, but should be kept consistent across
            different interfaces.
    :param noise_angle: the maximum opening angle/surface normal angle that will be used to generate the random noise.
                    This is used to keep the height of the roughness for some noise_angle consistent with diferent n_points
    :param n_points: number of points to use to define the surface (in total, not per side). Noise will be added
            to the height of each point so a large number means a texture with smaller features
    :param regular_grid: if True, the points will be placed on a regular grid. If False, the points will be placed
            randomly in the unit cell. n_points will be rounded up to the nearest square number if
            regular_grid=True.
    :return: a list of two RTSurface objects: [front_incidence, rear_incidence]
    """

    if regular_grid:
        n_per_side = int(np.ceil(np.sqrt(n_points)))
        x = np.linspace(0, size, n_per_side)
        y = np.linspace(0, size, n_per_side)
        coords = np.meshgrid(x, y)
        coords = np.vstack(
            (coords[0].flatten(), coords[1].flatten(), np.zeros(n_per_side**2))
        ).T
        ds = np.diff(x)[0]
        noise_height = ds * np.tan(noise_angle)
        coords[:, 2] = noise_height * np.random.random_sample(n_per_side**2)

    else:
        coords = size * np.random.random_sample((n_points - 4, 3))
        ds = size / np.sqrt(n_points)
        noise_height = ds * np.tan(noise_angle)
        coords[:, 2] = noise_height * np.random.random_sample(n_points - 4)

    if regular_grid is False:
        corners = np.array([[0, 0, 0], [0, size, 0], [size, size, 0], [size, 0, 0]])
        coords = np.vstack((coords, corners))

    # set heights all around edges to zero, otherwise there will be holes in the periodic surface!

    coords[:, 2][coords[:, 0] == 0] = 0
    coords[:, 2][coords[:, 1] == 0] = 0

    coords[:, 2][coords[:, 0] == size] = 0
    coords[:, 2][coords[:, 1] == size] = 0

    surfs = xyz_texture(*coords.T, **kwargs)

    return surfs


def hemisphere_surface(
    size=1, n_per_side=20, radius=0.5, offset=0, noise_angle=0, stretch=1,
        upright=True, **kwargs
):

    """Creates a planar surface with a hemispherical cap embedded in it. The planar part of the surface can be
    rough. The hemisphere is centered at the center of the unit cell.

    :param size: size of the unit cell; the units are arbitrary, but should be kept consistent across
    :param n_per_side: the number of grid points per side (total number of points will be n_per_side**2)
    :param radius: radius of the hemispherical cap
    :param offset: the hemisphere is shifted DOWN by this value (any points which end up below the z = 0 surface will be
                    removed from the surface)
    :param noise_angle: the maximum opening angle/surface normal angle (radians) that will be used to generate the random noise.
                    This is used to keep the height of the roughness for some noise_angle consistent with diferent n_points
    :param stretch: factor by which the height of the hemispherical cap is stretched (for ellipsoid rather than spheres)
    :return:
    """

    # throw error if offset < 0: the hemisphere will be ABOVE the plane!
    if offset < 0:
        raise ValueError("The hemisphere cannot be above the plane!")

    x = np.linspace(-size / 2, size / 2, n_per_side)
    y = np.linspace(-size / 2, size / 2, n_per_side)

    ds = np.diff(x.flatten())[0]
    noise_height = ds * np.tan(noise_angle)

    xs, ys = np.meshgrid(x, y)

    points = np.vstack((xs.flatten(), ys.flatten())).T

    all_points = np.vstack((xs.flatten(), ys.flatten(), np.zeros(n_per_side**2))).T

    include = np.zeros_like(all_points[:, 0], dtype=bool)

    for i1, coord in enumerate(points):

        if np.sum(coord**2) < (radius**2 - offset**2):

            theta = np.arcsin(np.sqrt(np.sum(coord**2)) / radius)
            all_points[i1, 2] = stretch * (radius * np.cos(theta) - offset)
            include[i1] = True

        elif (
            coord[0] == -size / 2
            or coord[0] == size / 2
            or coord[1] == -size / 2
            or coord[1] == size / 2
        ):

            include[i1] = True

        else:
            if noise_height > 0:
                include[i1] = True
                all_points[i1, 2] += noise_height * np.random.random_sample()

    all_points = all_points[include, :]

    all_points = all_points[all_points[:, 2] >= 0, :]

    if noise_height == 0:
        rad_at_zero = np.sqrt(radius**2 - offset**2)

        phis = np.linspace(0, 2 * np.pi, 5 * n_per_side)

        x_circle = rad_at_zero * np.cos(phis)
        y_circle = rad_at_zero * np.sin(phis)
        z_circle = np.zeros_like(phis)

        # x_edge = [-size/2, size/2, size/2, -size/2]
        # y_edge = [-size/2, -size/2, size/2, size/2]
        # z_edge = np.zeros(4)

        circle_points = np.vstack((x_circle, y_circle, z_circle)).T

        # edge_points = np.vstack((x_edge, y_edge, z_edge)).T

        all_points = np.vstack((all_points, circle_points))  # , edge_points))

    all_points[:, 0] = all_points[:, 0] + size / 2
    all_points[:, 1] = all_points[:, 1] + size / 2

    if not upright:
        all_points[:, 2] = -all_points[:, 2]

    surfs = xyz_texture(*all_points.T, coverage_height=0, **kwargs)

    return surfs
