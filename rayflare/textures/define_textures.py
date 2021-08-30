# Copyright (C) 2021 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU General Public License (GPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: pmp31@cam.ac.uk

from rayflare.ray_tracing.rt import RTSurface
import numpy as np


def xyz_texture(x, y, z):
    """

    :param x: list of x (in-plane) coordinates of points on the surface texture (1D numpy array)
    :param y: list of y (in-plane) coordinates of points on the surface texture (1D numpy array)
    :param z: list of z (height) coordinates of points on the surface texture (1D numpy array)
    :return:

    """
    Points = np.vstack([x, y, z]).T
    surf_fi = RTSurface(Points)

    Points_ri = np.vstack([x, y, -z]).T
    surf_ri = RTSurface(Points_ri)

    return [surf_fi, surf_ri]


def heights_texture(z_points, x_width, y_width):
    """

    :param z_points: list of z (height) coordinates of points on the surface texture (2D numpy array)
    :param x_width: width along the x direction
    :param y_width: width along the y direction
    :return:
    """
    x = np.linspace(0, x_width, z_points.shape[0])
    y = np.linspace(0, y_width, z_points.shape[1])

    xy = np.meshgrid(x, y)

    x = xy[0].flatten()
    y = xy[1].flatten()

    z = z_points.flatten()

    return xyz_texture(x, y, z)


def triangulation_texture(points, triangles):

    surf_fi = RTSurface(Points)
