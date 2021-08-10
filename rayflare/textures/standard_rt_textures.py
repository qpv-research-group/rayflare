# Copyright (C) 2021 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU General Public License (GPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: pmp31@cam.ac.uk

from rayflare.ray_tracing.rt import RTSurface
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
    x = np.array([0, Lx, Lx, 0])
    y = np.array([0, Ly, 0, Ly])
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