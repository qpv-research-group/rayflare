from rayflare.ray_tracing.rt import RTSurface
import math
import numpy as np
import os

def regular_pyramids(elevation_angle=55, upright=True, size=1):
    """

    :param elevation_angle:
    :param upright:
    :param size:
    :return:
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
    """

    :param size:
    :return:
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
    """

    :return:
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
    """

    :param elevation_angle:
    :param width:
    :param direction:
    :return:
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