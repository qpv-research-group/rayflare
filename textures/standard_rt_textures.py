from ray_tracing.rt_lookup import RTSurface
import math
import numpy as np

def regular_pyramids(elevation_angle=55, upright=True):

    char_angle = math.radians(elevation_angle)
    Lx = 1
    Ly = 1
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

def planar_surface():
    Lx = 1
    Ly = 1
    x = np.array([0, Lx, Lx, 0])
    y = np.array([0, Ly, 0, Ly])
    z = np.array([0, 0, 0, 0])

    Points = np.vstack([x, y, z]).T
    surf_fi = RTSurface(Points)
    surf_ri = RTSurface(Points)

    return [surf_fi, surf_ri]