import numpy as np

def text_regular_pryramids():
    from rayflare.textures import regular_pyramids

    size = 4.8
    char_angle = 0.3
    Lx = size*1
    Ly = size*1
    h = Lx*np.tan(char_angle)/2
    x = np.array([0, Lx/2, Lx, 0, Lx])
    y = np.array([0, Ly/2, 0, Ly, Ly])
    z = np.array([0, h, 0, 0, 0])
    points = np.vstack([x, y, z]).T
    points_r = np.vstack([x, y, -z]).T

    [front, back] = regular_pyramids(char_angle*180/np.pi, True, size)

    assert np.all(front.Points == points)
    assert np.all(back.Points == points_r)

    [front, back] = regular_pyramids(char_angle * 180 / np.pi, False, size)

    assert np.all(front.Points == points_r)
    assert np.all(back.Points == points)


def test_planar_surface():
    from rayflare.textures import planar_surface
    [f, b] = planar_surface(5)

    assert np.all(f.Points[:,2] == 0)
    assert np.all(b.Points[:,2] == 0)
    assert np.all(f.P_0s[:,2] == 0)
    assert np.all(f.P_1s[:, 2] == 0)
    assert np.all(f.P_2s[:, 2] == 0)
    assert np.all(b.P_0s[:, 2] == 0)
    assert np.all(b.P_1s[:, 2] == 0)
    assert np.all(b.P_2s[:, 2] == 0)


def test_V_grooves():
    from rayflare.textures import V_grooves

    width = 1.7
    char_angle = 0.3
    h = width*np.tan(char_angle)/2
    x = np.array([0, width, 0, width, width / 2, width / 2])
    y = np.array([0, 0, width, width, 0, 1])

    z = np.array([0, 0, 0, 0, h, h])

    points = np.vstack([x, y, z]).T
    points_r = np.vstack([x, y, -z]).T

    [f, b] = V_grooves(0.3*180/np.pi, width, 'y')

    assert np.all(f.Points == points)
    assert np.all(b.Points == points_r)

    [f, b] = V_grooves(0.3 * 180 / np.pi, width, 'x')

    points = np.vstack([y, x, z]).T
    points_r = np.vstack([y, x, -z]).T

    assert np.all(f.Points == points)
    assert np.all(b.Points == points_r)



def test_xyz_texture():
    from rayflare.textures import xyz_texture
    from rayflare.ray_tracing.rt import RTSurface

    x = np.array([0, 0, 1, 1, 0.5, 0.5])
    y = np.array([0, 1, 0, 1, 0, 1])
    z = np.array([0, 0, 0, 0, 1, 1])
    [front, back] = xyz_texture(x, y, z)

    assert isinstance(front, RTSurface)
    assert isinstance(back, RTSurface)

    assert np.all(front.Points[:,2] == -back.Points[:,2])


def test_heights_texture():
    from rayflare.textures import heights_texture
    from rayflare.ray_tracing.rt import RTSurface
    import os

    cur_path = os.path.dirname(os.path.abspath(__file__))

    AFM_data = np.loadtxt(os.path.join(cur_path, 'data/pyramids.csv'), delimiter=',')
    # AFM scan data: grid of heights (z coordinates), x and y dimensions are 20 x 20 um

    [front, back] =  heights_texture(AFM_data, 20, 20)

    assert isinstance(front, RTSurface)
    assert isinstance(back, RTSurface)

    assert np.all(front.Points[:,2] == -back.Points[:,2])

