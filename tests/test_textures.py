import numpy as np
from pytest import approx

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

    assert front.Points == approx(points)
    assert back.Points == approx(points_r)

    [front, back] = regular_pyramids(char_angle * 180 / np.pi, False, size)

    assert front.Points == approx(points_r)
    assert back.Points == approx(points)


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

    assert f.Points == approx(points)
    assert b.Points == approx(points_r)

    [f, b] = V_grooves(0.3 * 180 / np.pi, width, 'x')

    points = np.vstack([y, x, z]).T
    points_r = np.vstack([y, x, -z]).T

    assert f.Points == approx(points)
    assert b.Points == approx(points_r)



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


def test_hyperhemisphere():
    from rayflare.textures import hyperhemisphere
    from rayflare.textures import xyz_texture
    from solcore import material
    from rayflare.ray_tracing import rt_structure
    from rayflare.options import default_options

    n_points = 2**10

    hh = hyperhemisphere(n_points, 1, 0.8)

    assert len(hh[0].Points == n_points)
    assert len(hh[1].Points == n_points)

    d_bulk = 0

    r = 1.6
    h = 0.484

    GaAs = material('GaAs')()
    Air = material('Air')()

    [front, back] = hyperhemisphere(n_points, r, h)

    hyperhemi = [back, front]

    # now want to make closed, flat top surface: find z == 0 points of hyperhemisphere surface

    edge_points = back.Points[back.Points[:, 2] == 0]

    edge_points = np.vstack([edge_points, [0, 0, 0]])  # add point at centre

    flat_surf = xyz_texture(edge_points[:, 0], edge_points[:, 1], edge_points[:, 2])
    # this is a flat surface which extends to the edges of the sphere but not beyond.

    rtstr = rt_structure(textures=[flat_surf, hyperhemi],
                         materials=[GaAs],
                         widths=[d_bulk], incidence=Air, transmission=Air)

    # structure:

    # Air above lens
    # ----------- planar interface between air and GaAs
    # |         |
    # |        |
    #  \_____/     hyperhemisphere pointing down
    # Air below lens.

    options = default_options()

    options.x_limits = [-0.1, 0.1]  # area of the 'diode'
    options.y_limits = [-0.1, 0.1]

    options.initial_material = 1  # the rays start in the GaAs (material index 1) rather than in the air above the cell (material index 0)
    options.initial_direction = 1  # default initial direction, which is 1 (downwards)

    options.periodic = 0

    options.wavelengths = np.array([6e-6])
    options.parallel = False

    options.theta_in = 0.1
    options.nx = 30
    options.ny = 30
    options.pol = 'u'
    options.n_rays = 30**2

    result = rtstr.calculate(options)

    assert result['R'] + result['T'] == approx(1)
    assert result['T'] == approx(0.8, rel=0.075)





