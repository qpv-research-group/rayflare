import numpy as np
from pytest import approx, mark
import itertools

@mark.parametrize("upright", [True, False])
def test_regular_pryramids(upright):
    from rayflare.textures import regular_pyramids

    if upright:
        sign = 1
    else:
        sign = -1

    size = 4.8
    char_angle = 0.3
    Lx = size * 1
    Ly = size * 1
    h = Lx * np.tan(char_angle) / 2
    x = np.array([0, Lx / 2, Lx, 0, Lx])
    y = np.array([0, Ly / 2, 0, Ly, Ly])
    z = np.array([0, h, 0, 0, 0])
    points = np.vstack([x, y, sign*z]).T
    points_r = np.vstack([x, y, -sign*z]).T

    [front, back] = regular_pyramids(char_angle * 180 / np.pi, upright, size)

    assert front.Points == approx(points)
    assert back.Points == approx(points_r)


def test_planar_surface():
    from rayflare.textures import planar_surface

    [f, b] = planar_surface(5)

    assert np.all(f.Points[:, 2] == 0)
    assert np.all(b.Points[:, 2] == 0)
    assert np.all(f.P_0s[:, 2] == 0)
    assert np.all(f.P_1s[:, 2] == 0)
    assert np.all(f.P_2s[:, 2] == 0)
    assert np.all(b.P_0s[:, 2] == 0)
    assert np.all(b.P_1s[:, 2] == 0)
    assert np.all(b.P_2s[:, 2] == 0)


def test_V_grooves():
    from rayflare.textures import V_grooves

    width = 1.7
    char_angle = 0.3
    h = width * np.tan(char_angle) / 2
    x = np.array([0, width, 0, width, width / 2, width / 2])
    y = np.array([0, 0, width, width, 0, 1])

    z = np.array([0, 0, 0, 0, h, h])

    points = np.vstack([x, y, z]).T
    points_r = np.vstack([x, y, -z]).T

    [f, b] = V_grooves(0.3 * 180 / np.pi, width, "y")

    assert f.Points == approx(points)
    assert b.Points == approx(points_r)

    [f, b] = V_grooves(0.3 * 180 / np.pi, width, "x")

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

    assert np.all(front.Points[:, 2] == -back.Points[:, 2])


def test_heights_texture():
    from rayflare.textures import heights_texture
    from rayflare.ray_tracing.rt import RTSurface
    import os

    cur_path = os.path.dirname(os.path.abspath(__file__))

    AFM_data = np.loadtxt(os.path.join(cur_path, "data/pyramids.csv"), delimiter=",")
    # AFM scan data: grid of heights (z coordinates), x and y dimensions are 20 x 20 um

    [front, back] = heights_texture(AFM_data, 20, 20)

    assert isinstance(front, RTSurface)
    assert isinstance(back, RTSurface)

    assert np.all(front.Points[:, 2] == -back.Points[:, 2])


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

    GaAs = material("GaAs")()
    Air = material("Air")()

    [front, back] = hyperhemisphere(n_points, r, h)

    hyperhemi = [back, front]  # flip around

    # now want to make closed, flat top surface: find z == 0 points of hyperhemisphere surface

    edge_points = back.Points[back.Points[:, 2] == 0]

    edge_points = np.vstack([edge_points, [0, 0, 0]])  # add point at centre

    flat_surf = xyz_texture(edge_points[:, 0], edge_points[:, 1], edge_points[:, 2], coverage_height=0)
    # this is a flat surface which extends to the edges of the sphere but not beyond.

    rtstr = rt_structure(
        textures=[flat_surf, hyperhemi], materials=[GaAs], widths=[d_bulk], incidence=Air, transmission=Air
    )

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

    options.initial_material = (
        1  # the rays start in the GaAs (material index 1) rather than in the air above the cell (material index 0)
    )
    options.initial_direction = 1  # default initial direction, which is 1 (downwards)

    options.periodic = 0

    options.wavelength = np.array([6e-6])
    options.parallel = False

    options.theta_in = 0.1
    options.nx = 30
    options.ny = 30
    options.pol = "u"
    options.n_rays = 30**2

    result = rtstr.calculate(options)

    assert result["R"] + result["T"] == approx(1)
    assert result["T"] == approx(0.8, rel=0.075)

@mark.parametrize("upright", [True, False])
def test_rough_pyramids(upright):
    from rayflare.textures import rough_pyramids

    Lx = 5
    n_per_side_list = [31, 41, 51]
    noise_angle_list = [0.1, 0.2, 0.3]

    for n_noise in itertools.product(n_per_side_list, noise_angle_list):
        rough_pyramid_regular = rough_pyramids(30, upright, Lx, n_noise[1], n_noise[0] ** 2, True)
        rough_pyramid_random = rough_pyramids(30, upright, Lx, n_noise[1], n_noise[0] ** 2, False)

        assert rough_pyramid_regular[0].Points.shape == rough_pyramid_random[0].Points.shape
        assert rough_pyramid_regular[1].Points.shape == rough_pyramid_random[1].Points.shape

        assert np.mean(rough_pyramid_regular[0].Points[:, 2]) == approx(
            np.mean(rough_pyramid_regular[0].Points[:, 2]), rel=0.15
        )

        normal_regular = rough_pyramid_regular[0].N
        normal_random = rough_pyramid_random[0].N

        o_t_reg = np.histogram(
            np.abs(np.real(np.arccos(normal_regular[:, 2] / (np.linalg.norm(normal_regular, axis=1))))),
            bins=40,
            range=[0, 1.2],
            density=True,
        )[0]
        o_t_rand = np.histogram(
            np.abs(np.real(np.arccos(normal_random[:, 2] / (np.linalg.norm(normal_random, axis=1))))),
            bins=40,
            range=[0, 1.2],
            density=True,
        )[0]

        assert np.argmax(o_t_reg) == approx(np.argmax(o_t_rand), abs=3.01)
        assert np.max(o_t_reg) == approx(np.max(o_t_rand), rel=0.3)



def test_rough_pyramids_size():
    from rayflare.textures import rough_pyramids

    size_list = [1, 1.5, 3, 5.5]

    reg_array = np.zeros((len(size_list), 40))
    rand_array = np.zeros((len(size_list), 40))

    for i1, Lx in enumerate(size_list):
        rough_pyramid_regular = rough_pyramids(30, True, Lx, 0.3, 101**2, True)  # x, y points on regular grid
        rough_pyramid_random = rough_pyramids(30, True, Lx, 0.3, 101**2, False)  # x, y points randomly generated

        assert rough_pyramid_regular[0].Points.shape == rough_pyramid_random[0].Points.shape
        assert rough_pyramid_regular[1].Points.shape == rough_pyramid_random[1].Points.shape

        assert np.mean(rough_pyramid_regular[0].Points[:, 2]) == approx(
            np.mean(rough_pyramid_regular[0].Points[:, 2]), rel=0.15
        )

        normal_regular = rough_pyramid_regular[0].N
        normal_random = rough_pyramid_random[0].N

        o_t_reg = np.histogram(
            np.abs(np.real(np.arccos(normal_regular[:, 2] / (np.linalg.norm(normal_regular, axis=1))))),
            bins=40,
            range=[0, 1.2],
            density=True,
        )[0]
        o_t_rand = np.histogram(
            np.abs(np.real(np.arccos(normal_random[:, 2] / (np.linalg.norm(normal_random, axis=1))))),
            bins=40,
            range=[0, 1.2],
            density=True,
        )[0]

        assert np.argmax(o_t_reg) == approx(np.argmax(o_t_rand), abs=3.01)
        assert np.max(o_t_reg) == approx(np.max(o_t_rand), rel=0.35)

        reg_array[i1] = o_t_reg
        rand_array[i1] = o_t_rand

    assert reg_array[0][reg_array[0] > 0.3] == approx(reg_array[1][reg_array[0] > 0.3], rel=0.2)
    assert reg_array[0][reg_array[0] > 0.3] == approx(reg_array[2][reg_array[0] > 0.3], rel=0.2)

    assert rand_array[0][rand_array[0] > 0.3] == approx(rand_array[1][rand_array[0] > 0.3], rel=0.3)
    assert rand_array[0][rand_array[0] > 0.3] == approx(rand_array[2][rand_array[0] > 0.3], rel=0.3)


def test_rough_planar_size():
    from rayflare.textures import rough_planar_surface

    size_list = [1, 1.5, 3, 5.5]

    reg_array = np.zeros((len(size_list), 40))
    rand_array = np.zeros((len(size_list), 40))

    for i1, Lx in enumerate(size_list):
        rough_planar_regular = rough_planar_surface(Lx, 0.3, 101**2, True)  # x, y points on regular grid
        rough_planar_random = rough_planar_surface(Lx, 0.3, 101**2, False)  # x, y points randomly generated

        assert rough_planar_regular[0].Points.shape == rough_planar_random[0].Points.shape
        assert rough_planar_regular[1].Points.shape == rough_planar_random[1].Points.shape

        assert np.mean(rough_planar_regular[0].Points[:, 2]) == approx(
            np.mean(rough_planar_regular[0].Points[:, 2]), rel=0.15
        )

        normal_regular = rough_planar_regular[0].N
        normal_random = rough_planar_random[0].N

        o_t_reg = np.histogram(
            np.abs(np.real(np.arccos(normal_regular[:, 2] / (np.linalg.norm(normal_regular, axis=1))))),
            bins=40,
            range=[0, 1.2],
            density=True,
        )[0]
        o_t_rand = np.histogram(
            np.abs(np.real(np.arccos(normal_random[:, 2] / (np.linalg.norm(normal_random, axis=1))))),
            bins=40,
            range=[0, 1.2],
            density=True,
        )[0]

        reg_array[i1] = o_t_reg
        rand_array[i1] = o_t_rand

    assert reg_array[0][reg_array[0] > 1] == approx(reg_array[1][reg_array[0] > 1], rel=0.15)
    assert reg_array[0][reg_array[0] > 1] == approx(reg_array[2][reg_array[0] > 1], rel=0.15)

    assert rand_array[0][rand_array[0] > 1] == approx(rand_array[1][rand_array[0] > 1], rel=0.2)
    assert rand_array[0][rand_array[0] > 1] == approx(rand_array[2][rand_array[0] > 1], rel=0.2)


def test_rough_hemisphere_size():
    from rayflare.textures import hemisphere_surface

    size_list = [1, 1.5, 3, 5.5]

    reg_array = np.zeros((len(size_list), 40))

    for i1, Lx in enumerate(size_list):
        rough_planar_regular = hemisphere_surface(Lx, 101, Lx / 3, Lx / 5, 0.2, 1)  # x, y points on regular grid

        normal_regular = rough_planar_regular[0].N
        o_t_reg = np.histogram(
            np.abs(np.real(np.arccos(normal_regular[:, 2] / (np.linalg.norm(normal_regular, axis=1))))),
            bins=40,
            range=[0, 1.4],
            density=True,
        )[0]

        reg_array[i1] = o_t_reg

    assert reg_array[0][reg_array[0] > 0.75] == approx(reg_array[1][reg_array[0] > 0.75], rel=0.15)
    assert reg_array[0][reg_array[0] > 0.75] == approx(reg_array[2][reg_array[0] > 0.75], rel=0.15)

def test_hemisphere_cap_surface():
    from rayflare.textures import hemisphere_surface

    Lx = np.random.uniform(3, 5, 1)
    offset = np.random.uniform(0, 1, 1)

    front, _ = hemisphere_surface(Lx, 101, Lx / 3, offset, 0, 1)

    normals = front.N
    # radius of sphere with an offset:
    # radius is Lx/3
    assert np.max(front.Points[:,2]) == Lx / 3 - offset
    assert np.min(front.Points[:,2]) == 0
    assert np.sum(normals[:,2] ==  1) > 0

    # fraction of points not inside


def test_rough_hemsiphere_vars():
    from rayflare.textures import hemisphere_surface

    Lx = 3

    hemisphere_0 = hemisphere_surface(Lx, 101, Lx / 3, 0, 0.2, 1)
    hemisphere_offset = hemisphere_surface(Lx, 101, Lx / 3, Lx / 5, 0.2, 1)
    hemisphere_noisier = hemisphere_surface(Lx, 101, Lx / 3, Lx / 5, 0.4, 1)
    hemisphere_noisier_stretched = hemisphere_surface(Lx, 101, Lx / 3, Lx / 5, 0.4, 1.5)

    o_t_array = np.empty((4, 40))

    for i1, front in enumerate(
        [hemisphere_0[0], hemisphere_offset[0], hemisphere_noisier[0], hemisphere_noisier_stretched[0]]
    ):
        normals = front.N
        o_t_array[i1] = np.histogram(
            np.abs(np.real(np.arccos(normals[:, 2] / (np.linalg.norm(normals, axis=1))))),
            bins=40,
            range=[0, 1.4],
            density=True,
        )[0]

    assert np.max(o_t_array[0][o_t_array[0] > 0.5]) < np.max(
        o_t_array[1][o_t_array[0] > 0.5]
    )  # adding offset means noisy
    # planar area takes up larger % of the space
    assert np.argmax(o_t_array[1][o_t_array[1] > 0.5]) < np.argmax(
        o_t_array[2][o_t_array[1] > 0.5]
    )  # adding noise means higher avg opening angle
    assert np.argmax(o_t_array[1][o_t_array[1] > 0.5]) < np.argmax(
        o_t_array[3][o_t_array[2] > 0.5]
    )  # adding noise means higher avg opening angle

    # offset takes out higher angles (sides of hemisphere), so highest angle observed is lower

    assert np.sum(o_t_array[1] == 0) > np.sum(o_t_array[0] == 0)

    # adding noise does not affect this

    assert np.sum(o_t_array[2] == 0) == np.sum(o_t_array[1] == 0)

    # adding stretch means higher angles again

    assert np.sum(o_t_array[3] == 0) < np.sum(o_t_array[2] == 0)


def test_distribution_refresh():
    from rayflare.textures import regular_pyramids

    mean_height = 4
    stdv = 1.5
    heights = np.linspace(0, 10, 50)
    probs = np.exp(-0.5*((heights - mean_height)/stdv)**2)

    probs = probs/np.sum(probs)

    trisurf, _ = regular_pyramids(height_distribution={"p": probs, "h": heights})

    n_trials = 1000

    current_height = np.zeros(n_trials)

    for i1 in range(n_trials):
        trisurf.refresh()
        current_height[i1] = trisurf.z_max

    assert np.mean(current_height) == approx(mean_height, rel=0.05)
    assert np.std(current_height) == approx(stdv, rel=0.05)