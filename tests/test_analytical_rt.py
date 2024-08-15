import numpy as np
from pytest import approx, mark

def check_total_RAT_Fresnel():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import regular_pyramids
    from solcore import material
    from rayflare.options import default_options

    Si = material("Si")()
    Air = material("Air")()

    options = default_options()
    options.wavelength = np.linspace(300, 1000, 5) * 1e-9
    options.nx = 10
    options.ny = 10
    options.n_rays = 1 * options.nx ** 2
    options.pol = 's'
    options.analytical_ray_tracing = 2
    options.project_name = 'test_analytical'
    options.theta_in = 7 * np.pi / 180

    pyramids = regular_pyramids(50, True)

    rtstr = rt_structure(
        textures=[pyramids],
        materials=[],
        widths=[],
        incidence=Air, transmission=Si,
        use_TMM=False, options=options,
    )

    RAT = rtstr.calculate(options)

    # add up all contributions per wavelength:
    total_int = RAT['R'] + np.sum(RAT['A_per_layer'], 1) + RAT['T']
    assert np.all(RAT['R'] >= 0)
    assert np.all(RAT['T'] >= 0)
    assert np.all(RAT['A_per_layer'] >= 0)

    assert np.all(RAT['R'] <= 1)
    assert np.all(RAT['T'] <= 1)
    assert np.all(RAT['A_per_layer'] <= 1)

    assert total_int == approx(1, abs=options.I_thresh)

def check_total_RAT_TMM():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import regular_pyramids, planar_surface
    from solcore import material
    from rayflare.options import default_options
    from solcore.structure import Layer

    Si = material("Si")()
    Air = material("Air")()
    MgF2 = material("MgF2")()
    GaAs = material("GaAs")()
    Ge = material("Ge")()

    options = default_options()
    options.wavelength = np.linspace(300, 1400, 10)
    options.nx = 10
    options.ny = 10
    options.n_rays = 1 * options.nx ** 2
    options.pol = 's'
    options.analytical_ray_tracing = 2
    options.project_name = 'test_analytical'
    options.theta_in = 7*np.pi/180

    interface_layers = [Layer(70e-9, MgF2), Layer(500e-9, GaAs)]

    pyramids = regular_pyramids(50, True, interface_layers=interface_layers)
    planar = planar_surface(interface_layers=[Layer(200e-9, Ge)])
    planar_2 = planar_surface()

    rtstr = rt_structure(
        textures=[pyramids, planar, planar_2],
        materials=[Si, Ge],
        widths=[50e-6, 100e-6],
        incidence=Air, transmission=Air,
        use_TMM=True, options=options,
        save_location="current",
        overwrite=True,
    )

    RAT = rtstr.calculate(options)

    # add up all contributions per wavelength:
    total_int = RAT['R'] + np.sum(RAT['A_per_layer'], 1) + \
                np.sum(RAT['A_per_interface'][0], 1) + RAT['T']

    assert np.all(RAT['R'] >= 0)
    assert np.all(RAT['T'] >= 0)
    assert np.all(RAT['A_per_layer'] >= 0)
    assert np.all(RAT['A_per_interface'][0] >= 0)

    assert np.all(RAT['R'] <= 1)
    assert np.all(RAT['T'] <= 1)
    assert np.all(RAT['A_per_layer'] <= 1)
    assert np.all(RAT['A_per_interface'][0] <= 1)

    assert total_int == approx(1, abs=options.I_thresh)


def check_integrated_A_Fresnel():
    pass

def check_integrated_A_TMM():
    pass

def check_lambertian_scattering():
    pass

def check_lambertian_scattering_integrated():
    pass