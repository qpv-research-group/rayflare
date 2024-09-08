import numpy as np
from pytest import approx

def test_total_RAT_Fresnel():
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

    R_from_angles = np.array([np.sum(theta < np.pi/2) for theta in RAT['thetas']])/options.n_rays
    T_from_angles = np.array([np.sum(theta > np.pi/2) for theta in RAT['thetas']])/options.n_rays

    assert R_from_angles == approx(RAT['R'], abs=1/options.n_rays)
    assert T_from_angles == approx(RAT['T'], abs=1/options.n_rays)

def test_total_RAT_TMM():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import regular_pyramids, planar_surface
    from solcore import material
    from rayflare.options import default_options
    from solcore.structure import Layer
    import numpy as np
    from pytest import approx

    Si = material("Si")()
    Air = material("Air")()
    MgF2 = material("MgF2")()
    GaAs = material("GaAs")()
    Ge = material("Ge")()

    options = default_options()
    options.wavelength = np.linspace(300, 1400, 5) * 1e-9
    options.nx = 10
    options.ny = 10
    options.n_rays = 1 * options.nx ** 2
    options.pol = 's'
    options.analytical_ray_tracing = 2
    options.project_name = 'test_analytical'
    options.theta_in = 7 * np.pi / 180
    options.parallel = False

    interface_layers = [Layer(70e-9, MgF2), Layer(500e-9, GaAs)]

    pyramids = regular_pyramids(50, True, interface_layers=interface_layers)
    planar = planar_surface()
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

def test_compare_Fresnel():
    # calculate same structure with TMM and Fresnel

    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import regular_pyramids, planar_surface
    from solcore import material
    from rayflare.options import default_options

    Si = material("Si")()
    Air = material("Air")()
    MgF2 = material("coverglass_JJ")()

    options = default_options()

    options.wavelength = np.linspace(300, 1150, 80) * 1e-9

    options.nx = 10
    options.ny = 10
    options.n_rays = 2000
    options.project_name = 'fdsf'

    options.pol = 's'

    planar_surf = planar_surface()
    pyramids = regular_pyramids(46, True)
    pyramids_rear = regular_pyramids(20, True)

    rt_strt = rt_structure(
        textures=[planar_surf, pyramids, pyramids_rear],
        materials=[MgF2, Si],
        widths=[10e-6, 50e-6],
        incidence=Air, transmission=Air,
        use_TMM=False,
    )

    RAT_Fresnel_f = rt_strt.calculate(options)

    planar_surf = planar_surface(analytical=True)
    pyramids = regular_pyramids(46, True, analytical=True)
    pyramids_rear = regular_pyramids(20, True)

    rt_strt = rt_structure(
        textures=[planar_surf, pyramids, pyramids_rear],
        materials=[MgF2, Si],
        widths=[10e-6, 50e-6],
        incidence=Air, transmission=Air,
        use_TMM=False,
    )

    RAT_Fresnel_a = rt_strt.calculate(options)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(options.wavelength * 1e9, RAT_Fresnel_f['R'], '-k', label="R Fresnel")
    plt.plot(options.wavelength * 1e9, RAT_Fresnel_f['A_per_layer'], '-r', label="A Fresnel")
    plt.plot(options.wavelength * 1e9, RAT_Fresnel_f['T'], '-b', label="T Fresnel")
    plt.plot(options.wavelength * 1e9, RAT_Fresnel_a['R'], '--k', label="R Fresnel")
    plt.plot(options.wavelength * 1e9, RAT_Fresnel_a['A_per_layer'], '--r', label="A Fresnel")
    plt.plot(options.wavelength * 1e9, RAT_Fresnel_a['T'], '--b', label="T Fresnel")

    plt.show()

def test_compare_TMM():
    pass

def test_integrated_A_Fresnel():
    pass

def test_integrated_A_TMM():
    pass

def test_lambertian_scattering():
    pass

def test_lambertian_scattering_integrated():
    pass

# should have a test to check is Is, thetas calculate to correct R and T

def test_phong_scattering():
    pass

