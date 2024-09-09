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

    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import regular_pyramids, planar_surface
    from solcore import material
    from rayflare.options import default_options

    Si = material("Si")()
    Air = material("Air")()
    MgF2 = material("MgF2")()

    options = default_options()

    options.wavelength = np.linspace(300, 1100, 6) * 1e-9

    options.nx = 20
    options.ny = 20
    options.n_rays = 1000
    options.randomize_surface = True
    options.depth_spacing_bulk = 10e-9

    options.pol = 's'

    top_angle = 50

    planar_surf = planar_surface(analytical=True)
    pyramids = regular_pyramids(top_angle, True, analytical=True)
    pyramids_rear = regular_pyramids(20, True)

    all_args = dict(
        materials=[MgF2, Si],
        widths=[1e-6, 20e-6],
        incidence=Air, transmission=Air,
        use_TMM=False,
    )

    rt_strt = rt_structure(
        textures=[planar_surf, pyramids, pyramids_rear],
        **all_args,
    )

    RAT_Fresnel_a = rt_strt.calculate(options)

    planar_surf = planar_surface()
    pyramids = regular_pyramids(top_angle, True)
    pyramids_rear = regular_pyramids(20, True)

    rt_strt = rt_structure(
        textures=[planar_surf, pyramids, pyramids_rear],
        **all_args,
    )

    RAT_Fresnel_f = rt_strt.calculate(options)

    assert RAT_Fresnel_a['R'] == approx(RAT_Fresnel_f['R'], rel=0.05, abs=0.05)
    assert RAT_Fresnel_a['T'] == approx(RAT_Fresnel_f['T'], rel=0.05, abs=0.05)
    assert RAT_Fresnel_a['A_per_layer'] == approx(RAT_Fresnel_f['A_per_layer'], rel=0.05, abs=0.05)
    assert RAT_Fresnel_a['R0'] == approx(RAT_Fresnel_f['R0'], rel=0.05, abs=0.05)

    anlt_profile = RAT_Fresnel_a['profile']
    full_profile = RAT_Fresnel_f['profile']

    full_profile = full_profile[anlt_profile > 1e-6]
    anlt_profile = anlt_profile[anlt_profile > 1e-6]

    assert full_profile == approx(anlt_profile, rel=0.05, abs=1e-5)

def test_compare_TMM():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import regular_pyramids, planar_surface
    from solcore import material
    from solcore.structure import Layer
    from rayflare.options import default_options

    Si = material("Si")()
    GaAs = material("GaAs")()
    Air = material("Air")()
    MgF2 = material("MgF2")()
    SiN = material("Si3N4")()

    options = default_options()

    options.wavelength = np.linspace(300, 1100, 6) * 1e-9

    options.nx = 20
    options.ny = 20
    options.n_rays = 1000
    options.randomize_surface = True
    options.depth_spacing_bulk = 10e-9
    options.project_name = 'test_analytical'

    options.pol = 's'

    top_angle = 50

    front_surf_layers = [Layer(70e-9, MgF2), Layer(500e-9, GaAs)]
    int_surf_layers = [Layer(70e-9, SiN)]

    planar_surf = planar_surface(analytical=True,
                                 interface_layers=front_surf_layers)
    pyramids = regular_pyramids(top_angle, True, analytical=True,
                                interface_layers=int_surf_layers)
    pyramids_rear = regular_pyramids(20, True)

    all_args = dict(
        materials=[GaAs, Si],
        widths=[5e-6, 20e-6],
        incidence=Air, transmission=Air,
        use_TMM=True,
        options=options,
        overwrite=True,
    )

    rt_strt = rt_structure(
        textures=[planar_surf, pyramids, pyramids_rear],
        **all_args,
    )

    RAT_Fresnel_a = rt_strt.calculate(options)

    planar_surf = planar_surface(interface_layers=front_surf_layers)
    pyramids = regular_pyramids(top_angle, True,
                                interface_layers=int_surf_layers)
    pyramids_rear = regular_pyramids(20, True)

    rt_strt = rt_structure(
        textures=[planar_surf, pyramids, pyramids_rear],
        **all_args,
    )

    RAT_Fresnel_f = rt_strt.calculate(options)

    assert RAT_Fresnel_a['R'] == approx(RAT_Fresnel_f['R'], rel=0.05, abs=0.05)
    assert RAT_Fresnel_a['T'] == approx(RAT_Fresnel_f['T'], rel=0.05, abs=0.05)
    assert RAT_Fresnel_a['A_per_layer'] == approx(RAT_Fresnel_f['A_per_layer'], rel=0.05, abs=0.05)
    assert RAT_Fresnel_a['R0'] == approx(RAT_Fresnel_f['R0'], rel=0.05, abs=0.05)

    anlt_profile = RAT_Fresnel_a['profile']
    full_profile = RAT_Fresnel_f['profile']

    full_profile = full_profile[anlt_profile > 1e-5]
    anlt_profile = anlt_profile[anlt_profile > 1e-5]

    assert full_profile == approx(anlt_profile, rel=0.15, abs=1e-5)

def test_lambertian_scattering():
    pass

def test_lambertian_scattering_integrated():
    pass

# should have a test to check is Is, thetas calculate to correct R and T

def test_phong_scattering():
    # phong scattering should give same result whether used with analytical or old ray-tracing.
    # NOTE: currently, phong scattering is only applied when transferring to next surface! So need
    # to do two surfaces to check.

    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import regular_pyramids, planar_surface
    from solcore import material
    from rayflare.options import default_options

    Si = material("Si")()
    Air = material("Air")()

    options = default_options()
    options.wavelength = np.linspace(300, 1150, 25)*1e-9
    options.pol = 's'

    alpha = 50*np.random.rand()
    x = np.linspace(0, np.pi/2, 30)
    # analytical power law:
    power_law = np.cos(x)**alpha

    x = x[power_law > 5e-3]

    options.nx = 10
    options.ny = 10
    options.n_rays = 2000

    planar_surf = planar_surface(phong=True, phong_options=[alpha, True], analytical=True)

    rt_strt_anlt = rt_structure(
        textures=[planar_surf, planar_surf],
        materials=[Si],
        widths=[10e-6],
        incidence=Air, transmission=Air,
        use_TMM=False,
        options=options,
    )

    RAT_a = rt_strt_anlt.calculate(options)

    planar_surf = planar_surface(phong=True, phong_options=[alpha, True], analytical=False)

    rt_strt_full = rt_structure(
        textures=[planar_surf, planar_surf],
        materials=[Si],
        widths=[10e-6],
        incidence=Air, transmission=Air,
        use_TMM=False,
        options=options,
    )

    RAT_f = rt_strt_full.calculate(options)

    # if there is transmission:
    for i in np.where(RAT_a['T'] > 0.4)[0]:

        R_thetas_a = RAT_a['thetas'][i][RAT_a['thetas'][i] > np.pi/2]
        R_thetas_f = RAT_f['thetas'][i][RAT_f['thetas'][i] > np.pi/2]

        R_thetas_a = np.pi - R_thetas_a
        R_thetas_f = np.pi - R_thetas_f

    # TODO: no actual tests!
