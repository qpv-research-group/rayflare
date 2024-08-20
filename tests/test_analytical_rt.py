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

def test_compare_Fresnel():
    pass

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
    import numpy as np
    from rayflare.textures import planar_surface
    from solcore import material
    from rayflare.options import default_options
    from solcore.structure import Layer
    from rayflare.ray_tracing import rt_structure

    options = default_options()
    wl = np.linspace(400, 400, 1) * 1e-9

    options.wavelength = wl

    options.nx = 50
    options.ny = 50
    options.n_rays = 5 * options.nx ** 2
    # options.x_limits = [2.5, 7.5]
    # options.y_limits = [2.5, 7.5]
    options.project_name = 'thin_textured_Si'
    options.lambertian_approximation = 0
    options.randomize_surface = True
    options.analytical_ray_tracing = 2
    options.theta_in = 20 * np.pi / 180
    options.phi_in = 0 * np.pi / 180
    options.I_thresh = 0.002
    options.parallel = True

    SiOx = material("SiO2")()
    Si = material("Si")()
    Air = material("Air")()
    Ag = material("Ag_Jiang")()
    SiN = material("Si3N4")()
    GaAs = material("GaAs")()
    glass = material("coverglass_JJ")()
    Ge = material("Ge")()

    optim_surf_mat = material('optim_surf_mat')()

    options.analytical_ray_tracing = 0

    layers = [Layer(70e-9, SiN), Layer(100e-9, GaAs), Layer(100e-9, GaAs)]
    layers = [Layer(20e-9, SiN)]

    glass_text = planar_surface(interface_layers=layers)

    glass_text[0].phong = True
    # glass_text[0].phong_options = [0.1, False, True]

    rtstr_text = rt_structure(
        textures=[glass_text],
        materials=[],
        widths=[],
        incidence=Air, transmission=Air,
        use_TMM=True, options=options
    )

    RAT = rtstr_text.calculate(options)

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.hist(RAT['thetas'][0], bins=50)
    # plt.show()
    #
    # plt.figure()
    # plt.hist(RAT['thetas'][0], bins=50, range=(0, 0.5))
    # plt.show()
    #
    # plt.figure()
    # plt.hist(RAT['phis'][0] , bins=50)
    # plt.show()
    # alpha = 20
    #
    # rndn = np.random.rand(5000)
    # arccos_dist = rndn ** (1 / (alpha + 1))
    # phongs = np.arccos(arccos_dist)


    # plt.figure()
    # plt.hist(rndn, bins=50, density=True)
    # # Trueplt.plot(np.linspace(0, np.pi / 2), np.cos(np.linspace(0, np.pi / 2)) ** (1 + alpha))
    # plt.show()
    #
    # x = np.linspace(0, np.pi/2, 200)
    # pdf = (np.cos(x))**(1+alpha)
    # int_pdf = np.trapz(pdf, x)
    #
    # maxval = np.max(phongs)
    #
    # plt.figure()
    # plt.plot(x, pdf/int_pdf)
    # plt.show()
