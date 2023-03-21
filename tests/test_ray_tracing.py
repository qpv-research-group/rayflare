from pytest import approx
import numpy as np
import itertools


def test_parallel():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import regular_pyramids
    from rayflare.options import default_options
    from solcore import material
    from solcore import si

    Air = material("Air")()
    Si = material("Si")()
    GaAs = material("GaAs")()
    Ge = material("Ge")()

    triangle_surf = regular_pyramids(30)

    options = default_options()

    options.wavelengths = np.linspace(700, 1700, 5) * 1e-9
    options.theta_in = 45 * np.pi / 180
    options.nx = 5
    options.ny = 5
    options.pol = "p"
    options.n_rays = 3000
    options.depth_spacing = 1e-6
    options.parallel = True

    rtstr = rt_structure(
        textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
        materials=[GaAs, Si, Ge],
        widths=[si("100um"), si("70um"), si("50um")],
        incidence=Air,
        transmission=Air,
    )
    result_new = rtstr.calculate(options)

    options.parallel = False

    rtstr = rt_structure(
        textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
        materials=[GaAs, Si, Ge],
        widths=[si("100um"), si("70um"), si("50um")],
        incidence=Air,
        transmission=Air,
    )

    result_old = rtstr.calculate(options)

    assert sorted(list(result_old.keys())) == sorted(list(result_new.keys()))

    rel_error = 0.3  # Monte Carlo (random) simulations so will never be exactly equal)
    assert result_new["R"] == approx(result_old["R"], rel=rel_error)
    assert result_new["R0"] == approx(result_old["R0"], rel=rel_error)
    assert result_new["A_per_layer"] == approx(result_old["A_per_layer"], rel=rel_error)
    assert result_new["profile"] == approx(result_old["profile"], rel=rel_error)
    assert np.nanmean(result_new["thetas"], 1) == approx(
        np.nanmean(result_old["thetas"], 1), rel=rel_error
    )
    # assert np.nanmean(result_new['phis'], 1) == approx(np.nanmean(result_old['phis'], 1), rel=rel_error)
    assert np.nanmean(result_new["n_interactions"], 1) == approx(
        np.nanmean(result_old["n_interactions"], 1), rel=rel_error
    )
    assert np.nanmean(result_new["n_passes"], 1) == approx(
        np.nanmean(result_old["n_passes"], 1), rel=rel_error
    )
    assert result_new["R"] + result_new["T"] + np.sum(
        result_new["A_per_layer"], 1
    ) == approx(1, rel=options.I_thresh)
    assert result_old["R"] + result_old["T"] + np.sum(
        result_old["A_per_layer"], 1
    ) == approx(1, rel=options.I_thresh)


def test_flip():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import regular_pyramids
    from rayflare.options import default_options
    from solcore import material
    from solcore import si

    Air = material("Air")()
    Si = material("Si")()
    GaAs = material("GaAs")()
    Ge = material("Ge")()

    triangle_surf = regular_pyramids(70)

    options = default_options()

    options.wavelengths = np.linspace(700, 1700, 8) * 1e-9
    options.nx = 5
    options.ny = 5
    options.pol = "s"
    options.n_rays = 200
    options.parallel = True

    rtstr_1 = rt_structure(
        textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
        materials=[GaAs, Si, Ge],
        widths=[si("100um"), si("70um"), si("50um")],
        incidence=Air,
        transmission=Air,
    )
    result_up = rtstr_1.calculate(options)

    options.initial_direction = -1
    options.initial_material = 4
    triangle_surf = regular_pyramids(70, upright=False)

    rtstr_2 = rt_structure(
        textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
        materials=[Ge, Si, GaAs],
        widths=[si("50um"), si("70um"), si("100um")],
        incidence=Air,
        transmission=Air,
    )

    result_down = rtstr_2.calculate(options)

    abs_error = 0.1

    assert result_up["R"] + result_up["T"] + np.sum(
        result_up["A_per_layer"], 1
    ) == approx(1, rel=options.I_thresh)
    assert result_down["R"] + result_down["T"] + np.sum(
        result_down["A_per_layer"], 1
    ) == approx(1, rel=options.I_thresh)

    assert result_up["T"] == approx(result_down["R"], abs=abs_error)
    assert result_up["A_per_layer"][:, 0] == approx(
        result_down["A_per_layer"][:, 2], abs=abs_error
    )
    assert result_up["A_per_layer"][:, 1] == approx(
        result_down["A_per_layer"][:, 1], abs=abs_error
    )
    assert result_up["A_per_layer"][:, 2] == approx(
        result_down["A_per_layer"][:, 0], abs=abs_error
    )


def test_periodic():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import regular_pyramids
    from rayflare.options import default_options
    from solcore import material

    Air = material("Air")()
    Si = material("Si")()

    triangle_surf = regular_pyramids(60)

    options = default_options()

    options.wavelengths = np.linspace(700, 1700, 8) * 1e-9
    options.nx = 5
    options.ny = 5
    options.pol = "s"
    options.n_rays = 200
    options.parallel = True
    options.periodic = True

    rtstr_1 = rt_structure(
        textures=[triangle_surf],
        materials=[],
        widths=[],
        incidence=Air,
        transmission=Si,
    )
    result_periodic = rtstr_1.calculate(options)

    options.periodic = False

    rtstr_2 = rt_structure(
        textures=[triangle_surf],
        materials=[],
        widths=[],
        incidence=Air,
        transmission=Si,
    )

    result_single = rtstr_2.calculate(options)

    assert result_periodic["R"] + result_periodic["T"] + np.sum(
        result_periodic["A_per_layer"], 1
    ) == approx(1, rel=options.I_thresh)
    assert result_single["R"] + result_single["T"] + np.sum(
        result_single["A_per_layer"], 1
    ) == approx(1, rel=options.I_thresh)
    assert np.all(result_periodic["R"] >= result_single["R"])
    assert np.all(result_single["R"] == 0)


def test_interface_absorption():

    from rayflare.ray_tracing import rt_structure
    from solcore import material
    from solcore.structure import Layer
    from rayflare.options import default_options
    from rayflare.textures import planar_surface

    opts = default_options()
    opts.project_name = "method_comparison_test_prof"
    opts.n_rays = 1000
    opts.parallel = False
    # opts.nx = 5
    # opts.ny = 5

    thetas_in = np.linspace(0, 1.2, 3)
    pol_in = ["s", "p", "u"]

    d_Ge = 50e-6
    d_Si = 30e-6

    GaAs = material("GaAs")()
    Air = material("Air")()
    Si = material("Si")()
    Ge = material("Ge")()
    MgF2 = material("MgF2")()
    Ta2O5 = material("TaOx1")()
    ITO = material("ITO2")()
    Ag = material("Ag")()

    front_layers = [Layer(100e-9, MgF2), Layer(50e-9, Ta2O5), Layer(502e-9, GaAs)]
    back_layers = [Layer(100e-9, ITO), Layer(50e-9, Ag)]

    front_surf = planar_surface(
        interface_layers=front_layers, prof_layers=[3]
    )  # pyramid size in microns
    back_surf = planar_surface(
        interface_layers=back_layers, prof_layers=[1, 2]
    )  # pyramid size in microns
    middle_surf = planar_surface()

    opts.wavelengths = np.linspace(300, 1900, 5) * 1e-9
    opts.depth_spacing = 1e-9
    opts.depth_spacing_bulk = 1e-8
    # opts.parallel = False

    rtstr = rt_structure(
        [front_surf, middle_surf, back_surf],
        [Si, Ge],
        [d_Si, d_Ge],
        Air,
        Air,
        opts,
        use_TMM=True,
    )
    # import matplotlib.pyplot as plt

    for angle_pol in itertools.product(thetas_in, pol_in):
        print(angle_pol)
        opts.theta_in = angle_pol[0]
        opts.pol_in = angle_pol[1]
        rt_res = rtstr.calculate(opts)

        total_A = np.concatenate(
            (
                rt_res["A_per_layer"],
                rt_res["A_per_interface"][0],
                rt_res["A_per_interface"][2],
                rt_res["R"][:, None],
                rt_res["T"][:, None],
            ),
            axis=1,
        )

        prof_int_front = np.trapz(rt_res["interface_profiles"][0], dx=1, axis=1)
        prof_int_back = np.trapz(rt_res["interface_profiles"][2], dx=1, axis=1)

        prof_int_bulk = np.trapz(rt_res["profile"], dx=10, axis=1)

        assert prof_int_bulk == approx(np.sum(rt_res["A_per_layer"], 1), abs=0.01)
        assert prof_int_front == approx(
            np.sum(rt_res["A_per_interface"][0], 1), abs=0.01
        )
        assert prof_int_back == approx(
            np.sum(rt_res["A_per_interface"][2], 1), abs=0.01
        )

        # plt.figure()
        # plt.plot(opts.wavelengths*1e9, total_A)
        # plt.legend(["Si", "Ge", "MGF2", "Ta2O5", "GaAs", "ITO", "Ag", "R", "T"])
        # plt.show()
        #
        # plt.figure()
        # plt.plot(rt_res["interface_profiles"][0].T)
        # plt.show()
        #
        # plt.figure()
        # plt.plot(rt_res["interface_profiles"][2].T)
        # plt.show()

        assert np.sum(total_A, 1) == approx(1, abs=opts.I_thresh)


def test_random_position():
    from rayflare.ray_tracing import rt_structure

    from rayflare.textures import regular_pyramids
    from solcore import material

    from rayflare.options import default_options

    options = default_options()
    options.n_rays = 20000
    options.nx = 50
    options.ny = 50

    options.wavelengths = np.array([500e-9])

    Si = material("Si")()
    Air = material("Air")()

    triangle_surf = regular_pyramids()

    rtstr_1 = rt_structure([triangle_surf], [], [], Air, Si)

    options.random_ray_position = False

    res_regular = rtstr_1.calculate(options)

    options.random_ray_position = True

    res_random = rtstr_1.calculate(options)

    assert res_regular["R"] == approx(res_random["R"], rel=0.03)
    assert res_regular["T"] == approx(res_random["T"], rel=0.03)
    assert np.mean(res_regular["n_interactions"]) == approx(
        np.mean(res_random["n_interactions"]), rel=0.03
    )


def test_inverted():
    from rayflare.ray_tracing import rt_structure

    from rayflare.textures import regular_pyramids
    from solcore import material

    from rayflare.options import default_options

    options = default_options()
    options.n_rays = 20000
    options.nx = 50
    options.ny = 50

    options.wavelengths = np.array([500e-9])

    Si = material("Si")()
    Air = material("Air")()

    triangle_surf = regular_pyramids(elevation_angle=40, upright=True)
    triangle_surf_inverted = regular_pyramids(elevation_angle=40, upright=False)

    rtstr_1 = rt_structure([triangle_surf], [], [], Air, Si)


    res_upright = rtstr_1.calculate(options)

    rtstr_2 = rt_structure([triangle_surf_inverted], [], [], Si, Air)

    options.initial_material = 1
    options.initial_direction = -1

    res_inverted = rtstr_2.calculate(options)

    assert res_inverted["R"] == approx(res_upright["T"], rel=0.03)
    assert res_inverted["T"] == approx(res_upright["R"], rel=0.03)
    assert np.mean(res_inverted["n_interactions"]) == approx(
        np.mean(res_upright["n_interactions"]), rel=0.03
    )
