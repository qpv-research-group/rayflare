from pytest import approx, mark
import numpy as np
import sys

@mark.skipif(
    sys.platform == "win32",
    reason="S4 (RCWA) only installed for tests under Linux and macOS",
)
def test_tmm_rcwa_profile():

    from rayflare.utilities import make_absorption_function
    from rayflare.transfer_matrix_method import tmm_structure
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
    from rayflare.options import default_options

    from solcore import material, si
    from solcore.solar_cell import SolarCell, Layer, Junction
    from solcore.solar_cell_solver import solar_cell_solver
    from solcore.solar_cell_solver import default_options as defaults_solcore

    # setting up Solcore materials
    Air = material("Air")()
    Si = material("Si")()
    GaAs = material("GaAs")()

    wl = np.linspace(550, 1201, 10) * 1e-9

    # setting options
    options = default_options()
    options.wavelengths = wl
    options.depth_spacing = si("1nm")
    options.theta_in = 0
    options.orders = 2

    GaAs_total_d = si("3um")
    Si_total_d = si("300um")

    options.coherent = False
    options.coherency_list = ["c", "i"]

    tmmstr = tmm_structure(
        [Layer(GaAs_total_d, GaAs), Layer(Si_total_d, Si)],
        incidence=Air,
        transmission=Air,
    )
    result_tmm = tmmstr.calculate_profile(options)

    rcwastr = rcwa_structure(
        [Layer(GaAs_total_d, GaAs), Layer(Si_total_d, Si)],
        incidence=Air,
        transmission=Air,
        size=((50, 0), (0,50)),
        options=options,
    )
    result_rcwa = rcwastr.calculate_profile(options)

    total_R_tmm = result_tmm["R"]

    Si_SC = material("Si")
    GaAs_SC = material("GaAs")
    T = 300

    n_material_GaAs_width = si("300nm")
    p_material_GaAs_width = tmmstr.layer_stack.widths[0] * 1e-9 - n_material_GaAs_width

    n_material_GaAs = GaAs_SC(
        Nd=si(3e18, "cm-3"),
        hole_diffusion_length=si("400nm"),
        electron_mobility=50e-4,
        relative_permittivity=12.4,
    )
    p_material_GaAs = GaAs_SC(
        Na=si(1e17, "cm-3"),
        electron_diffusion_length=si("1um"),
        electron_mobility=100e-4,
        relative_permittivity=12.4,
    )

    n_material_Si_width = si("500nm")
    p_material_Si_width = tmmstr.layer_stack.widths[1] * 1e-9 - n_material_Si_width

    n_material_Si = Si_SC(
        T=T,
        Nd=si(1e21, "cm-3"),
        hole_diffusion_length=si("10um"),
        electron_mobility=50e-4,
        relative_permittivity=11.68,
    )
    p_material_Si = Si_SC(
        T=T,
        Na=si(1e16, "cm-3"),
        electron_diffusion_length=si("290um"),
        hole_mobility=400e-4,
        relative_permittivity=11.68,
    )

    options_sc = defaults_solcore
    options_sc.optics_method = "external"
    options_sc.position = np.arange(0, tmmstr.width, options.depth_spacing)
    options_sc.wavelength = wl
    options_sc.theta = options.theta_in * 180 / np.pi
    options_sc.coherency_list = ["c", "c", "i", "i"]
    options_sc.no_back_reflection = False
    V = np.linspace(0, 2, 200)
    options_sc.voltages = V

    pos_tmm, diff_absorb_fn = make_absorption_function(
        result_tmm, tmmstr, options
    )

    pos_rcwa, diff_absorb_fn_rcwa = make_absorption_function(
        result_rcwa, rcwastr, options
    )

    regen_tmm = diff_absorb_fn(pos_tmm)
    regen_rcwa = diff_absorb_fn_rcwa(pos_rcwa)

    assert np.all(pos_tmm == pos_rcwa)
    assert regen_tmm[regen_tmm > 1e4] == approx(regen_rcwa[regen_tmm > 1e4], rel=0.03)
    # incoherent layer so won't be exactly the same at all wavelengths/positions,
    # but this doesn't matter for highly absorbing layers

    solar_cell = SolarCell(
        [
            Junction(
                [
                    Layer(
                        width=n_material_GaAs_width,
                        material=n_material_GaAs,
                        role="emitter",
                    ),
                    Layer(
                        width=p_material_GaAs_width,
                        material=p_material_GaAs,
                        role="base",
                    ),
                ],
                sn=2,
                sp=2,
                kind="DA",
            ),
            Junction(
                [
                    Layer(
                        width=n_material_Si_width,
                        material=n_material_Si,
                        role="emitter",
                    ),
                    Layer(
                        width=p_material_Si_width, material=p_material_Si, role="base"
                    ),
                ],
                sn=1,
                sp=1,
                kind="DA",
            ),
        ],
        external_reflected=total_R_tmm,
        external_absorbed=diff_absorb_fn,
    )

    solar_cell_solver(solar_cell, "qe", options_sc)

    solar_cell_SC = SolarCell(
        [
            Junction(
                [
                    Layer(
                        width=n_material_GaAs_width,
                        material=n_material_GaAs,
                        role="emitter",
                    ),
                    Layer(
                        width=p_material_GaAs_width,
                        material=p_material_GaAs,
                        role="base",
                    ),
                ],
                sn=2,
                sp=2,
                kind="DA",
            ),
            Junction(
                [
                    Layer(
                        width=n_material_Si_width,
                        material=n_material_Si,
                        role="emitter",
                    ),
                    Layer(
                        width=p_material_Si_width, material=p_material_Si, role="base"
                    ),
                ],
                sn=1,
                sp=1,
                kind="DA",
            ),
        ],
        substrate=Air,
    )

    options_sc.optics_method = "TMM"

    solar_cell_solver(solar_cell_SC, "qe", options_sc)

    assert solar_cell_SC[0].qe["EQE"] == approx(solar_cell[0].qe["EQE"], abs=1e-2)
    assert solar_cell_SC[1].qe["EQE"] == approx(solar_cell[1].qe["EQE"], abs=1e-3)


def test_matrix_method_profile():

    from rayflare.textures import regular_pyramids
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import calculate_RAT, process_structure
    from rayflare.options import default_options
    from rayflare.angles import make_angle_vector
    from rayflare.utilities import make_absorption_function

    from solcore import material, si
    from solcore.solar_cell import Layer

    wavelengths = np.linspace(250, 1200, 6) * 1e-9

    options = default_options()
    options.wavelengths = wavelengths
    options.project_name = "GaAs_GaAs_Si_test"
    options.n_theta_bins = 20
    options.nx = 5
    options.ny = 5
    options.depth_spacing = si("1nm")
    options.depth_spacing_bulk = si("10um")
    _, _, angle_vector = make_angle_vector(
        options["n_theta_bins"], options["phi_symmetry"], options["c_azimuth"]
    )
    options.bulk_profile = True
    options.n_rays = options.nx**2 * int(len(angle_vector) / 2)

    Air = material("Air")()
    Al2O3 = material("Al2O3")()
    Ag = material("Ag")()
    GaAs = material("GaAs")()
    InAlP = material("AlInP")(Al=0.5)
    GaInP = material("GaInP")(In=0.5)
    Si = material("Si")()
    MgF2 = material("MgF2")()
    Ta2O5 = material("TAOX1")()

    GaAs_1_th = 120e-9
    GaAs_2_th = 1200e-9

    front_materials = [
        Layer(50e-9, MgF2),
        Layer(40e-9, Ta2O5),
        Layer(30e-9, GaInP),
        Layer(GaAs_1_th, GaAs),
        Layer(30e-9, InAlP),
        Layer(20e-9, GaAs),
        Layer(30e-9, GaInP),
        Layer(GaAs_2_th, GaAs),
        Layer(30e-9, InAlP),
        Layer(100e-9, GaAs),
    ]
    back_materials = [Layer(62e-9, Ag), Layer(240e-9, Al2O3)]

    surf_back = regular_pyramids(elevation_angle=55, upright=False)

    front_surf = Interface(
        "TMM",
        layers=front_materials,
        name="GaAs_GaAs",
        coherent=True,
        prof_layers=np.arange(1, len(front_materials) + 1),
    )
    back_surf = Interface(
        "RT_TMM",
        texture=surf_back,
        layers=back_materials,
        name="Si_HIT_rear",
        prof_layers=np.arange(1, len(back_materials) + 1),
        coherent=True,
    )

    bulk_Si = BulkLayer(250e-6, Si, name="Si_bulk")  # bulk thickness in m

    SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results = calculate_RAT(SC, options)

    widths_front = front_surf.widths
    width_bulk = bulk_Si.width
    widths_back = back_surf.widths

    positions, absorb_fn = make_absorption_function(
        results, SC, options
    )

    pos_front = np.arange(
        0, np.sum(widths_front) * 1e9 - 1e-12, options.depth_spacing * 1e9
    )
    pos_bulk = np.sum(widths_front) * 1e6 + np.arange(
        0, width_bulk * 1e6 - 1e-12, options.depth_spacing_bulk * 1e6
    )
    pos_back = (
        np.sum(widths_front) * 1e9
        + width_bulk * 1e9
        + np.arange(0, np.sum(widths_back) * 1e9, options.depth_spacing * 1e9)
    )

    positions_expected = np.hstack((pos_front / 1e9, pos_bulk / 1e6, pos_back / 1e9))

    generated = absorb_fn(positions_expected)

    profile_front = results[2][0]
    profile_Si = results[3][0]
    profile_back = results[2][1]

    generated_expected = np.hstack(
        (
            profile_front[:, : len(pos_front)] * 1e9,
            profile_Si[:, : len(pos_bulk)],
            profile_back[:, : len(pos_back)] * 1e9,
        )
    )

    # Do one for a selection of layers, one for all layers profile calculated

    assert positions == approx(positions_expected)
    assert generated == approx(generated_expected, abs=1e-4)


def test_rt_tmm_profile():

    from rayflare.textures import regular_pyramids, planar_surface
    from rayflare.options import default_options
    from rayflare.ray_tracing import rt_structure

    from solcore import material, si
    from rayflare.utilities import make_absorption_function

    from solcore.solar_cell import SolarCell, Layer, Junction
    from solcore.solar_cell_solver import solar_cell_solver
    from solcore.solar_cell_solver import default_options as defaults_solcore

    wavelengths = np.linspace(500, 1200, 6) * 1e-9

    options = default_options()
    options.nx = 10
    options.ny = 10
    options.n_rays = 500
    options.randomize_surface = True
    options.project_name = "GaInP_GaAs_Si_spacer_tmm_rt_test"
    options.wavelengths = wavelengths

    GaAs = material("GaAs")
    GaAs_opt = GaAs()
    Si = material("Si")
    Si_opt = Si()
    Al = material("Al")()
    Air = material("Air")()
    epoxy = material("BK7")()

    GaInP = material("GaInP")(In=0.49)

    ITO = material("ITO2")()

    w1 = 50e-9
    w2 = 120e-9
    d1 = 1000e-9
    d2 = 2000e-9

    d_epoxy = 200e-6
    d_Si = 100e-6

    front_layers = [
        Layer(w1, GaInP),  # tunnel junction/cell 1 window layer
        Layer(d1, GaAs_opt),  # cell 1
        Layer(w2, GaInP),  # tunnel junction/cell 2 window layer
        Layer(d2, GaAs_opt),  # cell 2
        Layer(w2, GaInP),  # cell 2 BSF
    ]

    cell_layer_ind = [2, 4]

    Si_back_layers = [Layer(500e-9, Si_opt), Layer(240e-9, ITO)]


    front_surface = planar_surface(
        interface_layers=front_layers, prof_layers=cell_layer_ind
    )
    Si_surface = regular_pyramids(upright=True) # Fresnel instead of TMM

    back_surface = regular_pyramids(upright=False, interface_layers=Si_back_layers,
                                    prof_layers=[1])

    rt_str = rt_structure(
        textures=[front_surface, Si_surface, back_surface],
        materials=[epoxy, Si_opt],
        widths=[d_epoxy, d_Si],
        incidence=Air,
        transmission=Al,
        use_TMM=True,
        options=options,
    )

    result = rt_str.calculate(options)

    positions, absorb_func = make_absorption_function(result, rt_str, options)

    n_material_GaAs = GaAs(
        Nd=si(3e18, "cm-3"),
        hole_diffusion_length=si("400nm"),
        electron_mobility=50e-4,
        relative_permittivity=12.4,
    )
    p_material_GaAs = GaAs(
        Na=si(1e17, "cm-3"),
        electron_diffusion_length=si("1um"),
        electron_mobility=100e-4,
        relative_permittivity=12.4,
    )

    GaAs_junc_1 = Junction(
                [Layer(
                        width=d1 / 4,
                        material=n_material_GaAs,
                        role="emitter"),
                Layer(
                        width=3*d1 / 4,
                        material=p_material_GaAs,
                        role="base")], sn=2, sp=2, kind="DA")

    GaAs_junc_2 = Junction(
                [Layer(
                        width=d2 / 4,
                        material=n_material_GaAs,
                        role="emitter"),
                Layer(
                        width=3*d2 / 4,
                        material=p_material_GaAs,
                        role="base")], sn=2, sp=2, kind="DA")

    n_material_Si_width = si("500nm")
    p_material_Si_width = d_Si - n_material_Si_width

    n_material_Si = Si(
        Nd=si(1e21, "cm-3"),
        hole_diffusion_length=si("10um"),
        electron_mobility=50e-4,
        relative_permittivity=11.68,
    )
    p_material_Si = Si(
        Na=si(1e16, "cm-3"),
        electron_diffusion_length=si("290um"),
        hole_mobility=400e-4,
        relative_permittivity=11.68,
    )

    Si_junc = Junction(
                [
                    Layer(
                        width=n_material_Si_width,
                        material=n_material_Si,
                        role="emitter",
                    ),
                    Layer(
                        width=p_material_Si_width, material=p_material_Si, role="base"
                    ),
                ],
                sn=1,
                sp=1,
                kind="DA",
            )

    solar_cell = SolarCell(
        [
            Layer(w1, GaInP),
            GaAs_junc_1,
            Layer(w2, GaInP),
            GaAs_junc_2,
            Layer(w2, GaInP),
            Layer(d_epoxy, epoxy),
            Si_junc,
            Layer(500e-9, Si),
            Layer(240e-9, ITO)
        ],
        external_reflected=result["R"],
        external_absorbed=absorb_func,
    )

    options_sc = defaults_solcore
    options_sc.optics_method = "external"
    options_sc.position = positions
    options_sc.wavelength = wavelengths

    solar_cell_solver(solar_cell, "qe", options_sc)

    GaAs_1 = result["A_per_interface"][0][:,1]
    GaAs_2 = result["A_per_interface"][0][:,3]
    Si_abs = result["A_per_layer"][:,1]

    assert np.all(solar_cell[1].qe["EQE"] < 1)
    assert np.all(solar_cell[3].qe["EQE"] < 1)
    assert np.all(solar_cell[6].qe["EQE"] < 1)

    assert np.all(solar_cell[1].qe["EQE"] >= 0)
    assert np.all(solar_cell[3].qe["EQE"] >= 0)
    assert np.all(solar_cell[6].qe["EQE"] >= 0)

    assert np.all(solar_cell[1].qe["EQE"] <= GaAs_1)
    assert np.all(solar_cell[3].qe["EQE"] <= GaAs_2)
    assert np.all(solar_cell[6].qe["EQE"] <= Si_abs)
