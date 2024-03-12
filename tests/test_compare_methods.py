from pytest import approx, mark
import numpy as np
import sys
import itertools
from .conftest import skip_s4_test

@mark.skipif(skip_s4_test(), reason="Only works if S4 installed")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_tmm_rcwa_structure_comparison(RCWA_method):
    from solcore import si, material
    from solcore.structure import Layer
    from solcore.solar_cell import SolarCell

    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
    from rayflare.transfer_matrix_method import tmm_structure
    from rayflare.options import default_options

    InGaP = material("GaInP")(In=0.5)
    GaAs = material("GaAs")()
    Ge = material("Ge")()
    Ag = material("Ag")()
    Air = material("Air")()

    Al2O3 = material("Al2O3")()

    wavelengths = np.linspace(250, 1900, 500) * 1e-9

    options = default_options()

    options.wavelengths = wavelengths
    options.orders = 2
    options.RCWA_method = RCWA_method

    size = ((100, 0), (0, 100))

    # anti-reflection coating
    ARC = [Layer(si("80nm"), Al2O3)]

    solar_cell = SolarCell(
        ARC
        + [
            Layer(material=InGaP, width=si("400nm")),
            Layer(material=GaAs, width=si("4000nm")),
            Layer(material=Ge, width=si("3000nm")),
        ],
        substrate=Ag,
    )

    rcwa_setup = rcwa_structure(solar_cell, size=size, options=options, incidence=Air, transmission=Ag)
    tmm_setup = tmm_structure(solar_cell, incidence=Air, transmission=Ag, no_back_reflection=False)

    for pol in ["s", "p", "u"]:
        for angle in [0, np.pi / 3]:
            options["pol"] = pol
            options["theta_in"] = angle

            rcwa_result = rcwa_setup.calculate(options)
            tmm_result = tmm_setup.calculate(options)

            assert tmm_result["A_per_layer"] == approx(rcwa_result["A_per_layer"])
            assert tmm_result["R"] == approx(rcwa_result["R"])
            assert tmm_result["T"] == approx(rcwa_result["T"])

            assert np.sum(tmm_result["A_per_layer"], 1) + tmm_result["R"] + tmm_result["T"] == approx(1)
            assert np.sum(rcwa_result["A_per_layer"], 1) + rcwa_result["R"] + rcwa_result["T"] == approx(1)


@mark.skipif(skip_s4_test(), reason="Only works if S4 installed")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_planar_structure(RCWA_method):
    # solcore imports
    from solcore.structure import Layer
    from solcore import material
    from solcore.absorption_calculator import calculate_rat, OptiStack

    # rayflare imports
    from rayflare.textures.standard_rt_textures import planar_surface
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options

    # Thickness of bottom Ge layer
    bulkthick = 300e-6

    wavelengths = np.linspace(300, 1850, 50) * 1e-9

    # set options
    options = default_options()
    options.wavelengths = wavelengths
    options.project_name = "method_comparison_test"
    options.n_rays = 250
    options.n_theta_bins = 3
    options.lookuptable_angles = 100
    options.parallel = True
    options.c_azimuth = 0.001
    options.I_thresh = 1e-8
    options.bulk_profile = False
    options.RCWA_method = RCWA_method

    # set up Solcore materials
    Ge = material("Ge")()
    GaAs = material("GaAs")()
    GaInP = material("GaInP")(In=0.5)
    Ag = material("Ag")()
    SiN = material("Si3N4")()
    Air = material("Air")()
    Ta2O5 = material("TaOx1")()  # Ta2O5 (SOPRA database)
    MgF2 = material("MgF2")()  # MgF2 (SOPRA database)

    front_materials = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5), Layer(464e-9, GaInP), Layer(1682e-9, GaAs)]
    back_materials = [Layer(100e-9, SiN)]

    # TMM, matrix framework

    front_surf = Interface("TMM", layers=front_materials, name="GaInP_GaAs_TMM", coherent=True)
    back_surf = Interface("TMM", layers=back_materials, name="SiN_Ag_TMM", coherent=True)

    bulk_Ge = BulkLayer(bulkthick, Ge, name="Ge_bulk")  # bulk thickness in m

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_TMM_Matrix = calculate_RAT(SC, options)

    results_per_pass_TMM_matrix = results_TMM_Matrix[1]

    results_per_layer_front_TMM_matrix = np.sum(results_per_pass_TMM_matrix["a"][0], 0)

    ## RT with TMM lookup tables

    surf = planar_surface()  # [texture, flipped texture]

    front_surf = Interface("RT_TMM", layers=front_materials, texture=surf, name="GaInP_GaAs_RT", coherent=True)
    back_surf = Interface("RT_TMM", layers=back_materials, texture=surf, name="SiN_Ag_RT_50k", coherent=True)

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_RT = calculate_RAT(SC, options)

    results_per_pass_RT = results_RT[1]

    # only select absorbing layers, sum over passes
    results_per_layer_front_RT = np.sum(results_per_pass_RT["a"][0], 0)

    ## RCWA

    front_surf = Interface(
        "RCWA",
        layers=front_materials,
        name="GaInP_GaAs_RCWA",
        coherent=True,
        d_vectors=((500, 0), (0, 500)),
        rcwa_orders=2,
    )
    back_surf = Interface(
        "RCWA", layers=back_materials, name="SiN_Ag_RCWA", coherent=True, d_vectors=((500, 0), (0, 500)), rcwa_orders=2
    )

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_RCWA_Matrix = calculate_RAT(SC, options)

    results_per_pass_RCWA = results_RCWA_Matrix[1]

    # only select absorbing layers, sum over passes
    results_per_layer_front_RCWA = np.sum(results_per_pass_RCWA["a"][0], 0)

    ## pure TMM (from Solcore)
    all_layers = front_materials + [Layer(bulkthick, Ge)] + back_materials

    coh_list = len(front_materials) * ["c"] + ["i"] + ["c"]

    OS_layers = OptiStack(all_layers, substrate=Ag, no_back_reflection=False)

    TMM_res = calculate_rat(
        OS_layers,
        wavelength=wavelengths * 1e9,
        no_back_reflection=False,
        angle=options["theta_in"] * 180 / np.pi,
        coherent=False,
        coherency_list=coh_list,
        pol=options["pol"],
    )

    # stack results for comparison
    TMM_reference = TMM_res["A_per_layer"][1:-2].T
    TMM_matrix = np.hstack((results_per_layer_front_TMM_matrix, results_TMM_Matrix[0].A_bulk[0].data[:, None]))
    RCWA_matrix = np.hstack((results_per_layer_front_RCWA, results_RCWA_Matrix[0].A_bulk[0].data[:, None]))
    RT_matrix = np.hstack((results_per_layer_front_RT, results_RT[0].A_bulk[0].data[:, None]))

    assert TMM_reference == approx(TMM_matrix, abs=0.02)
    assert TMM_reference == approx(RCWA_matrix, abs=0.02)
    assert TMM_reference == approx(RT_matrix, abs=0.23)

    # check normalization

    assert (
        results_TMM_Matrix[0].R[0]
        + results_TMM_Matrix[0].T[0]
        + np.sum(results_per_layer_front_TMM_matrix, 1)
        + results_TMM_Matrix[0].A_bulk[0]
    ).data == approx(1)
    assert (
        results_RCWA_Matrix[0].R[0]
        + results_RCWA_Matrix[0].T[0]
        + np.sum(results_per_layer_front_RCWA, 1)
        + results_RCWA_Matrix[0].A_bulk[0]
    ).data == approx(1)
    assert (
        results_RT[0].R[0] + results_RT[0].T[0] + np.sum(results_per_layer_front_RT, 1) + results_RT[0].A_bulk[0]
    ).data == approx(1)


@mark.skipif(skip_s4_test(), reason="Only works if S4 installed")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_planar_structure_45deg(RCWA_method):
    # solcore imports
    from solcore.structure import Layer
    from solcore import material
    from solcore.absorption_calculator import calculate_rat, OptiStack

    # rayflare imports
    from rayflare.textures.standard_rt_textures import planar_surface
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options

    # Thickness of bottom Ge layer
    bulkthick = 300e-6

    wavelengths = np.linspace(300, 1850, 30) * 1e-9

    # set options
    options = default_options()
    options.wavelengths = wavelengths
    options.project_name = "method_comparison_test_45deg"
    options.n_rays = 500
    options.n_theta_bins = 20
    options.lookuptable_angles = 100
    options.parallel = True
    options.c_azimuth = 0.001
    options.theta_in = 0.6 * np.pi / 2
    options.nx = 1
    options.ny = 1
    options.I_thresh = 1e-8
    options.bulk_profile = False
    options.RCWA_method = RCWA_method

    # set up Solcore materials
    Ge = material("Ge")()
    GaAs = material("GaAs")()
    GaInP = material("GaInP")(In=0.5)
    Ag = material("Ag")()
    SiN = material("Si3N4")()
    Air = material("Air")()
    Ta2O5 = material("TaOx1")()  # Ta2O5 (SOPRA database)
    MgF2 = material("MgF2")()  # MgF2 (SOPRA database)

    front_materials = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5), Layer(464e-9, GaInP), Layer(1682e-9, GaAs)]
    back_materials = [Layer(100e-9, SiN)]

    # TMM, matrix framework

    front_surf = Interface("TMM", layers=front_materials, name="GaInP_GaAs_TMM", coherent=True)
    back_surf = Interface("TMM", layers=back_materials, name="SiN_Ag_TMM", coherent=True)

    bulk_Ge = BulkLayer(bulkthick, Ge, name="Ge_bulk")  # bulk thickness in m

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_TMM_Matrix = calculate_RAT(SC, options)

    results_per_pass_TMM_matrix = results_TMM_Matrix[1]

    results_per_layer_front_TMM_matrix = np.sum(results_per_pass_TMM_matrix["a"][0], 0)

    ## RT with TMM lookup tables

    surf = planar_surface()  # [texture, flipped texture]

    front_surf = Interface("RT_TMM", layers=front_materials, texture=surf, name="GaInP_GaAs_RT", coherent=True)
    back_surf = Interface("RT_TMM", layers=back_materials, texture=surf, name="SiN_Ag_RT_50k", coherent=True)

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_RT = calculate_RAT(SC, options)

    results_per_pass_RT = results_RT[1]

    # only select absorbing layers, sum over passes
    results_per_layer_front_RT = np.sum(results_per_pass_RT["a"][0], 0)

    ## RCWA

    front_surf = Interface(
        "RCWA",
        layers=front_materials,
        name="GaInP_GaAs_RCWA",
        coherent=True,
        d_vectors=((500, 0), (0, 500)),
        rcwa_orders=2,
    )
    back_surf = Interface(
        "RCWA", layers=back_materials, name="SiN_Ag_RCWA", coherent=True, d_vectors=((500, 0), (0, 500)), rcwa_orders=2
    )

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_RCWA_Matrix = calculate_RAT(SC, options)

    results_per_pass_RCWA = results_RCWA_Matrix[1]

    # only select absorbing layers, sum over passes
    results_per_layer_front_RCWA = np.sum(results_per_pass_RCWA["a"][0], 0)

    ## pure TMM (from Solcore)
    all_layers = front_materials + [Layer(bulkthick, Ge)] + back_materials

    coh_list = len(front_materials) * ["c"] + ["i"] + ["c"]

    OS_layers = OptiStack(all_layers, substrate=Ag, no_back_reflection=False)

    TMM_res = calculate_rat(
        OS_layers,
        wavelength=wavelengths * 1e9,
        no_back_reflection=False,
        angle=options["theta_in"] * 180 / np.pi,
        coherent=False,
        coherency_list=coh_list,
        pol=options["pol"],
    )

    # stack results for comparison
    TMM_reference = TMM_res["A_per_layer"][1:-2].T
    TMM_matrix = np.hstack((results_per_layer_front_TMM_matrix, results_TMM_Matrix[0].A_bulk[0].data[:, None]))
    RCWA_matrix = np.hstack((results_per_layer_front_RCWA, results_RCWA_Matrix[0].A_bulk[0].data[:, None]))
    RT_matrix = np.hstack((results_per_layer_front_RT, results_RT[0].A_bulk[0].data[:, None]))

    assert TMM_reference == approx(TMM_matrix, abs=0.05)
    assert TMM_reference == approx(RCWA_matrix, abs=0.05)
    assert TMM_reference == approx(RT_matrix, abs=0.4)

    assert (
        results_TMM_Matrix[0].R[0]
        + results_TMM_Matrix[0].T[0]
        + np.sum(results_per_layer_front_TMM_matrix, 1)
        + results_TMM_Matrix[0].A_bulk[0]
    ).data == approx(1)
    assert (
        results_RCWA_Matrix[0].R[0]
        + results_RCWA_Matrix[0].T[0]
        + np.sum(results_per_layer_front_RCWA, 1)
        + results_RCWA_Matrix[0].A_bulk[0]
    ).data == approx(1)
    assert (
        results_RT[0].R[0] + results_RT[0].T[0] + np.sum(results_per_layer_front_RT, 1) + results_RT[0].A_bulk[0]
    ).data == approx(1)


@mark.skipif(skip_s4_test(), reason="Only works if S4 installed")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_tmm_rcwa_pol_angle(RCWA_method):
    # solcore imports
    from solcore.structure import Layer
    from solcore import material
    from solcore.absorption_calculator import calculate_rat, OptiStack

    # rayflare imports
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options

    # Thickness of bottom Ge layer
    bulkthick = 300e-6

    wavelengths = np.linspace(300, 1850, 30) * 1e-9

    # set options
    options = default_options()
    options.wavelengths = wavelengths
    options.project_name = "method_comparison_test_angle_pol"
    options.n_theta_bins = 50
    options.lookuptable_angles = 100
    options.parallel = True
    options.c_azimuth = 0.001
    options.I_thresh = 1e-8
    options.only_incidence_angle = False
    options.bulk_profile = False
    options.RCWA_method = RCWA_method

    # set up Solcore materials
    Ge = material("Ge")()
    GaAs = material("GaAs")()
    GaInP = material("GaInP")(In=0.5)
    Ag = material("Ag")()
    SiN = material("Si3N4")()
    Air = material("Air")()
    Ta2O5 = material("TaOx1")()  # Ta2O5 (SOPRA database)
    MgF2 = material("MgF2")()  # MgF2 (SOPRA database)

    front_materials = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5), Layer(464e-9, GaInP), Layer(1682e-9, GaAs)]
    back_materials = [Layer(100e-9, SiN)]

    angles = [0, np.pi / 5, np.pi / 3]
    pols = ["s", "p", "u"]
    # TMM, matrix framework

    for angle in angles:
        for pol in pols:
            options.pol = pol
            options.theta_in = angle
            options.phi_in = angle

            front_surf = Interface("TMM", layers=front_materials, name="GaInP_GaAs_TMM" + str(pol), coherent=True)
            back_surf = Interface("TMM", layers=back_materials, name="SiN_Ag_TMM" + str(pol), coherent=True)

            bulk_Ge = BulkLayer(bulkthick, Ge, name="Ge_bulk")  # bulk thickness in m

            SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

            process_structure(SC, options)

            results_TMM_Matrix = calculate_RAT(SC, options)

            results_per_pass_TMM_matrix = results_TMM_Matrix[1]

            results_per_layer_front_TMM_matrix = np.sum(results_per_pass_TMM_matrix["a"][0], 0)

            ## RCWA

            front_surf = Interface(
                "RCWA",
                layers=front_materials,
                name="GaInP_GaAs_RCWA" + str(pol),
                coherent=True,
                d_vectors=((500, 0), (0, 500)),
                rcwa_orders=2,
            )
            back_surf = Interface(
                "RCWA",
                layers=back_materials,
                name="SiN_Ag_RCWA" + str(pol),
                coherent=True,
                d_vectors=((500, 0), (0, 500)),
                rcwa_orders=2,
            )

            SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

            process_structure(SC, options)

            results_RCWA_Matrix = calculate_RAT(SC, options)

            results_per_pass_RCWA = results_RCWA_Matrix[1]

            # only select absorbing layers, sum over passes
            results_per_layer_front_RCWA = np.sum(results_per_pass_RCWA["a"][0], 0)

            ## pure TMM (from Solcore)
            all_layers = front_materials + [Layer(bulkthick, Ge)] + back_materials

            coh_list = len(front_materials) * ["c"] + ["i"] + ["c"]

            OS_layers = OptiStack(all_layers, substrate=Ag, no_back_reflection=False)

            TMM_res = calculate_rat(
                OS_layers,
                wavelength=wavelengths * 1e9,
                no_back_reflection=False,
                angle=options.theta_in * 180 / np.pi,
                coherent=False,
                coherency_list=coh_list,
                pol=options.pol,
            )

            # stack results for comparison
            TMM_reference = TMM_res["A_per_layer"][1:-2].T
            TMM_matrix = np.hstack((results_per_layer_front_TMM_matrix, results_TMM_Matrix[0].A_bulk[0].data[:, None]))

            RCWA_matrix = np.hstack((results_per_layer_front_RCWA, results_RCWA_Matrix[0].A_bulk[0].data[:, None]))

            assert TMM_reference == approx(TMM_matrix, abs=0.05)
            assert TMM_reference == approx(RCWA_matrix, abs=0.05)

            assert (
                results_TMM_Matrix[0].R[0]
                + results_TMM_Matrix[0].T[0]
                + np.sum(results_per_layer_front_TMM_matrix, 1)
                + results_TMM_Matrix[0].A_bulk[0]
            ).data == approx(1)
            assert (
                results_RCWA_Matrix[0].R[0]
                + results_RCWA_Matrix[0].T[0]
                + np.sum(results_per_layer_front_RCWA, 1)
                + results_RCWA_Matrix[0].A_bulk[0]
            ).data == approx(1)


def test_absorption_profile():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import planar_surface
    from rayflare.options import default_options
    from solcore import material
    from solcore import si
    from rayflare.transfer_matrix_method import tmm_structure
    from solcore.structure import Layer

    Air = material("Air")()
    Si = material("Si")()
    GaAs = material("GaAs")()
    Ge = material("Ge")()

    triangle_surf = planar_surface()

    options = default_options()

    options.wavelengths = np.linspace(700, 1400, 4) * 1e-9
    options.theta_in = 45 * np.pi / 180
    options.nx = 5
    options.ny = 5
    options.pol = "u"
    options.n_rays = 2000
    options.depth_spacing = 1e-6

    rtstr = rt_structure(
        textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
        materials=[GaAs, Si, Ge],
        widths=[si("100um"), si("70um"), si("50um")],
        incidence=Air,
        transmission=Air,
    )
    result_rt = rtstr.calculate(options)

    stack = [Layer(si("100um"), GaAs), Layer(si("70um"), Si), Layer(si("50um"), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air, no_back_reflection=False)

    output = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile = output["profile"][output["profile"] > 1e-7]
    rt_profile = result_rt["profile"][output["profile"] > 1e-7]

    assert output["profile"].shape == result_rt["profile"].shape
    assert rt_profile == approx(tmm_profile, rel=0.4)


def test_absorption_profile_incoh_angles():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import planar_surface
    from rayflare.options import default_options
    from solcore import material
    from solcore import si
    from rayflare.transfer_matrix_method import tmm_structure
    from solcore.structure import Layer

    Air = material("Air")()
    Si = material("Si")()
    GaAs = material("GaAs")()
    Ge = material("Ge")()

    triangle_surf = planar_surface()

    options = default_options()

    options.wavelengths = np.linspace(700, 1400, 4) * 1e-9
    options.nx = 5
    options.ny = 5
    options.n_rays = 2000
    options.depth_spacing = 1e-7
    options.coherent = False
    options.coherency_list = ["c", "i", "i"]
    options.theta_in = np.pi / 4
    options.phi_in = np.pi / 3
    options.pol = "s"
    options.depth_spacing_bulk = 1e-7

    rtstr = rt_structure(
        textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
        materials=[GaAs, Si, Ge],
        widths=[si("100um"), si("70um"), si("50um")],
        incidence=Air,
        transmission=Air,
    )
    result_rt_s = rtstr.calculate(options)

    stack = [Layer(si("100um"), GaAs), Layer(si("70um"), Si), Layer(si("50um"), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air, no_back_reflection=False)

    output_s = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_s = output_s["profile"][output_s["profile"] > 1e-7]
    rt_profile_s = result_rt_s["profile"][output_s["profile"] > 1e-7]

    assert output_s["profile"].shape == result_rt_s["profile"].shape
    assert rt_profile_s == approx(tmm_profile_s, rel=0.2)

    options.pol = "p"

    result_rt_p = rtstr.calculate(options)
    output_p = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_p = output_p["profile"][output_p["profile"] > 1e-7]
    rt_profile_p = result_rt_p["profile"][output_p["profile"] > 1e-7]

    assert output_p["profile"].shape == result_rt_p["profile"].shape
    assert rt_profile_p == approx(tmm_profile_p, rel=0.2)

    options.pol = "u"

    result_rt_u = rtstr.calculate(options)
    output_u = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_u = output_u["profile"][output_u["profile"] > 1e-7]
    rt_profile_u = result_rt_u["profile"][output_u["profile"] > 1e-7]

    assert output_u["profile"].shape == result_rt_u["profile"].shape
    assert rt_profile_u == approx(tmm_profile_u, rel=0.2)


def test_absorption_profile_coh_angles():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import planar_surface
    from rayflare.options import default_options
    from solcore import material
    from solcore import si
    from rayflare.transfer_matrix_method import tmm_structure
    from solcore.structure import Layer

    Air = material("Air")()
    Si = material("Si")()
    GaAs = material("GaAs")()
    Ge = material("Ge")()

    triangle_surf = planar_surface()

    options = default_options()

    options.wavelengths = np.linspace(700, 1400, 4) * 1e-9
    options.nx = 5
    options.ny = 5
    options.n_rays = 2000
    options.depth_spacing = 1e-6
    options.theta_in = np.pi / 4
    options.phi_in = np.pi / 3
    options.pol = "s"

    rtstr = rt_structure(
        textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
        materials=[GaAs, Si, Ge],
        widths=[si("100um"), si("70um"), si("50um")],
        incidence=Air,
        transmission=Air,
    )
    result_rt_s = rtstr.calculate(options)

    stack = [Layer(si("100um"), GaAs), Layer(si("70um"), Si), Layer(si("50um"), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air, no_back_reflection=False)

    output_s = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_s = output_s["profile"][output_s["profile"] > 1e-7]
    rt_profile_s = result_rt_s["profile"][output_s["profile"] > 1e-7]

    assert output_s["profile"].shape == result_rt_s["profile"].shape
    assert rt_profile_s == approx(tmm_profile_s, rel=0.4)

    options.pol = "p"

    result_rt_p = rtstr.calculate(options)
    output_p = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_p = output_p["profile"][output_p["profile"] > 1e-7]
    rt_profile_p = result_rt_p["profile"][output_p["profile"] > 1e-7]

    assert output_p["profile"].shape == result_rt_p["profile"].shape
    assert rt_profile_p == approx(tmm_profile_p, rel=0.4)

    options.pol = "u"

    result_rt_u = rtstr.calculate(options)
    output_u = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_u = output_u["profile"][output_u["profile"] > 1e-7]
    rt_profile_u = result_rt_u["profile"][output_u["profile"] > 1e-7]

    assert output_u["profile"].shape == result_rt_u["profile"].shape
    assert rt_profile_u == approx(tmm_profile_u, rel=0.4)


@mark.skipif(skip_s4_test(), reason="Only works if S4 installed")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_rcwa_tmm_profiles_coh(RCWA_method):
    from rayflare.options import default_options
    from solcore import material
    from solcore import si
    from rayflare.transfer_matrix_method import tmm_structure
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
    from solcore.structure import Layer

    Air = material("Air")()
    Si = material("Si")()
    GaAs = material("GaAs")()
    Ge = material("Ge")()

    options = default_options()

    options.wavelengths = np.linspace(700, 1400, 4) * 1e-9

    options.depth_spacing = 10e-9
    options.theta_in = np.pi / 3
    options.phi_in = np.pi / 4
    options.pol = "s"
    options.RCWA_method = RCWA_method

    stack = [Layer(si("500nm"), GaAs), Layer(si("1.1um"), Si), Layer(si("0.834um"), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air, no_back_reflection=False)

    strt_rcwa = rcwa_structure(stack, ((100, 0), (0, 100)), options, Air, Air)
    strt_rcwa.calculate(options)
    output_rcwa_s = strt_rcwa.calculate_profile(options)["profile"]

    output_s = strt.calculate_profile(options)["profile"]

    tmm_profile_s = output_s[output_s > 1e-5]
    rcwa_profile_s = output_rcwa_s[output_s > 1e-5]

    assert rcwa_profile_s == approx(tmm_profile_s, rel=0.025)

    options.pol = "p"

    stack = [Layer(si("500nm"), GaAs), Layer(si("1.1um"), Si), Layer(si("0.834um"), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air, no_back_reflection=False)

    strt_rcwa = rcwa_structure(stack, ((100, 0), (0, 100)), options, Air, Air)
    strt_rcwa.calculate(options)
    output_rcwa_p = strt_rcwa.calculate_profile(options)["profile"]

    output_p = strt.calculate_profile(options)["profile"]

    tmm_profile_p = output_p[output_p > 1e-5]
    rcwa_profile_p = output_rcwa_p[output_p > 1e-5]

    assert rcwa_profile_p == approx(tmm_profile_p, rel=0.025)

    options.pol = "u"

    stack = [Layer(si("500nm"), GaAs), Layer(si("1.1um"), Si), Layer(si("0.834um"), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air, no_back_reflection=False)

    strt_rcwa = rcwa_structure(stack, ((100, 0), (0, 100)), options, Air, Air)
    strt_rcwa.calculate(options)
    output_rcwa_u = strt_rcwa.calculate_profile(options)["profile"]

    output_u = strt.calculate_profile(options)["profile"]

    tmm_profile_u = output_u[output_u > 5e-5]
    rcwa_profile_u = output_rcwa_u[output_u > 5e-5]

    assert rcwa_profile_u == approx(tmm_profile_u, rel=0.025)


@mark.skipif(skip_s4_test(), reason="Only works if S4 installed")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_rcwa_tmm_matrix_check_sums(RCWA_method):
    from solcore.structure import Layer
    from solcore import material

    # rayflare imports
    from rayflare.textures import planar_surface
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options
    from rayflare.transfer_matrix_method import tmm_structure
    from rayflare.angles import make_angle_vector

    # Thickness of bulk Ge layer
    bulkthick = 30e-9

    wavelengths = np.linspace(640, 850, 6) * 1e-9

    # set options
    options = default_options()
    options.wavelengths = wavelengths
    options.n_rays = 4000
    options.n_theta_bins = 20
    options.lookuptable_angles = 100
    options.depth_spacing = 5e-10
    options.only_incidence_angle = False
    options.nx = 4
    options.ny = 4
    options.bulk_profile = False
    options.RCWA_method = RCWA_method

    _, _, angle_vector = make_angle_vector(options.n_theta_bins, options.phi_symmetry, options.c_azimuth)

    for pol in ["s", "p", "u"]:
        options.project_name = "rcwa_tmm_matrix_profiles_" + pol + RCWA_method
        options.pol = pol

        SiN = material("Si3N4")()
        GaAs = material("GaAs")()
        GaInP = material("GaInP")(In=0.5)
        Ag = material("Ag")()
        Air = material("Air")()
        Ta2O5 = material("TaOx1")()  # Ta2O5 (SOPRA database)
        MgF2 = material("MgF2")()  # MgF2 (SOPRA database)

        front_materials = [Layer(100e-9, MgF2), Layer(50e-9, GaInP), Layer(100e-9, Ta2O5), Layer(200e-9, GaAs)]
        back_materials = [Layer(50e-9, GaInP)]

        prof_layers = np.arange(len(front_materials)) + 1

        ## pure TMM with incoherent thick layer
        all_layers = front_materials + [Layer(bulkthick, SiN)] + back_materials

        coh_list = len(front_materials) * ["c"] + ["i"] + ["c"]

        options.coherent = False
        options.coherency_list = coh_list

        OS_layers = tmm_structure(all_layers, incidence=Air, transmission=Ag, no_back_reflection=False)

        th_ind = np.random.randint(0, 4)
        phi_in = np.random.uniform(0, np.pi)

        options.theta_in = angle_vector[th_ind, 1]
        options.phi_in = phi_in

        TMM_res = OS_layers.calculate(options, profile=True, layers=[1, 2, 3, 4, 5, 6])

        ## TMM Matrix method

        front_surf = Interface("TMM", layers=front_materials, name="TMM_f", coherent=True, prof_layers=prof_layers)
        back_surf = Interface("TMM", layers=back_materials, name="TMM_b", coherent=True, prof_layers=[1])

        bulk_Ge = BulkLayer(bulkthick, SiN, name="SiN_bulk")  # bulk thickness in m

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_TMM_Matrix = calculate_RAT(SC, options)

        results_per_pass = results_TMM_Matrix[1]

        results_per_layer_front_TMM = np.sum(results_per_pass["a"][0], 0)
        results_per_layer_back_TMM = np.sum(results_per_pass["a"][1], 0)[:, 0]

        assert np.all(results_TMM_Matrix[0]["A_interface"][0].data == np.sum(results_per_pass["a"][0], (0, 2)))

        assert np.all(results_TMM_Matrix[0]["A_interface"][1].data == np.sum(results_per_pass["a"][1], (0, 2)))

        assert np.all(
            (
                results_TMM_Matrix[0]["R"]
                + results_TMM_Matrix[0]["T"]
                + results_TMM_Matrix[0]["A_bulk"]
                + np.sum(results_TMM_Matrix[0]["A_interface"], 0)
            )
            == approx(1, abs=0.01)
        )

        surf = planar_surface()

        front_surf = Interface(
            "RT_TMM", layers=front_materials, texture=surf, name="RT_TMM_f", coherent=True, prof_layers=prof_layers
        )
        back_surf = Interface(
            "RT_TMM", layers=back_materials, texture=surf, name="RT_TMM_b", coherent=True, prof_layers=[1]
        )

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_RT = calculate_RAT(SC, options)

        results_per_pass = results_RT[1]

        # only select absorbing layers, sum over passes
        results_per_layer_front_RT = np.sum(results_per_pass["a"][0], 0)
        # results_per_layer_back_RT = np.sum(results_per_pass['a'][1], 0)[:,0]

        assert np.all(results_RT[0]["A_interface"][0].data == np.sum(results_per_pass["a"][0], (0, 2)))

        assert np.all(results_RT[0]["A_interface"][1].data == np.sum(results_per_pass["a"][1], (0, 2)))

        assert np.all(
            (
                results_RT[0]["R"]
                + results_RT[0]["T"]
                + results_RT[0]["A_bulk"]
                + np.sum(results_RT[0]["A_interface"], 0)
            )
            == approx(1, abs=0.01)
        )

        ## RCWA Matrix

        front_surf = Interface(
            "RCWA",
            layers=front_materials,
            name="RCWA_f",
            d_vectors=((500, 0), (0, 500)),
            rcwa_orders=2,
            prof_layers=prof_layers,
        )
        back_surf = Interface(
            "RCWA", layers=back_materials, name="RCWA_b", d_vectors=((500, 0), (0, 500)), rcwa_orders=2, prof_layers=[1]
        )

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_RCWA_Matrix = calculate_RAT(SC, options)

        results_per_pass = results_RCWA_Matrix[1]

        results_per_layer_front_RCWA = np.sum(results_per_pass["a"][0], 0)
        results_per_layer_back_RCWA = np.sum(results_per_pass["a"][1], 0)[:, 0]

        assert np.all(results_RCWA_Matrix[0]["A_interface"][0] == np.sum(results_per_pass["a"][0], (0, 2)))

        assert np.all(results_RCWA_Matrix[0]["A_interface"][1] == np.sum(results_per_pass["a"][1], (0, 2)))

        assert np.all(
            (
                results_RCWA_Matrix[0]["R"]
                + results_RCWA_Matrix[0]["T"]
                + results_RCWA_Matrix[0]["A_bulk"]
                + np.sum(results_RCWA_Matrix[0]["A_interface"], 0)
            )
            == approx(1, abs=0.01)
        )

        results_per_layer_front_TMM_ref = TMM_res["A_per_layer"][:, : len(front_materials)]
        results_per_layer_back_TMM_ref = TMM_res["A_per_layer"][:, -1]

        c_i = results_per_layer_front_TMM_ref > 1e-2

        assert results_per_layer_front_TMM[c_i] == approx(results_per_layer_front_TMM_ref[c_i], rel=0.05)
        assert results_per_layer_front_RCWA[c_i] == approx(results_per_layer_front_TMM_ref[c_i], rel=0.05)
        assert results_per_layer_front_RT[c_i] == approx(results_per_layer_front_TMM_ref[c_i], rel=0.8)

        c_i = results_per_layer_back_TMM_ref > 1e-2

        assert results_per_layer_back_TMM[c_i] == approx(results_per_layer_back_TMM_ref[c_i], rel=0.05)
        assert results_per_layer_back_RCWA[c_i] == approx(results_per_layer_back_TMM_ref[c_i], rel=0.05)


@mark.skipif(skip_s4_test(), reason="Only works if S4 installed")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_rcwa_tmm_matrix_profiles(RCWA_method):
    from solcore.structure import Layer
    from solcore import material

    # rayflare imports
    from rayflare.textures.standard_rt_textures import planar_surface
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options
    from rayflare.transfer_matrix_method import tmm_structure
    from rayflare.angles import make_angle_vector

    # Thickness of bulk Ge layer
    bulkthick = 30e-9

    wavelengths = np.linspace(640, 850, 6) * 1e-9

    # set options
    options = default_options()
    options.wavelengths = wavelengths
    options.n_rays = 4000
    options.n_theta_bins = 20
    options.lookuptable_angles = 100
    options.parallel = True
    options.c_azimuth = 0.25
    options.depth_spacing = 5e-10
    options.only_incidence_angle = False
    options.nx = 4
    options.ny = 4
    options.RCWA_method = RCWA_method

    _, _, angle_vector = make_angle_vector(options.n_theta_bins, options.phi_symmetry, options.c_azimuth)

    for pol in ["s", "p", "u"]:
        options.project_name = "rcwa_tmm_matrix_profiles_" + pol + RCWA_method
        options.pol = pol

        SiN = material("Si3N4")()
        GaAs = material("GaAs")()
        GaInP = material("GaInP")(In=0.5)
        Ag = material("Ag")()
        Air = material("Air")()
        Ta2O5 = material("TaOx1")()  # Ta2O5 (SOPRA database)
        MgF2 = material("MgF2")()  # MgF2 (SOPRA database)

        front_materials = [Layer(100e-9, MgF2), Layer(50e-9, GaInP), Layer(100e-9, Ta2O5), Layer(200e-9, GaAs)]
        back_materials = [Layer(50e-9, GaInP)]

        prof_layers = np.arange(len(front_materials)) + 1

        ## pure TMM with incoherent thick layer
        all_layers = front_materials + [Layer(bulkthick, SiN)] + back_materials

        coh_list = len(front_materials) * ["c"] + ["i"] + ["c"]

        options.coherent = False
        options.coherency_list = coh_list

        OS_layers = tmm_structure(all_layers, incidence=Air, transmission=Ag, no_back_reflection=False)

        th_ind = np.random.randint(0, 4)
        phi_in = np.random.uniform(0, np.pi)

        options.theta_in = angle_vector[th_ind, 1]
        options.phi_in = phi_in

        # set up Solcore materials

        TMM_res = OS_layers.calculate(options, profile=True, layers=[1, 2, 3, 4, 5, 6])

        ## TMM Matrix method

        front_surf = Interface("TMM", layers=front_materials, name="TMM_f", coherent=True, prof_layers=prof_layers)
        back_surf = Interface("TMM", layers=back_materials, name="TMM_b", coherent=True, prof_layers=[1])

        bulk_Ge = BulkLayer(bulkthick, SiN, name="SiN_bulk")  # bulk thickness in m

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_TMM_Matrix = calculate_RAT(SC, options)

        profile = results_TMM_Matrix[2]

        prof_plot_TMM = profile[0]
        prof_plot_TMM_back = profile[1]

        depths = np.linspace(0, len(prof_plot_TMM[0, :]) * options["depth_spacing"] * 1e9, len(prof_plot_TMM[0, :]))
        integrated_A = np.trapz(prof_plot_TMM, depths, axis=1)

        depths_back = np.linspace(
            0, len(prof_plot_TMM_back[0, :]) * options["depth_spacing"] * 1e9, len(prof_plot_TMM_back[0, :])
        )
        integrated_A_back = np.trapz(prof_plot_TMM_back, depths_back, axis=1)

        assert integrated_A == approx(results_TMM_Matrix[0]["A_interface"][0].data, abs=0.01)
        assert integrated_A_back == approx(results_TMM_Matrix[0]["A_interface"][1].data, abs=0.01)

        ## RT_TMM Matrix method
        surf = planar_surface()  # [texture, flipped texture]

        front_surf = Interface(
            "RT_TMM", layers=front_materials, texture=surf, name="RT_TMM_f", coherent=True, prof_layers=prof_layers
        )
        back_surf = Interface(
            "RT_TMM", layers=back_materials, texture=surf, name="RT_TMM_b", coherent=True, prof_layers=[1]
        )

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_RT = calculate_RAT(SC, options)

        profile = results_RT[2]

        prof_plot_RT = profile[0]
        prof_plot_RT_back = profile[1]

        depths = np.linspace(0, len(prof_plot_RT[0, :]) * options["depth_spacing"] * 1e9, len(prof_plot_RT[0, :]))
        integrated_A = np.trapz(prof_plot_RT, depths, axis=1)

        depths_back = np.linspace(
            0, len(prof_plot_RT_back[0, :]) * options["depth_spacing"] * 1e9, len(prof_plot_RT_back[0, :])
        )
        integrated_A_back = np.trapz(prof_plot_RT_back, depths_back, axis=1)

        assert integrated_A == approx(results_RT[0]["A_interface"][0].data, abs=0.01)
        assert integrated_A_back == approx(results_RT[0]["A_interface"][1].data, abs=0.01)

        ## RCWA Matrix

        front_surf = Interface(
            "RCWA",
            layers=front_materials,
            name="RCWA_f",
            d_vectors=((500, 0), (0, 500)),
            rcwa_orders=2,
            prof_layers=prof_layers,
        )
        back_surf = Interface(
            "RCWA", layers=back_materials, name="RCWA_b", d_vectors=((500, 0), (0, 500)), rcwa_orders=2, prof_layers=[1]
        )

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_RCWA_Matrix = calculate_RAT(SC, options)

        profile = results_RCWA_Matrix[2]

        prof_plot_RCWA = profile[0]
        prof_plot_RCWA_back = profile[1]

        depths = np.linspace(0, len(prof_plot_RCWA[0, :]) * options["depth_spacing"] * 1e9, len(prof_plot_RCWA[0, :]))
        integrated_A = np.trapz(prof_plot_RCWA, depths, axis=1)

        depths_back = np.linspace(
            0, len(prof_plot_RCWA_back[0, :]) * options["depth_spacing"] * 1e9, len(prof_plot_RCWA_back[0, :])
        )
        integrated_A_back = np.trapz(prof_plot_RCWA_back, depths_back, axis=1)

        assert integrated_A == approx(results_RCWA_Matrix[0]["A_interface"][0].data, abs=0.01)
        assert integrated_A_back == approx(results_RCWA_Matrix[0]["A_interface"][1].data, abs=0.01)

        front_profile_TMM = TMM_res["profile"][:, : len(depths)]
        c_i = front_profile_TMM > 1e-4

        assert prof_plot_TMM.data[c_i] == approx(front_profile_TMM[c_i], rel=0.15)
        assert prof_plot_RCWA.data[c_i] == approx(front_profile_TMM[c_i], rel=0.15)

        back_profile_TMM = TMM_res["profile"][:, -len(depths_back) :]
        c_i = back_profile_TMM > 1e-4

        assert prof_plot_TMM_back.data[c_i] == approx(back_profile_TMM[c_i], rel=0.15)
        assert prof_plot_RCWA_back.data[c_i] == approx(back_profile_TMM[c_i], rel=0.15)


@mark.skipif(skip_s4_test(), reason="Only works if S4 installed")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_profile_integration(RCWA_method):
    from rayflare.utilities import get_savepath
    import os
    import xarray as xr

    ds = 0.5

    for pol in ["s", "p", "u"]:
        for method in ["TMM_f", "RCWA_f"]:
            project_name = "rcwa_tmm_matrix_profiles_" + pol + RCWA_method
            pth = get_savepath("default", project_name)

            profdatapath = os.path.join(pth, method + "frontprofmat.nc")
            profdatapath_r = os.path.join(pth, method + "rearprofmat.nc")

            prof_dataset = xr.load_dataset(profdatapath)
            prof_dataset_r = xr.load_dataset(profdatapath_r)

            intgr = prof_dataset["intgr"]
            prof = prof_dataset["profile"]
            intgr_r = prof_dataset_r["intgr"]
            prof_r = prof_dataset_r["profile"]

            depths = np.arange(0, len(prof["z"]) * ds, ds)
            depths_r = np.arange(0, len(prof_r["z"]) * ds, ds)

            integrated_prof = np.trapz(prof.data, depths, axis=1)
            integrated_prof_r = np.trapz(prof_r.data, depths_r, axis=1)

            assert integrated_prof == approx(intgr.data, rel=0.03)
            assert integrated_prof_r == approx(intgr_r.data, rel=0.03)

    for pol in ["s", "p", "u"]:
        for method in ["TMM_b", "RCWA_b"]:
            project_name = "rcwa_tmm_matrix_profiles_" + pol + RCWA_method
            pth = get_savepath("default", project_name)

            profdatapath = os.path.join(pth, method + "frontprofmat.nc")

            prof_dataset = xr.load_dataset(profdatapath)

            intgr = prof_dataset["intgr"]
            prof = prof_dataset["profile"]

            depths = np.arange(0, len(prof["z"]) * ds, ds)

            integrated_prof = np.trapz(prof.data, depths, axis=1)

            c_i = intgr.data > 1e-4

            assert integrated_prof[c_i] == approx(intgr.data[c_i], rel=0.04)


def test_compare_RT_TMM_Fresnel():
    from solcore.structure import Layer
    from solcore import material

    # rayflare imports
    from rayflare.textures import regular_pyramids, planar_surface
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options

    Si = material("Si")()
    Air = material("Air")()

    options = default_options()
    options.project_name = "RT_Fresnel_TMM"
    options.n_theta_bins = 20
    options.wavelengths = np.linspace(950, 1130, 4) * 1e-9
    options.bulk_profile = False

    flat_surf = planar_surface(size=2)  # pyramid size in microns
    triangle_surf = regular_pyramids(55, upright=False, size=2)

    front_surf = Interface("RT_Fresnel", name="RT_F_f", texture=triangle_surf)
    back_surf = Interface("RT_Fresnel", name="RT_F_b", texture=flat_surf)

    bulk = BulkLayer(300e-6, Si, name="Si_bulk")  # bulk thickness in m

    SC_F = Structure([front_surf, bulk, back_surf], incidence=Air, transmission=Air)

    process_structure(SC_F, options, "current")

    res_fresnel = calculate_RAT(SC_F, options, "current")

    front_surf = Interface(
        "RT_TMM", name="RT_TMM_f", texture=triangle_surf, layers=[Layer(1e-9, Si)], coherent=False, coherency_list=["i"]
    )
    back_surf = Interface(
        "RT_TMM", name="RT_TMM_b", texture=flat_surf, layers=[Layer(1e-9, Si)], coherent=False, coherency_list=["i"]
    )

    bulk = BulkLayer(300e-6, Si, name="Si_bulk")  # bulk thickness in m

    SC_TMM = Structure([front_surf, bulk, back_surf], incidence=Air, transmission=Air)

    process_structure(SC_TMM, options, "current")

    res_TMM = calculate_RAT(SC_TMM, options, "current")

    T_nz = res_TMM[0]["T"][0].data > 5e-3

    assert res_fresnel[0]["A_bulk"][0].data == approx(res_TMM[0]["A_bulk"][0].data, abs=0.15)
    assert res_fresnel[0]["R"][0].data == approx(res_TMM[0]["R"][0].data, abs=0.15)
    assert res_fresnel[0]["T"][0].data[T_nz] == approx(res_TMM[0]["T"][0].data[T_nz], abs=0.15)


def compare_rt_integrated_tmm():
    from rayflare.ray_tracing import rt_structure
    from rayflare.transfer_matrix_method import tmm_structure
    from solcore import material
    from solcore.structure import Layer
    from rayflare.options import default_options
    from rayflare.textures import planar_surface

    opts = default_options()
    opts.project_name = "method_comparison_test"
    opts.n_rays = 1000

    thetas_in = np.linspace(0, 1.2, 3)
    pol_in = ["s", "p", "u"]

    d_Si = 100e-6

    GaAs = material("GaAs")()
    Air = material("Air")()
    Si = material("Si")()
    MgF2 = material("MgF2")()
    Ag = material("Ag")()

    front_layers = [Layer(100e-9, MgF2), Layer(502e-9, GaAs)]
    back_layers = [Layer(50e-9, Ag)]

    coherency_list_TMM = ["c", "c", "i", "c"]
    front_surf = planar_surface(interface_layers=front_layers)  # pyramid size in microns
    back_surf = planar_surface(interface_layers=back_layers)  # pyramid size in microns

    opts.coherent = False
    opts.coherency_list = coherency_list_TMM
    opts.wavelengths = np.linspace(300, 1200, 5) * 1e-9

    rtstr = rt_structure([front_surf, back_surf], [Si], [d_Si], Air, Air, opts, use_TMM=True)
    tmmstr = tmm_structure(front_layers + [Layer(d_Si, Si)] + back_layers, Air, Air)

    for angle_pol in itertools.product(thetas_in, pol_in):
        opts.theta_in = angle_pol[0]
        opts.pol_in = angle_pol[1]

        rt_res = rtstr.calculate(opts)
        tmm_res = tmmstr.calculate(opts)

        # reflection and transmission:
        assert rt_res["R"] == approx(tmm_res["R"], abs=0.05)
        assert rt_res["T"] == approx(tmm_res["T"], abs=0.05)

        # Si absorption:
        assert rt_res["A_per_layer"][:, 0] == approx(tmm_res["A_per_layer"][:, 2], abs=0.05)
        # MgF2 + GaAs absorption:
        assert rt_res["A_per_interface"][0] == approx(tmm_res["A_per_layer"][:, :2], abs=0.05)
        # Ag absorption:
        assert rt_res["A_per_interface"][1] == approx(tmm_res["A_per_layer"][:, -1:], abs=0.05)


def compare_rt_integrated_tmm_profile():
    from rayflare.ray_tracing import rt_structure
    from rayflare.transfer_matrix_method import tmm_structure
    from solcore import material
    from solcore.structure import Layer
    from rayflare.options import default_options
    from rayflare.textures import planar_surface

    opts = default_options()
    opts.project_name = "method_comparison_int_tmm"
    opts.n_rays = 4000
    opts.lookuptable_angles = 500
    # opts.nx = 5
    # opts.ny = 5

    thetas_in = np.linspace(0, 0.8, 2)
    # thetas_in = [0]
    pol_in = ["s", "p", "u"]

    d_Si = 10e-6

    GaAs = material("GaAs")()
    Air = material("Air")()
    Si = material("Si")()
    MgF2 = material("MgF2")()
    ITO = material("ITO2")()
    Ag = material("Ag")()

    front_layers = [Layer(100e-9, MgF2), Layer(502e-9, GaAs)]
    back_layers = [Layer(100e-9, ITO), Layer(50e-9, Ag)]

    coherency_list_TMM = ["c", "c", "i", "c", "c"]
    front_surf = planar_surface(interface_layers=front_layers, prof_layers=[2])  # pyramid size in microns
    back_surf = planar_surface(interface_layers=back_layers, prof_layers=[1])  # pyramid size in microns

    opts.coherent = False
    opts.coherency_list = coherency_list_TMM
    opts.wavelengths = np.linspace(300, 1100, 5) * 1e-9
    opts.depth_spacing = 1e-9
    opts.depth_spacing_bulk = 1e-9
    opts.parallel = True

    rtstr = rt_structure([front_surf, back_surf], [Si], [d_Si], Air, Air, opts, use_TMM=True)
    tmmstr = tmm_structure(front_layers + [Layer(d_Si, Si)] + back_layers, Air, Air)

    for angle_pol in itertools.product(thetas_in, pol_in):
        # print(angle_pol)
        opts.theta_in = angle_pol[0]
        opts.pol_in = angle_pol[1]
        rt_res = rtstr.calculate(opts)
        tmm_res = tmmstr.calculate(opts, profile=True, layers=[2, 3, 4])

        rt_front = rt_res["interface_profiles"][0]
        # rt_Si = rt_res["profile"]
        rt_back = rt_res["interface_profiles"][1]

        # rt_profile = np.hstack([rt_front, rt_Si, rt_back])
        # tmm_profile = tmm_res["profile"]

        ratio = rt_front[rt_front > 3e-5] / tmm_res["profile"][:, : rt_front.shape[1]][rt_front > 3e-5]

        assert ratio == approx(np.ones_like(ratio), abs=0.07)

        ratio = rt_back[rt_back > 3e-5] / tmm_res["profile"][:, -rt_back.shape[1] :][rt_back > 3e-5]

        assert ratio == approx(np.ones_like(ratio), abs=0.2)

        # reflection and transmission:
        assert rt_res["R"] == approx(tmm_res["R"], abs=0.05)
        assert rt_res["T"] == approx(tmm_res["T"], abs=0.05)

        # Si absorption:
        assert rt_res["A_per_layer"][:, 0] == approx(tmm_res["A_per_layer"][:, 2], abs=0.05)
        # MgF2 + GaAs absorption:
        assert rt_res["A_per_interface"][0] == approx(tmm_res["A_per_layer"][:, :2], abs=0.05)
        # Ag absorption:
        assert rt_res["A_per_interface"][1] == approx(tmm_res["A_per_layer"][:, -2:], abs=0.05)


def test_tmm_arm():
    from solcore.structure import Layer
    from solcore import material

    # rayflare imports
    from rayflare.textures.standard_rt_textures import planar_surface
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options
    from rayflare.transfer_matrix_method import tmm_structure

    # Thickness of bottom Ge layer
    bulkthick = 0.3e-6

    wavelengths = np.linspace(700, 1400, 6) * 1e-9

    # set options
    options = default_options()
    options.wavelengths = wavelengths
    options.project_name = "rt_tmm_comparisons_2"
    options.n_rays = 10000
    options.n_theta_bins = 20
    options.lookuptable_angles = 300
    options.I_thresh = 1e-3
    options.bulk_profile = True
    options.randomize_surface = True
    options.c_azimuth = 0.0001
    options.periodic = True
    options.bulk_profile = True
    options.depth_spacing_bulk = 1e-10
    options.depth_spacing = 1e-10
    options.only_incidence_angle = True
    options.parallel = True
    options.theta_in = 0.4

    # set up Solcore materials
    Ge = material("Ge")()
    GaAs = material("GaAs")()
    GaInP = material("GaInP")(In=0.5)
    Ag = material("Ag")()
    SiN = material("Si3N4")()
    Air = material("Air")()
    Ta2O5 = material("TaOx1")()  # Ta2O5 (SOPRA database)
    MgF2 = material("MgF2")()  # MgF2 (SOPRA database)

    front_materials = [
        Layer(120e-9, MgF2),
        Layer(74e-9, Ta2O5),
        Layer(464e-9, GaInP),
        Layer(1682e-9, GaAs),
        Layer(200e-9, Ge),
    ]

    back_materials = [Layer(200e-9, Ge), Layer(100e-9, SiN)]

    # RT/TMM, matrix framework

    bulk_Ge = BulkLayer(bulkthick, Ge, name="Ge_bulk")  # bulk thickness in m

    ## RT with TMM lookup tables

    surf_planar = planar_surface()

    front_surf = Interface(
        "TMM", layers=front_materials, texture=surf_planar, name="GaInP_GaAs_TMM", coherent=True, prof_layers=[3, 4, 5]
    )

    back_surf = Interface(
        "TMM", layers=back_materials, texture=surf_planar, name="SiN_Ag_TMM", coherent=True, prof_layers=[1, 2]
    )

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_RT = calculate_RAT(SC, options)

    prof_front = results_RT[2][0]
    prof_back = results_RT[2][1]
    prof_Ge = results_RT[3][0]

    results_per_pass_RT = results_RT[1]

    # sum over passes
    results_per_layer_front_RT = np.sum(results_per_pass_RT["a"][0], 0)
    results_per_layer_back_RT = np.sum(results_per_pass_RT["a"][1], 0)

    options.coherency_list = ["c"] * len(front_materials) + ["i"] + ["c"] * len(back_materials)
    options.coherent = False

    tmmstrt = tmm_structure(front_materials + [Layer(bulkthick, Ge)] + back_materials, incidence=Air, transmission=Ag)

    res_tmm = tmmstrt.calculate(options, profile=True, layers=[3, 4, 5, 6, 7, 8])

    tmm_front = res_tmm["profile"][:, : prof_front.shape[1]]
    tmm_back = res_tmm["profile"][:, -prof_back.shape[1] :]
    tmm_Ge = res_tmm["profile"][:, prof_front.shape[1] : -prof_back.shape[1]]

    tmm_Ge_front = tmm_front[:, -1900:]
    prof_front_Ge = prof_front[:, -1900:]

    # at interface can get a spike, but this doesn't really mean anything (points in different materials)

    ratio = tmm_Ge_front[tmm_Ge_front > 1e-6] / prof_front_Ge.data[tmm_Ge_front > 1e-6]

    assert np.allclose(ratio, 1, atol=0.05)

    ratio = tmm_back[tmm_back > 1e-4] / prof_back.data[tmm_back > 1e-4]

    assert np.allclose(ratio, 1, atol=0.5)

    ratio = 1e9 * tmm_Ge[tmm_Ge > 1e-4] / prof_Ge[tmm_Ge > 1e-4]

    assert np.allclose(ratio, 1, atol=0.15)

    int_front_rt = np.trapz(tmm_front, dx=0.1, axis=1)
    int_front_arm = np.trapz(prof_front, dx=0.1, axis=1)

    assert int_front_rt == approx(int_front_arm, rel=0.02)

    assert res_tmm["R"] == approx(results_RT[0]["R"][0], abs=0.05)
    assert res_tmm["T"] == approx(results_RT[0]["T"][0], abs=0.05)
    assert res_tmm["A_per_layer"][:, :5] == approx(results_per_layer_front_RT, abs=0.05)
    assert res_tmm["A_per_layer"][:, 5] == approx(results_RT[0]["A_bulk"][0], abs=0.05)
    assert res_tmm["A_per_layer"][:, 6:] == approx(results_per_layer_back_RT, abs=0.05)


def test_tmm_rt_methods():
    from solcore.structure import Layer
    from solcore import material

    # rayflare imports
    from rayflare.textures.standard_rt_textures import regular_pyramids
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options
    from rayflare.ray_tracing import rt_structure

    # Thickness of bottom Ge layer
    bulkthick = 0.3e-6

    wavelengths = np.linspace(700, 1400, 4) * 1e-9

    # set options
    options = default_options()
    options.wavelengths = wavelengths
    options.project_name = "rt_tmm_comparisons_3"
    options.nx = 10
    options.ny = 10
    options.n_rays = 420 * 10 * 10
    options.n_theta_bins = 40
    options.lookuptable_angles = 300
    options.parallel = True
    options.I_thresh = 1e-3
    options.bulk_profile = True
    options.randomize_surface = True
    options.c_azimuth = 0.5
    options.periodic = True
    options.bulk_profile = True
    options.depth_spacing_bulk = 1e-9
    options.only_incidence_angle = True
    options.theta_in = 0.5

    # set up Solcore materials
    Ge = material("Ge")()

    GaAs = material("GaAs")()
    GaInP = material("GaInP")(In=0.5)
    Ag = material("Ag")()
    SiN = material("Si3N4")()
    Air = material("Air")()
    Ta2O5 = material("TaOx1")()  # Ta2O5 (SOPRA database)
    MgF2 = material("MgF2")()  # MgF2 (SOPRA database)

    front_materials = [
        Layer(120e-9, MgF2),
        Layer(74e-9, Ta2O5),
        Layer(464e-9, GaInP),
        Layer(1682e-9, GaAs),
        Layer(200e-9, Ge),
    ]

    back_materials = [Layer(200e-9, Ge), Layer(100e-9, SiN)]

    # RT/TMM, matrix framework

    bulk_Ge = BulkLayer(bulkthick, Ge, name="Ge_bulk")  # bulk thickness in m

    ## RT with TMM lookup tables

    pyramid_front = regular_pyramids(elevation_angle=20)
    pyramid_back = regular_pyramids(elevation_angle=20, upright=False)

    front_surf = Interface(
        "RT_TMM",
        layers=front_materials,
        texture=pyramid_front,
        name="GaInP_GaAs_RT",
        coherent=True,
        prof_layers=[3, 4, 5],
    )

    back_surf = Interface(
        "RT_TMM", layers=back_materials, texture=pyramid_back, name="SiN_Ag", coherent=True, prof_layers=[1, 2]
    )

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options, overwrite=True)

    results_RT = calculate_RAT(SC, options)

    prof_front = results_RT[2][0]
    prof_back = results_RT[2][1]
    prof_Ge = results_RT[3][0]

    results_per_pass_RT = results_RT[1]

    # sum over passes
    results_per_layer_front_RT = np.sum(results_per_pass_RT["a"][0], 0)

    front_surf = regular_pyramids(
        elevation_angle=20, interface_layers=front_materials, prof_layers=[3, 4, 5]
    )  # pyramid size in microns
    back_surf = regular_pyramids(
        elevation_angle=20, upright=False, interface_layers=back_materials, prof_layers=[1, 2]
    )  # pyramid size in microns

    rtstr = rt_structure([front_surf, back_surf], [Ge], [bulkthick], Air, Ag, options, use_TMM=True)
    # RT + TMM

    options.n_rays = 2000

    result_RT_only = rtstr.calculate(options)

    rt_front = result_RT_only["interface_profiles"][0]
    rt_back = result_RT_only["interface_profiles"][1]
    rt_Ge = result_RT_only["profile"]

    ratio = rt_front[rt_front > 1e-6] / prof_front.data[rt_front > 1e-6]

    assert np.allclose(ratio, 1, atol=0.25)

    ratio = rt_back[rt_back > 1e-3] / prof_back.data[rt_back > 1e-3]

    assert np.allclose(ratio, 1, atol=0.25)

    ratio = 1e9 * rt_Ge[rt_Ge > 5e-4] / prof_Ge[rt_Ge > 5e-4]

    assert np.allclose(ratio, 1, atol=0.15)

    int_front_rt = np.trapz(rt_front, dx=1, axis=1)
    int_front_arm = np.trapz(prof_front, dx=1, axis=1)

    # import matplotlib.pyplot as plt
    #
    # pos = np.arange(0, options.depth_spacing * 1e9 * len(rt_front.T), options.depth_spacing * 1e9)
    # plt.figure()
    # plt.plot(pos, rt_front.T)
    # plt.plot(prof_front.T, '--')
    # plt.xlabel("Position (nm)")
    # plt.ylabel("a(z)")
    # plt.legend([str(np.round(wl * 1e9, 0)) for wl in wavelengths])
    # plt.tight_layout()
    # plt.show()
    #
    # pos = np.arange(0, options.depth_spacing * 1e9 * len(rt_back.T), options.depth_spacing * 1e9)
    # plt.figure()
    # plt.plot(pos, rt_back.T)
    # plt.plot(prof_back.T, '--')
    # plt.xlabel("Position (nm)")
    # plt.ylabel("a(z)")
    # plt.legend([str(np.round(wl * 1e9, 0)) for wl in wavelengths])
    # plt.tight_layout()
    # plt.show()

    assert int_front_rt == approx(np.sum(result_RT_only["A_per_interface"][0], axis=1), rel=0.02)
    assert int_front_arm == approx(np.sum(results_per_layer_front_RT, axis=1), rel=0.02)
    assert int_front_rt == approx(int_front_arm, abs=0.075)

    assert result_RT_only["R"] == approx(results_RT[0]["R"][0], abs=0.075)
    assert result_RT_only["T"] == approx(results_RT[0]["T"][0], abs=0.075)
    assert result_RT_only["A_per_interface"][0] == approx(results_per_layer_front_RT, abs=0.075)
    assert result_RT_only["A_per_layer"][:, 0] == approx(results_RT[0]["A_bulk"][0], abs=0.075)
