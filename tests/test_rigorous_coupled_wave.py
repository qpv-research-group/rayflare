from pytest import approx, mark
import numpy as np
import sys


@mark.skipif(sys.platform == "win32", reason="S4 (RCWA) only installed for tests under Linux and macOS")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_RAT(RCWA_method):
    from solcore import si, material
    from solcore.structure import Layer
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure

    from solcore.solar_cell import SolarCell

    InAlP_hole_barrier = material("AlInP")(Al=0.5)
    GaAs_pn_junction = material("GaAs")()
    InGaP_e_barrier = material("GaInP")(In=0.5)
    Air = material("Air")()
    Ag = material("Ag")()

    wavelengths = np.linspace(303, 1000, 10) * 1e-9

    # define the problem

    x = 500

    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    RCWA_wl = wavelengths

    options = {
        "nm_spacing": 0.5,
        "n_theta_bins": 100,
        "c_azimuth": 1e-7,
        "pol": "u",
        "wavelengths": RCWA_wl,
        "theta_in": 0,
        "phi_in": 0,
        "parallel": True,
        "n_jobs": -1,
        "phi_symmetry": np.pi / 2,
        "project_name": "ultrathin",
        "orders": 30,
        "RCWA_method": RCWA_method,
    }

    if RCWA_method == "S4":
        options["A_per_order"] = True

    else:
        options["A_per_order"] = False

    ropt = dict(
        LatticeTruncation="Circular",
        DiscretizedEpsilon=False,
        DiscretizationResolution=8,
        PolarizationDecomposition=False,
        PolarizationBasis="Default",
        LanczosSmoothing=True,
        SubpixelSmoothing=True,
        ConserveMemory=False,
        WeismannFormulation=True,
        Verbosity=0,
    )

    options["S4_options"] = ropt

    SiN = material("Si3N4")()

    grating1 = [Layer(si(20, "nm"), SiN)]
    grating2 = [
        Layer(si(80, "nm"), SiN, geometry=[{"type": "circle", "mat": Air, "center": (0, 0), "radius": 115, "angle": 0}])
    ]

    solar_cell = SolarCell(
        [
            Layer(material=InGaP_e_barrier, width=si("19nm")),
            Layer(material=GaAs_pn_junction, width=si("85nm")),
            Layer(material=InAlP_hole_barrier, width=si("19nm")),
        ]
        + grating1
        + grating2,
        substrate=Air,
    )

    S4_setup = rcwa_structure(solar_cell, size, options, Air, Ag)

    RAT = S4_setup.calculate(options)

    abs_diff = 0.001 if RCWA_method == "S4" else 0.02

    assert RAT["R"] == approx(
        np.array([0.419, 0.439, 0.258, 0.222, 0.753, 0.822, 0.796, 0.682, 0.972, 0.988]), abs=abs_diff
    )

    assert RAT["T"] == approx(
        np.array([0.0, 0.0, 0.014, 0.047, 0.016, 0.011, 0.025, 0.053, 0.028, 0.012]), abs=abs_diff
    )
    assert RAT["A_per_layer"] == approx(
        np.array(
            [
                [0.45, 0.131, 0.0, 0.0, 0.0],
                [0.371, 0.189, 0.0, 0.0, 0.0],
                [0.157, 0.55, 0.02, 0.0, 0.0],
                [0.153, 0.575, 0.003, 0.0, 0.0],
                [0.013, 0.218, 0.0, 0.0, 0.0],
                [0.0, 0.167, 0.0, 0.0, 0.0],
                [0.0, 0.18, 0.0, 0.0, 0.0],
                [0.0, 0.265, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        abs=abs_diff,
    )

    if RCWA_method == "S4":
        assert len(RAT["basis_set"]) == 19

        assert np.array(RAT["reciprocal"]) == approx(
            np.array([[0.002, -0.0011547005383792516], [-0.0, 0.002309401076758503]])
        )


@mark.skipif(sys.platform == "win32", reason="S4 (RCWA) only installed for tests under Linux and macOS")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_RAT_angle_pol(RCWA_method):
    from solcore import si, material
    from solcore.structure import Layer
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure

    from solcore.solar_cell import SolarCell
    from rayflare.options import default_options

    InAlP_hole_barrier = material("AlInP")(Al=0.5)
    GaAs_pn_junction = material("GaAs")()
    InGaP_e_barrier = material("GaInP")(In=0.5)
    Air = material("Air")()
    Ag = material("Ag")()

    wavelengths = np.linspace(303, 1000, 10) * 1e-9

    # define the problem

    x = 500

    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    options = default_options()
    options.wavelengths = wavelengths
    options.orders = 2
    options.RCWA_method = RCWA_method

    ropt = dict(
        LatticeTruncation="Circular",
        DiscretizedEpsilon=False,
        DiscretizationResolution=8,
        PolarizationDecomposition=False,
        PolarizationBasis="Default",
        LanczosSmoothing=True,
        SubpixelSmoothing=True,
        ConserveMemory=False,
        WeismannFormulation=True,
        Verbosity=0,
    )

    options.S4_options = ropt

    SiN = material("Si3N4")()

    grating1 = [Layer(si(20, "nm"), SiN)]
    grating2 = [
        Layer(si(80, "nm"), SiN, geometry=[{"type": "circle", "mat": Ag, "center": (0, 0), "radius": 115, "angle": 0}])
    ]

    solar_cell = SolarCell(
        [
            Layer(material=InGaP_e_barrier, width=si("19nm")),
            Layer(material=GaAs_pn_junction, width=si("85nm")),
            Layer(material=InAlP_hole_barrier, width=si("19nm")),
        ]
        + grating1
        + grating2,
        substrate=Ag,
    )

    S4_setup = rcwa_structure(solar_cell, size, options, Air, Ag)

    angles = [0, np.pi / 5, np.pi / 3]
    pols = ["s", "p", "u"]

    for angle in angles:
        for pol in pols:
            options.pol = pol
            options.theta_in = angle
            options.phi_in = angle
            RAT = S4_setup.calculate(options)

            assert RAT["R"] + RAT["T"] + np.sum(RAT["A_per_layer"], 1) == approx(1)


@mark.skipif(sys.platform == "win32", reason="S4 (RCWA) only installed for tests under Linux and macOS")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_RAT_angle_pol_ninc(RCWA_method):
    from solcore import si, material
    from solcore.structure import Layer
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure

    from solcore.solar_cell import SolarCell
    from rayflare.options import default_options

    InAlP_hole_barrier = material("AlInP")(Al=0.5)
    GaAs_pn_junction = material("GaAs")()
    InGaP_e_barrier = material("GaInP")(In=0.5)
    Ag = material("Ag")()

    wavelengths = np.linspace(303, 1000, 10) * 1e-9

    # define the problem

    x = 500

    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    options = default_options()
    options.wavelengths = wavelengths
    options.orders = 2
    options.RCWA_method = RCWA_method

    ropt = dict(
        LatticeTruncation="Circular",
        DiscretizedEpsilon=False,
        DiscretizationResolution=8,
        PolarizationDecomposition=False,
        PolarizationBasis="Default",
        LanczosSmoothing=True,
        SubpixelSmoothing=True,
        ConserveMemory=False,
        WeismannFormulation=True,
        Verbosity=0,
    )

    options.S4_options = ropt

    SiN = material("Si3N4")()

    grating1 = [Layer(si(20, "nm"), SiN)]
    grating2 = [
        Layer(si(80, "nm"), SiN, geometry=[{"type": "circle", "mat": Ag, "center": (0, 0), "radius": 115, "angle": 0}])
    ]

    solar_cell = SolarCell(
        [
            Layer(material=InGaP_e_barrier, width=si("19nm")),
            Layer(material=GaAs_pn_junction, width=si("85nm")),
            Layer(material=InAlP_hole_barrier, width=si("19nm")),
        ]
        + grating1
        + grating2,
        substrate=Ag,
    )

    S4_setup = rcwa_structure(solar_cell, size, options, SiN, Ag)

    angles = [0, np.pi / 5, np.pi / 3]
    pols = ["s", "p", "u"]

    # import matplotlib.pyplot as plt
    for angle in angles:
        for pol in pols:
            options.pol = pol
            options.theta_in = angle
            options.phi_in = angle
            RAT = S4_setup.calculate(options)

            assert RAT["R"] + RAT["T"] + np.sum(RAT["A_per_layer"], 1) == approx(1)


@mark.skipif(sys.platform == "win32", reason="S4 (RCWA) only installed for tests under Linux and macOS")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_shapes(RCWA_method):
    from solcore import material
    from solcore.structure import Layer
    from solcore.solar_cell import SolarCell

    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.options import default_options
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure

    wavelengths = np.linspace(800, 1150, 4) * 1e-9

    options = default_options()
    options.wavelengths = wavelengths
    options.project_name = "rcwa_mat_test"
    options.RCWA_method = RCWA_method
    options.parallel = False

    Ag = material("Ag")()
    Au = material("Au")()
    Si = material("Si")()
    SiN = material("Si3N4")()
    Air = material("Air")()

    grating_circles = [{"type": "circle", "mat": Ag, "center": (0, 0), "radius": 300}]
    grating_circles = [{"type": "circle", "mat": Ag, "center": (0, 0), "radius": 300}]

    grating_squares = [{"type": "rectangle", "mat": Ag, "center": (0, 0), "halfwidths": [300, 300], "angle": 20}]

    grating_ellipse = [{"type": "ellipse", "mat": Ag, "center": (0, 0), "halfwidths": [300, 200], "angle": 20}]

    grating_polygon = [
        {"type": "polygon", "mat": Ag, "center": (0, 0), "angle": 0, "vertices": ((300, 0), (0, 300), (-300, 0))}
    ]

    grating_circle_polygon = [
        {"type": "circle", "mat": Ag, "center": (0, 0), "radius": 100},
        {"type": "polygon", "mat": Au, "center": (200, 200), "angle": 0, "vertices": ((200, 0), (0, 200), (-200, 0))},
    ]

    grating_list = [grating_circles, grating_squares, grating_ellipse, grating_polygon, grating_circle_polygon, None]

    bulk_Si = BulkLayer(100e-6, Si)

    A_bulk = []
    A_back = []
    R = []
    T = []

    d_v = ((1000, 0), (0, 1000))

    for i1, geometry in enumerate(grating_list):
        back_materials = [Layer(200e-9, SiN, geometry=geometry)]
        front_surf = Interface("TMM", layers=[], name="planar_front", coherent=True)
        back_surf = Interface(
            "RCWA", layers=back_materials, name="grating_" + str(i1), coherent=True, d_vectors=d_v, rcwa_orders=9
        )

        SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

        if i1 == 2:
            options.parallel = False

        process_structure(SC, options)

        res = calculate_RAT(SC, options)

        A_bulk.append(res[0]["A_bulk"][0])
        A_back.append(np.sum(res[1]["a"][1], 0).T[0])
        R.append(res[0]["R"][0])
        T.append(res[0]["T"][0])

        solar_cell = SolarCell(back_materials)
        S4_setup = rcwa_structure(solar_cell, size=d_v, options=options, incidence=Air, transmission=Ag)

        S4_setup.get_fourier_epsilon(layer_index=1, wavelength=500, options=options, plot=False)

    for i1 in range(len(grating_list)):
        assert (A_bulk[i1] + A_back[i1] + R[i1] + T[i1]).data == approx(1, abs=0.01)

    for i1 in range(len(grating_list) - 1):
        assert np.all(A_back[i1] > A_back[-1])


@mark.skipif(sys.platform == "win32", reason="S4 (RCWA) only installed for tests under Linux and macOS")
def test_reciprocal_lattice():
    from rayflare.rigorous_coupled_wave_analysis.rcwa import get_reciprocal_lattice

    size = ((200, 0), (0, 200))

    a = get_reciprocal_lattice(size, 3)

    assert a[0] == approx((1 / 200, 0))
    assert a[1] == approx((0, 1 / 200))


@mark.skipif(sys.platform == "win32", reason="S4 (RCWA) only installed for tests under Linux and macOS")
def test_plotting_funcs():
    from solcore import si, material
    from solcore.structure import Layer
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
    import os

    from solcore.solar_cell import SolarCell
    from rayflare.options import default_options

    InAlP_hole_barrier = material("AlInP")(Al=0.5)
    GaAs_pn_junction = material("GaAs")()
    InGaP_e_barrier = material("GaInP")(In=0.5)
    Ag = material("Ag")()
    SiN = material("Si3N4")()

    wl_plot = 400
    e_SiN = (SiN.n(wl_plot * 1e-9) + 1j * SiN.k(wl_plot * 1e-9)) ** 2
    e_Ag = (Ag.n(wl_plot * 1e-9) + 1j * Ag.k(wl_plot * 1e-9)) ** 2

    wavelengths = np.linspace(300, 500, 3) * 1e-9

    # define the problem

    x = 500

    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    options = default_options()
    options.wavelengths = wavelengths
    options.orders = 50
    options.pol = "s"

    ropt = dict(
        LatticeTruncation="Circular",
        DiscretizedEpsilon=False,
        DiscretizationResolution=8,
        PolarizationDecomposition=False,
        PolarizationBasis="Default",
        LanczosSmoothing=True,
        SubpixelSmoothing=False,
        ConserveMemory=False,
        WeismannFormulation=True,
        Verbosity=0,
    )

    options.S4_options = ropt

    grating = [
        Layer(si(100, "nm"), SiN, geometry=[{"type": "circle", "mat": Ag, "center": (0, 0), "radius": 115, "angle": 0}])
    ]

    solar_cell = SolarCell(
        [
            Layer(material=InGaP_e_barrier, width=si("19nm")),
            Layer(material=GaAs_pn_junction, width=si("85nm")),
            Layer(material=InAlP_hole_barrier, width=si("19nm")),
        ]
        + grating,
        substrate=Ag,
    )

    S4_setup = rcwa_structure(solar_cell, size, options, SiN, Ag)

    S4_setup.save_layer_postscript(4, options, "test")

    current_dir = os.getcwd()

    assert os.path.isfile(os.path.join(current_dir, "test.ps"))

    xs, ys, a_r, a_i = S4_setup.get_fourier_epsilon(4, wl_plot, options, plot=False)

    assert np.min(a_r) == approx(np.real(e_Ag), rel=0.2)
    assert np.max(a_r) == approx(np.real(e_SiN), rel=0.2)
    assert np.min(a_i) == approx(np.imag(e_SiN), abs=0.1)
    assert np.max(a_i) == approx(np.imag(e_Ag), rel=0.2)

    xs, ys, a_r, a_i = S4_setup.get_fourier_epsilon(
        3, wl_plot, options, extent=[[-10, 10], [-20, 20]], n_points=10, plot=False
    )

    e_InAlP = (InAlP_hole_barrier.n(wl_plot * 1e-9) + 1j * InAlP_hole_barrier.k(wl_plot * 1e-9)) ** 2
    assert a_r == approx(np.real(e_InAlP))
    assert a_i == approx(np.imag(e_InAlP))

    assert [np.min(xs), np.max(xs)] == [-10, 10]
    assert [np.min(ys), np.max(ys)] == [-20, 20]
    assert len(xs) == 10
    assert len(ys) == 10

    options.pol = (0.5, 0.5)

    xs, ys, E, H, E_mag, H_mag = S4_setup.get_fields(
        4, wl_plot, options, extent=[[-100, 100], [-150, 150]], n_points=10, plot=False
    )

    assert len(xs) == 10
    assert len(ys) == 10
    assert np.all(E_mag > 0)
    assert np.all(H_mag > 0)
    assert E.shape == (len(xs), len(ys), 3)
    assert H.shape == (len(xs), len(ys), 3)

    options.pol = "s"

    xs, ys, E_1, H_1, E_mag, H_mag = S4_setup.get_fields(4, wl_plot, options, plot=False)

    assert np.all(E_mag > 0)
    assert np.all(H_mag > 0)
    assert E_1.shape == (len(xs), len(ys), 3)
    assert H_1.shape == (len(xs), len(ys), 3)

    E_2, H_2 = S4_setup.get_fields_unit_cell(4, wl_plot, options, n_points=50)

    assert np.array(E_2).shape == (50, 50, 3)
    assert np.array(H_2).shape == (50, 50, 3)

    assert (np.min(E_1), np.max(E_1), np.min(H_1), np.max(H_1)) == approx(
        (np.min(E_2), np.max(E_2), np.min(H_2), np.max(H_2)), rel=0.05
    )

    options.order = 7
    xs, ys, E, H, E_mag, H_mag = S4_setup.get_fields_z_integral(4, wl_plot, options, n_points=10, plot=False)

    assert len(xs) == 10
    assert len(ys) == 10
    assert np.all(E_mag > 0)
    assert np.all(H_mag > 0)
    assert E.shape == (len(xs), len(ys), 3)
    assert H.shape == (len(xs), len(ys), 3)


@mark.skipif(sys.platform == "win32", reason="S4 (RCWA) only installed for tests under Linux and macOS")
@mark.parametrize("RCWA_method", ["S4", "Inkstone"])
def test_matrix_generation(RCWA_method):
    from rayflare.rigorous_coupled_wave_analysis import RCWA
    from solcore.structure import Layer
    from solcore import material

    # rayflare imports
    from rayflare.options import default_options

    # Thickness of bottom Ge layer

    wavelengths = np.linspace(300, 1850, 50) * 1e-9

    # set options
    options = default_options()
    options.wavelengths = wavelengths
    options.project_name = "method_comparison_test"
    options.n_rays = 250
    options.n_theta_bins = 3
    options.lookuptable_angles = 100
    options.parallel = False
    options.c_azimuth = 0.001
    options.I_thresh = 1e-8
    options.bulk_profile = False
    options.RCWA_method = RCWA_method

    # set up Solcore materials
    Ge = material("Ge")()
    GaAs = material("GaAs")()
    GaInP = material("GaInP")(In=0.5)
    Air = material("Air")()
    Ta2O5 = material("TaOx1")()  # Ta2O5 (SOPRA database)
    MgF2 = material("MgF2")()  # MgF2 (SOPRA database)

    front_materials = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5), Layer(464e-9, GaInP), Layer(1682e-9, GaAs)]

    size = ((500, 0), (0, 500))

    full_mat, A_mat = RCWA(
        front_materials, size, 2, options, "test", Air, Ge, False, None, "front", "RCWA_test", False, False
    )

    assert full_mat.shape == (len(wavelengths), 6, options.n_theta_bins)
    assert A_mat.shape == (len(wavelengths), 4, options.n_theta_bins)
