from pytest import approx
import numpy as np

def test_plotting_funcs():
    from solcore import si, material
    from solcore.structure import Layer
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure

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
    options.wavelength = wavelengths
    options.orders = 50
    options.pol = "s"
    options.RCWA_method = "Inkstone"

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

    xs, ys, a_r, a_i = S4_setup.get_fourier_epsilon(4, wl_plot, options, plot=False)

    assert np.min(a_r) == approx(np.real(e_Ag), rel=0.2)
    assert np.max(a_r) == approx(np.real(e_SiN), rel=0.2)
    assert np.min(a_i) == approx(np.imag(e_SiN), abs=0.1)
    assert np.max(a_i) == approx(np.imag(e_Ag), rel=0.2)

    xs, ys, E, H, E_mag, H_mag = S4_setup.get_fields(
        4, wl_plot, options, extent=[[-100, 100], [-150, 150]], n_points=10, plot=False
    )

    assert len(xs) == 10
    assert len(ys) == 10
    assert np.all(E_mag > 0)
    assert np.all(H_mag > 0)
    assert E.shape == (len(xs), len(ys), 3)
    assert H.shape == (len(xs), len(ys), 3)