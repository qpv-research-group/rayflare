from pytest import approx, mark
import numpy as np
import sys

@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_RAT():

    from solcore import si, material
    from solcore.structure import Layer
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure

    from solcore.solar_cell import SolarCell

    InAlP_hole_barrier = material('AlInP')(Al=0.5)
    GaAs_pn_junction = material('GaAs')()
    InGaP_e_barrier = material('GaInP')(In=0.5)
    Air = material('Air')()
    Ag = material('Ag')()

    wavelengths = np.linspace(303, 1000, 10) * 1e-9

    # define the problem

    x = 500

    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    RCWA_wl = wavelengths

    options = {'nm_spacing': 0.5,
               'n_theta_bins': 100,
               'c_azimuth': 1e-7,
               'pol': 'u',
               'wavelengths': RCWA_wl,
               'theta_in': 0, 'phi_in': 0,
               'parallel': True, 'n_jobs': -1,
               'phi_symmetry': np.pi / 2,
               'project_name': 'ultrathin',
               'A_per_order': True,
               'orders': 19
               }

    ropt = dict(LatticeTruncation='Circular',
                DiscretizedEpsilon=False,
                DiscretizationResolution=8,
                PolarizationDecomposition=False,
                PolarizationBasis='Default',
                LanczosSmoothing=True,
                SubpixelSmoothing=True,
                ConserveMemory=False,
                WeismannFormulation=True,
                Verbosity=0)

    options['S4_options'] = ropt


    SiN = material('Si3N4')()

    grating1 = [Layer(si(20, 'nm'), SiN)]
    grating2 = [Layer(si(80, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                      'radius': 115, 'angle': 0}])]

    solar_cell = SolarCell([Layer(material=InGaP_e_barrier, width=si('19nm')),
                            Layer(material=GaAs_pn_junction, width=si('85nm')),
                            Layer(material=InAlP_hole_barrier, width=si('19nm'))] + grating1 + grating2,
                           substrate=Ag)


    S4_setup = rcwa_structure(solar_cell, size, options, Air, Ag)

    RAT = S4_setup.calculate(options)

    assert RAT['R'] == approx(np.array([0.41892354, 0.43912852, 0.27507833, 0.18905804, 0.73269615,
                        0.73388915, 0.41056721, 0.63470072, 0.74084271, 0.90495267]))
    assert RAT['T'] == approx(np.array([2.13232413e-05, 3.27735248e-05, 1.09710895e-02, 4.60876152e-02,
                        7.56617334e-03, 1.31928539e-02, 1.70901324e-02, 1.69188028e-02,
                        4.71873992e-03, 2.69254910e-03]))
    assert RAT['A_per_layer'] == approx(np.array([
        [4.49927488e-01, 1.31031272e-01, 8.95650077e-05, 1.69406589e-20, 6.81598463e-06],
       [3.71341253e-01, 1.89132605e-01, 3.33630990e-04, 5.42101086e-20, 3.12191293e-05],
       [1.50714067e-01, 5.37243082e-01, 2.06429874e-02, 2.08166817e-17, 5.35044186e-03],
       [1.48012843e-01, 5.87066327e-01, 5.52266828e-03, 3.33066907e-16, 2.42525093e-02],
       [2.13228632e-02, 2.22185973e-01, 5.63157887e-04, 1.80411242e-16, 1.56656866e-02],
       [9.57566167e-06, 2.40269898e-01, 0.00000000e+00, 1.24900090e-16, 1.26385188e-02],
       [1.33226763e-15, 4.25140163e-01, 5.55111512e-16, 8.04911693e-16, 1.47202500e-01],
       [4.44089210e-16, 1.91128661e-01, 2.33146835e-15, 0.00000000e+00, 1.57251821e-01],
       [0.00000000e+00, 5.55111512e-17, 7.21644966e-16, 3.05311332e-16, 2.54438554e-01],
       [2.77555756e-16, 0.00000000e+00, 5.41233725e-16, 1.24900090e-16, 9.23547831e-02]]))

    assert len(RAT['basis_set']) == 19

    assert RAT['reciprocal'] == ((0.002, -0.0011547005383792516), (-0.0, 0.002309401076758503))



@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_RAT_angle_pol():

    from solcore import si, material
    from solcore.structure import Layer
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure

    from solcore.solar_cell import SolarCell
    from rayflare.options import default_options

    InAlP_hole_barrier = material('AlInP')(Al=0.5)
    GaAs_pn_junction = material('GaAs')()
    InGaP_e_barrier = material('GaInP')(In=0.5)
    Air = material('Air')()
    Ag = material('Ag')()

    wavelengths = np.linspace(303, 1000, 10) * 1e-9

    # define the problem

    x = 500

    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    options = default_options()
    options.wavelengths = wavelengths
    options.orders = 2

    ropt = dict(LatticeTruncation='Circular',
                DiscretizedEpsilon=False,
                DiscretizationResolution=8,
                PolarizationDecomposition=False,
                PolarizationBasis='Default',
                LanczosSmoothing=True,
                SubpixelSmoothing=True,
                ConserveMemory=False,
                WeismannFormulation=True,
                Verbosity=0)

    options.S4_options = ropt


    SiN = material('Si3N4')()

    grating1 = [Layer(si(20, 'nm'), SiN)]
    grating2 = [Layer(si(80, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                      'radius': 115, 'angle': 0}])]

    solar_cell = SolarCell([Layer(material=InGaP_e_barrier, width=si('19nm')),
                            Layer(material=GaAs_pn_junction, width=si('85nm')),
                            Layer(material=InAlP_hole_barrier, width=si('19nm'))] + grating1 + grating2,
                           substrate=Ag)


    S4_setup = rcwa_structure(solar_cell, size, options, Air, Ag)

    angles = [0, np.pi/5, np.pi/3]
    pols = ['s', 'p', 'u']


    for angle in angles:
        for pol in pols:
            options.pol = pol
            options.theta_in = angle
            options.phi_in = angle
            RAT = S4_setup.calculate(options)

            assert RAT['R'] + RAT['T'] + np.sum(RAT['A_per_layer'], 1) == approx(1)



@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_RAT_angle_pol_ninc():

    from solcore import si, material
    from solcore.structure import Layer
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure

    from solcore.solar_cell import SolarCell
    from rayflare.options import default_options

    InAlP_hole_barrier = material('AlInP')(Al=0.5)
    GaAs_pn_junction = material('GaAs')()
    InGaP_e_barrier = material('GaInP')(In=0.5)
    Ag = material('Ag')()

    wavelengths = np.linspace(303, 1000, 10) * 1e-9

    # define the problem

    x = 500

    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    options = default_options()
    options.wavelengths = wavelengths
    options.orders = 2

    ropt = dict(LatticeTruncation='Circular',
                DiscretizedEpsilon=False,
                DiscretizationResolution=8,
                PolarizationDecomposition=False,
                PolarizationBasis='Default',
                LanczosSmoothing=True,
                SubpixelSmoothing=True,
                ConserveMemory=False,
                WeismannFormulation=True,
                Verbosity=0)

    options.S4_options = ropt


    SiN = material('Si3N4')()

    grating1 = [Layer(si(20, 'nm'), SiN)]
    grating2 = [Layer(si(80, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                      'radius': 115, 'angle': 0}])]

    solar_cell = SolarCell([Layer(material=InGaP_e_barrier, width=si('19nm')),
                            Layer(material=GaAs_pn_junction, width=si('85nm')),
                            Layer(material=InAlP_hole_barrier, width=si('19nm'))] + grating1 + grating2,
                           substrate=Ag)


    S4_setup = rcwa_structure(solar_cell, size, options, SiN, Ag)

    angles = [0, np.pi/5, np.pi/3]
    pols = ['s', 'p', 'u']

    # import matplotlib.pyplot as plt
    for angle in angles:
        for pol in pols:
            options.pol = pol
            options.theta_in = angle
            options.phi_in = angle
            RAT = S4_setup.calculate(options)

            assert RAT['R'] + RAT['T'] + np.sum(RAT['A_per_layer'], 1) == approx(1)



@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_shapes():
    from solcore import material
    from solcore.structure import Layer
    from solcore.solar_cell import SolarCell

    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.options import default_options
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure

    wavelengths = np.linspace(800, 1150, 4)*1e-9

    options = default_options()
    options.wavelengths = wavelengths
    options.project_name = 'rcwa_mat_test'

    Ag = material("Ag")()
    Au = material("Au")()
    Si = material("Si")()
    SiN = material("Si3N4")()
    Air = material("Air")()

    grating_circles = [{'type': 'circle', 'mat': Ag, 'center': (0, 0), 'radius': 300}]

    grating_squares = [{'type': 'rectangle', 'mat': Ag, 'center': (0, 0), 'halfwidths': [300, 300], 'angle': 20}]

    grating_ellipse = [{'type': 'ellipse', 'mat': Ag, 'center': (0, 0), 'halfwidths': [300, 200], 'angle': 20}]

    grating_polygon = [{'type': 'polygon', 'mat': Ag, 'center': (0, 0), 'angle': 0,
                        'vertices': ((300, 0), (0, 300), (-300, 0))}]

    grating_circle_polygon = [{'type': 'circle', 'mat': Ag, 'center': (0, 0), 'radius': 100},
                              {'type': 'polygon', 'mat': Au, 'center': (600, 600), 'angle': -20,
                               'vertices': ((100, 0), (0, 100), (-100, 0))}]

    grating_list = [grating_circles, grating_squares, grating_ellipse, grating_polygon, grating_circle_polygon, None]

    bulk_Si = BulkLayer(100e-6, Si)

    A_bulk = []
    A_back = []
    R = []
    T= []

    d_v = ((1000, 0), (0, 1000))

    for i1, geometry in enumerate(grating_list):

        back_materials = [Layer(200e-9, SiN, geometry=geometry)]
        front_surf = Interface('TMM', layers=[], name='planar_front', coherent=True)
        back_surf = Interface('RCWA', layers=back_materials, name='grating_' + str(i1),
                              coherent=True, d_vectors=d_v, rcwa_orders=9)

        SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

        if i1 == 2:
            options.parallel = False

        process_structure(SC, options)

        res = calculate_RAT(SC, options)

        A_bulk.append(res[0]['A_bulk'][0])
        A_back.append(np.sum(res[1]['a'][1], 0).T[0])
        R.append(res[0]['R'][0])
        T.append(res[0]['T'][0])

        solar_cell = SolarCell(back_materials)
        S4_setup = rcwa_structure(solar_cell, size=d_v, options=options,
                                  incidence=Air, transmission=Ag)

        S4_setup.get_fourier_epsilon(layer_index=1, wavelength=500, options=options)

        if i1 == 2:
            options.parallel = False


    for i1 in range(len(grating_list)):

        assert (A_bulk[i1] + A_back[i1] + R[i1] + T[i1]).data == approx(1, abs=0.01)

    for i1 in range(len(grating_list)-1):

        assert np.all(A_back[i1] > A_back[-1])


@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_reciprocal_lattice():
    from rayflare.rigorous_coupled_wave_analysis.rcwa import get_reciprocal_lattice

    size = ((200, 0), (0, 200))

    a = get_reciprocal_lattice(size, 3)

    assert a[0] == approx((1/200, 0))
    assert a[1] == approx((0, 1/200))


@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_plotting_funcs():

    from solcore import si, material
    from solcore.structure import Layer
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
    import os

    from solcore.solar_cell import SolarCell
    from rayflare.options import default_options

    InAlP_hole_barrier = material('AlInP')(Al=0.5)
    GaAs_pn_junction = material('GaAs')()
    InGaP_e_barrier = material('GaInP')(In=0.5)
    Ag = material('Ag')()
    SiN = material('Si3N4')()

    wl_plot = 400
    e_SiN = (SiN.n(wl_plot*1e-9) + 1j*SiN.k(wl_plot*1e-9))**2
    e_Ag = (Ag.n(wl_plot*1e-9) + 1j*Ag.k(wl_plot*1e-9))**2

    wavelengths = np.linspace(300, 500, 3) * 1e-9

    # define the problem

    x = 500

    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    options = default_options()
    options.wavelengths = wavelengths
    options.orders = 50
    options.pol = 's'

    ropt = dict(LatticeTruncation='Circular',
                DiscretizedEpsilon=False,
                DiscretizationResolution=8,
                PolarizationDecomposition=False,
                PolarizationBasis='Default',
                LanczosSmoothing=True,
                SubpixelSmoothing=False,
                ConserveMemory=False,
                WeismannFormulation=True,
                Verbosity=0)

    options.S4_options = ropt

    grating = [Layer(si(100, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                      'radius': 115, 'angle': 0}])]

    solar_cell = SolarCell([Layer(material=InGaP_e_barrier, width=si('19nm')),
                            Layer(material=GaAs_pn_junction, width=si('85nm')),
                            Layer(material=InAlP_hole_barrier, width=si('19nm'))] + grating,
                           substrate=Ag)


    S4_setup = rcwa_structure(solar_cell, size, options, SiN, Ag)

    S4_setup.save_layer_postscript(4, options, 'test')

    current_dir = os.getcwd()

    assert os.path.isfile(os.path.join(current_dir, 'test.ps'))

    xs, ys, a_r, a_i = S4_setup.get_fourier_epsilon(4, wl_plot, options)

    assert np.min(a_r) == approx(np.real(e_Ag), rel=0.2)
    assert np.max(a_r) == approx(np.real(e_SiN), rel=0.2)
    assert np.min(a_i) == approx(np.imag(e_SiN), abs=0.1)
    assert np.max(a_i) == approx(np.imag(e_Ag), rel=0.2)

    xs, ys, a_r, a_i = S4_setup.get_fourier_epsilon(3, wl_plot, options, extent=[[-10, 10], [-20, 20]],
                                                    n_points=10, plot=False)

    e_InAlP = (InAlP_hole_barrier.n(wl_plot * 1e-9) + 1j * InAlP_hole_barrier.k(wl_plot * 1e-9)) ** 2
    assert a_r == approx(np.real(e_InAlP))
    assert a_i == approx(np.imag(e_InAlP))

    assert [np.min(xs), np.max(xs)] == [-10, 10]
    assert [np.min(ys), np.max(ys)] == [-20, 20]
    assert len(xs) == 10
    assert len(ys) == 10

    options.pol = (0.5, 0.5)

    xs, ys, E, H, E_mag, H_mag = S4_setup.get_fields(4, wl_plot, options, extent = [[-100, 100], [-150, 150]],
                                                     n_points=10, plot=False)

    assert len(xs) == 10
    assert len(ys) == 10
    assert np.all(E_mag > 0)
    assert np.all(H_mag > 0)
    assert E.shape == (len(xs), len(ys), 3)
    assert H.shape == (len(xs), len(ys), 3)

    options.pol = 's'

    xs, ys, E_1, H_1, E_mag, H_mag = S4_setup.get_fields(4, wl_plot, options)

    assert np.all(E_mag > 0)
    assert np.all(H_mag > 0)
    assert E_1.shape == (len(xs), len(ys), 3)
    assert H_1.shape == (len(xs), len(ys), 3)

    E_2, H_2 = S4_setup.get_fields_unit_cell(4, wl_plot, options, n_points=50)

    assert np.array(E_2).shape == (50, 50, 3)
    assert np.array(H_2).shape == (50, 50, 3)

    assert (np.min(E_1), np.max(E_1), np.min(H_1), np.max(H_1)) == approx((np.min(E_2), np.max(E_2), np.min(H_2), np.max(H_2)), rel=0.05)

    options.order = 7
    xs, ys, E, H, E_mag, H_mag = S4_setup.get_fields_z_integral(4, wl_plot, options, n_points=10)

    assert len(xs) == 10
    assert len(ys) == 10
    assert np.all(E_mag > 0)
    assert np.all(H_mag > 0)
    assert E.shape == (len(xs), len(ys), 3)
    assert H.shape == (len(xs), len(ys), 3)




