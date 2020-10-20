from pytest import approx, mark
import numpy as np
import sys

@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_RAT():

    from solcore import si, material
    from solcore.structure import Layer
    from rayflare.rigorous_coupled_wave_analysis.rcwa import rcwa_structure

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
                WeismannFormulation=True)

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

    assert RAT['R'] == approx(np.array([0.41892312, 0.43915471, 0.26330785, 0.24965168, 0.72211567,
                                        0.73748424, 0.25604396, 0.61168331, 0.96145068, 0.97934733]))
    assert RAT['T'] == approx(np.array([2.24966117e-05, 3.43826894e-05, 1.11130804e-02, 3.86384787e-02,
                                        1.00939851e-02, 1.23302304e-02, 2.50071959e-02, 3.87165301e-02,
                                        1.06092446e-02, 3.54956143e-03]))
    assert RAT['A_per_layer'] == approx(np.array([
        [4.49928331e-01, 1.31031340e-01, 8.91098628e-05, 0.00000000e+00, 5.60522288e-06],
       [3.71321936e-01, 1.89120147e-01, 3.48001882e-04, 2.57498016e-19,  2.08207117e-05],
       [1.57660477e-01, 5.41950476e-01, 2.27547898e-02, 1.24900090e-16,  3.21332766e-03],
       [1.34987013e-01, 5.60525158e-01, 4.64399153e-03, 5.55111512e-16,  1.15536768e-02],
       [2.30511802e-02, 2.33130079e-01, 5.40139122e-04, 5.55111512e-17,  1.10689505e-02],
       [1.00274859e-05, 2.45390360e-01, 0.00000000e+00, 1.38777878e-16,  4.78513883e-03],
       [0.00000000e+00, 5.40475093e-01, 0.00000000e+00, 9.15933995e-16,  1.78473754e-01],
       [0.00000000e+00, 2.76288948e-01, 1.55431223e-15, 0.00000000e+00,  7.33112124e-02],
       [8.88178420e-16, 9.99200722e-16, 4.99600361e-16, 1.66533454e-16,  2.79400793e-02],
       [3.88578059e-16, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,  1.71031114e-02]]))

    assert len(RAT['basis_set']) == 19

    assert RAT['reciprocal'] == ((0.002, -0.0011547005383792516), (-0.0, 0.002309401076758503))

