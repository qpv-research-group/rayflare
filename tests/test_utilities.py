from pytest import approx
import numpy as np

def test_TMM_profile():

    from rayflare.utilities import make_absorption_function
    from rayflare.transfer_matrix_method import tmm_structure
    from rayflare.options import default_options

    from solcore import material, si
    from solcore.solar_cell import SolarCell, Layer, Junction
    from solcore.solar_cell_solver import solar_cell_solver
    from solcore.solar_cell_solver import default_options as defaults_solcore

    # setting up Solcore materials
    Air = material('Air')()
    Si = material('Si')()
    GaAs = material('GaAs')()

    wl = np.linspace(550, 1201, 30) * 1e-9

    # setting options
    options = default_options()
    options.wavelengths = wl
    options.depth_spacing = si('1nm')
    options.theta_in = 0

    GaAs_total_d = si('3um')
    Si_total_d = si('300um')

    options.coherent = False
    options.coherency_list = ['c', 'i']

    tmmstr = tmm_structure([Layer(GaAs_total_d, GaAs), Layer(Si_total_d, Si)],
                           incidence=Air, transmission=Air)
    result_tmm = tmmstr.calculate_profile(options)

    total_R_tmm = result_tmm['R']
    profile_tmm = result_tmm['profile']

    Si_SC = material("Si")
    GaAs_SC = material("GaAs")
    T = 300

    n_material_GaAs_width = si("300nm")
    p_material_GaAs_width = tmmstr.stack.widths[0]*1e-9 - n_material_GaAs_width

    n_material_GaAs = GaAs_SC(Nd=si(3e18, "cm-3"), hole_diffusion_length=si("400nm"),
                              electron_mobility=50e-4, relative_permittivity=12.4)
    p_material_GaAs = GaAs_SC(Na=si(1e17, "cm-3"), electron_diffusion_length=si("1um"),
                              electron_mobility=100e-4, relative_permittivity=12.4)

    n_material_Si_width = si("500nm")
    p_material_Si_width = tmmstr.stack.widths[1]*1e-9  - n_material_Si_width

    n_material_Si = Si_SC(T=T, Nd=si(1e21, "cm-3"), hole_diffusion_length=si("10um"),
                          electron_mobility=50e-4, relative_permittivity=11.68)
    p_material_Si = Si_SC(T=T, Na=si(1e16, "cm-3"), electron_diffusion_length=si("290um"),
                          hole_mobility=400e-4, relative_permittivity=11.68)

    options_sc = defaults_solcore
    options_sc.optics_method = "external"
    options_sc.position = np.arange(0, tmmstr.width, options.depth_spacing)
    options_sc.wavelength = wl
    options_sc.theta = options.theta_in * 180 / np.pi
    options_sc.coherency_list = ['c', 'c', 'i', 'i']
    options_sc.no_back_reflection = False
    V = np.linspace(0, 2, 200)
    options_sc.voltages = V

    _, diff_absorb_fn = make_absorption_function(profile_tmm, tmmstr, options, matrix_method=False)

    solar_cell = SolarCell(
        [
            Junction([Layer(width=n_material_GaAs_width, material=n_material_GaAs, role='emitter'),
                      Layer(width=p_material_GaAs_width, material=p_material_GaAs, role='base')],
                     sn=2, sp=2, kind='DA'),
            Junction([Layer(width=n_material_Si_width, material=n_material_Si, role='emitter'),
                      Layer(width=p_material_Si_width, material=p_material_Si, role='base')],
                     sn=1, sp=1, kind='DA')
        ],
        external_reflected=total_R_tmm,
        external_absorbed=diff_absorb_fn)

    solar_cell_solver(solar_cell, 'qe', options_sc)

    solar_cell_SC = SolarCell(
        [
            Junction([Layer(width=n_material_GaAs_width, material=n_material_GaAs, role='emitter'),
                      Layer(width=p_material_GaAs_width, material=p_material_GaAs, role='base')],
                     sn=2, sp=2, kind='DA'),
            Junction([Layer(width=n_material_Si_width, material=n_material_Si, role='emitter'),
                      Layer(width=p_material_Si_width, material=p_material_Si, role='base')],
                     sn=1, sp=1, kind='DA')
        ], substrate=Air)

    options_sc.optics_method = "TMM"

    solar_cell_solver(solar_cell_SC, 'qe', options_sc)

    assert solar_cell_SC[0].qe["EQE"] == approx(solar_cell[0].qe["EQE"], abs = 1e-2)
    assert solar_cell_SC[1].qe["EQE"] == approx(solar_cell[1].qe["EQE"], abs = 1e-3)



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
    options.project_name = 'GaAs_GaAs_Si_test'
    options.n_theta_bins = 20
    options.nx = 5
    options.ny = 5
    options.depth_spacing = si('1nm')
    options.depth_spacing_bulk = si('10um')
    _, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                           options['c_azimuth'])
    options.bulk_profile = True
    options.n_rays = options.nx ** 2 * int(len(angle_vector) / 2)

    Air = material('Air')()
    Al2O3 = material("Al2O3")()
    Ag = material("Ag")()
    GaAs = material("GaAs")()
    InAlP = material("AlInP")(Al=0.5)
    GaInP = material("GaInP")(In=0.5)
    Si = material("Si")()
    MgF2 = material("203", nk_db=True)()
    Ta2O5 = material("410", nk_db=True)()

    GaAs_1_th = 120e-9
    GaAs_2_th = 1200e-9

    front_materials = [Layer(50e-9, MgF2), Layer(40e-9, Ta2O5),
                       Layer(30e-9, GaInP), Layer(GaAs_1_th, GaAs), Layer(30e-9, InAlP),
                       Layer(20e-9, GaAs),
                       Layer(30e-9, GaInP), Layer(GaAs_2_th, GaAs), Layer(30e-9, InAlP),
                       Layer(100e-9, GaAs)]
    back_materials = [Layer(62e-9, Ag), Layer(240e-9, Al2O3)]

    surf_back = regular_pyramids(elevation_angle=55, upright=False)

    front_surf = Interface('TMM', layers=front_materials, name='GaAs_GaAs',
                           coherent=True, prof_layers=np.arange(1, len(front_materials)+1))
    back_surf = Interface('RT_TMM', texture=surf_back, layers=back_materials,
                          name='Si_HIT_rear', prof_layers=np.arange(1, len(back_materials)+1),
                          coherent=True)

    bulk_Si = BulkLayer(250e-6, Si, name='Si_bulk')  # bulk thickness in m

    SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results = calculate_RAT(SC, options)

    profile_front = results[2][0]
    profile_Si = results[3][0]
    profile_back = results[2][1]

    widths_front = front_surf.widths
    width_bulk = bulk_Si.width
    widths_back = back_surf.widths

    positions, absorb_fn = make_absorption_function([profile_front, profile_Si, profile_back], SC, options, True)

    pos_front = np.arange(0, np.sum(widths_front)*1e9-1e-12, options.depth_spacing*1e9)
    pos_bulk = np.sum(widths_front)*1e6 + np.arange(0, width_bulk*1e6 - 1e-12,
                         options.depth_spacing_bulk*1e6)
    pos_back = np.sum(widths_front)*1e9 + width_bulk*1e9 + np.arange(0, np.sum(widths_back)*1e9,
                         options.depth_spacing*1e9)

    positions_expected = np.hstack((pos_front/1e9, pos_bulk/1e6, pos_back/1e9))

    generated = absorb_fn(positions_expected)

    generated_expected = np.hstack((profile_front[:,:len(pos_front)]*1e9,
                                    profile_Si[:,:len(pos_bulk)],
                                    profile_back[:,:len(pos_back)]*1e9))

    # Do one for a selection of layers, one for all layers profile calculated

    assert positions == approx(positions_expected)
    assert generated == approx(generated_expected, abs=1e-4)