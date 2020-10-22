from pytest import approx, mark
import numpy as np
import sys

@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_tmm_rcwa_structure_comparison():
    import numpy as np
    from solcore import si, material
    from solcore.structure import Layer
    from solcore.solar_cell import SolarCell

    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
    from rayflare.transfer_matrix_method import tmm_structure
    from rayflare.options import default_options

    InGaP = material('GaInP')(In=0.5)
    GaAs = material('GaAs')()
    Ge = material('Ge')()
    Ag = material('Ag')()
    Air = material('Air')()

    Al2O3 = material('Al2O3')()

    wavelengths = np.linspace(250, 1900, 500) * 1e-9

    options = default_options()

    options.wavelengths = wavelengths
    options.orders = 2

    size = ((100, 0), (0, 100))

    # anti-reflection coating
    ARC = [Layer(si('80nm'), Al2O3)]

    solar_cell = SolarCell(ARC + [Layer(material=InGaP, width=si('400nm')),
                                  Layer(material=GaAs, width=si('4000nm')),
                                  Layer(material=Ge, width=si('3000nm'))], substrate=Ag)

    rcwa_setup = rcwa_structure(solar_cell, size=size, options=options, incidence=Air, transmission=Ag)
    tmm_setup = tmm_structure(solar_cell, incidence=Air, transmission=Ag, no_back_reflection=False)

    for pol in ['s', 'p', 'u']:
        for angle in [0, np.pi / 3]:
            options['pol'] = pol
            options['theta_in'] = angle

            rcwa_result = rcwa_setup.calculate(options)
            tmm_result = tmm_setup.calculate(options)

            assert tmm_result['A_per_layer'] == approx(rcwa_result['A_per_layer'])
            assert tmm_result['R'] == approx(rcwa_result['R'])
            assert tmm_result['T'] == approx(rcwa_result['T'])

            assert np.sum(tmm_result['A_per_layer'], 1) + tmm_result['R'] + tmm_result['T'] == approx(1)
            assert np.sum(rcwa_result['A_per_layer'], 1) + rcwa_result['R'] + rcwa_result['T'] == approx(1)


@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_planar_structure():

    # solcore imports
    from solcore.structure import Layer
    from solcore import material
    from solcore.absorption_calculator import calculate_rat, OptiStack

    # rayflare imports
    from rayflare.textures.standard_rt_textures import planar_surface
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism.process_structure import process_structure
    from rayflare.matrix_formalism.multiply_matrices import calculate_RAT
    from rayflare.options import default_options

    # Thickness of bottom Ge layer
    bulkthick = 300e-6

    wavelengths = np.linspace(300, 1850, 50) * 1e-9

    # set options
    options = default_options()
    options.wavelengths = wavelengths
    options.project_name = 'method_comparison_test'
    options.n_rays = 250
    options.n_theta_bins = 3
    options.lookuptable_angles = 100
    options.parallel = True
    options.c_azimuth = 0.001

    # set up Solcore materials
    Ge = material('Ge')()
    GaAs = material('GaAs')()
    GaInP = material('GaInP')(In=0.5)
    Ag = material('Ag')()
    SiN = material('Si3N4')()
    Air = material('Air')()
    Ta2O5 = material('TaOx1')() # Ta2O5 (SOPRA database)
    MgF2 = material('MgF2')() # MgF2 (SOPRA database)

    front_materials = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5), Layer(464e-9, GaInP),
                       Layer(1682e-9, GaAs)]
    back_materials = [Layer(100E-9, SiN)]

    # TMM, matrix framework

    front_surf = Interface('TMM', layers=front_materials, name = 'GaInP_GaAs_TMM',
                           coherent=True)
    back_surf = Interface('TMM', layers=back_materials, name = 'SiN_Ag_TMM',
                          coherent=True)

    bulk_Ge = BulkLayer(bulkthick, Ge, name = 'Ge_bulk') # bulk thickness in m

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_TMM_Matrix = calculate_RAT(SC, options)

    results_per_pass_TMM_matrix = results_TMM_Matrix[1]

    results_per_layer_front_TMM_matrix = np.sum(results_per_pass_TMM_matrix['a'][0], 0)

    ## RT with TMM lookup tables

    surf = planar_surface() # [texture, flipped texture]

    front_surf = Interface('RT_TMM', layers=front_materials, texture=surf, name = 'GaInP_GaAs_RT',
                           coherent=True)
    back_surf = Interface('RT_TMM', layers=back_materials, texture = surf, name = 'SiN_Ag_RT_50k',
                          coherent=True)

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_RT = calculate_RAT(SC, options)

    results_per_pass_RT = results_RT[1]

    # only select absorbing layers, sum over passes
    results_per_layer_front_RT = np.sum(results_per_pass_RT['a'][0], 0)

    ## RCWA

    front_surf = Interface('RCWA', layers=front_materials, name = 'GaInP_GaAs_RCWA',
                           coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=2)
    back_surf = Interface('RCWA', layers=back_materials, name = 'SiN_Ag_RCWA',
                          coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=2)


    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_RCWA_Matrix = calculate_RAT(SC, options)

    results_per_pass_RCWA = results_RCWA_Matrix[1]

    # only select absorbing layers, sum over passes
    results_per_layer_front_RCWA = np.sum(results_per_pass_RCWA['a'][0], 0)


    ## pure TMM (from Solcore)
    all_layers = front_materials + [Layer(bulkthick, Ge)] + back_materials

    coh_list = len(front_materials)*['c'] + ['i'] + ['c']

    OS_layers = OptiStack(all_layers, substrate=Ag, no_back_reflection=False)

    TMM_res = calculate_rat(OS_layers, wavelength=wavelengths*1e9,
                            no_back_reflection=False, angle=options['theta_in']*180/np.pi, coherent=False,
                            coherency_list=coh_list, pol=options['pol'])

    # stack results for comparison
    TMM_reference = TMM_res['A_per_layer'][1:-2].T
    TMM_matrix = np.hstack((results_per_layer_front_TMM_matrix, results_TMM_Matrix[0].A_bulk[0].data[:,None]))
    RCWA_matrix = np.hstack((results_per_layer_front_RCWA, results_RCWA_Matrix[0].A_bulk[0].data[:, None]))
    RT_matrix = np.hstack((results_per_layer_front_RT, results_RT[0].A_bulk[0].data[:, None]))

    assert TMM_reference == approx(TMM_matrix, abs=0.01)
    assert TMM_reference == approx(RCWA_matrix, abs=0.01)
    assert TMM_reference == approx(RT_matrix, abs=0.2)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # cols = sns.cubehelix_palette(3)
    # plt.figure()
    # plt.plot(wavelengths, TMM_matrix, color=cols[0])
    # plt.plot(wavelengths, RCWA_matrix, color=cols[1])
    # plt.plot(wavelengths, RT_matrix, color=cols[2])
    # plt.show()


@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_planar_structure_45deg():

    # solcore imports
    from solcore.structure import Layer
    from solcore import material
    from solcore.absorption_calculator import calculate_rat, OptiStack

    # rayflare imports
    from rayflare.textures.standard_rt_textures import planar_surface
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism.process_structure import process_structure
    from rayflare.matrix_formalism.multiply_matrices import calculate_RAT
    from rayflare.options import default_options

    # Thickness of bottom Ge layer
    bulkthick = 300e-6

    wavelengths = np.linspace(300, 1850, 30) * 1e-9

    # set options
    options = default_options()
    options.wavelengths = wavelengths
    options.project_name = 'method_comparison_test_45deg'
    options.n_rays = 500
    options.n_theta_bins = 20
    options.lookuptable_angles = 100
    options.parallel = True
    options.c_azimuth = 0.001
    options.theta_in = 0.6*np.pi/2
    options.nx = 1
    options.ny = 1
    options.pol = 's'

    # set up Solcore materials
    Ge = material('Ge')()
    GaAs = material('GaAs')()
    GaInP = material('GaInP')(In=0.5)
    Ag = material('Ag')()
    SiN = material('Si3N4')()
    Air = material('Air')()
    Ta2O5 = material('TaOx1')() # Ta2O5 (SOPRA database)
    MgF2 = material('MgF2')() # MgF2 (SOPRA database)

    front_materials = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5), Layer(464e-9, GaInP),
                       Layer(1682e-9, GaAs)]
    back_materials = [Layer(100E-9, SiN)]

    # TMM, matrix framework

    front_surf = Interface('TMM', layers=front_materials, name = 'GaInP_GaAs_TMM',
                           coherent=True)
    back_surf = Interface('TMM', layers=back_materials, name = 'SiN_Ag_TMM',
                          coherent=True)

    bulk_Ge = BulkLayer(bulkthick, Ge, name = 'Ge_bulk') # bulk thickness in m

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_TMM_Matrix = calculate_RAT(SC, options)

    results_per_pass_TMM_matrix = results_TMM_Matrix[1]

    results_per_layer_front_TMM_matrix = np.sum(results_per_pass_TMM_matrix['a'][0], 0)

    ## RT with TMM lookup tables

    surf = planar_surface() # [texture, flipped texture]

    front_surf = Interface('RT_TMM', layers=front_materials, texture=surf, name = 'GaInP_GaAs_RT',
                           coherent=True)
    back_surf = Interface('RT_TMM', layers=back_materials, texture = surf, name = 'SiN_Ag_RT_50k',
                          coherent=True)

    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_RT = calculate_RAT(SC, options)

    results_per_pass_RT = results_RT[1]

    # only select absorbing layers, sum over passes
    results_per_layer_front_RT = np.sum(results_per_pass_RT['a'][0], 0)

    ## RCWA

    front_surf = Interface('RCWA', layers=front_materials, name = 'GaInP_GaAs_RCWA',
                           coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=2)
    back_surf = Interface('RCWA', layers=back_materials, name = 'SiN_Ag_RCWA',
                          coherent=True, d_vectors = ((500,0), (0,500)), rcwa_orders=2)


    SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

    process_structure(SC, options)

    results_RCWA_Matrix = calculate_RAT(SC, options)

    results_per_pass_RCWA = results_RCWA_Matrix[1]

    # only select absorbing layers, sum over passes
    results_per_layer_front_RCWA = np.sum(results_per_pass_RCWA['a'][0], 0)


    ## pure TMM (from Solcore)
    all_layers = front_materials + [Layer(bulkthick, Ge)] + back_materials

    coh_list = len(front_materials)*['c'] + ['i'] + ['c']

    OS_layers = OptiStack(all_layers, substrate=Ag, no_back_reflection=False)

    TMM_res = calculate_rat(OS_layers, wavelength=wavelengths*1e9,
                            no_back_reflection=False, angle=options['theta_in']*180/np.pi, coherent=False,
                            coherency_list=coh_list, pol=options['pol'])

    # stack results for comparison
    TMM_reference = TMM_res['A_per_layer'][1:-2].T
    TMM_matrix = np.hstack((results_per_layer_front_TMM_matrix, results_TMM_Matrix[0].A_bulk[0].data[:,None]))
    RCWA_matrix = np.hstack((results_per_layer_front_RCWA, results_RCWA_Matrix[0].A_bulk[0].data[:, None]))
    RT_matrix = np.hstack((results_per_layer_front_RT, results_RT[0].A_bulk[0].data[:, None]))

    assert TMM_reference == approx(TMM_matrix, abs=0.05)
    assert TMM_reference == approx(RCWA_matrix, abs=0.05)
    assert TMM_reference == approx(RT_matrix, abs=0.3)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # cols = sns.cubehelix_palette(4)
    # plt.figure()
    # plt.plot(wavelengths, TMM_reference, '--', color=cols[0])
    # plt.plot(wavelengths, TMM_matrix, color=cols[1])
    # plt.plot(wavelengths, RCWA_matrix, color=cols[2])
    # plt.plot(wavelengths, RT_matrix, color=cols[3])
    # plt.show()


def test_absorption_profile():
    from rayflare.ray_tracing.rt import rt_structure
    from rayflare.textures import planar_surface
    from rayflare.options import default_options
    from solcore import material
    from solcore import si
    from rayflare.transfer_matrix_method.tmm import tmm_structure
    from solcore.structure import Layer

    Air = material('Air')()
    Si = material('Si')()
    GaAs = material('GaAs')()
    Ge = material('Ge')()

    triangle_surf = planar_surface()

    options = default_options()

    options.wavelengths = np.linspace(700, 1400, 4)*1e-9
    options.theta_in = 45*np.pi/180
    options.nx = 5
    options.ny = 5
    options.pol = 'u'
    options.n_rays = 2000
    options.depth_spacing = 1e-6

    rtstr = rt_structure(textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
                        materials = [GaAs, Si, Ge],
                        widths=[si('100um'), si('70um'), si('50um')], incidence=Air, transmission=Air)
    result_rt = rtstr.calculate(options)


    stack = [Layer(si('100um'), GaAs), Layer(si('70um'), Si), Layer(si('50um'), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air,
                         no_back_reflection=False)

    output = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile = output['profile'][output['profile'] > 1e-7]
    rt_profile = result_rt['profile'][output['profile'] > 1e-7]

    assert output['profile'].shape == result_rt['profile'].shape
    assert rt_profile == approx(tmm_profile, rel=0.4)


def test_absorption_profile_incoh_angles():
    from rayflare.ray_tracing.rt import rt_structure
    from rayflare.textures import planar_surface
    from rayflare.options import default_options
    from solcore import material
    from solcore import si
    from rayflare.transfer_matrix_method.tmm import tmm_structure
    from solcore.structure import Layer

    Air = material('Air')()
    Si = material('Si')()
    GaAs = material('GaAs')()
    Ge = material('Ge')()

    triangle_surf = planar_surface()

    options = default_options()

    options.wavelengths = np.linspace(700, 1400, 4)*1e-9
    options.nx = 5
    options.ny = 5
    options.n_rays = 2000
    options.depth_spacing = 1e-6
    options.coherent = False
    options.coherency_list = ['c', 'i', 'i']
    options.theta_in = np.pi/4
    options.phi_in = np.pi/3
    options.pol = 's'

    rtstr = rt_structure(textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
                        materials = [GaAs, Si, Ge],
                        widths=[si('100um'), si('70um'), si('50um')], incidence=Air, transmission=Air)
    result_rt_s = rtstr.calculate(options)


    stack = [Layer(si('100um'), GaAs), Layer(si('70um'), Si), Layer(si('50um'), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air,
                         no_back_reflection=False)

    output_s = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_s = output_s['profile'][output_s['profile'] > 1e-7]
    rt_profile_s = result_rt_s['profile'][output_s['profile'] > 1e-7]

    assert output_s['profile'].shape == result_rt_s['profile'].shape
    assert rt_profile_s == approx(tmm_profile_s, rel=0.2)

    options.pol = 'p'

    result_rt_p = rtstr.calculate(options)
    output_p = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_p = output_p['profile'][output_p['profile'] > 1e-7]
    rt_profile_p = result_rt_p['profile'][output_p['profile'] > 1e-7]

    assert output_p['profile'].shape == result_rt_p['profile'].shape
    assert rt_profile_p == approx(tmm_profile_p, rel=0.2)

    options.pol = 'u'

    result_rt_u = rtstr.calculate(options)
    output_u = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_u = output_u['profile'][output_u['profile'] > 1e-7]
    rt_profile_u = result_rt_u['profile'][output_u['profile'] > 1e-7]

    assert output_u['profile'].shape == result_rt_u['profile'].shape
    assert rt_profile_u == approx(tmm_profile_u, rel=0.2)


def test_absorption_profile_coh_angles():
    from rayflare.ray_tracing.rt import rt_structure
    from rayflare.textures import planar_surface
    from rayflare.options import default_options
    from solcore import material
    from solcore import si
    from rayflare.transfer_matrix_method.tmm import tmm_structure
    from solcore.structure import Layer

    Air = material('Air')()
    Si = material('Si')()
    GaAs = material('GaAs')()
    Ge = material('Ge')()

    triangle_surf = planar_surface()

    options = default_options()

    options.wavelengths = np.linspace(700, 1400, 4)*1e-9
    options.nx = 5
    options.ny = 5
    options.n_rays = 2000
    options.depth_spacing = 1e-6
    options.theta_in = np.pi/4
    options.phi_in = np.pi/3
    options.pol = 's'

    rtstr = rt_structure(textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
                        materials = [GaAs, Si, Ge],
                        widths=[si('100um'), si('70um'), si('50um')], incidence=Air, transmission=Air)
    result_rt_s = rtstr.calculate(options)


    stack = [Layer(si('100um'), GaAs), Layer(si('70um'), Si), Layer(si('50um'), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air,
                         no_back_reflection=False)

    output_s = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_s = output_s['profile'][output_s['profile'] > 1e-7]
    rt_profile_s = result_rt_s['profile'][output_s['profile'] > 1e-7]

    assert output_s['profile'].shape == result_rt_s['profile'].shape
    assert rt_profile_s == approx(tmm_profile_s, rel=0.4)

    options.pol = 'p'

    result_rt_p = rtstr.calculate(options)
    output_p = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_p = output_p['profile'][output_p['profile'] > 1e-7]
    rt_profile_p = result_rt_p['profile'][output_p['profile'] > 1e-7]

    assert output_p['profile'].shape == result_rt_p['profile'].shape
    assert rt_profile_p == approx(tmm_profile_p, rel=0.4)

    options.pol = 'u'

    result_rt_u = rtstr.calculate(options)
    output_u = strt.calculate(options, profile=True, layers=[1, 2, 3])

    tmm_profile_u = output_u['profile'][output_u['profile'] > 1e-7]
    rt_profile_u = result_rt_u['profile'][output_u['profile'] > 1e-7]

    assert output_u['profile'].shape == result_rt_u['profile'].shape
    assert rt_profile_u == approx(tmm_profile_u, rel=0.4)


@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_rcwa_tmm_profiles_coh():
    from rayflare.options import default_options
    from solcore import material
    from solcore import si
    from rayflare.transfer_matrix_method.tmm import tmm_structure
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
    from solcore.structure import Layer

    Air = material('Air')()
    Si = material('Si')()
    GaAs = material('GaAs')()
    Ge = material('Ge')()

    options = default_options()

    options.wavelengths = np.linspace(700, 1400, 4)*1e-9

    options.depth_spacing = 10e-9
    options.theta_in = np.pi/3
    options.phi_in = np.pi/4
    options.pol = 's'

    stack = [Layer(si('500nm'), GaAs), Layer(si('1.1um'), Si), Layer(si('0.834um'), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air,
                         no_back_reflection=False)

    strt_rcwa = rcwa_structure(stack, ((100, 0), (0, 100)), options, Air, Air)
    strt_rcwa.calculate(options)
    output_rcwa_s = strt_rcwa.calculate_profile(options)

    output_s = strt.calculate_profile(options)

    tmm_profile_s = output_s[output_s > 1e-7]
    rcwa_profile_s = output_rcwa_s[output_s > 1e-7]

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.plot(output_s[2], '--')
    # plt.plot(output_rcwa_s[2], '-.')
    # plt.plot()
    # plt.show()
    #
    # plt.figure()
    # plt.show()

    assert rcwa_profile_s == approx(tmm_profile_s, rel=0.02)

    options.pol = 'p'

    stack = [Layer(si('500nm'), GaAs), Layer(si('1.1um'), Si), Layer(si('0.834um'), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air,
                         no_back_reflection=False)

    strt_rcwa = rcwa_structure(stack, ((100, 0), (0, 100)), options, Air, Air)
    strt_rcwa.calculate(options)
    output_rcwa_p = strt_rcwa.calculate_profile(options)

    output_p = strt.calculate_profile(options)

    tmm_profile_p = output_p[output_p > 1e-7]
    rcwa_profile_p = output_rcwa_p[output_p > 1e-7]

    assert rcwa_profile_p == approx(tmm_profile_p, rel=0.02)

    options.pol = 'u'

    stack = [Layer(si('500nm'), GaAs), Layer(si('1.1um'), Si), Layer(si('0.834um'), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air,
                         no_back_reflection=False)

    strt_rcwa = rcwa_structure(stack, ((100, 0), (0, 100)), options, Air, Air)
    strt_rcwa.calculate(options)
    output_rcwa_u = strt_rcwa.calculate_profile(options)

    output_u = strt.calculate_profile(options)

    tmm_profile_u = output_u[output_u > 1e-7]
    rcwa_profile_u = output_rcwa_u[output_u > 1e-7]

    assert rcwa_profile_u == approx(tmm_profile_u, rel=0.02)