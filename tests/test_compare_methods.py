from pytest import approx, mark
import numpy as np
import sys

@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_tmm_rcwa_structure_comparison():
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
    from rayflare.matrix_formalism import process_structure, calculate_RAT
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
    options.I_thresh = 1e-8
    options.bulk_profile = False

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

    assert TMM_reference == approx(TMM_matrix, abs=0.02)
    assert TMM_reference == approx(RCWA_matrix, abs=0.02)
    assert TMM_reference == approx(RT_matrix, abs=0.2)

    # check normalization

    assert (results_TMM_Matrix[0].R[0] + results_TMM_Matrix[0].T[0] + np.sum(results_per_layer_front_TMM_matrix, 1) + results_TMM_Matrix[0].A_bulk[0]).data == approx(1)
    assert (results_RCWA_Matrix[0].R[0] + results_RCWA_Matrix[0].T[0] + np.sum(results_per_layer_front_RCWA, 1) + results_RCWA_Matrix[0].A_bulk[0]).data == approx(1)
    assert (results_RT[0].R[0] + results_RT[0].T[0] + np.sum(results_per_layer_front_RT, 1) +
            results_RT[0].A_bulk[0]).data == approx(1)


@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_planar_structure_45deg():

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
    options.project_name = 'method_comparison_test_45deg'
    options.n_rays = 500
    options.n_theta_bins = 20
    options.lookuptable_angles = 100
    options.parallel = True
    options.c_azimuth = 0.001
    options.theta_in = 0.6*np.pi/2
    options.nx = 1
    options.ny = 1
    options.I_thresh = 1e-8
    options.bulk_profile = False

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

    assert (results_TMM_Matrix[0].R[0] + results_TMM_Matrix[0].T[0] + np.sum(results_per_layer_front_TMM_matrix, 1) +
            results_TMM_Matrix[0].A_bulk[0]).data == approx(1)
    assert (results_RCWA_Matrix[0].R[0] + results_RCWA_Matrix[0].T[0] + np.sum(results_per_layer_front_RCWA, 1) +
            results_RCWA_Matrix[0].A_bulk[0]).data == approx(1)
    assert (results_RT[0].R[0] + results_RT[0].T[0] + np.sum(results_per_layer_front_RT, 1) +
            results_RT[0].A_bulk[0]).data == approx(1)


@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_tmm_rcwa_pol_angle():
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
    options.project_name = 'method_comparison_test_angle_pol'
    options.n_theta_bins = 50
    options.lookuptable_angles = 100
    options.parallel = True
    options.c_azimuth = 0.001
    options.I_thresh = 1e-8
    options.only_incidence_angle = False
    options.bulk_profile = False

    # set up Solcore materials
    Ge = material('Ge')()
    GaAs = material('GaAs')()
    GaInP = material('GaInP')(In=0.5)
    Ag = material('Ag')()
    SiN = material('Si3N4')()
    Air = material('Air')()
    Ta2O5 = material('TaOx1')()  # Ta2O5 (SOPRA database)
    MgF2 = material('MgF2')()  # MgF2 (SOPRA database)

    front_materials = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5), Layer(464e-9, GaInP),
                       Layer(1682e-9, GaAs)]
    back_materials = [Layer(100E-9, SiN)]

    angles = [0, np.pi/5, np.pi/3]
    pols = ['s', 'p', 'u']
    # TMM, matrix framework

    for angle in angles:
        for pol in pols:

            options.pol = pol
            options.theta_in = angle
            options.phi_in = angle

            front_surf = Interface('TMM', layers=front_materials, name='GaInP_GaAs_TMM'+str(pol),
                                   coherent=True)
            back_surf = Interface('TMM', layers=back_materials, name='SiN_Ag_TMM'+str(pol),
                                  coherent=True)

            bulk_Ge = BulkLayer(bulkthick, Ge, name='Ge_bulk')  # bulk thickness in m

            SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

            process_structure(SC, options)

            results_TMM_Matrix = calculate_RAT(SC, options)

            results_per_pass_TMM_matrix = results_TMM_Matrix[1]

            results_per_layer_front_TMM_matrix = np.sum(results_per_pass_TMM_matrix['a'][0], 0)


            ## RCWA

            front_surf = Interface('RCWA', layers=front_materials, name='GaInP_GaAs_RCWA'+str(pol),
                                   coherent=True, d_vectors=((500, 0), (0, 500)), rcwa_orders=2)
            back_surf = Interface('RCWA', layers=back_materials, name='SiN_Ag_RCWA'+str(pol),
                                  coherent=True, d_vectors=((500, 0), (0, 500)), rcwa_orders=2)

            SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

            process_structure(SC, options)

            results_RCWA_Matrix = calculate_RAT(SC, options)

            results_per_pass_RCWA = results_RCWA_Matrix[1]

            # only select absorbing layers, sum over passes
            results_per_layer_front_RCWA = np.sum(results_per_pass_RCWA['a'][0], 0)

            ## pure TMM (from Solcore)
            all_layers = front_materials + [Layer(bulkthick, Ge)] + back_materials

            coh_list = len(front_materials) * ['c'] + ['i'] + ['c']

            OS_layers = OptiStack(all_layers, substrate=Ag, no_back_reflection=False)

            TMM_res = calculate_rat(OS_layers, wavelength=wavelengths * 1e9,
                                    no_back_reflection=False, angle=options.theta_in * 180 / np.pi, coherent=False,
                                    coherency_list=coh_list, pol=options.pol)

            # stack results for comparison
            TMM_reference = TMM_res['A_per_layer'][1:-2].T
            TMM_matrix = np.hstack((results_per_layer_front_TMM_matrix, results_TMM_Matrix[0].A_bulk[0].data[:, None]))
            RCWA_matrix = np.hstack((results_per_layer_front_RCWA, results_RCWA_Matrix[0].A_bulk[0].data[:, None]))

            print(pol, angle)

            assert TMM_reference == approx(TMM_matrix, abs=0.05)
            assert TMM_reference == approx(RCWA_matrix, abs=0.05)

            assert (results_TMM_Matrix[0].R[0] + results_TMM_Matrix[0].T[0] + np.sum(results_per_layer_front_TMM_matrix, 1) +
                    results_TMM_Matrix[0].A_bulk[0]).data == approx(1)
            assert (results_RCWA_Matrix[0].R[0] + results_RCWA_Matrix[0].T[0] + np.sum(results_per_layer_front_RCWA, 1) +
                    results_RCWA_Matrix[0].A_bulk[0]).data == approx(1)



def test_absorption_profile():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import planar_surface
    from rayflare.options import default_options
    from solcore import material
    from solcore import si
    from rayflare.transfer_matrix_method import tmm_structure
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
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import planar_surface
    from rayflare.options import default_options
    from solcore import material
    from solcore import si
    from rayflare.transfer_matrix_method import tmm_structure
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
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import planar_surface
    from rayflare.options import default_options
    from solcore import material
    from solcore import si
    from rayflare.transfer_matrix_method import tmm_structure
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
    from rayflare.transfer_matrix_method import tmm_structure
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
    output_rcwa_s = strt_rcwa.calculate_profile(options)['profile']

    output_s = strt.calculate_profile(options)['profile']

    tmm_profile_s = output_s[output_s > 1e-7]
    rcwa_profile_s = output_rcwa_s[output_s > 1e-7]

    assert rcwa_profile_s == approx(tmm_profile_s, rel=0.02)

    options.pol = 'p'

    stack = [Layer(si('500nm'), GaAs), Layer(si('1.1um'), Si), Layer(si('0.834um'), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air,
                         no_back_reflection=False)

    strt_rcwa = rcwa_structure(stack, ((100, 0), (0, 100)), options, Air, Air)
    strt_rcwa.calculate(options)
    output_rcwa_p = strt_rcwa.calculate_profile(options)['profile']

    output_p = strt.calculate_profile(options)['profile']

    tmm_profile_p = output_p[output_p > 1e-7]
    rcwa_profile_p = output_rcwa_p[output_p > 1e-7]

    assert rcwa_profile_p == approx(tmm_profile_p, rel=0.02)

    options.pol = 'u'

    stack = [Layer(si('500nm'), GaAs), Layer(si('1.1um'), Si), Layer(si('0.834um'), Ge)]

    strt = tmm_structure(stack, incidence=Air, transmission=Air,
                         no_back_reflection=False)

    strt_rcwa = rcwa_structure(stack, ((100, 0), (0, 100)), options, Air, Air)
    strt_rcwa.calculate(options)
    output_rcwa_u = strt_rcwa.calculate_profile(options)['profile']

    output_u = strt.calculate_profile(options)['profile']

    tmm_profile_u = output_u[output_u > 1e-7]
    rcwa_profile_u = output_rcwa_u[output_u > 1e-7]

    assert rcwa_profile_u == approx(tmm_profile_u, rel=0.02)


@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_rcwa_tmm_matrix_check_sums():
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
    options.depth_spacing = 1e-9
    options.only_incidence_angle = False
    options.nx = 4
    options.ny = 4
    options.bulk_profile = False

    _, _, angle_vector = make_angle_vector(options.n_theta_bins,
                                           options.phi_symmetry, options.c_azimuth)

    for pol in ['s', 'p', 'u']:
        options.project_name = 'rcwa_tmm_matrix_profiles_' + pol
        options.pol = pol

        SiN = material('Si3N4')()
        GaAs = material('GaAs')()
        GaInP = material('GaInP')(In=0.5)
        Ag = material('Ag')()
        Air = material('Air')()
        Ta2O5 = material('TaOx1')()  # Ta2O5 (SOPRA database)
        MgF2 = material('MgF2')()  # MgF2 (SOPRA database)

        front_materials = [Layer(100e-9, MgF2), Layer(50e-9, GaInP), Layer(100e-9, Ta2O5), Layer(200e-9, GaAs)]
        back_materials = [Layer(50E-9, GaInP)]

        prof_layers = np.arange(len(front_materials)) + 1

        ## pure TMM with incoherent thick layer
        all_layers = front_materials + [Layer(bulkthick, SiN)] + back_materials

        coh_list = len(front_materials) * ['c'] + ['i'] + ['c']

        options.coherent = False
        options.coherency_list = coh_list

        OS_layers = tmm_structure(all_layers, incidence=Air,
                                  transmission=Ag, no_back_reflection=False)


        th_ind = np.random.randint(0, 4)
        phi_in = np.random.uniform(0, np.pi)

        options.theta_in = angle_vector[th_ind, 1]
        options.phi_in = phi_in


        TMM_res = OS_layers.calculate(options, profile=True, layers=[1, 2, 3, 4, 5, 6])

        ## TMM Matrix method

        front_surf = Interface('TMM', layers=front_materials, name='TMM_f',
                               coherent=True, prof_layers=prof_layers)
        back_surf = Interface('TMM', layers=back_materials, name='TMM_b',
                              coherent=True, prof_layers=[1])

        bulk_Ge = BulkLayer(bulkthick, SiN, name='SiN_bulk')  # bulk thickness in m

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_TMM_Matrix = calculate_RAT(SC, options)

        results_per_pass = results_TMM_Matrix[1]

        results_per_layer_front_TMM = np.sum(results_per_pass['a'][0], 0)
        results_per_layer_back_TMM = np.sum(results_per_pass['a'][1], 0)[:,0]


        assert np.all(results_TMM_Matrix[0]['A_interface'][0].data == np.sum(results_per_pass['a'][0], (0, 2)))

        assert np.all(results_TMM_Matrix[0]['A_interface'][1].data == np.sum(results_per_pass['a'][1], (0,2)))

        assert np.all((results_TMM_Matrix[0]['R'] + results_TMM_Matrix[0]['T'] + results_TMM_Matrix[0]['A_bulk'] + np.sum(
            results_TMM_Matrix[0]['A_interface'], 0)) == approx(1, abs=0.01))

        surf = planar_surface()

        front_surf = Interface('RT_TMM', layers=front_materials, texture=surf, name='RT_TMM_f',
                               coherent=True, prof_layers=prof_layers)
        back_surf = Interface('RT_TMM', layers=back_materials, texture=surf, name='RT_TMM_b',
                              coherent=True, prof_layers=[1])

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_RT = calculate_RAT(SC, options)

        results_per_pass = results_RT[1]

        # only select absorbing layers, sum over passes
        results_per_layer_front_RT = np.sum(results_per_pass['a'][0], 0)
        # results_per_layer_back_RT = np.sum(results_per_pass['a'][1], 0)[:,0]

        assert np.all(results_RT[0]['A_interface'][0].data == np.sum(results_per_pass['a'][0], (0, 2)))

        assert np.all(results_RT[0]['A_interface'][1].data == np.sum(results_per_pass['a'][1], (0, 2)))

        assert np.all((results_RT[0]['R'] + results_RT[0]['T'] + results_RT[0]['A_bulk'] + np.sum(
            results_RT[0]['A_interface'], 0)) == approx(1, abs=0.01))


        ## RCWA Matrix

        front_surf = Interface('RCWA', layers=front_materials, name='RCWA_f', d_vectors=((500, 0), (0, 500)),
                               rcwa_orders=2, prof_layers=prof_layers)
        back_surf = Interface('RCWA', layers=back_materials, name='RCWA_b', d_vectors=((500, 0), (0, 500)),
                              rcwa_orders=2, prof_layers=[1])

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_RCWA_Matrix = calculate_RAT(SC, options)

        results_per_pass = results_RCWA_Matrix[1]

        results_per_layer_front_RCWA = np.sum(results_per_pass['a'][0], 0)
        results_per_layer_back_RCWA = np.sum(results_per_pass['a'][1], 0)[:,0]

        assert np.all(results_RCWA_Matrix[0]['A_interface'][0] == np.sum(results_per_pass['a'][0], (0, 2)))

        assert np.all(results_RCWA_Matrix[0]['A_interface'][1] == np.sum(results_per_pass['a'][1], (0, 2)))

        assert np.all((results_RCWA_Matrix[0]['R'] + results_RCWA_Matrix[0]['T'] + results_RCWA_Matrix[0]['A_bulk'] + np.sum(
            results_RCWA_Matrix[0]['A_interface'], 0)) == approx(1, abs=0.01))


        results_per_layer_front_TMM_ref = TMM_res['A_per_layer'][:,:len(front_materials)]
        results_per_layer_back_TMM_ref = TMM_res['A_per_layer'][:, -1]

        c_i = results_per_layer_front_TMM_ref > 1e-2

        assert results_per_layer_front_TMM[c_i] == approx(results_per_layer_front_TMM_ref[c_i], rel=0.05)
        assert results_per_layer_front_RCWA[c_i] == approx(results_per_layer_front_TMM_ref[c_i], rel=0.05)
        assert results_per_layer_front_RT[c_i] == approx(results_per_layer_front_TMM_ref[c_i], rel=0.8)


        c_i = results_per_layer_back_TMM_ref > 1e-2

        assert results_per_layer_back_TMM[c_i] == approx(results_per_layer_back_TMM_ref[c_i], rel=0.05)
        assert results_per_layer_back_RCWA[c_i] == approx(results_per_layer_back_TMM_ref[c_i], rel=0.05)


@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_rcwa_tmm_matrix_profiles():
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
    options.depth_spacing = 1e-9
    options.only_incidence_angle = False
    options.nx = 4
    options.ny = 4

    _, _, angle_vector = make_angle_vector(options.n_theta_bins,
                                           options.phi_symmetry, options.c_azimuth)

    for pol in ['s', 'p', 'u']:
        options.project_name = 'rcwa_tmm_matrix_profiles_' + pol
        options.pol = pol

        SiN = material('Si3N4')()
        GaAs = material('GaAs')()
        GaInP = material('GaInP')(In=0.5)
        Ag = material('Ag')()
        Air = material('Air')()
        Ta2O5 = material('TaOx1')()  # Ta2O5 (SOPRA database)
        MgF2 = material('MgF2')()  # MgF2 (SOPRA database)

        front_materials = [Layer(100e-9, MgF2), Layer(50e-9, GaInP), Layer(100e-9, Ta2O5), Layer(200e-9, GaAs)]
        back_materials = [Layer(50E-9, GaInP)]

        prof_layers = np.arange(len(front_materials)) + 1

        ## pure TMM with incoherent thick layer
        all_layers = front_materials + [Layer(bulkthick, SiN)] + back_materials

        coh_list = len(front_materials) * ['c'] + ['i'] + ['c']

        options.coherent = False
        options.coherency_list = coh_list

        OS_layers = tmm_structure(all_layers, incidence=Air,
                                  transmission=Ag, no_back_reflection=False)


        th_ind = np.random.randint(0, 4)
        phi_in = np.random.uniform(0, np.pi)

        options.theta_in = angle_vector[th_ind, 1]
        options.phi_in = phi_in

        # set up Solcore materials

        TMM_res = OS_layers.calculate(options, profile=True, layers=[1, 2, 3, 4, 5, 6])

        ## TMM Matrix method

        front_surf = Interface('TMM', layers=front_materials, name='TMM_f',
                               coherent=True, prof_layers=prof_layers)
        back_surf = Interface('TMM', layers=back_materials, name='TMM_b',
                              coherent=True, prof_layers=[1])

        bulk_Ge = BulkLayer(bulkthick, SiN, name='SiN_bulk')  # bulk thickness in m

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_TMM_Matrix = calculate_RAT(SC, options)

        profile = results_TMM_Matrix[2]

        prof_plot_TMM = profile[0]
        prof_plot_TMM_back = profile[1]

        depths = np.linspace(0, len(prof_plot_TMM[0, :]) * options['depth_spacing'] * 1e9, len(prof_plot_TMM[0, :]))
        integrated_A = np.trapz(prof_plot_TMM, depths, axis=1)

        depths_back = np.linspace(0, len(prof_plot_TMM_back[0, :]) * options['depth_spacing'] * 1e9, len(prof_plot_TMM_back[0, :]))
        integrated_A_back = np.trapz(prof_plot_TMM_back, depths_back, axis=1)

        assert approx(integrated_A == results_TMM_Matrix[0]['A_interface'][0].data)
        assert approx(integrated_A_back == results_TMM_Matrix[0]['A_interface'][1].data)

        ## RT_TMM Matrix method
        surf = planar_surface()  # [texture, flipped texture]

        front_surf = Interface('RT_TMM', layers=front_materials, texture=surf, name='RT_TMM_f',
                               coherent=True, prof_layers=prof_layers)
        back_surf = Interface('RT_TMM', layers=back_materials, texture=surf, name='RT_TMM_b',
                              coherent=True, prof_layers=[1])

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_RT = calculate_RAT(SC, options)


        profile = results_RT[2]

        prof_plot_RT = profile[0]
        prof_plot_RT_back = profile[1]

        depths = np.linspace(0, len(prof_plot_RT[0, :]) * options['depth_spacing'] * 1e9, len(prof_plot_RT[0, :]))
        integrated_A = np.trapz(prof_plot_RT, depths, axis=1)

        depths_back = np.linspace(0, len(prof_plot_RT_back[0, :]) * options['depth_spacing'] * 1e9,
                                  len(prof_plot_RT_back[0, :]))
        integrated_A_back = np.trapz(prof_plot_RT_back, depths_back, axis=1)

        assert approx(integrated_A == results_RT[0]['A_interface'][0].data)
        assert approx(integrated_A_back == results_RT[0]['A_interface'][1].data)

        ## RCWA Matrix

        front_surf = Interface('RCWA', layers=front_materials, name='RCWA_f', d_vectors=((500, 0), (0, 500)),
                               rcwa_orders=2, prof_layers=prof_layers)
        back_surf = Interface('RCWA', layers=back_materials, name='RCWA_b', d_vectors=((500, 0), (0, 500)),
                              rcwa_orders=2, prof_layers=[1])

        SC = Structure([front_surf, bulk_Ge, back_surf], incidence=Air, transmission=Ag)

        process_structure(SC, options)

        results_RCWA_Matrix = calculate_RAT(SC, options)

        profile = results_RCWA_Matrix[2]

        prof_plot_RCWA = profile[0]
        prof_plot_RCWA_back = profile[1]

        depths = np.linspace(0, len(prof_plot_RCWA[0, :]) * options['depth_spacing'] * 1e9, len(prof_plot_RCWA[0, :]))
        integrated_A = np.trapz(prof_plot_RCWA, depths, axis=1)

        depths_back = np.linspace(0, len(prof_plot_RCWA_back[0, :]) * options['depth_spacing'] * 1e9,
                                  len(prof_plot_RCWA_back[0, :]))
        integrated_A_back = np.trapz(prof_plot_RCWA_back, depths_back, axis=1)

        assert approx(integrated_A == results_RCWA_Matrix[0]['A_interface'][0].data)
        assert approx(integrated_A_back == results_RCWA_Matrix[0]['A_interface'][1].data)

        front_profile_TMM = TMM_res['profile'][:, :len(depths)]
        c_i  = front_profile_TMM > 1e-4

        assert prof_plot_TMM.data[c_i] == approx(front_profile_TMM[c_i], rel=0.15)
        assert prof_plot_RCWA.data[c_i] == approx(front_profile_TMM[c_i], rel=0.15)

        back_profile_TMM = TMM_res['profile'][:, -len(depths_back):]
        c_i  = back_profile_TMM > 1e-4

        assert prof_plot_TMM_back.data[c_i] == approx(back_profile_TMM[c_i], rel=0.15)
        assert prof_plot_RCWA_back.data[c_i] == approx(back_profile_TMM[c_i], rel=0.15)


@mark.skipif(sys.platform != "linux", reason="S4 (RCWA) only installed for tests under Linux")
def test_profile_integration():
    from rayflare.matrix_formalism.process_structure import get_savepath
    import os
    import xarray as xr

    for pol in ['s', 'p', 'u']:
        for method in ['TMM_f', 'RCWA_f']:
            project_name = 'rcwa_tmm_matrix_profiles_' + pol
            pth = get_savepath('default', project_name)

            profdatapath = os.path.join(pth, method + 'frontprofmat.nc')
            profdatapath_r = os.path.join(pth, method + 'rearprofmat.nc')

            prof_dataset = xr.load_dataset(profdatapath)
            prof_dataset_r = xr.load_dataset(profdatapath_r)

            intgr = prof_dataset['intgr']
            prof = prof_dataset['profile']
            intgr_r = prof_dataset_r['intgr']
            prof_r = prof_dataset_r['profile']

            depths = np.arange(0, len(prof['z']))
            depths_r = np.arange(0, len(prof_r['z']))

            integrated_prof = np.trapz(prof.data, depths, axis=1)
            integrated_prof_r = np.trapz(prof_r.data, depths_r, axis=1)

            assert integrated_prof == approx(intgr.data, rel=0.03)
            assert integrated_prof_r == approx(intgr_r.data, rel=0.03)

    for pol in ['s', 'p', 'u']:
        for method in ['TMM_b', 'RCWA_b']:
            project_name = 'rcwa_tmm_matrix_profiles_' + pol
            pth = get_savepath('default', project_name)

            profdatapath = os.path.join(pth, method + 'frontprofmat.nc')

            prof_dataset = xr.load_dataset(profdatapath)

            intgr = prof_dataset['intgr']
            prof = prof_dataset['profile']

            depths = np.arange(0, len(prof['z']))

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
    options.project_name = 'RT_Fresnel_TMM'
    options.n_theta_bins = 20
    options.wavelengths = np.linspace(950, 1130, 4)*1e-9
    options.bulk_profile = False

    flat_surf = planar_surface(size=2)  # pyramid size in microns
    triangle_surf = regular_pyramids(55, upright=False, size=2)

    front_surf = Interface('RT_Fresnel', name='RT_F_f', texture=triangle_surf)
    back_surf = Interface('RT_Fresnel', name='RT_F_b', texture=flat_surf)

    bulk = BulkLayer(300e-6, Si, name='Si_bulk')  # bulk thickness in m

    SC_F = Structure([front_surf, bulk, back_surf], incidence=Air, transmission=Air)

    process_structure(SC_F, options, 'current')

    res_fresnel = calculate_RAT(SC_F, options, 'current')

    front_surf = Interface('RT_TMM', name='RT_TMM_f', texture=triangle_surf, layers=[Layer(1e-9, Si)],
                           coherent=False, coherency_list=['i'])
    back_surf = Interface('RT_TMM', name='RT_TMM_b', texture=flat_surf, layers=[Layer(1e-9, Si)],
                          coherent=False, coherency_list=['i'])

    bulk = BulkLayer(300e-6, Si, name='Si_bulk')  # bulk thickness in m

    SC_TMM = Structure([front_surf, bulk, back_surf], incidence=Air, transmission=Air)

    process_structure(SC_TMM, options, 'current')

    res_TMM = calculate_RAT(SC_TMM, options, 'current')

    T_nz = res_TMM[0]['T'][0].data > 5e-3

    assert res_fresnel[0]['A_bulk'][0].data == approx(res_TMM[0]['A_bulk'][0].data, abs=0.15)
    assert res_fresnel[0]['R'][0].data == approx(res_TMM[0]['R'][0].data, abs=0.15)
    assert res_fresnel[0]['T'][0].data[T_nz] == approx(res_TMM[0]['T'][0].data[T_nz], abs=0.15)



