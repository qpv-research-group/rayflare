from pytest import approx
import numpy as np


def test_bulk_profile():
    from solcore import material

    # rayflare imports
    from rayflare.textures import regular_pyramids
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options

    Si = material("Si")()
    Air = material("Air")()

    options = default_options()
    options.project_name = "bulk_profile_test"
    options.n_theta_bins = 20
    options.wavelengths = np.linspace(950, 1130, 3) * 1e-9
    options.bulk_profile = True
    options.depth_spacing_bulk = 1e-6
    options.c_azimuth = 0.25
    options.nx = 30
    options.ny = 30
    options.n_rays = 54000
    options.only_incidence_angle = False
    options.phi_in = 0.3
    options.phi_symmetry = np.pi / 2
    options.theta_in = 0.5

    triangle_surf = regular_pyramids(55, upright=True)

    front_surf = Interface("RT_Fresnel", name="RT_f", texture=triangle_surf)
    back_surf = Interface("TMM", name="TMM_b", layers=[])

    bulk = BulkLayer(20e-6, Si, name="Si_bulk")  # bulk thickness in m

    SC_F = Structure([front_surf, bulk, back_surf], incidence=Air, transmission=Air)

    process_structure(SC_F, options)

    res_fresnel = calculate_RAT(SC_F, options)

    depths = np.arange(0, bulk.width, options.depth_spacing_bulk)

    check = np.trapz(res_fresnel[3][0], depths, axis=1)

    assert check == approx(res_fresnel[0]["A_bulk"][0].data)

    for phi_in in options.phi_in + np.array([np.pi / 2, np.pi, 3 * np.pi / 2]):
        options.phi_in = phi_in
        res_fresnel_phi = calculate_RAT(SC_F, options)

        for key in res_fresnel[0].keys():
            assert res_fresnel_phi[0][key].data == approx(res_fresnel[0][key].data)


def test_phi_all():
    from solcore import material

    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options

    GaAs = material("GaAs")()
    Air = material("Air")()

    options = default_options()
    options.project_name = "test_arm_phi"
    options.n_theta_bins = 20
    options.wavelengths = np.linspace(950, 1130, 4) * 1e-9
    options.bulk_profile = True
    options.depth_spacing_bulk = 1e-6
    options.c_azimuth = 0.25
    options.only_incidence_angle = False
    options.phi_in = 0
    options.phi_symmetry = np.pi / 2
    options.theta_in = 0

    front_surf = Interface("TMM", name="TMM_f", layers=[])
    back_surf = Interface("TMM", name="TMM_b", layers=[])

    bulk = BulkLayer(3e-6, GaAs, name="GaAs_bulk")  # bulk thickness in m

    SC_F = Structure([front_surf, bulk, back_surf], incidence=Air, transmission=Air)

    process_structure(SC_F, options)

    res = calculate_RAT(SC_F, options)

    depths = np.arange(0, bulk.width, options.depth_spacing_bulk)

    check = np.trapz(res[3][0], depths, axis=1)

    assert check == approx(res[0]["A_bulk"][0].data)

    options.phi_in = "all"

    res_2 = calculate_RAT(SC_F, options)

    for key in res_2[0].keys():
        assert res_2[0][key].data == approx(res[0][key].data)


def test_random_position():
    from solcore import material

    # rayflare imports
    from rayflare.textures import regular_pyramids
    from rayflare.structure import Interface, BulkLayer, Structure
    from rayflare.matrix_formalism import process_structure, calculate_RAT
    from rayflare.options import default_options

    Si = material("Si")()
    Air = material("Air")()

    options = default_options()
    options.project_name = "bulk_profile_test_random"
    options.n_theta_bins = 20
    options.wavelengths = np.linspace(950, 1130, 3) * 1e-9
    options.bulk_profile = True
    options.depth_spacing_bulk = 1e-6
    options.c_azimuth = 0.25
    options.nx = 30
    options.ny = 30
    options.n_rays = 54000
    options.only_incidence_angle = False
    options.phi_in = 0
    options.phi_symmetry = np.pi / 2
    options.theta_in = 0
    options.random_ray_position = True
    options.only_incidence_angle = True

    triangle_surf = regular_pyramids(55, upright=True)

    front_surf = Interface("RT_Fresnel", name="RT_f", texture=triangle_surf)
    back_surf = Interface("TMM", name="TMM_b", layers=[])

    bulk = BulkLayer(20e-6, Si, name="Si_bulk")  # bulk thickness in m

    SC_F = Structure([front_surf, bulk, back_surf], incidence=Air, transmission=Air)

    process_structure(SC_F, options, overwrite=True)

    res_random = calculate_RAT(SC_F, options)

    options.random_ray_position = False
    options.project_name = "bulk_profile_test"

    res_regular = calculate_RAT(SC_F, options)

    for key in res_regular[0].keys():
        assert res_random[0][key].data == approx(res_regular[0][key].data, abs=0.07)
