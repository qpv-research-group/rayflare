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
    options.project_name = 'bulk_profile_test'
    options.n_theta_bins = 20
    options.wavelengths = np.linspace(950, 1130, 4) * 1e-9
    options.bulk_profile = True
    options.depth_spacing_bulk = 1e-6
    options.nx = 5
    options.ny = 5
    options.n_rays = 2000

    triangle_surf = regular_pyramids(55, upright=True)
    triangle_surf_back = regular_pyramids(55, upright=False, size=2)

    front_surf = Interface('RT_Fresnel', name='RT_F_f', texture=triangle_surf)
    back_surf = Interface('RT_Fresnel', name='RT_F_b', texture=triangle_surf_back)

    bulk = BulkLayer(300e-6, Si, name='Si_bulk')  # bulk thickness in m

    SC_F = Structure([front_surf, bulk, back_surf], incidence=Air, transmission=Air)

    process_structure(SC_F, options)

    res_fresnel = calculate_RAT(SC_F, options)

    depths = np.arange(0, 300e-6, 1e-6)

    check = np.trapz(res_fresnel[3][0], depths, axis=1)

    assert check == approx(res_fresnel[0]['A_bulk'][0].data)