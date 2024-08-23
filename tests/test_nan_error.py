from pytest import approx
import numpy as np

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
    options.wavelength = np.linspace(950, 1130, 4) * 1e-9
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

