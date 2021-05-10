from pytest import approx
import numpy as np

def test_parallel():
    from rayflare.ray_tracing import rt_structure
    from rayflare.textures import regular_pyramids
    from rayflare.options import default_options
    from solcore import material
    from solcore import si

    Air = material('Air')()
    Si = material('Si')()
    GaAs = material('GaAs')()
    Ge = material('Ge')()

    triangle_surf = regular_pyramids(30)

    options = default_options()

    options.wavelengths = np.linspace(700, 1700, 5)*1e-9
    options.theta_in = 45*np.pi/180
    options.nx = 5
    options.ny = 5
    options.pol = 'p'
    options.n_rays = 3000
    options.depth_spacing = 1e-6
    options.parallel = True

    rtstr = rt_structure(textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
                        materials = [GaAs, Si, Ge],
                        widths=[si('100um'), si('70um'), si('50um')], incidence=Air, transmission=Air)
    result_new = rtstr.calculate(options)

    rtstr = rt_structure(textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
                        materials = [GaAs, Si, Ge],
                        widths=[si('100um'), si('70um'), si('50um')], incidence=Air, transmission=Air)

    result_old = rtstr.calculate(options)


    assert sorted(list(result_old.keys())) == sorted(list(result_new.keys()))

    rel_error = 0.3  # Monte Carlo (random) simulations so will never be exactly equal)
    assert result_new['R'] == approx(result_old['R'], rel=rel_error)
    assert result_new['R0'] == approx(result_old['R0'], rel=rel_error)
    assert result_new['A_per_layer'] == approx(result_old['A_per_layer'], rel=rel_error)
    assert result_new['profile'] == approx(result_old['profile'], rel=rel_error)
    assert np.nanmean(result_new['thetas'], 1) == approx(np.nanmean(result_old['thetas'], 1), rel=rel_error)
    # assert np.nanmean(result_new['phis'], 1) == approx(np.nanmean(result_old['phis'], 1), rel=rel_error)
    assert np.nanmean(result_new['n_interactions'], 1) == approx(np.nanmean(result_old['n_interactions'], 1), rel=rel_error)
    assert np.nanmean(result_new['n_passes'], 1) == approx(np.nanmean(result_old['n_passes'], 1), rel=rel_error)