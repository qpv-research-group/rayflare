import numpy as np
import pytest
from numpy.testing import assert_allclose
from tmm import absorp_in_each_layer, coh_tmm


pytest.importorskip("fmmax", reason="optional FMMAX dependency is not installed")


def _planar_result(pol, theta=35, profile_depths=None):
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    return fmmax_backend.solve(
        wavelength=650.0,
        geom_list=[None, None, None, None],
        layer_permittivities=np.asarray(
            [1 + 0j, (1.45 + 0.02j) ** 2, (3.4 + 0.12j) ** 2, 1.5**2]
        ),
        shape_permittivities=[],
        shape_names=[],
        pol=pol,
        theta=theta,
        phi=23,
        widths=[np.inf, 80, 160, np.inf],
        size=((500, 0), (0, 500)),
        orders=1,
        user_options={"grid_spacing": 20},
        profile_depths=profile_depths,
    )


@pytest.mark.parametrize("theta", [0, 35, 60])
@pytest.mark.parametrize("pol", ["s", "p", "u"])
def test_planar_matches_tmm(theta, pol):
    indices = [1, 1.45 + 0.02j, 3.4 + 0.12j, 1.5]
    thicknesses = [np.inf, 80, 160, np.inf]
    reference = {}
    for component in ("s", "p"):
        result = coh_tmm(component, indices, thicknesses, np.deg2rad(theta), 650)
        reference[component] = np.r_[
            result["R"], result["T"], absorp_in_each_layer(result)[1:-1]
        ]
    if pol == "u":
        expected = (reference["s"] + reference["p"]) / 2
    else:
        expected = reference[pol]

    result = _planar_result(pol, theta)
    actual = np.r_[result["R"], result["T"], result["A_per_layer"]]
    assert_allclose(actual, expected, atol=2e-6, rtol=0)
    assert_allclose(np.sum(actual), 1, atol=2e-6, rtol=0)


def test_profile_integrates_to_layer_absorption():
    spacing = 0.5
    depths = np.arange(0, 240, spacing)
    result = _planar_result("u", profile_depths=depths)
    layer_index = np.searchsorted(np.cumsum([80, 160]), depths, side="right")
    integrated = np.asarray(
        [np.sum(result["profile"][layer_index == i]) * spacing for i in range(2)]
    )
    assert_allclose(integrated, result["A_per_layer"], atol=3e-4, rtol=0)


def _patterned_result(pol, orders=25, grid_spacing=4):
    from solcore import material
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    air = material("Air")()
    air_name = str(air)
    air_epsilon = (air.n(600e-9) + 1j * air.k(600e-9)) ** 2
    return fmmax_backend.solve(
        wavelength=600,
        geom_list=[
            None,
            [{"type": "circle", "mat": air_name, "center": (0, 0), "radius": 100}],
            None,
        ],
        layer_permittivities=np.asarray([1 + 0j, (2.2 + 0.04j) ** 2, 1 + 0j]),
        shape_permittivities=np.asarray([air_epsilon]),
        shape_names=[air_name],
        pol=pol,
        theta=np.degrees(0.35),
        phi=np.degrees(0.2),
        widths=[np.inf, 120, np.inf],
        size=((400, 0), (0, 400)),
        orders=orders,
        user_options={"grid_spacing": grid_spacing},
    )


@pytest.mark.parametrize("pol", ["s", "p"])
def test_patterned_matches_inkstone(pol):
    from solcore import material
    from rayflare.rigorous_coupled_wave_analysis.rcwa import RCWA_structure_wl_inkstone
    from rayflare.utilities import process_pol

    air = material("Air")()
    air_name = str(air)
    air_epsilon = (air.n(600e-9) + 1j * air.k(600e-9)) ** 2
    geometry = [
        None,
        [{"type": "circle", "mat": air_name, "center": (0, 0), "radius": 100}],
        None,
    ]
    reference = RCWA_structure_wl_inkstone(
        600,
        geometry,
        np.asarray([1 + 0j, (2.2 + 0.04j) ** 2, 1 + 0j]),
        np.asarray([air_epsilon]),
        [air_name],
        process_pol(pol),
        np.degrees(0.35),
        np.degrees(0.2),
        [np.inf, 120, np.inf],
        ((400, 0), (0, 400)),
        25,
    )
    result = _patterned_result(pol)
    actual = np.r_[result["R"], result["T"], result["A_per_layer"]]
    expected = np.r_[reference[0], reference[1], reference[2]]
    assert_allclose(actual, expected, atol=0.02, rtol=0)
    assert_allclose(np.sum(actual), 1, atol=2e-6, rtol=0)


def test_patterned_unpolarized_and_convergence():
    result_s = _patterned_result("s", orders=81, grid_spacing=2)
    result_p = _patterned_result("p", orders=81, grid_spacing=2)
    result_u = _patterned_result("u", orders=81, grid_spacing=2)
    for key in ("R", "T", "A_per_layer"):
        assert_allclose(result_u[key], (result_s[key] + result_p[key]) / 2, atol=2e-6)

    order_49 = _patterned_result("p", orders=49, grid_spacing=2)
    coarse_grid = _patterned_result("p", orders=81, grid_spacing=4)
    fine = np.r_[result_p["R"], result_p["T"], result_p["A_per_layer"]]
    medium_order = np.r_[order_49["R"], order_49["T"], order_49["A_per_layer"]]
    coarse = np.r_[coarse_grid["R"], coarse_grid["T"], coarse_grid["A_per_layer"]]
    assert np.max(np.abs(fine - medium_order)) < 0.006
    assert np.max(np.abs(fine - coarse)) < 0.003


def test_rcwa_structure_selects_fmmax():
    from solcore import material
    from rayflare.options import default_options
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure

    air = material("Air")()
    opts = default_options()
    opts.RCWA_method = "FMMAX"
    opts.parallel = False
    opts.orders = 1
    opts.wavelength = np.asarray([649e-9, 650e-9])
    opts.pol = "s"
    structure = rcwa_structure(
        [[100, 2.0, 0.1, []]], ((500, 0), (0, 500)), opts, air, air
    )
    result = structure.calculate(opts)
    assert result["R"].shape == (2,)
    assert result["A_per_layer"].shape == (2, 1)
    assert_allclose(
        result["R"] + result["T"] + np.sum(result["A_per_layer"], axis=1),
        1,
        atol=2e-6,
    )


def test_rcwa_matrix_interface_selects_fmmax(tmp_path):
    from solcore import material
    from solcore.structure import Layer
    from rayflare.options import default_options
    from rayflare.rigorous_coupled_wave_analysis import RCWA

    air = material("Air")()
    gaas = material("GaAs")()
    opts = default_options()
    opts.RCWA_method = "FMMAX"
    opts.parallel = False
    opts.orders = 1
    opts.wavelength = np.asarray([700e-9])
    opts.n_theta_bins = 3
    opts.c_azimuth = 0.1
    opts.pol = "s"
    redistribution, absorption = RCWA(
        [Layer(100e-9, gaas)],
        ((500, 0), (0, 500)),
        1,
        opts,
        str(tmp_path),
        air,
        air,
        save=False,
        overwrite=True,
    )
    energy = np.sum(redistribution.todense(), axis=1) + np.sum(
        absorption.todense(), axis=1
    )
    assert_allclose(energy, 1, atol=2e-6)


def test_unsupported_geometry_is_rejected():
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    with pytest.raises(
        NotImplementedError, match="circle, ellipse, rectangle, and polygon"
    ):
        fmmax_backend.solve(
            wavelength=600,
            geom_list=[None, [{"type": "sphere", "mat": "air"}], None],
            layer_permittivities=np.asarray([1 + 0j, 2.25 + 0j, 1 + 0j]),
            shape_permittivities=np.asarray([1 + 0j]),
            shape_names=["air"],
            pol="s",
            theta=0,
            phi=0,
            widths=[np.inf, 100, np.inf],
            size=((400, 0), (0, 400)),
            orders=1,
        )
