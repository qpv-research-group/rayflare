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


@pytest.mark.parametrize("pol", ["s", "p", "u"])
def test_profile_integrates_to_layer_absorption(pol):
    spacing = 0.5
    depths = np.arange(0, 240, spacing)
    result = _planar_result(pol, profile_depths=depths)
    layer_index = np.searchsorted(np.cumsum([80, 160]), depths, side="right")
    integrated = np.asarray(
        [np.sum(result["profile"][layer_index == i]) * spacing for i in range(2)]
    )
    assert_allclose(integrated, result["A_per_layer"], atol=3e-4, rtol=0)


def test_physical_incident_flux_matches_plane_wave():
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    jnp, basis, fields, fmm, _ = fmmax_backend._imports()
    theta, phi, n = np.deg2rad(35), np.deg2rad(23), 1.5
    lattice = basis.LatticeVectors(u=jnp.asarray([500, 0]), v=jnp.asarray([0, 500]))
    expansion = basis.generate_expansion(lattice, 1, basis.Truncation.CIRCULAR)
    wavevector = basis.plane_wave_in_plane_wavevector(
        jnp.asarray(650), jnp.asarray(theta), jnp.asarray(phi), jnp.asarray(n**2)
    )
    layer = fmm.eigensolve_isotropic_media(
        wavelength=jnp.asarray(650),
        in_plane_wavevector=wavevector,
        primitive_lattice_vectors=lattice,
        permittivity=jnp.asarray([[n**2]]),
        expansion=expansion,
        formulation=fmm.Formulation.FFT,
    )
    amplitudes, _ = fmmax_backend._incident_amplitudes(
        layer, theta, phi, "u", fields, jnp
    )
    flux = fmmax_backend._physical_incident_flux(amplitudes, layer, fields, jnp)
    assert_allclose(flux, n * np.cos(theta) / 2, atol=2e-6, rtol=0)


def test_material_grid_matches_solver_sampling_origin():
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    jnp, basis, _, _, _ = fmmax_backend._imports()
    lattice = np.asarray([[400, 0], [80, 300]], dtype=float)
    shape = (13, 17)
    vectors = basis.LatticeVectors(u=jnp.asarray(lattice[0]), v=jnp.asarray(lattice[1]))
    expected = basis.unit_cell_coordinates(vectors, shape, num_unit_cells=(1, 1))
    actual = fmmax_backend._cell_coordinates(lattice, shape)
    for coordinates, reference in zip(actual, expected):
        assert_allclose(coordinates, reference, atol=1e-6, rtol=0)


def test_minimum_grid_helper_matches_installed_api():
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    jnp, basis, _, _, _ = fmmax_backend._imports()
    vectors = basis.LatticeVectors(u=jnp.asarray([400, 0]), v=jnp.asarray([0, 400]))
    expansion = basis.generate_expansion(vectors, 25, basis.Truncation.CIRCULAR)
    shape = fmmax_backend._grid_shape(
        np.asarray([[400, 0], [0, 400]]),
        expansion,
        fmmax_backend.options({"grid_spacing": 10000}),
        basis,
    )
    expected = tuple(2 * np.max(np.abs(expansion.basis_coefficients), axis=0) + 1)
    assert shape == expected


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


@pytest.mark.parametrize("side", [1, -1])
def test_normal_incidence_matrix_retains_reflection_and_transmission(side):
    from rayflare.angles import make_angle_vector
    from rayflare.rigorous_coupled_wave_analysis.rcwa import RCWA_wl_fmmax

    theta_edges, phi_edges, angles = make_angle_vector(3, 2 * np.pi, 1)
    result = RCWA_wl_fmmax(
        wl=650,
        geom_list=[None, None, None],
        l_oc=np.asarray([1 + 0j, 4 + 0j, 1 + 0j]),
        s_oc=[],
        s_names=[],
        pol="s",
        theta=np.asarray([0.0]),
        phi=np.asarray([0.0]),
        widths=[np.inf, 100, np.inf],
        size=((500, 0), (0, 500)),
        orders=1,
        phi_sym=2 * np.pi,
        theta_intv=theta_edges,
        phi_intv=phi_edges,
        angle_vector_0=angles[:, 0],
        fmmax_options={},
        side=side,
    )
    reflection, transmission, absorption, matrix, _ = result
    column = 0 if side == 1 else np.flatnonzero(angles[:, 0] == 5)[0] - len(angles) // 2
    power = matrix.todense()[:, column]
    half = len(angles) // 2
    reflected = np.sum(power[:half] if side == 1 else power[half:])
    transmitted = np.sum(power[half:] if side == 1 else power[:half])
    assert reflection[column] > 0.3  # Nonzero reflection must not be discarded.
    assert_allclose(reflected, reflection[column], atol=2e-6, rtol=0)
    assert_allclose(transmitted, transmission[column], atol=2e-6, rtol=0)
    assert_allclose(np.sum(power) + np.sum(absorption[:, column]), 1, atol=2e-6, rtol=0)


def test_rear_incidence_keeps_patterned_geometry_with_its_layer(tmp_path):
    from solcore import material
    from solcore.structure import Layer
    from rayflare.options import default_options
    from rayflare.rigorous_coupled_wave_analysis import RCWA

    air = material("Air")()
    layers = [
        Layer(
            120e-9,
            material("GaAs")(),
            geometry=[{"type": "circle", "mat": air, "center": (0, 0), "radius": 80}],
        ),
        Layer(
            80e-9,
            material("Si")(),
            geometry=[
                {
                    "type": "rectangle",
                    "mat": air,
                    "center": (100, 50),
                    "halfwidths": (40, 60),
                }
            ],
        ),
    ]
    opts = default_options()
    opts.RCWA_method = "FMMAX"
    opts.parallel = False
    opts.orders = 1
    opts.wavelength = np.asarray([650e-9])
    opts.n_theta_bins = 3
    opts.c_azimuth = 0.1
    opts.theta_in = opts.phi_in = 0
    opts.pol = "s"
    arguments = dict(
        size=((400, 0), (0, 400)),
        orders=1,
        options=opts,
        structpath=str(tmp_path),
        incidence=air,
        transmission=air,
        only_incidence_angle=True,
        save=False,
        overwrite=True,
    )
    reference_rt, reference_a = RCWA(layers[::-1], front_or_rear="front", **arguments)
    rear_rt, rear_a = RCWA(layers, front_or_rear="rear", **arguments)
    reference = reference_rt.todense()[0, :, 0]
    rear = rear_rt.todense()[0, :, -1]
    half = len(reference) // 2
    assert_allclose(np.sum(rear[:half]), np.sum(reference[half:]), atol=2e-6, rtol=0)
    assert_allclose(np.sum(rear[half:]), np.sum(reference[:half]), atol=2e-6, rtol=0)
    assert_allclose(
        rear_a.todense()[0, :, -1], reference_a.todense()[0, ::-1, 0], atol=2e-6, rtol=0
    )
    assert_allclose(
        np.sum(rear) + np.sum(rear_a.todense()[0, :, -1]), 1, atol=2e-6, rtol=0
    )


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
