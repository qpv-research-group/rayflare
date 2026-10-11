import numpy as np
import os
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


@pytest.mark.skipif(
    os.environ.get("RAYFLARE_TEST_GPU") != "1",
    reason="opt-in NVIDIA GPU precision regression",
)
def test_gpu_solve_uses_full_precision_without_changing_global_config():
    import jax

    assert jax.default_backend() == "gpu", "Force JAX_PLATFORMS=cuda for this test"
    # A100 defaults can use reduced-mantissa TF32 products. The adapter must
    # retain its CPU accuracy contract even when the caller selects that default.
    with jax.default_matmul_precision("default"):
        test_planar_matches_tmm(35, "s")
        assert jax.config.jax_default_matmul_precision == "default"


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


@pytest.fixture
def full_precision_primitives():
    import jax

    # These tests call FMMAX primitives outside adapter.solve(); use the same
    # multiply precision as production rather than GPU's reduced-mantissa default.
    with jax.default_matmul_precision("highest"):
        yield


def test_physical_incident_flux_matches_plane_wave(full_precision_primitives):
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


def test_complex_excitation_preserves_relative_phase_in_fields(
    full_precision_primitives,
):
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    jnp, basis, fields, fmm, _ = fmmax_backend._imports()
    theta, phi = np.deg2rad([35, 23])
    lattice = basis.LatticeVectors(u=jnp.asarray([500, 0]), v=jnp.asarray([0, 500]))
    layer = fmm.eigensolve_isotropic_media(
        wavelength=jnp.asarray(650),
        in_plane_wavevector=basis.plane_wave_in_plane_wavevector(
            jnp.asarray(650), jnp.asarray(theta), jnp.asarray(phi), jnp.asarray(1.0)
        ),
        primitive_lattice_vectors=lattice,
        permittivity=jnp.asarray([[1.0]]),
        expansion=basis.generate_expansion(lattice, 1, basis.Truncation.CIRCULAR),
        formulation=fmm.Formulation.FFT,
    )
    s, p = 1 + 0.4j, 0.3 - 0.7j
    amplitude, weights = fmmax_backend._incident_amplitudes(
        layer, theta, phi, (s, p), fields, jnp
    )
    electric, _ = fields.fields_from_wave_amplitudes(
        amplitude, jnp.zeros_like(amplitude), layer
    )
    expected = [
        s * (-np.sin(phi)) + p * np.cos(theta) * np.cos(phi),
        s * np.cos(phi) + p * np.cos(theta) * np.sin(phi),
    ]
    assert amplitude.shape == (2, 1)  # one coherent field, not two incoherent channels
    assert_allclose(
        [np.asarray(electric[0])[0, 0], np.asarray(electric[1])[0, 0]],
        expected,
        atol=2e-6,
        rtol=0,
    )
    assert_allclose(weights, [1], atol=0, rtol=0)


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


@pytest.mark.parametrize("lossless", [False, True])
def test_public_multilayer_profile_matches_tmm(lossless):
    from solcore import material
    from rayflare.options import default_options
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
    from tmm import position_resolved

    opts = default_options()
    opts.RCWA_method = "FMMAX"
    opts.parallel = False
    opts.n_jobs = 1
    opts.orders = 1
    opts.wavelength = np.array([450, 650, 850]) * 1e-9
    opts.depth_spacing = 0.5e-9
    opts.pol = "p"
    opts.theta_in, opts.phi_in = np.deg2rad([35, 23])
    indices = [
        1,
        1.45 + (0 if lossless else 0.02j),
        3.4 + (0 if lossless else 0.12j),
        1,
    ]
    air = material("Air")()
    structure = rcwa_structure(
        [
            [80, indices[1].real, indices[1].imag, []],
            [160, indices[2].real, indices[2].imag, []],
        ],
        ((500, 0), (0, 500)),
        opts,
        air,
        air,
    )
    result = structure.calculate_profile(opts)
    depths = np.arange(0, 240, 0.5)
    assert_allclose(structure.dist, depths, atol=1e-10, rtol=0)
    assert result["profile"].shape == (3, len(depths))
    assert np.isfinite(result["profile"]).all()
    for i, wavelength in enumerate([450, 650, 850]):
        reference = coh_tmm(
            "p", indices, [np.inf, 80, 160, np.inf], opts.theta_in, wavelength
        )
        reference_a = absorp_in_each_layer(reference)[1:-1]
        integrated = [
            0.5 * result["profile"][i, depths < 80].sum(),
            0.5 * result["profile"][i, depths >= 80].sum(),
        ]
        assert_allclose(result["A_per_layer"][i], reference_a, atol=2e-6, rtol=0)
        expected_profile = [
            position_resolved(1 if z < 80 else 2, z if z < 80 else z - 80, reference)[
                "absor"
            ]
            for z in depths
        ]
        assert_allclose(result["profile"][i], expected_profile, atol=2e-8, rtol=2e-5)
        reference_quad = np.array(
            [
                0.5 * np.sum(np.array(expected_profile)[depths < 80]),
                0.5 * np.sum(np.array(expected_profile)[depths >= 80]),
            ]
        )
        # Bound rectangular-rule error using the independent analytic profile.
        assert np.max(np.abs(reference_quad - reference_a)) < 8e-4
        assert np.all(
            np.abs(np.array(integrated) - reference_a)
            <= np.abs(reference_quad - reference_a) + 3e-6
        )
        assert_allclose(integrated, result["A_per_layer"][i], atol=8e-4, rtol=0)
    if lossless:
        assert_allclose(result["profile"], 0, atol=1e-12, rtol=0)


def _lossy_pattern_result(orders=25, grid_spacing=4, spacing=2, eig_backend="default"):
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    return fmmax_backend.solve(
        wavelength=600,
        geom_list=[
            None,
            [
                dict(
                    type="ellipse",
                    mat="feature",
                    center=(113, 77),
                    halfwidths=(65, 45),
                    angle=27,
                )
            ],
            None,
        ],
        layer_permittivities=np.array([1, (1.7 + 0.04j) ** 2, 1], dtype=complex),
        shape_permittivities=np.array([(2 + 0.13j) ** 2]),
        shape_names=["feature"],
        pol=(1, 0.5j),
        theta=20,
        phi=31,
        widths=[np.inf, 100, np.inf],
        size=((300, 0), (0, 300)),
        orders=orders,
        user_options=dict(grid_spacing=grid_spacing, eig_backend=eig_backend),
        profile_depths=np.arange(spacing / 2, 100, spacing),
    )


def test_lossy_pattern_profile_convergence():
    coarse = _lossy_pattern_result(25, 4, 2)
    fourier = _lossy_pattern_result(49, 4, 2)
    raster = _lossy_pattern_result(49, 2, 2)
    depth = _lossy_pattern_result(49, 2, 1)
    integrals = []
    for result, spacing in zip([coarse, fourier, raster, depth], [2, 2, 2, 1]):
        assert np.isfinite(result["profile"]).all()
        assert result["profile"].min() > 0
        assert result["A_per_layer"][0] > 0.03
        assert_allclose(
            result["R"] + result["T"] + result["A_per_layer"].sum(),
            1,
            atol=2e-6,
            rtol=0,
        )
        integrals.append(spacing * result["profile"].sum())
    # Midpoint quadrature: refinement must resolve absorption, not hide a scale error.
    assert abs(integrals[-1] - depth["A_per_layer"][0]) < 3e-4
    assert abs(integrals[-1] - integrals[-2]) < 2e-5
    assert abs(integrals[-2] - integrals[-3]) < 5e-4
    assert abs(integrals[-3] - integrals[-4]) < 2e-3
    # Bound the whole profile, including positive/negative differences that
    # cancel in its integral. The pointwise budgets are the existing integrated
    # budgets divided by this fixture's 100 nm thickness; L1 uses those budgets
    # directly. They are convergence tolerances, not floating-point tolerances.
    for left, right, budget in ((coarse, fourier, 2e-3), (fourier, raster, 5e-4)):
        delta = np.abs(left["profile"] - right["profile"])
        assert np.max(delta) < budget / 100
        assert 2 * np.sum(delta) < budget
    # Compare the common 2 nm sampling with pairs of 1 nm midpoint samples.
    delta = np.abs(raster["profile"] - depth["profile"].reshape(-1, 2).mean(axis=1))
    assert np.max(delta) < 2e-5 / 100
    assert 2 * np.sum(delta) < 2e-5


def test_complex_coherent_excitation_matches_weighted_planar_tmm():
    # Planar isotropy has no s/p cross term in power; relative phase remains coherent.
    result = _planar_result((1 + 0.4j, 0.3 - 0.7j))
    weights = np.abs([1 + 0.4j, 0.3 - 0.7j]) ** 2
    references = [
        coh_tmm(
            pol,
            [1, 1.45 + 0.02j, 3.4 + 0.12j, 1.5],
            [np.inf, 80, 160, np.inf],
            np.deg2rad(35),
            650,
        )
        for pol in ("s", "p")
    ]
    expected = sum(
        w * np.r_[r["R"], r["T"], absorp_in_each_layer(r)[1:-1]]
        for w, r in zip(weights, references)
    ) / sum(weights)
    assert_allclose(
        np.r_[result["R"], result["T"], result["A_per_layer"]],
        expected,
        atol=2e-6,
        rtol=0,
    )
    assert_allclose(
        [result["R_per_order"].sum(), result["T_per_order"].sum()],
        expected[:2],
        atol=2e-6,
        rtol=0,
    )


@pytest.mark.parametrize(
    "settings,exception",
    [
        ({"grid_shape": (1, 1)}, ValueError),
        ({"max_grid_points": 1}, ValueError),
    ],
)
def test_undersized_pattern_grid_rejected(settings, exception):
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    with pytest.raises(exception):
        fmmax_backend.solve(
            600,
            [None, None, None],
            np.array([1, 4, 1]),
            [],
            [],
            "s",
            0,
            0,
            [np.inf, 100, np.inf],
            ((300, 0), (0, 300)),
            25,
            settings,
        )


def test_absorbing_incidence_rejected():
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    with pytest.raises(NotImplementedError, match="non-absorbing incidence"):
        fmmax_backend.solve(
            600,
            [None, None, None],
            np.array([1 + 0.1j, 4, 1]),
            [],
            [],
            "s",
            0,
            0,
            [np.inf, 100, np.inf],
            ((300, 0), (0, 300)),
            1,
        )


def test_installed_eigensolver_selection_policy():
    import inspect
    from fmmax import fmm
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    settings = fmmax_backend.options({"eig_backend": "cusolver"})
    if "eig_backend" in inspect.signature(fmm.eigensolve_isotropic_media).parameters:
        from fmmax import utils

        assert (
            fmmax_backend._eigensolve_options(fmm, settings)["eig_backend"]
            == utils.EigBackend.CUSOLVER
        )
    else:
        with pytest.raises(NotImplementedError, match="does not support eig_backend"):
            fmmax_backend._eigensolve_options(fmm, settings)


@pytest.mark.parametrize("option", ["A_per_order", "detailed_rcwa"])
def test_public_unsupported_outputs_rejected(option):
    from solcore import material
    from rayflare.options import default_options
    from rayflare.rigorous_coupled_wave_analysis import rcwa_structure

    opts = default_options()
    opts.RCWA_method = "FMMAX"
    opts.parallel = False
    opts.orders = 1
    opts.wavelength = np.array([650e-9])
    opts[option] = True
    air = material("Air")()
    structure = rcwa_structure(
        [[100, 2, 0.1, []]], ((500, 0), (0, 500)), opts, air, air
    )
    with pytest.raises(NotImplementedError, match="A_per_order and detailed_rcwa"):
        structure.calculate(opts)


def _diffracting_result(side=1, eig_backend="default"):
    from rayflare.angles import make_angle_vector
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend
    from rayflare.rigorous_coupled_wave_analysis.rcwa import RCWA_wl_fmmax

    edges, phi_edges, angles = make_angle_vector(8, 2 * np.pi, 4)
    args = dict(
        wavelength=500,
        geom_list=[
            None,
            [
                dict(
                    type="rectangle",
                    mat="feature",
                    center=(173, 91),
                    halfwidths=(180, 110),
                    angle=19,
                )
            ],
            None,
        ],
        layer_permittivities=np.array([1, 2.25**2, 1], dtype=complex),
        shape_permittivities=np.array([1.3**2]),
        shape_names=["feature"],
        pol="s",
        theta=12,
        phi=37,
        widths=[np.inf, 130, np.inf],
        size=((900, 0), (0, 900)),
        orders=25,
        user_options=dict(grid_spacing=10, eig_backend=eig_backend),
    )
    orders = fmmax_backend.solve(**args)
    matrix = RCWA_wl_fmmax(
        args["wavelength"],
        args["geom_list"],
        args["layer_permittivities"],
        args["shape_permittivities"],
        args["shape_names"],
        args["pol"],
        np.array([args["theta"]]),
        np.array([args["phi"]]),
        args["widths"],
        args["size"],
        args["orders"],
        2 * np.pi,
        edges,
        phi_edges,
        angles[:, 0],
        args["user_options"],
        side=side,
    )
    return orders, matrix, (edges, phi_edges, angles)


def _analytic_bin(theta, phi, edges, phi_edges, angles):
    # Independent interval membership, not the adapter's bin helper.
    ring = next(i for i in range(len(edges) - 1) if edges[i] <= theta < edges[i + 1])
    azimuth = phi % (2 * np.pi)
    sector = next(
        i
        for i in range(len(phi_edges[ring]) - 1)
        if phi_edges[ring][i] <= azimuth < phi_edges[ring][i + 1]
    )
    return np.flatnonzero(angles[:, 0] == ring)[sector]


@pytest.mark.parametrize("side", [1, -1])
def test_higher_diffraction_orders_land_in_grating_equation_bins(side):
    orders, (reflection, transmission, absorption, matrix, _), bins = (
        _diffracting_result(side)
    )
    edges, phi_edges, angles = bins
    theta, phi = np.deg2rad([12, 37])
    column = _analytic_bin(theta if side == 1 else np.pi - theta, phi, *bins)
    if side == -1:
        column -= len(angles) // 2
    # k_parallel/k0 = incident direction + wavelength * integer order / period.
    kxy = (
        np.sin(theta) * np.array([np.cos(phi), np.sin(phi)])
        + 500 / 900 * orders["basis_set"]
    )
    assert_allclose(
        orders["transverse_wavevectors"] / (2 * np.pi / 500), kxy, atol=2e-7, rtol=0
    )
    propagating = np.linalg.norm(kxy, axis=1) < 1
    higher = np.any(orders["basis_set"] != 0, axis=1) & propagating
    assert np.sum(orders["T_per_order"][higher]) > 0.01
    expected = np.zeros(len(angles))
    for i in np.flatnonzero(propagating):
        polar = np.arcsin(np.linalg.norm(kxy[i]))
        azimuth = np.arctan2(kxy[i, 1], kxy[i, 0])
        r_theta = polar if side == 1 else np.pi - polar
        t_theta = np.pi - polar if side == 1 else polar
        expected[_analytic_bin(r_theta, azimuth, *bins)] += orders["R_per_order"][i]
        expected[_analytic_bin(t_theta, azimuth, *bins)] += orders["T_per_order"][i]
    assert_allclose(matrix.todense()[:, column], expected, atol=2e-6, rtol=0)
    assert np.count_nonzero(expected > 1e-5) > 4
    assert_allclose(expected.sum() + absorption[:, column].sum(), 1, atol=2e-6, rtol=0)
    assert_allclose(
        [reflection[column], transmission[column]],
        [orders["R"], orders["T"]],
        atol=2e-6,
        rtol=0,
    )
