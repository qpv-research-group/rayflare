import builtins
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_rcwa_import_does_not_import_jax():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rayflare.rigorous_coupled_wave_analysis; "
            "assert 'jax' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_fmmax_error_is_actionable(monkeypatch):
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    real_import = builtins.__import__

    def fail_fmmax_import(name, *args, **kwargs):
        if name == "jax.numpy" or name.startswith("fmmax"):
            raise ImportError("simulated missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_fmmax_import)
    with pytest.raises(ImportError, match=r"rayflare\[fmmax\]"):
        fmmax_backend._imports()


def test_default_eigensolver_requires_no_fork_specific_api():
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    assert (
        fmmax_backend._eigensolve_options(SimpleNamespace(), fmmax_backend.options())
        == {}
    )


def test_unavailable_eigensolver_selection_is_rejected():
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    def eigensolve_isotropic_media(wavelength):
        pass

    solver = SimpleNamespace(eigensolve_isotropic_media=eigensolve_isotropic_media)
    with pytest.raises(NotImplementedError, match="does not support eig_backend"):
        fmmax_backend._eigensolve_options(
            solver, fmmax_backend.options({"eig_backend": "cusolver"})
        )


def test_physical_incident_flux_uses_fields_not_modal_prefactors():
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend

    # Two orthogonal excitations of a homogeneous medium with n=1.5.
    # The physical flux is n |E|^2 / 2 for either polarization.
    electric = (np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), np.zeros((1, 2)))
    magnetic = (np.array([[0.0, -1.5]]), np.array([[1.5, 0.0]]), np.zeros((1, 2)))

    def fields_from_wave_amplitudes(forward, backward, result):
        np.testing.assert_array_equal(backward, np.zeros_like(forward))
        return electric, magnetic

    fields = SimpleNamespace(fields_from_wave_amplitudes=fields_from_wave_amplitudes)
    actual = fmmax_backend._physical_incident_flux(np.ones((2, 2)), None, fields, np)
    np.testing.assert_allclose(actual, [0.75, 0.75], atol=1e-15, rtol=0)


def test_fmmax_matrix_angle_bins_include_zero_boundaries():
    from rayflare.angles import make_angle_vector
    from rayflare.rigorous_coupled_wave_analysis.rcwa import _fmmax_angle_bins

    theta_edges, phi_edges, angles = make_angle_vector(3, 2 * np.pi, 1)
    bins = _fmmax_angle_bins(
        np.array([0.0, np.pi]), np.zeros(2), theta_edges, phi_edges, angles[:, 0]
    )
    assert bins[0] == 0
    assert bins[1] == np.flatnonzero(angles[:, 0] == 5)[0]
    # The normal grid directions retain their original ordering.
    bins = _fmmax_angle_bins(
        angles[:, 1], angles[:, 2], theta_edges, phi_edges, angles[:, 0]
    )
    np.testing.assert_array_equal(bins, np.arange(len(angles)))


@pytest.mark.parametrize("kind", ["ellipse", "rectangle", "polygon"])
def test_rotated_shapes_wrap_and_translate_in_skew_cell(kind):
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend as backend

    # Points away from edges: 90-degree rotation makes the long axis vertical.
    lattice = np.array([[100.0, 0.0], [30.0, 100.0]])
    center = np.array([5.0, 5.0])
    points = np.array([[5, 5], [5, 20], [20, 5], [5, 95], [95, 5]])
    shape = dict(
        type=kind,
        center=center,
        angle=90,
        halfwidths=(20, 8),
        vertices=[(-20, -8), (20, -8), (20, 8), (-20, 8)],
    )
    expected = [True, True, False, False, False]
    for offset in (np.zeros(2), lattice[0], lattice[1], -lattice[0] + lattice[1]):
        translated = dict(shape, center=center + offset)
        np.testing.assert_array_equal(
            backend._shape_mask(translated, points[:, 0], points[:, 1], lattice),
            expected,
        )
    # Equivalent periodic image at the opposite x boundary.
    wrapped = np.array([[99.0, 5.0], [-1.0, 5.0]])
    assert backend._shape_mask(shape, wrapped[:, 0], wrapped[:, 1], lattice).all()


def test_later_geometry_overwrites_overlap(monkeypatch):
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend as backend

    # Supply explicit sample points so this inexpensive raster test needs no JAX.
    x, y = np.array([[10.0, 20.0, 40.0]]), np.array([[10.0, 10.0, 10.0]])
    monkeypatch.setattr(backend, "_cell_coordinates", lambda lattice, shape: (x, y))
    shapes = [
        dict(type="circle", mat="first", center=(10, 10), radius=15),
        dict(type="rectangle", mat="last", center=(20, 10), halfwidths=(5, 5)),
    ]
    grids = backend._permittivity_grids(
        [shapes], [1], [2 + 1j, 3 + 2j], ["first", "last"], np.eye(2) * 100, x.shape
    )
    np.testing.assert_array_equal(grids[0], [[2 + 1j, 3 + 2j, 1]])


@pytest.mark.parametrize("kind", ["ellipse", "rectangle", "polygon"])
def test_oblique_rotation_has_counterclockwise_sign(kind):
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend as backend

    shape = dict(
        type=kind,
        center=(50, 50),
        angle=45,
        halfwidths=(20, 5),
        vertices=[(-20, -5), (20, -5), (20, 5), (-20, 5)],
    )
    x, y = np.array([60.0, 60.0, 50.0]), np.array([60.0, 40.0, 50.0])
    np.testing.assert_array_equal(
        backend._shape_mask(shape, x, y, np.eye(2) * 100), [True, False, True]
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"unknown": 1},
        {"grid_spacing": 0},
        {"grid_shape": (0, 5)},
        {"grid_shape": (2.5, 5)},
        {"grid_shape": (3,)},
        {"max_grid_points": 0},
    ],
)
def test_invalid_backend_options_are_rejected(settings):
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend as backend

    with pytest.raises(ValueError):
        backend.options(settings)


def _validation_pair(tmp_path, values, difference=None):
    import json
    from pathlib import Path

    paths = [tmp_path / "cpu.npz", tmp_path / "gpu.npz"]
    record = dict(
        sha="test-sha",
        source={},
        versions={"jax": "test"},
        x64=False,
        matmul_precision="None",
        solve_matmul_precision="highest",
        dirty=False,
        python_version="3.12.14",
        runtime_environment={"XLA_FLAGS": None},
        num_terms={"planar": 1, "pattern": 45, "diffraction_1": 21},
        geometry_grids=[],
    )
    for index, path in enumerate(paths):
        np.savez(path, **values)
        provenance = record.copy()
        if index and difference:
            provenance.update(difference)
        Path(str(path) + ".json").write_text(json.dumps(provenance))
    return paths


@pytest.mark.parametrize(
    "key,values",
    [
        ("planar_A_per_layer", [-0.1, 0.2]),
        ("planar_R", [1.1]),
        ("pattern_profile", [-0.001]),
        ("matrix_1", [[-0.01, 1.01]]),
    ],
)
def test_comparison_rejects_matching_unphysical_results(tmp_path, key, values):
    from tests.fmmax_validation import compare

    paths = _validation_pair(tmp_path, {key: np.array(values)})
    with pytest.raises(AssertionError, match="physical"):
        compare(*paths)


@pytest.mark.parametrize(
    "difference",
    [
        {"python_version": "3.11.9"},
        {"runtime_environment": {"XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false"}},
        {"solve_matmul_precision": "default"},
        {"geometry_grids": ["different-material-raster"]},
    ],
)
def test_comparison_rejects_mismatched_runtime(tmp_path, difference):
    from tests.fmmax_validation import compare

    paths = _validation_pair(tmp_path, {"planar_R": np.array(0.2)}, difference)
    with pytest.raises(AssertionError):
        compare(*paths)


def test_snapshot_metadata_allows_absent_optional_cuda_packages():
    from tests.fmmax_validation import _installed_version

    assert (
        _installed_version("__rayflare_nonexistent_validation_optional_dependency__")
        is None
    )
    assert _installed_version("numpy") is not None


def test_comparison_uses_precision_and_modal_dimension_roundoff_scale(tmp_path):
    import json
    from pathlib import Path
    from tests.fmmax_validation import compare

    paths = _validation_pair(tmp_path, {"pattern_R": np.array(0.2)})
    np.savez(paths[1], pattern_R=np.array(0.200004))
    # Plausible float32 roundoff for 90 modes, but not double precision.
    compare(*paths)
    for path in paths:
        record = json.loads(Path(str(path) + ".json").read_text())
        record["x64"] = True
        Path(str(path) + ".json").write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        compare(*paths)
    np.savez(paths[1], pattern_R=np.array(0.2001))
    for path in paths:
        record = json.loads(Path(str(path) + ".json").read_text())
        record["x64"] = False
        Path(str(path) + ".json").write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        compare(*paths)
