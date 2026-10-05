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
