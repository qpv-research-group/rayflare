"""Separate-process CPU/GPU snapshots of the regression fixtures.

Run from the checkout, in each source's environment::

    JAX_PLATFORMS=cpu python tests/fmmax_validation.py snapshot cpu.npz --backend cpu
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu \
        python tests/fmmax_validation.py snapshot gpu.npz --backend gpu
    python tests/fmmax_validation.py compare cpu.npz gpu.npz

Maintained source also accepts ``--eig-backend cusolver`` for GPU snapshots.
Outputs and hardware/environment records belong outside tracked source.
This is test infrastructure, not a new solver or a timing benchmark.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib import metadata
from functools import wraps
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np


def _installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _physical_bounds(data):
    for key, value in data.items():
        value = np.asarray(value)
        assert np.isfinite(value).all(), ("physical finite output", key)
        slack = 2e-8 if key.endswith("profile") else 2e-6
        assert np.all(value >= -slack), ("physical lower bound", key, value.min())
        if not key.endswith("profile"):
            assert np.all(value <= 1 + slack), (
                "physical upper bound",
                key,
                value.max(),
            )


def _power_tolerance(record, key):
    # A fixture-specific roundoff scale, not an eigensolver error theorem.
    # A solve has 2N modal amplitudes. gamma_(2N) bounds accumulation of 2N
    # rounded terms; float64 controls and eigenpair diagnostics independently
    # check the suitability of this scale for these moderately conditioned cases.
    # Keep the original planar/power floor and absolute comparisons near zero.
    if key.startswith("matrix_"):
        case = "diffraction_" + key.split("_", 1)[1]
    else:
        case = next(name for name in record["num_terms"] if key.startswith(name + "_"))
    n = 2 * record["num_terms"][case]
    eps = np.finfo(np.float64 if record["x64"] else np.float32).eps
    return max(2e-6, n * eps / (1 - n * eps))


def snapshot(path, backend, eig_backend):
    import jax
    from fmmax import fields, fmm, scattering
    from rayflare.rigorous_coupled_wave_analysis import fmmax_backend as adapter
    from test_fmmax import (
        _diffracting_result,
        _lossy_pattern_result,
        _planar_result,
        _patterned_result,
    )

    assert jax.default_backend() == backend, (jax.default_backend(), backend)
    devices = jax.devices()
    assert len(devices) == 1, devices
    witnesses = {}
    lowering = {}
    counts = {}
    eigen_diagnostics = {}
    geometry_grids = []
    original_grids = adapter._permittivity_grids

    @wraps(original_grids)
    def material_grids(*args, **kwargs):
        grids = original_grids(*args, **kwargs)
        geometry_grids.append(
            [
                dict(
                    shape=list(np.shape(grid)),
                    sha256=hashlib.sha256(
                        np.ascontiguousarray(grid, dtype="<c16").tobytes()
                    ).hexdigest(),
                )
                for grid in grids
            ]
        )
        return grids

    def witness(name, function, expected_backend=backend):
        @wraps(function)
        def wrapped(*args, **kwargs):
            result = function(*args, **kwargs)
            arrays = [
                a for a in jax.tree_util.tree_leaves(result) if isinstance(a, jax.Array)
            ]
            assert arrays, name
            for a in arrays:
                a.block_until_ready()
                assert all(d.platform == expected_backend for d in a.devices()), (
                    name,
                    a.devices(),
                )
            witnesses[name] = sorted({(str(a.dtype), str(a.device)) for a in arrays})
            counts[name] = counts.get(name, 0) + 1
            return result

        return wrapped

    def eig_witness(name, function):
        wrapped = witness(name, function)

        @wraps(function)
        def checked(matrix):
            if name not in lowering:
                lowering[name] = jax.jit(function).lower(matrix).as_text()
            result = wrapped(matrix)
            m = np.asarray(matrix, dtype=complex)
            w, v = [np.asarray(x, dtype=complex) for x in result]
            residual = np.linalg.norm(m @ v - v * w[np.newaxis, :]) / (
                np.linalg.norm(m) * np.linalg.norm(v)
            )
            entry = dict(
                dimension=len(m),
                eigenvector_condition=float(np.linalg.cond(v)),
                normalized_eigenpair_residual=float(residual),
            )
            eigen_diagnostics.setdefault(name, []).append(entry)
            return result

        return checked

    original_solve = adapter.solve

    def solve(*args, **kwargs):
        kwargs["user_options"] = dict(
            kwargs.get("user_options") or {}, eig_backend=eig_backend
        )
        return original_solve(*args, **kwargs)

    data = {}
    with ExitStack() as stack:
        stack.enter_context(patch.object(adapter, "solve", solve))
        stack.enter_context(
            patch.object(adapter, "_permittivity_grids", material_grids)
        )
        for module, name in [
            (fmm, "eigensolve_isotropic_media"),
            (scattering, "stack_s_matrices_interior"),
            (fields, "fields_on_grid"),
        ]:
            stack.enter_context(
                patch.object(module, name, witness(name, getattr(module, name)))
            )
        try:
            from fmmax import eig as eig_module

            selected = "_eig_jax"
        except ImportError:
            from fmmax import utils as eig_module

            selected = (
                "_eig_cusolver"
                if eig_backend == "cusolver"
                else ("_eig_callback" if backend == "gpu" else "_eig_jax")
            )
        if selected != "_eig_callback":
            stack.enter_context(
                patch.object(
                    eig_module,
                    selected,
                    eig_witness(selected, getattr(eig_module, selected)),
                )
            )
        else:
            # The maintained default explicitly transfers its matrix to a host callback.
            stack.enter_context(
                patch.object(
                    eig_module,
                    selected,
                    eig_witness(selected, getattr(eig_module, selected)),
                )
            )
            if backend == "gpu":
                stack.enter_context(
                    patch.object(
                        eig_module,
                        "_eig_jax_cpu",
                        witness("host_eigensolve", eig_module._eig_jax_cpu, "cpu"),
                    )
                )
        results = {
            "planar": _planar_result("p", profile_depths=np.arange(0.5, 240, 1)),
            "patterned": _patterned_result("p", orders=49, grid_spacing=2),
            "pattern": _lossy_pattern_result(49, 2, 1, eig_backend),
        }
        for side in (1, -1):
            order, matrix, _ = _diffracting_result(side, eig_backend)
            results[f"diffraction_{side}"] = order
            data[f"matrix_{side}"] = matrix[3].todense()
        for name, result in results.items():
            for key in (
                "R",
                "T",
                "A_per_layer",
                "R_per_order",
                "T_per_order",
                "profile",
            ):
                if key in result:
                    value = np.asarray(result[key])
                    assert np.isfinite(value).all(), (name, key)
                    data[f"{name}_{key}"] = value
            assert (
                abs(result["R"] + result["T"] + sum(result["A_per_layer"]) - 1) < 2e-6
            )
        assert all(
            name in witnesses
            for name in (
                "eigensolve_isotropic_media",
                "stack_s_matrices_interior",
                "fields_on_grid",
            )
        )
        assert counts.get(selected, 0) > 0, (
            "No selected eigensolver witness; check optional jeig",
            selected,
        )
        if eig_backend == "cusolver":
            assert counts.get("_eig_cusolver", 0) > 0
            assert "cusolver_geev_ffi" in lowering["_eig_cusolver"]
        elif backend == "gpu" and selected == "_eig_callback":
            assert counts.get("host_eigensolve", 0) > 0

    _physical_bounds(data)

    distribution = metadata.distribution("fmmax")
    record = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        backend=backend,
        eig_backend=eig_backend,
        x64=jax.config.x64_enabled,
        matmul_precision=str(jax.config.jax_default_matmul_precision),
        solve_matmul_precision="highest",
        python_version=platform.python_version(),
        runtime_environment={
            name: os.environ.get(name)
            for name in (
                "XLA_FLAGS",
                "JAX_ENABLE_X64",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "JAX_DISABLE_JIT",
                "NVIDIA_TF32_OVERRIDE",
            )
        },
        devices=[str(d) for d in devices],
        device_models=[d.device_kind for d in devices],
        witnesses=witnesses,
        counts=counts,
        eigen_diagnostics=eigen_diagnostics,
        geometry_grids=geometry_grids,
        num_terms={name: len(result["basis_set"]) for name, result in results.items()},
        source=json.loads(distribution.read_text("direct_url.json") or "{}"),
        versions={
            name: metadata.version(name)
            for name in (
                "fmmax",
                "jax",
                "jaxlib",
                "numpy",
            )
        },
        cuda_versions={
            name: _installed_version(name)
            for name in (
                "jax-cuda12-plugin",
                "jax-cuda12-pjrt",
                "jax-cuda13-plugin",
                "jax-cuda13-pjrt",
            )
        },
        environment={
            name: os.environ.get(name)
            for name in (
                "JAX_PLATFORMS",
                "CUDA_VISIBLE_DEVICES",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "JAX_ENABLE_X64",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "LD_LIBRARY_PATH",
            )
        },
    )
    np.savez(path, **data)
    Path(str(path) + ".json").write_text(json.dumps(record, indent=2) + "\n")
    for name, text in lowering.items():
        Path(str(path) + f".{name}.mlir").write_text(text)
    print(json.dumps(record, indent=2))


def compare(left, right):
    a, b = np.load(left), np.load(right)
    ma, mb = [json.loads(Path(str(p) + ".json").read_text()) for p in (left, right)]
    for key in (
        "sha",
        "source",
        "versions",
        "x64",
        "matmul_precision",
        "solve_matmul_precision",
        "python_version",
        "runtime_environment",
        "num_terms",
        "geometry_grids",
    ):
        assert ma[key] == mb[key], (key, ma[key], mb[key])
    assert not ma["dirty"] and not mb["dirty"], (
        "Commit the candidate before recording final evidence"
    )
    assert set(a.files) == set(b.files)
    if ma.get("backend") == mb.get("backend") == "gpu":
        assert ma["cuda_versions"] == mb["cuda_versions"]
    _physical_bounds(a)
    _physical_bounds(b)
    errors = {}
    for key in a.files:
        assert a[key].shape == b[key].shape
        assert np.isfinite(a[key]).all() and np.isfinite(b[key]).all()
        error = float(np.max(np.abs(a[key] - b[key])))
        errors[key] = dict(max_absolute=error)
        if key.endswith("profile"):
            errors[key]["integrated_absolute"] = float(
                abs(np.sum(a[key] - b[key]))
            )  # 1 nm sampling
            errors[key]["integrated_l1"] = float(np.sum(np.abs(a[key] - b[key])))
            if key.startswith("planar"):
                errors[key]["per_layer_integrated_absolute"] = [
                    float(abs(np.sum(a[key][lo:hi] - b[key][lo:hi])))
                    for lo, hi in ((0, 80), (80, 240))
                ]
            assert errors[key]["integrated_absolute"] < 2e-6, key
            assert errors[key]["integrated_l1"] < 2e-6, key
        tolerance = 2e-8 if key.endswith("profile") else _power_tolerance(ma, key)
        errors[key]["absolute_tolerance"] = float(tolerance)
        np.testing.assert_allclose(
            a[key],
            b[key],
            atol=tolerance,
            rtol=0,
            err_msg=key,
        )
    print(json.dumps(errors, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("snapshot")
    run.add_argument("output", type=Path)
    run.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    run.add_argument(
        "--eig-backend", choices=("default", "cusolver"), default="default"
    )
    diff = sub.add_parser("compare")
    diff.add_argument("left", type=Path)
    diff.add_argument("right", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.output, args.backend, args.eig_backend)
    else:
        compare(args.left, args.right)
