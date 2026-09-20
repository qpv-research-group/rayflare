import builtins
import subprocess
import sys

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
