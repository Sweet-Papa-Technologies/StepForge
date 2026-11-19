"""The engine self-test (`python -m beatforge.selftest`): every failure is
reported with its own real cause, and a working install charts a clip."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from beatforge import selftest

TOOLS = Path(__file__).resolve().parents[2]
MODEL = TOOLS.parent / "models" / "stepforge_flow"
needs_model = pytest.mark.skipif(not (MODEL / "model_meta.json").exists(),
                                 reason="the StepForge Flow model is not in this checkout")


def test_a_module_that_does_not_import_is_reported_with_its_real_error(monkeypatch):
    # What `import onnxruntime` does on a PC without the VC++ runtime.
    def broken(name, *a, **k):
        if name == "onnxruntime":
            raise ImportError("DLL load failed while importing onnxruntime_pybind11_state: "
                              "The specified module could not be found.")
        return real(name, *a, **k)
    real = selftest.importlib.import_module
    monkeypatch.setattr(selftest.importlib, "import_module", broken)
    mods = selftest.check_modules()
    assert mods["numpy"]["ok"] and mods["numpy"]["version"]
    assert mods["onnxruntime"] == {
        "ok": False,
        "error": "ImportError: DLL load failed while importing onnxruntime_pybind11_state: "
                 "The specified module could not be found.",
        "used_for": "the StepForge Flow model",
    }


def test_each_failure_gets_its_own_line_in_the_report(monkeypatch):
    monkeypatch.setattr(selftest, "check_modules", lambda: {
        "numpy": {"ok": True, "version": "2"},
        "soundfile": {"ok": False, "error": "OSError: cannot load library 'libsndfile.so'",
                      "used_for": "reading and writing audio"}})
    monkeypatch.setattr(selftest, "check_model", lambda d=None: {
        "ok": False, "weights_present": False, "error": "no model: model_meta.json is not in the model folder"})
    r = selftest.run_selftest(chart=True)
    assert r["ok"] is False
    assert r["errors"] == [
        "soundfile did not import (reading and writing audio needs it): "
        "OSError: cannot load library 'libsndfile.so'",
        "the model did not load: no model: model_meta.json is not in the model folder",
    ]
    assert r["chart"] == {"ok": False, "error": "skipped: the checks above failed"}


def test_a_folder_without_a_model_is_a_missing_model_not_a_crash(tmp_path):
    m = selftest.check_model(tmp_path)
    assert m["ok"] is False and m["weights_present"] is False
    assert "model_meta.json" in m["error"]


def test_vc_runtime_loaded_from_windows_is_found_and_bundled_copies_are_not():
    root = r"C:\Program Files\StepForge\runtime\python"
    loaded = [
        root + r"\python312.dll",
        root + r"\vcruntime140.dll",                                  # the app's own
        root + r"\Lib\site-packages\numpy.libs\msvcp140-a4c2229b.dll",  # numpy's private copy
        r"C:\Windows\System32\MSVCP140.dll",                           # the machine's
        r"C:\WINDOWS\SYSTEM32\msvcp140_1.dll",
        r"C:\Windows\System32\msvcp_win.dll",                         # part of Windows, not of VC++
        r"C:\Windows\System32\kernel32.dll",
    ]
    assert selftest.vc_runtime_outside(loaded, root) == ["msvcp140.dll", "msvcp140_1.dll"]
    assert selftest.vc_runtime_outside(loaded[:3], root) == []


def test_status_tells_a_broken_onnxruntime_from_a_missing_model(monkeypatch):
    """`installed` is false in both cases; the UI must be able to tell them apart."""
    from beatforge.adapters.stepmania.neural import infer
    if not (MODEL / "model_meta.json").exists():
        pytest.skip("no model in this checkout")
    monkeypatch.setattr(infer, "_onnxruntime_error", lambda: "ImportError: DLL load failed")
    s = infer.neural_status(MODEL)
    assert s["installed"] is False and s["weights_present"] is True
    assert s["onnxruntime"] is False and s["onnxruntime_error"] == "ImportError: DLL load failed"
    assert "ImportError: DLL load failed" in s["error"] and "pip install" not in s["error"]


def test_status_for_a_missing_model_says_so(tmp_path):
    from beatforge.adapters.stepmania.neural import infer
    s = infer.neural_status(tmp_path)
    assert s["installed"] is False and s["weights_present"] is False
    assert "not installed" in s["error"]


def _cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(TOOLS))
    return subprocess.run([sys.executable, "-m", "beatforge.selftest", *args],
                          capture_output=True, text=True, env=env, timeout=900)


def test_the_cli_exits_nonzero_and_still_prints_one_json_report(tmp_path):
    res = _cli("--json", "--model-dir", str(tmp_path))
    assert res.returncode == 1
    report = json.loads(res.stdout.strip().splitlines()[-1])
    assert report["ok"] is False
    assert any("the model did not load" in e for e in report["errors"])


@needs_model
def test_a_working_install_charts_the_clip_end_to_end():
    pytest.importorskip("onnxruntime")
    res = _cli("--json", "--chart")
    report = json.loads(res.stdout.strip().splitlines()[-1])
    assert res.returncode == 0, report["errors"]
    assert report["ok"] is True and report["model"]["ok"] is True
    assert report["chart"]["ok"] is True and report["chart"]["seconds"] == 20
    assert set(report["chart"]["charts"]) == {"easy", "hard"}
    assert all(n > 0 for n in report["chart"]["charts"].values())
