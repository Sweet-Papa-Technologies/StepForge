"""Engine self-test: does this install actually work, and if not, exactly why?

    python -m beatforge.selftest --json            # imports + both models load
    python -m beatforge.selftest --json --chart    # ... and chart a 20 s clip

One JSON object on the last line of stdout; exit 0 when every check passed.

This is what the desktop app's "Copy diagnostics" runs, and what the release
smoke test runs on every installer before it is published. It is deliberately
a module of its own with nothing but the standard library imported at the top:
`python -m beatforge` imports numpy and scipy before it parses a single
argument, so an install where one of those is broken could never report which.
Here every dependency is imported inside its own try block and the real
exception (type and message) is kept.

Nothing here touches the user's library: the charted clip is synthesized, and
it is charted through the real command line in a scratch folder that is
deleted afterwards. The report carries no file names or folder names from the
user's machine except the runtime's own location, which the app redacts.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

#: The engine's hard requirements, then what only the neural model needs.
#: (import name, what breaks without it)
MODULES = (
    ("numpy", "everything"),
    ("scipy", "audio analysis"),
    ("scipy.signal", "audio analysis"),
    ("soundfile", "reading and writing audio"),
    ("simfile", "writing .ssc/.sm files"),
    ("PIL", "chart preview images"),
    ("onnxruntime", "the StepForge Flow model"),
)

CLIP_SECONDS = 20
CLIP_BPM = 120.0
CLIP_DIFFICULTIES = ("easy", "hard")

#: Microsoft Visual C++ runtime libraries. The app ships its own copies next to
#: python.exe; one loaded from anywhere else means the install only works on
#: machines that happen to have the VC++ redistributable.
VC_RUNTIME_PREFIXES = ("msvcp", "vcruntime", "concrt", "vcomp", "vccorlib")


def describe(e: BaseException) -> str:
    """`ImportError: DLL load failed while importing x: The specified module…`"""
    msg = str(e).strip()
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def check_modules() -> dict:
    out = {}
    for name, used_for in MODULES:
        try:
            mod = importlib.import_module(name)
            out[name] = {"ok": True, "version": getattr(mod, "__version__", None)}
        except BaseException as e:                 # noqa: BLE001 — the report IS the point
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            out[name] = {"ok": False, "error": describe(e), "used_for": used_for}
    return out


def check_model(model_dir=None) -> dict:
    """Load both networks the way a chart run does (two ONNX sessions)."""
    out = {"ok": False, "weights_present": False, "version": None, "error": None}
    try:
        from .adapters.stepmania.neural import infer
        d = infer.resolve_model_dir(model_dir)
        out["weights_present"] = (d / infer.META_FILE).exists()
        if not out["weights_present"]:
            out["error"] = f"no model: {infer.META_FILE} is not in the model folder"
            return out
        model = infer.FlowModel(model_dir)
        out.update(ok=True, version=model.meta.get("version"),
                   display_name=model.meta.get("display_name"),
                   quantization=model.quantization)
    except BaseException as e:                     # noqa: BLE001
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        out["error"] = describe(e)
    return out


def write_clip(path: Path, seconds: int = CLIP_SECONDS, bpm: float = CLIP_BPM) -> None:
    """A click on every beat: deterministic, a few hundred KB, nobody's song."""
    import numpy as np
    import soundfile as sf
    sr = 44100
    y = np.zeros(int(seconds * sr), np.float32)
    rng = np.random.default_rng(7)
    beat = 60.0 / bpm
    for b in range(int((seconds - 1) / beat)):
        k = int((beat + b * beat) * sr)
        y[k:k + 400] += 0.6 * rng.standard_normal(400).astype(np.float32) * np.hanning(800)[400:]
    sf.write(str(path), np.stack([y, y], axis=1), sr)


def read_charts(ssc: Path) -> dict:
    """{difficulty: note count} from a written .ssc, read back with the same
    library the game-facing writer uses. Raises if the file is not a simfile."""
    import simfile
    with open(ssc, encoding="utf-8") as f:
        sim = simfile.load(f)
    counts = {}
    for chart in sim.charts:
        rows = [r for r in str(chart.notes).replace(",", "\n").split() if len(r) == 4]
        counts[str(chart.difficulty).lower()] = sum(1 for r in rows if any(c in "124" for c in r))
    return counts


def check_chart(model_dir=None) -> dict:
    """Chart the clip through the real command line, in a scratch workspace:
    to-ogg (libsndfile), analysis (scipy), both networks (onnxruntime), the
    foot model, .ssc/.sm writing (simfile) and the preview image (Pillow)."""
    out = {"ok": False, "seconds": CLIP_SECONDS, "charts": {}, "error": None}
    package = Path(__file__).resolve().parent
    try:
        from .adapters.stepmania.neural import infer
        model = infer.resolve_model_dir(model_dir)
        with tempfile.TemporaryDirectory(prefix="stepforge-selftest-") as tmp:
            ws = Path(tmp)
            # The engine works relative to where its code is, so the scratch
            # workspace gets its own copy of the code (a few hundred KB).
            shutil.copytree(package, ws / "tools" / "beatforge",
                            ignore=shutil.ignore_patterns("__pycache__", "tests", "*.pyc"))
            for d in ("assets/tracks", "public/assets/tracks", "public/maps", "build"):
                (ws / d).mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, PYTHONPATH=str(ws / "tools"), PYTHONNOUSERSITE="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                       BEATFORGE_NEURAL_MODEL_DIR=str(model))

            def run(*args: str) -> None:
                r = subprocess.run([sys.executable, "-m", "beatforge", *args], cwd=ws, env=env,
                                   capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=900)
                if r.returncode != 0:
                    tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
                    raise RuntimeError(f"`beatforge {args[0]}` exited {r.returncode}: "
                                       + " | ".join(tail))

            wav = ws / "assets/tracks/selftest.wav"
            write_clip(wav)
            run("to-ogg", "--in", str(wav), "--out", str(ws / "public/assets/tracks/selftest.ogg"))
            wav.unlink()
            run("stepforge", "--track", "selftest", "--difficulties", ",".join(CLIP_DIFFICULTIES),
                "--engine", "neural", "--backend", "local")
            song = ws / "build" / "stepmania" / "selftest"
            missing = [n for n in ("selftest.ssc", "selftest.sm", "hard.notefield.png")
                       if not (song / n).is_file()]
            if missing:
                raise RuntimeError(f"the chart run did not write {', '.join(missing)}")
            out["charts"] = read_charts(song / "selftest.ssc")
            empty = [d for d in CLIP_DIFFICULTIES if not out["charts"].get(d)]
            if empty:
                raise RuntimeError(f"the .ssc has no notes for: {', '.join(empty)}")
            out["ok"] = True
    except BaseException as e:                     # noqa: BLE001
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        out["error"] = describe(e)
    return out


def loaded_libraries() -> list[str]:
    """Paths of the native libraries loaded into this process (Windows only;
    empty elsewhere and on any failure)."""
    if os.name != "nt":
        return []
    try:
        import ctypes
        from ctypes import wintypes
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                             ctypes.POINTER(wintypes.DWORD)]
        psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE,
                                               wintypes.LPWSTR, wintypes.DWORD]
        proc = kernel32.GetCurrentProcess()
        handles = (wintypes.HMODULE * 2048)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(proc, handles, ctypes.sizeof(handles), ctypes.byref(needed)):
            return []
        buf = ctypes.create_unicode_buffer(32768)
        paths = []
        for h in handles[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            if psapi.GetModuleFileNameExW(proc, h, buf, len(buf)):
                paths.append(buf.value)
        return paths
    except Exception:                              # noqa: BLE001 — informational
        return []


def vc_runtime_outside(paths: list[str], runtime_root: str) -> list[str]:
    """Of `paths`, the VC++ runtime libraries that are NOT inside `runtime_root`
    (file names only). Renamed private copies inside wheels (numpy's
    `msvcp140-<hash>.dll`) live under the runtime and so never count.
    Windows path rules whatever the platform: this only ever sees Windows paths."""
    import ntpath
    norm = lambda p: ntpath.normcase(ntpath.normpath(p))    # noqa: E731
    root = norm(runtime_root).rstrip("\\") + "\\"
    out = []
    for p in paths:
        name = ntpath.basename(p).lower()
        if not name.startswith(VC_RUNTIME_PREFIXES) or not name.endswith(".dll"):
            continue
        if not norm(p).startswith(root):
            out.append(name)
    return sorted(set(out))


def run_selftest(chart: bool = False, model_dir=None, self_contained: bool = False) -> dict:
    report = {
        "ok": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "modules": check_modules(),
        "model": None, "chart": None, "system_vc_runtime": [], "errors": [],
    }
    for name, m in report["modules"].items():
        if not m["ok"]:
            report["errors"].append(f"{name} did not import ({m['used_for']} needs it): {m['error']}")
    try:
        from . import __version__
        report["engine"] = __version__
    except Exception:                              # noqa: BLE001
        pass
    report["model"] = check_model(model_dir)
    if not report["model"]["ok"]:
        report["errors"].append(f"the model did not load: {report['model']['error']}")
    if chart:
        if report["errors"]:
            report["chart"] = {"ok": False, "error": "skipped: the checks above failed"}
        else:
            report["chart"] = check_chart(model_dir)
            if not report["chart"]["ok"]:
                report["errors"].append(f"charting the test clip failed: {report['chart']['error']}")
    report["system_vc_runtime"] = vc_runtime_outside(loaded_libraries(), sys.base_prefix)
    if self_contained and report["system_vc_runtime"]:
        report["errors"].append(
            "these Visual C++ runtime libraries were loaded from the system, not from the app "
            "(the install would fail on a PC without the VC++ redistributable): "
            + ", ".join(report["system_vc_runtime"]))
    report["ok"] = not report["errors"]
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m beatforge.selftest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print one JSON object")
    ap.add_argument("--chart", action="store_true",
                    help=f"also chart a synthesized {CLIP_SECONDS}-second clip end to end")
    ap.add_argument("--model-dir", dest="model_dir")
    ap.add_argument("--self-contained", action="store_true", dest="self_contained",
                    help="fail if a VC++ runtime library was loaded from outside this Python "
                         "(release builds: nothing may depend on what the PC has installed)")
    a = ap.parse_args(argv)
    try:
        report = run_selftest(chart=a.chart, model_dir=a.model_dir, self_contained=a.self_contained)
    except Exception as e:                         # noqa: BLE001 — never a bare traceback
        report = {"ok": False, "errors": [f"the self-test itself failed: {describe(e)}"],
                  "traceback": traceback.format_exc(limit=6)}
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                          # noqa: BLE001
            pass
    if a.json:
        print(json.dumps(report))
    else:
        for name, m in (report.get("modules") or {}).items():
            print(f"{name:<14} {'ok ' + str(m.get('version') or '') if m['ok'] else m['error']}")
        model = report.get("model") or {}
        print(f"{'model':<14} {'ok ' + str(model.get('version') or '') if model.get('ok') else model.get('error')}")
        if report.get("chart"):
            c = report["chart"]
            print(f"{'chart':<14} {'ok ' + json.dumps(c['charts']) if c['ok'] else c['error']}")
        for e in report["errors"]:
            print(f"FAILED: {e}")
        print("self-test passed" if report["ok"] else "self-test FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
