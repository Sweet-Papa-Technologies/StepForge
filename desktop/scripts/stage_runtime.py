#!/usr/bin/env python3
"""stage_runtime.py — build the runtime StepForge bundles inside the app.

The desktop app shells out to the Python charting engine. Release builds ship
everything it needs so users install nothing else:

    desktop/src-tauri/runtime/
      python/   a self-contained CPython (python-build-standalone) with the
                engine's packages installed into it
      engine/   tools/beatforge, requirements.txt, package.json,
                models/stepforge_flow (the neural model), ENGINE_VERSION

The release config (`src-tauri/tauri.release.conf.json`) bundles that folder as
an app resource; on first run the app copies `engine/` into a writable
per-user workspace and runs it with `python/`.

    python3 desktop/scripts/stage_runtime.py                 # this machine's platform
    python3 desktop/scripts/stage_runtime.py --version 0.1.0-beta.1
    SIGN_IDENTITY="Developer ID Application: …" python3 desktop/scripts/stage_runtime.py

On macOS with SIGN_IDENTITY set, every Mach-O file in the runtime is signed
with the hardened runtime and a secure timestamp — notarization rejects an app
whose bundled binaries are not. The interpreter gets `python.entitlements`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESKTOP = HERE.parent
REPO = DESKTOP.parent
OUT = DESKTOP / "src-tauri" / "runtime"

#: python-build-standalone release and CPython version the builds pin.
PBS_TAG = "20261110"
PY_VERSION = "3.12.16"
PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/"
           "{tag}/cpython-{py}+{tag}-{triple}-install_only.tar.gz")

#: Directories of the standard library nobody running the engine needs.
STDLIB_PRUNE = ("test", "idlelib", "turtledemo", "ensurepip", "lib2to3", "pydoc_data")


def log(msg: str) -> None:
    print(f"[stage-runtime] {msg}", flush=True)


def host_triple() -> str:
    m = platform.machine().lower()
    arch = "aarch64" if m in ("arm64", "aarch64") else "x86_64"
    if sys.platform == "darwin":
        return f"{arch}-apple-darwin"
    if sys.platform.startswith("win"):
        return f"{arch}-pc-windows-msvc"
    return f"{arch}-unknown-linux-gnu"


def python_exe(root: Path) -> Path:
    return root / ("python.exe" if os.name == "nt" else "bin/python3")


def fetch_python(triple: str, cache: Path, dest: Path) -> None:
    url = PBS_URL.format(tag=PBS_TAG, py=PY_VERSION, triple=triple)
    cache.mkdir(parents=True, exist_ok=True)
    tgz = cache / url.rsplit("/", 1)[1]
    if not tgz.exists():
        log(f"downloading {url}")
        part = tgz.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
        part.replace(tgz)
    log(f"extracting {tgz.name}")
    tmp = dest.parent / ".python.extract"
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(tgz) as t:
        t.extractall(tmp, filter="tar") if hasattr(tarfile, "data_filter") else t.extractall(tmp)
    shutil.move(str(tmp / "python"), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)


def macos_minimum() -> str:
    """The oldest macOS the app declares (tauri.conf.json), e.g. "12.0"."""
    conf = json.loads((DESKTOP / "src-tauri" / "tauri.conf.json").read_text())
    return str(conf["bundle"]["macOS"]["minimumSystemVersion"])


def wheel_platform(triple: str) -> str | None:
    """On macOS, the wheel tag for the app's declared minimum, else None.

    pip on its own picks wheels for the BUILD machine's macOS: on a macOS 15
    runner that meant numpy, scipy and onnxruntime builds that need macOS 14,
    inside an app that says it runs on older systems. Pinning the platform tag
    to the declared minimum makes pip choose wheels every supported Mac can load
    (and fail the build if a dependency has none)."""
    if not triple.endswith("-apple-darwin"):
        return None
    # Wheel tags for macOS 11+ name the major version only (macosx_12_0 covers
    # every 12.x); check_macos_minimum() catches a binary built for a later minor.
    major = macos_minimum().split(".")[0]
    arch = "arm64" if triple.startswith("aarch64") else "x86_64"
    return f"macosx_{major}_0_{arch}"


def _version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def check_macos_minimum(py_root: Path) -> None:
    """Fail if any bundled binary needs a newer macOS than the app declares.

    A wheel's tag is a promise, not a guarantee: scipy's macosx_12_0 arm64
    wheel contains extensions built for 12.3. Read the real load-command minimum
    (LC_BUILD_VERSION / LC_VERSION_MIN_MACOSX) of every binary instead."""
    floor = macos_minimum()
    worst: dict[str, str] = {}
    for f in py_root.rglob("*"):
        if not f.is_file() or f.is_symlink() or not (f.suffix in (".so", ".dylib") or f.parent.name == "bin"):
            continue
        # -arch all: universal2 files carry one minimum per slice.
        out = subprocess.run(["otool", "-arch", "all", "-l", str(f)],
                             capture_output=True, text=True).stdout
        legacy = False                       # inside an LC_VERSION_MIN_MACOSX command
        for line in out.splitlines():
            s = line.split()
            if s[:2] == ["cmd", "LC_VERSION_MIN_MACOSX"]:
                legacy = True
            elif s[:1] == ["minos"] or (legacy and s[:1] == ["version"]):
                legacy = False
                if _version(s[1]) > _version(floor):
                    worst[str(f.relative_to(py_root))] = s[1]
    if worst:
        listing = "\n  ".join(f"{v}  {k}" for k, v in sorted(worst.items())[:20])
        raise SystemExit(f"{len(worst)} bundled binaries need a newer macOS than the declared "
                         f"minimum {floor} (tauri.conf.json bundle.macOS.minimumSystemVersion):\n  {listing}")
    log(f"every bundled binary loads on macOS {floor}")


def install_packages(py: Path, triple: str) -> None:
    reqs = [l.split("#", 1)[0].strip() for l in (REPO / "requirements.txt").read_text().splitlines()]
    reqs = [r for r in reqs if r]
    log(f"installing {len(reqs)} packages: {' '.join(reqs)}")
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1")
    cmd = [str(py), "-m", "pip", "install", "--no-cache-dir", "--no-compile",
           "--only-binary", ":all:"]
    plat = wheel_platform(triple)
    if plat:
        # A platform tag requires --target: install straight into the
        # interpreter's own site-packages.
        site = subprocess.run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                              check=True, capture_output=True, text=True).stdout.strip()
        log(f"wheels for {plat} (the app's minimum macOS)")
        cmd += ["--platform", plat, "--implementation", "cp",
                "--python-version", ".".join(PY_VERSION.split(".")[:2]), "--target", site]
    # setuptools < 81: `fs` (pulled in by simfile) still imports pkg_resources,
    # which setuptools 81 removed — without the pin simfile cannot be imported.
    subprocess.run([*cmd, *reqs, "setuptools<81"], check=True, env=env)


def prune(py_root: Path) -> None:
    """Drop what the engine never loads: pip itself, stdlib tests, bytecode
    caches (the app redirects bytecode outside the bundle anyway)."""
    lib = next((p for p in [py_root / "Lib", *sorted((py_root / "lib").glob("python3.*"))]
                if p.is_dir()), None)
    if lib:
        for name in STDLIB_PRUNE:
            shutil.rmtree(lib / name, ignore_errors=True)
        site = lib / "site-packages"
        # `bin` holds console scripts that `pip install --target` drops there.
        for d in list(site.glob("pip")) + list(site.glob("pip-*.dist-info")) + [site / "bin"]:
            shutil.rmtree(d, ignore_errors=True)
    for d in list(py_root.rglob("__pycache__")):
        shutil.rmtree(d, ignore_errors=True)
    # Package test suites (numpy/scipy/matplotlib ship theirs). Only `tests`:
    # `numpy.testing` and friends are imported at runtime.
    for d in list(py_root.rglob("tests")):
        if d.is_dir() and "site-packages" in d.parts:
            shutil.rmtree(d, ignore_errors=True)


def copy_engine(dest: Path, version: str) -> None:
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "tests", "*.pyc")
    shutil.copytree(REPO / "tools" / "beatforge", dest / "tools" / "beatforge", ignore=ignore)
    for f in ("requirements.txt", "package.json"):
        shutil.copy2(REPO / f, dest / f)
    shutil.copytree(REPO / "models" / "stepforge_flow", dest / "models" / "stepforge_flow")
    for d in ("assets/tracks", "public/assets/tracks", "public/maps", "build"):
        (dest / d).mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001 — not a checkout
        sha = "unknown"
    (dest / "ENGINE_VERSION").write_text(f"{version} {sha}\n")


def is_macho(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            magic = f.read(4)
    except OSError:
        return False
    return magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe",
                     b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce", b"\xbf\xba\xfe\xca")


def sign_macos(root: Path, identity: str) -> None:
    ents = HERE / "python.entitlements"
    files = [p for p in root.rglob("*") if p.is_file() and not p.is_symlink() and is_macho(p)]
    exes = {p for p in files if os.access(p, os.X_OK) and p.parent.name == "bin"}
    log(f"signing {len(files)} Mach-O files with '{identity}'")
    # libraries first, executables last
    for p in sorted(files, key=lambda p: p in exes):
        cmd = ["codesign", "--force", "--timestamp", "--options", "runtime", "--sign", identity]
        if p in exes:
            cmd += ["--entitlements", str(ents)]
        subprocess.run(cmd + [str(p)], check=True, capture_output=True)
    log("signatures applied")


INFER_CHECK = """
import sys
import numpy as np
import onnxruntime
from beatforge.adapters.stepmania.neural import conditioning as cond
from beatforge.adapters.stepmania.neural import grid_features as GF
from beatforge.adapters.stepmania.neural import infer
rng = np.random.default_rng(3)
y = np.zeros(int(30 * GF.SR), np.float32)
for b in range(140):                               # a click every 0.2 s (150 BPM)
    k = int((0.2 + b * 0.2) * GF.SR)
    y[k:k + 60] += rng.standard_normal(60).astype(np.float32)
g = GF.extract_grid_features(y, GF.TimingMap.constant(150.0, 0.2))
eng = infer.StepManiaNeuralEngine(sys.argv[1])
counts = {}
for d in ("easy", "hard"):
    ch = eng.generate(g.X, g.tick_times_sec, d, cond.resolve_style(d), 30.0, seed=1)
    counts[d] = len(infer.to_placements(ch))
assert all(counts.values()), counts
print(f"{onnxruntime.__version__} (charted easy {counts['easy']} / hard {counts['hard']} notes)")
"""


def smoke_test(py: Path, engine: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(engine / "tools"), PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([str(py), "-m", "beatforge", "neural-status", "--json"],
                         cwd=engine, env=env, capture_output=True, text=True, check=True)
    status = json.loads(out.stdout.strip().splitlines()[-1])
    if not status.get("installed"):
        raise SystemExit(f"the staged engine does not see its model: {status.get('error')}")
    # Loading is not running: chart a synthetic click track with the real
    # model, so an onnxruntime build that loads the model but cannot execute
    # it (each platform resolves its own onnxruntime version) fails here.
    run = subprocess.run([str(py), "-c", INFER_CHECK, str(engine / "models" / "stepforge_flow")],
                         cwd=engine, env=env, capture_output=True, text=True)
    if run.returncode != 0:
        raise SystemExit(f"the staged engine could not chart:\n{run.stderr[-3000:]}")
    log(f"smoke test ok: {status.get('display_name')} via onnxruntime "
        f"{run.stdout.strip().splitlines()[-1]}, "
        f"style index {'yes' if status.get('style_rag_available') else 'no'}")


def tree_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triple", default=host_triple(),
                    help="python-build-standalone target (default: this machine)")
    ap.add_argument("--version", default=os.environ.get("STEPFORGE_VERSION", "0.0.0-dev"),
                    help="app version stamped into ENGINE_VERSION")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cache", default=str(REPO / ".cache" / "runtime"))
    ap.add_argument("--keep-python", action="store_true",
                    help="reuse an already staged python/ (refresh the engine only)")
    a = ap.parse_args()

    out = Path(a.out)
    py_root, engine = out / "python", out / "engine"
    if not (a.keep_python and python_exe(py_root).exists()):
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        fetch_python(a.triple, Path(a.cache), py_root)
        install_packages(python_exe(py_root), a.triple)
        prune(py_root)
        if a.triple.endswith("-apple-darwin"):
            check_macos_minimum(py_root)
    shutil.rmtree(engine, ignore_errors=True)
    copy_engine(engine, a.version)
    smoke_test(python_exe(py_root), engine)
    for p in out.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)

    identity = os.environ.get("SIGN_IDENTITY", "").strip()
    if sys.platform == "darwin" and identity:
        sign_macos(out, identity)

    digest = hashlib.sha256((engine / "ENGINE_VERSION").read_bytes()).hexdigest()[:12]
    log(f"runtime ready at {out}: python {tree_size(py_root) / 1e6:.0f} MB, "
        f"engine {tree_size(engine) / 1e6:.0f} MB ({digest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
