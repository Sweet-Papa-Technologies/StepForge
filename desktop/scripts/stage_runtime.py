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
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native_audit  # noqa: E402

DESKTOP = HERE.parent
REPO = DESKTOP.parent
OUT = DESKTOP / "src-tauri" / "runtime"

#: python-build-standalone release and CPython version the builds pin.
PBS_TAG = "20261110"
PY_VERSION = "3.12.16"
PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/"
           "{tag}/cpython-{py}+{tag}-{triple}-install_only.tar.gz")

#: Directories of the standard library nobody running the engine needs.
STDLIB_PRUNE = ("test", "idlelib", "turtledemo", "ensurepip", "lib2to3", "pydoc_data", "tkinter")

#: Tcl/Tk, bundled by python-build-standalone for tkinter; the engine has no
#: GUI. Folders directly under python/lib (tcl9.0, tk9.0, itcl4.x, thread3.x)
#: and the shared libraries / extension module that load them.
TK_DIR_RE = re.compile(r"^(tcl|tk|itcl|thread|tdbc\w*)\d", re.I)
TK_FILE_RE = re.compile(r"^((lib)?(tcl|tk)\d[\w.]*\.(dll|dylib|so)|_tkinter\..*)$", re.I)

#: Parts of installed packages the engine never imports. Each file is one more
#: thing an installer writes and a virus scanner opens, so they cost install
#: time on Windows well beyond their size.
SITE_PRUNE = (
    "onnxruntime/transformers", "onnxruntime/tools", "onnxruntime/quantization",
    "onnxruntime/datasets", "numpy/_core/include",
)
# Not numpy/f2py: scipy imports it while loading (scipy.signal, .stats, …).
#: Type stubs, C headers and Cython sources: build-time only.
SITE_PRUNE_SUFFIXES = (".pyi", ".h", ".pxd", ".pxi", ".pyx")


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
    """Drop what the engine never loads: pip itself, stdlib tests, Tcl/Tk,
    C headers, type stubs, package tooling, bytecode caches (the app redirects
    bytecode outside the bundle anyway). smoke_test() then charts a song end to
    end with what is left, so an over-eager rule fails the build, not a user."""
    before = sum(1 for f in py_root.rglob("*") if f.is_file())
    # The stdlib folder is the one holding os.py: python/lib/python3.x on macOS
    # and Linux, python/Lib on Windows. (Testing for "Lib" alone matched
    # python/lib on macOS's case-insensitive disk, so nothing in site-packages
    # was pruned there and the Mac smoke test could not catch a bad rule.)
    lib = next((p for p in [*sorted((py_root / "lib").glob("python3.*")), py_root / "Lib"]
                if (p / "os.py").is_file()), None)
    if lib is None:
        raise SystemExit(f"no standard library found under {py_root}")
    if lib:
        for name in STDLIB_PRUNE:
            shutil.rmtree(lib / name, ignore_errors=True)
        (lib / "turtle.py").unlink(missing_ok=True)
        # `crypt` (deprecated, removed in Python 3.13, unused here): its
        # extension links libcrypt.so.1, which Arch-based distributions do not
        # install. Nothing loads it, but it is the one library in the bundle
        # that a supported system may not have, so it does not ship.
        (lib / "crypt.py").unlink(missing_ok=True)
        for f in (lib / "lib-dynload").glob("_crypt.*"):
            f.unlink()
        site = lib / "site-packages"
        # `bin` holds console scripts that `pip install --target` drops there.
        for d in list(site.glob("pip")) + list(site.glob("pip-*.dist-info")) + [site / "bin"]:
            shutil.rmtree(d, ignore_errors=True)
        for rel in SITE_PRUNE:
            shutil.rmtree(site / rel, ignore_errors=True)
        for f in list(site.rglob("*")):
            if f.is_file() and (f.suffix in SITE_PRUNE_SUFFIXES or f.name == "py.typed"):
                f.unlink()
    # Tcl/Tk: python/tcl on Windows, python/lib/{tcl9.0,tk9.0,...} elsewhere.
    shutil.rmtree(py_root / "tcl", ignore_errors=True)
    for d in list((py_root / "lib").glob("*")) if (py_root / "lib").is_dir() else []:
        if d.is_dir() and TK_DIR_RE.match(d.name):
            shutil.rmtree(d, ignore_errors=True)
    for f in list(py_root.rglob("*")):
        if f.is_file() and TK_FILE_RE.match(f.name):
            f.unlink()
    # Headers, import libraries and man pages for building against Python.
    for d in ("include", "libs", "share"):
        shutil.rmtree(py_root / d, ignore_errors=True)
    for d in list(py_root.rglob("__pycache__")):
        shutil.rmtree(d, ignore_errors=True)
    # Package test suites (numpy/scipy/matplotlib ship theirs). Only `tests`:
    # `numpy.testing` and friends are imported at runtime.
    for d in list(py_root.rglob("tests")):
        if d.is_dir() and "site-packages" in d.parts:
            shutil.rmtree(d, ignore_errors=True)
    after = sum(1 for f in py_root.rglob("*") if f.is_file())
    log(f"pruned the runtime: {before} -> {after} files")


def platform_of(triple: str) -> str:
    return ("windows" if "windows" in triple else "macos" if triple.endswith("-apple-darwin")
            else "linux")


def vc_redist_dirs() -> list[Path]:
    """Where this build machine keeps redistributable copies of the Visual C++
    runtime, best first: Visual Studio's own redist folder (newest toolset),
    then System32 (present when the VC++ redistributable is installed)."""
    dirs: list[Path] = []
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        r = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            base = Path(line.strip()) / "VC" / "Redist" / "MSVC"
            if base.is_dir():
                # 14.44.35112/x64/Microsoft.VC143.CRT — newest version first.
                found = [d for d in base.glob("*/x64/Microsoft.VC*.CRT") if d.is_dir()]
                dirs += sorted(found, key=lambda d: [int(x) if x.isdigit() else 0
                                                     for x in d.parts[-3].split(".")], reverse=True)
    dirs.append(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")
    return dirs


def bundle_vc_runtime(py_root: Path, sources: list[Path] | None = None) -> list[str]:
    """Copy the Visual C++ runtime DLLs the bundle imports but lacks to next to
    python.exe (the application directory, which Windows searches before
    System32). Returns the names copied.

    python-build-standalone ships vcruntime140.dll and vcruntime140_1.dll;
    numpy brings a private msvcp140. onnxruntime imports msvcp140.dll and
    msvcp140_1.dll and brings neither, so on a PC without the VC++
    redistributable `import onnxruntime` failed ("DLL load failed") while
    everything else worked. Driven by the audit, so a future dependency on
    another of these DLLs is picked up without editing a list."""
    sources = vc_redist_dirs() if sources is None else sources
    copied: list[str] = []
    for _ in range(4):                       # a copied DLL can import another
        missing = [n for n in native_audit.audit(py_root, "windows") if native_audit.VC_RUNTIME.match(n)]
        if not missing:
            break
        for name in missing:
            src = next((d / name for d in sources if (d / name).is_file()), None)
            if src is None:
                raise SystemExit(f"the bundle needs {name} and this machine has no copy to ship "
                                 f"(looked in: {', '.join(str(d) for d in sources)})")
            shutil.copy2(src, py_root / name)
            copied.append(name)
            log(f"bundled {name} from {src.parent}")
    return copied


def audit_native(py_root: Path, triple: str) -> None:
    """Fail the build if the runtime loads a library it does not contain and
    the operating system does not provide (native_audit.py)."""
    missing = native_audit.audit(py_root, platform_of(triple))
    if missing:
        raise SystemExit("the bundled runtime loads libraries it does not contain, so it would only "
                         "work on machines that happen to have them:\n" + native_audit.describe(missing))
    log("every native library the runtime loads is bundled or part of the OS")


def copy_engine(dest: Path, version: str, model_dir: Path | None = None) -> None:
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "tests", "*.pyc")
    shutil.copytree(REPO / "tools" / "beatforge", dest / "tools" / "beatforge", ignore=ignore)
    for f in ("requirements.txt", "package.json"):
        shutil.copy2(REPO / f, dest / f)
    # a candidate model can be staged for a local test build without
    # installing it in the repository (--model-dir)
    shutil.copytree(model_dir or REPO / "models" / "stepforge_flow",
                    dest / "models" / "stepforge_flow")
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


#: A 24 s click track, written with the bundled interpreter. check_appimage.py
#: uses it for the song it imports through the app.
E2E_AUDIO = """
import sys
import numpy as np
import soundfile as sf
sr = 44100
y = np.zeros(int(24 * sr), np.float32)
rng = np.random.default_rng(7)
for b in range(int(23 / 0.5)):                     # a beat every 0.5 s (120 BPM)
    k = int((0.5 + b * 0.5) * sr)
    y[k:k + 400] += 0.6 * rng.standard_normal(400).astype(np.float32) * np.hanning(800)[400:]
sf.write(sys.argv[1], np.stack([y, y], axis=1), sr)
"""


def smoke_test(py: Path, engine: Path, triple: str) -> None:
    """Run the engine's own self-test with the staged interpreter: every
    dependency imports, both networks load, and a 20-second clip is charted end
    to end through the real command line (to-ogg, analysis, the model, the foot
    model, .ssc/.sm writing, the preview PNG). An over-eager pruning rule or an
    onnxruntime build that loads but cannot run fails here, not on a user's PC.

    On Windows it also has to be self-contained: no Visual C++ runtime DLL may
    come from the build machine's System32."""
    env = dict(os.environ, PYTHONPATH=str(engine / "tools"), PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    env.pop("BEATFORGE_NEURAL_MODEL_DIR", None)
    cmd = [str(py), "-m", "beatforge.selftest", "--json", "--chart"]
    if platform_of(triple) == "windows":
        cmd.append("--self-contained")
    run = subprocess.run(cmd, cwd=engine, env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=1200)
    try:
        report = json.loads(run.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise SystemExit(f"the staged engine's self-test gave no report (exit {run.returncode}):\n"
                         f"{run.stdout[-1500:]}\n{run.stderr[-3000:]}") from None
    if not report.get("ok"):
        raise SystemExit("the staged engine failed its self-test:\n  "
                         + "\n  ".join(report.get("errors") or ["(no reason given)"])
                         + f"\n{run.stderr[-2000:]}")
    mods = report["modules"]
    log(f"smoke test ok: {report['model'].get('display_name')} via onnxruntime "
        f"{mods['onnxruntime'].get('version')}, numpy {mods['numpy'].get('version')}")
    log(f"smoke test ok: charted a {report['chart']['seconds']} s clip end to end "
        f"({report['chart']['charts']})")


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
    ap.add_argument("--model-dir",
                    help="bundle this model folder instead of models/stepforge_flow "
                         "(a local test build of a candidate model)")
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
    if platform_of(a.triple) == "windows":
        bundle_vc_runtime(py_root)
    audit_native(py_root, a.triple)
    shutil.rmtree(engine, ignore_errors=True)
    model_dir = Path(a.model_dir).expanduser().resolve() if a.model_dir else None
    if model_dir is not None and not (model_dir / "model_meta.json").is_file():
        raise SystemExit(f"--model-dir {model_dir} has no model_meta.json")
    copy_engine(engine, a.version, model_dir)
    smoke_test(python_exe(py_root), engine, a.triple)
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
