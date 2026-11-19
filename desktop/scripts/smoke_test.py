#!/usr/bin/env python3
"""smoke_test.py — install a release artifact and prove it can chart a song.

    python3 desktop/scripts/smoke_test.py dist/StepForge-<v>-macos-arm64.dmg
    python  desktop/scripts/smoke_test.py dist/StepForge-<v>-windows-x64-setup.exe
    python3 desktop/scripts/smoke_test.py path/to/StepForge.app        # an unpacked build

The release workflow runs this on a fresh runner for every macOS and Windows
installer, after the build and before anything is published; a failure blocks
the release. (Linux installers are tested inside distribution containers by
smoke_linux.sh, which needs no Python on the test machine.)

What it does, as a user's first launch would:

1. Installs the artifact: the .dmg is mounted and the app copied out of it;
   the Windows installer runs silently into a fresh folder.
2. Runs `StepForge --headless selftest` — the app's own launcher, interpreter
   and environment — with a new, empty profile each time. The self-test imports
   every engine dependency (onnxruntime among them), loads both models, charts
   a 20-second clip and checks that a valid .ssc came out.
3. Does it twice: once with a plain profile path, once with one that has
   spaces and non-ASCII characters, like a real user name.

On Windows the self-test must also be self-contained: no Visual C++ runtime
DLL may be loaded from the machine's System32 (CI runners have the VC++
redistributable installed; a new PC may not).
"""
from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: Profile folder names: plain, then what a real user name can look like.
PROFILES = (("plain", "profile"), ("non-ASCII", "Zoë Müller 用户"))

#: Variables that could prop up a broken bundle, or point the app at a
#: developer's engine, model or workspace.
SCRUB_PREFIXES = ("PYTHON", "BEATFORGE_", "CONDA", "VIRTUAL_ENV")


def log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def clean_env(profile: Path) -> dict[str, str]:
    """This machine's environment with nothing of ours in it, and the app's
    workspace (engine, model, songs) in a new empty folder under `profile`."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(SCRUB_PREFIXES)}
    profile.mkdir(parents=True, exist_ok=True)
    env["BEATFORGE_WORKSPACE"] = str(profile / "workspace")
    if os.name != "nt":
        # A whole new home: settings, caches and the workspace all start empty.
        env["HOME"] = str(profile)
        env["XDG_DATA_HOME"] = str(profile / ".local/share")
        env["XDG_CONFIG_HOME"] = str(profile / ".config")
        env["XDG_CACHE_HOME"] = str(profile / ".cache")
    return env


def install_dmg(dmg: Path, tmp: Path) -> Path:
    mount = tmp / "mount"
    mount.mkdir()
    subprocess.run(["hdiutil", "attach", str(dmg), "-nobrowse", "-readonly", "-mountpoint", str(mount)],
                   check=True, stdout=subprocess.DEVNULL)
    try:
        apps = sorted(mount.glob("*.app"))
        if not apps:
            raise SystemExit(f"no .app inside {dmg.name}")
        dest = tmp / "Applications" / apps[0].name
        dest.parent.mkdir()
        # ditto keeps the signature, symlinks and extended attributes intact.
        subprocess.run(["ditto", str(apps[0]), str(dest)], check=True)
    finally:
        subprocess.run(["hdiutil", "detach", str(mount), "-force"], stdout=subprocess.DEVNULL)
    return dest


def app_executable(app: Path) -> Path:
    with open(app / "Contents/Info.plist", "rb") as f:
        name = plistlib.load(f)["CFBundleExecutable"]
    return app / "Contents/MacOS" / name


def install_windows(setup: Path, tmp: Path) -> Path:
    dest = tmp / "app"
    # NSIS: /S is silent; /D= must be last and unquoted.
    r = subprocess.run(f'"{setup}" /S /D={dest}', timeout=1800)
    if r.returncode != 0:
        raise SystemExit(f"the installer exited {r.returncode}")
    exes = [p for p in dest.glob("*.exe") if "uninstall" not in p.name.lower()]
    if len(exes) != 1:
        raise SystemExit(f"expected one program in {dest}, found {[p.name for p in exes]}")
    count = sum(1 for p in dest.rglob("*") if p.is_file())
    log(f"installed {count} files to {dest}")
    return exes[0]


def install(artifact: Path, tmp: Path) -> Path:
    """Install `artifact`; returns the program to run."""
    name = artifact.name.lower()
    if name.endswith(".dmg"):
        return app_executable(install_dmg(artifact, tmp))
    if name.endswith(".app"):
        # Copied out like an install: an app left inside a source checkout
        # finds that checkout's engine next to itself and would test that.
        dest = tmp / "Applications" / artifact.name
        dest.parent.mkdir()
        subprocess.run(["ditto", str(artifact), str(dest)], check=True)
        return app_executable(dest)
    if name.endswith(".exe"):
        return install_windows(artifact, tmp)
    if artifact.is_file() and os.access(artifact, os.X_OK):
        return artifact                       # an already installed program
    raise SystemExit(f"do not know how to install {artifact.name}")


def selftest(exe: Path, profile: Path, extra: list[str]) -> tuple[bool, str]:
    """Run the app's self-test with a new profile. (passed, what happened)."""
    cmd = [str(exe), "--headless", "selftest", *extra]
    try:
        r = subprocess.run(cmd, env=clean_env(profile), cwd=str(profile), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=1800)
    except subprocess.TimeoutExpired:
        return False, "timed out after 30 minutes"
    result = None
    for line in r.stdout.splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("op") == "selftest":
            result = obj
    if result is None:
        return False, (f"no self-test result (exit {r.returncode})\nstdout:\n{r.stdout[-2000:]}"
                       f"\nstderr:\n{r.stderr[-4000:]}")
    report = result.get("report") or {}
    if not result.get("ok") or r.returncode != 0:
        errors = report.get("errors") or [result.get("error") or "(no reason given)"]
        media = result.get("media") or {}
        if media.get("checked") and not media.get("ok"):
            errors.append(f"audio playback is missing GStreamer elements: {media.get('missing')}")
        return False, "\n  ".join([f"self-test FAILED (exit {r.returncode}):", *errors,
                                   "stderr:", r.stderr[-3000:]])
    # A first launch provisions the engine into the new profile. If it is not
    # there, the app ran some other engine (a checkout, a saved setting) and
    # this was not a test of the installer.
    if not (profile / "workspace/tools/beatforge/__main__.py").is_file():
        return False, f"the app did not set up its engine in the new profile ({profile / 'workspace'})"
    chart = report.get("chart") or {}
    mods = report.get("modules") or {}
    return True, (f"onnxruntime {mods.get('onnxruntime', {}).get('version')}, "
                  f"{(report.get('model') or {}).get('display_name')}, "
                  f"charted {chart.get('seconds')} s -> {chart.get('charts')} "
                  f"on {report.get('platform')}")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    artifact = Path(sys.argv[1]).resolve()
    extra = sys.argv[2:]
    if os.name == "nt" and "--self-contained" not in extra:
        extra.append("--self-contained")
    failures = []
        # realpath below: macOS temp folders are symlinks, and the app resolves paths.
    t = tempfile.mkdtemp(prefix="stepforge-smoke-")
    try:
        tmp = Path(os.path.realpath(t))
        exe = install(artifact, tmp)
        log(f"program: {exe}")
        for label, folder in PROFILES:
            ok, what = selftest(exe, tmp / "profiles" / folder, extra)
            log(f"{label} profile path: {'ok — ' if ok else ''}{what}")
            if not ok:
                failures.append(label)
    finally:
        # Best effort: Windows may still hold files of the installed app.
        shutil.rmtree(t, ignore_errors=True)
    if failures:
        log(f"FAILED with: {', '.join(failures)} profile path")
        return 1
    log(f"{artifact.name}: installs and charts a song on a clean profile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
