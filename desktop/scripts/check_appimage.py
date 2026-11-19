#!/usr/bin/env python3
"""check_appimage.py — prove a built AppImage works before it is published.

    xvfb-run -a python3 desktop/scripts/check_appimage.py dist/StepForge-<v>-linux-x86_64.AppImage

1. Library boundary. The AppImage is built on Ubuntu 22.04 and linuxdeploy copies
   in everything the app links against. Some of those families belong to the
   host and break when an old copy is loaded on a newer distribution: an old
   libwayland-client in the bundle made Mesa abort with "Could not create
   default EGL display: EGL_BAD_PARAMETER" on Arch (CachyOS reported a black
   window). The build excludes them (LINUXDEPLOY_EXCLUDED_LIBRARIES in
   release.yml); this fails the release if one comes back.

2. End to end, through the AppImage's own launcher. AppRun exports variables for
   the app (PYTHONHOME=$APPDIR/usr/, LD_LIBRARY_PATH, PATH, ...) that also reach
   every program the app starts. In beta 5 that PYTHONHOME stopped the bundled
   Python before it ran a line, on every distribution. stage_runtime.py's smoke
   test runs the interpreter directly and could not see it; this charts a song
   with `AppRun --headless generate`, the same path the GUI takes.

3. Audio. The AppImage carries its own GStreamer (WebKitGTK links it), and a
   GStreamer without plugins cannot create a single element. Beta 6 shipped the
   library and no plugins: on the build distribution the system's plugins were
   found, so nothing looked wrong, and everywhere else the web view's process
   died as soon as the Results screen created its audio player (the window
   stopped taking clicks and went black). The plugins must be inside the
   AppImage, and the launcher must point GStreamer at them.

4. The app's own self-test through AppRun (`--headless selftest`): every engine
   dependency imports, both models load, a clip is charted, and the GStreamer
   elements the player needs can be created.

This runs on the build machine, where (3) can only be checked by looking at the
files. smoke_linux.sh runs the finished AppImage on other distributions.

Headless mode still initialises GTK, hence xvfb-run.
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_runtime import E2E_AUDIO  # noqa: E402

# Keep in sync with LINUXDEPLOY_EXCLUDED_LIBRARIES in .github/workflows/release.yml.
EXCLUDED = ("libsystemd.so*", "libudev.so*", "libdbus-1.so*", "libwayland-*.so*",
            "libepoxy.so*", "libnghttp2.so*")


#: GStreamer plugin files the Results screen's player needs (gst-plugins-base
#: and -good): sources and sinks for WebKit, decoding, Ogg Vorbis, conversion,
#: and something to play through.
GST_PLUGINS = ("libgstcoreelements.so", "libgstapp.so", "libgstplayback.so", "libgstgio.so",
               "libgsttypefindfunctions.so", "libgstogg.so", "libgstvorbis.so",
               "libgstaudioconvert.so", "libgstaudioresample.so", "libgstautodetect.so")
GST_SINKS = ("libgstpulseaudio.so", "libgstalsa.so")


def check_gstreamer(root: Path) -> None:
    if not any(root.glob("usr/lib/libgstreamer-1.0.so*")):
        print("[check-appimage] no GStreamer library bundled: the system's is used whole")
        return
    plugins = {p.name for p in root.rglob("gstreamer-1.0/*.so")}
    missing = [n for n in GST_PLUGINS if n not in plugins]
    if not any(n in plugins for n in GST_SINKS):
        missing.append(" or ".join(GST_SINKS))
    if missing:
        raise SystemExit("the AppImage bundles GStreamer without the plugins the audio player "
                         "needs (bundle.linux.appimage.bundleMediaFramework in tauri.conf.json, "
                         "and gstreamer1.0-plugins-base/-good on the build machine):\n  "
                         + "\n  ".join(missing))
    hooks = "".join(h.read_text(errors="replace") for h in (root / "apprun-hooks").glob("*"))
    if "GST_PLUGIN_SYSTEM_PATH" not in hooks:
        raise SystemExit("the AppImage's launcher does not point GStreamer at the bundled plugins "
                         "(no GST_PLUGIN_SYSTEM_PATH in apprun-hooks/)")
    print(f"[check-appimage] GStreamer plugins bundled ({len(plugins)}), launcher points at them")


def selftest_through_apprun(root: Path, tmp: Path) -> None:
    home = tmp / "selftest-home"
    home.mkdir()
    env = clean_env(BEATFORGE_WORKSPACE=str(home / "workspace"), HOME=str(home),
                    XDG_DATA_HOME=str(home / ".local/share"),
                    XDG_CONFIG_HOME=str(home / ".config"), XDG_CACHE_HOME=str(home / ".cache"))
    r = subprocess.run([str(root / "AppRun"), "--headless", "selftest"], env=env, cwd=tmp,
                       capture_output=True, text=True, timeout=1200)
    result = {}
    for line in r.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("op") == "selftest":
            result = obj
    if r.returncode != 0 or not result.get("ok"):
        errors = (result.get("report") or {}).get("errors") or [result.get("error")]
        raise SystemExit(f"the self-test through AppRun failed (exit {r.returncode}): {errors}\n"
                         f"media: {result.get('media')}\nstderr:\n{r.stderr[-4000:]}")
    media = result.get("media") or {}
    print(f"[check-appimage] self-test through AppRun passed "
          f"(chart {result['report']['chart']['charts']}, audio elements "
          f"{'ok' if media.get('checked') else 'not checked'})")


def check_libraries(root: Path) -> None:
    bad = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                 if p.is_file() and any(fnmatch.fnmatch(p.name, g) for g in EXCLUDED))
    if bad:
        raise SystemExit("the AppImage bundles host-owned libraries that break on newer "
                         "distributions:\n  " + "\n  ".join(bad))
    print(f"[check-appimage] no host-owned libraries bundled ({len(EXCLUDED)} families checked)")


def clean_env(**extra: str) -> dict[str, str]:
    """The CI runner's environment, minus anything that could prop up a broken bundle."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PYTHON") and k not in ("LD_LIBRARY_PATH", "APPDIR", "APPIMAGE")}
    env.update(extra)
    return env


def chart_through_apprun(root: Path, tmp: Path) -> None:
    runtime = root / "usr/lib/StepForge/runtime"
    py, tools = runtime / "python/bin/python3", runtime / "engine/tools"
    if not py.is_file() or not tools.is_dir():
        raise SystemExit(f"bundled runtime not found under {runtime}")

    ws, home = tmp / "workspace", tmp / "home"
    (ws / "public/assets/tracks").mkdir(parents=True)
    home.mkdir()

    # The input song, made with the bundled interpreter called directly (not
    # through AppRun): this step only prepares the test, it is not under test.
    wav = tmp / "smoke.wav"
    prep = clean_env(PYTHONPATH=str(tools), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    for args in (["-c", E2E_AUDIO, str(wav)],
                 ["-m", "beatforge", "to-ogg", "--in", str(wav),
                  "--out", str(ws / "public/assets/tracks/smoke.ogg")]):
        subprocess.run([str(py), *args], env=prep, check=True, capture_output=True, text=True)

    env = clean_env(BEATFORGE_WORKSPACE=str(ws), HOME=str(home),
                    XDG_DATA_HOME=str(home / ".local/share"),
                    XDG_CONFIG_HOME=str(home / ".config"))
    r = subprocess.run([str(root / "AppRun"), "--headless", "generate", "--track", "smoke",
                        "--difficulties", "easy,hard"],
                       env=env, cwd=tmp, capture_output=True, text=True, timeout=900)
    events = []
    for line in r.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    ok = any(e.get("op") == "generate" and e.get("ok") for e in events)
    song = ws / "build/stepmania/smoke"
    missing = [n for n in ("smoke.ssc", "smoke.sm") if not (song / n).is_file()]
    if r.returncode != 0 or not ok or missing:
        raise SystemExit(f"charting through AppRun failed (exit {r.returncode}, missing {missing})\n"
                         f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-4000:]}")
    charts = (song / "smoke.ssc").read_text(encoding="utf-8").count("#NOTEDATA:")
    print(f"[check-appimage] charted a song end to end through AppRun ({charts} charts)")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: xvfb-run -a python3 check_appimage.py <StepForge.AppImage>")
    appimage = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="stepforge-appimage-") as t:
        tmp = Path(t)
        img = tmp / "StepForge.AppImage"
        shutil.copy2(appimage, img)
        img.chmod(0o755)
        # --appimage-extract needs no FUSE; it unpacks to ./squashfs-root.
        subprocess.run([str(img), "--appimage-extract"], cwd=tmp, check=True,
                       stdout=subprocess.DEVNULL)
        root = tmp / "squashfs-root"
        check_libraries(root)
        check_gstreamer(root)
        selftest_through_apprun(root, tmp)
        chart_through_apprun(root, tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
