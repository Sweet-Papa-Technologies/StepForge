#!/usr/bin/env bash
# smoke_linux.sh — install a Linux release artifact on THIS machine and prove
# it can chart a song.
#
#   docker run --rm -v "$PWD:/w" -w /w archlinux:latest \
#       bash desktop/scripts/smoke_linux.sh dist/StepForge-<v>-linux-x86_64.AppImage
#   docker run --rm -v "$PWD:/w" -w /w ubuntu:24.04 \
#       bash desktop/scripts/smoke_linux.sh dist/StepForge-<v>-linux-amd64.deb
#
# Meant for a clean distribution container (the release workflow runs it in
# Arch and Ubuntu ones): no Python, no compilers, no StepForge files. It installs
# only what a desktop install of that distribution already has (GTK, a display
# server to run against), installs the artifact, and runs the app's self-test:
#
#   every engine dependency imports (onnxruntime among them), both models load,
#   a 20-second clip is charted and a valid .ssc is written, and the GStreamer
#   elements the Results screen needs to play audio exist.
#
# The last one is why this runs on other distributions than the build machine:
# the AppImage is built on Ubuntu 22.04, where a missing GStreamer plugin is
# invisible because the system's own are found. On Arch the app's web view
# crashed as soon as Results opened (the window froze, then went black).
#
# Bash on purpose. The test machine must not need Python for the app to work,
# so the test does not bring one.
set -euo pipefail

ARTIFACT="${1:?usage: smoke_linux.sh <StepForge .AppImage or .deb>}"
ARTIFACT="$(readlink -f "$ARTIFACT")"
say() { printf '[smoke] %s\n' "$*"; }

if [ "$(id -u)" != 0 ]; then
    echo "run as root in a throwaway container: this installs packages" >&2
    exit 2
fi

# ---- what a desktop install already has ------------------------------------
if command -v pacman >/dev/null; then
    DISTRO="$(. /etc/os-release && echo "$PRETTY_NAME")"
    pacman -Syu --noconfirm --needed xorg-server-xvfb xorg-xauth gtk3 mesa >/dev/null
elif command -v apt-get >/dev/null; then
    DISTRO="$(. /etc/os-release && echo "$PRETTY_NAME")"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends xvfb xauth >/dev/null
    case "$ARTIFACT" in
        # The AppImage carries its own WebKitGTK; these are the host-owned
        # libraries it leaves to the system (see release.yml) plus GTK's needs.
        *.AppImage) apt-get install -y -qq --no-install-recommends libgtk-3-0 libegl1 libgl1 \
                        libwayland-client0 libwayland-cursor0 libwayland-egl1 libwayland-server0 libepoxy0 \
                        libnghttp2-14 libdbus-1-3 >/dev/null ;;
    esac
else
    echo "unsupported distribution (no pacman or apt-get)" >&2
    exit 2
fi
say "distribution: $DISTRO"
for tool in python python3 gcc cc; do
    command -v "$tool" >/dev/null && say "note: this machine has '$tool' ($(command -v "$tool")); the app must not use it"
done

# ---- install ---------------------------------------------------------------
WORK="$(mktemp -d /tmp/stepforge-smoke-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
case "$ARTIFACT" in
    *.AppImage)
        cp "$ARTIFACT" "$WORK/StepForge.AppImage"
        chmod +x "$WORK/StepForge.AppImage"
        # Containers have no FUSE; --appimage-extract runs the same AppRun.
        (cd "$WORK" && ./StepForge.AppImage --appimage-extract >/dev/null)
        APP="$WORK/squashfs-root/AppRun"
        ;;
    *.deb)
        apt-get install -y -qq --no-install-recommends "$ARTIFACT" >/dev/null
        APP="$(command -v stepforge-desktop)"
        ;;
    *) echo "do not know how to install $ARTIFACT" >&2; exit 2 ;;
esac
say "program: $APP"

# ---- run the self-test with new, empty profiles -----------------------------
# Twice: a plain profile path, then one with spaces and non-ASCII characters.
# PYTHON* and BEATFORGE_* are unset so nothing props up a broken bundle.
FAILED=""
run_profile() {
    local label="$1" home="$2" out
    mkdir -p "$home"
    out="$WORK/$label.json"
    if env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
            LANG=C.UTF-8 HOME="$home" XDG_DATA_HOME="$home/.local/share" \
            XDG_CONFIG_HOME="$home/.config" XDG_CACHE_HOME="$home/.cache" \
            BEATFORGE_WORKSPACE="$home/workspace" \
            xvfb-run -a "$APP" --headless selftest > "$out"; then
        # A first launch provisions the engine into the new profile; if it is
        # not there, some other engine was tested.
        if [ -f "$home/workspace/tools/beatforge/__main__.py" ]; then
            say "$label profile path: ok"
        else
            say "$label profile path: FAILED (the app did not set up its engine in $home/workspace)"
            FAILED="$FAILED $label"
        fi
    else
        say "$label profile path: FAILED (exit $?)"
        FAILED="$FAILED $label"
    fi
    # The result line (one JSON object), for the log.
    grep '"op":"selftest"' "$out" | tail -1 | cut -c1-1500 || cat "$out"
}
run_profile plain "$WORK/home/tester"
run_profile non-ascii "$WORK/home/Zoë Müller 用户"

if [ -n "$FAILED" ]; then
    say "FAILED on $DISTRO with:$FAILED profile path"
    exit 1
fi
say "$(basename "$ARTIFACT") installs and charts a song on $DISTRO"
