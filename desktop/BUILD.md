# StepForge desktop — build and release

The StepForge app is a **Tauri v2 + Vue 3 + TypeScript** shell around the
charting engine in `tools/beatforge` (Python, run as `python -m beatforge …`).

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Node | 22 | `npm` |
| Rust | stable | `rustup` |
| Python | 3.10+ | for the engine during development (`pip install -r requirements.txt`) |
| ffmpeg | optional | only to import formats other than WAV/MP3/FLAC/OGG |

## Develop

```bash
python3 -m pip install -r requirements.txt   # repo root: the engine's packages
cd desktop
npm ci
npm run tauri dev       # hot-reloading app (Vite on :1420 + Rust shell)
npm run build           # icon check + vue-tsc type check + production bundle
cd src-tauri && cargo test
```

In development the app finds the engine by walking up from its working
directory to the repo (`tools/beatforge/__main__.py`), and uses the `python3` on
your PATH (or the one set in Settings).

## Release builds

Release builds carry their own runtime so users install nothing else:

```bash
python3 desktop/scripts/stage_runtime.py --version 0.1.0   # -> desktop/src-tauri/runtime/
cd desktop
npx tauri build --config src-tauri/tauri.release.conf.json
```

`stage_runtime.py` downloads a self-contained CPython
([python-build-standalone](https://github.com/astral-sh/python-build-standalone)),
installs `requirements.txt` into it, copies the engine and the model next to it,
stamps `ENGINE_VERSION`, and smoke-tests it: the model charts a synthetic click
track, then a synthetic song goes through the real command line end to end
(to-ogg, analysis, charting, .ssc/.sm, the preview PNG) in a scratch workspace.

The runtime is pruned to what the engine loads: no Tcl/Tk, no C headers or type
stubs, no onnxruntime conversion tooling, and only `requirements.txt` (matplotlib
and google-auth live in `requirements-optional.txt` and are not bundled). Fewer
files is what makes the Windows installer fast: Defender scans every one. `tauri.release.conf.json` bundles that folder as an app
resource.

On macOS the stager installs wheels for the app's declared minimum
(`bundle.macOS.minimumSystemVersion` in `tauri.conf.json`, currently 12.3), not
for the build machine's macOS, then checks every bundled binary's load-command
minimum and fails the build if one needs a newer system. (The macOS 12 floor is
set by scipy's Apple silicon wheels; onnxruntime 1.19, the newest with macOS 12
wheels, charts byte-for-byte the same as current releases.) To stage an Intel
runtime on an Apple silicon Mac, run the stager under Rosetta:
`arch -x86_64 /usr/bin/python3 desktop/scripts/stage_runtime.py`.

On first launch the app copies `runtime/engine/` into a writable per-user
workspace (the engine writes beside its code) and runs it with
`runtime/python/`:

| | Workspace |
|---|---|
| macOS | `~/Library/Application Support/StepForge/workspace` |
| Windows | `%APPDATA%\StepForge\workspace` |
| Linux | `$XDG_DATA_HOME/StepForge/workspace` (`~/.local/share/…`) |

When an update ships a different `ENGINE_VERSION`, the app refreshes the
workspace's engine code and model at startup; the user's songs and charts
(`assets/`, `public/`, `build/`) are left alone. With the bundled Python the
app sets `PYTHONPYCACHEPREFIX` to the workspace, so nothing is ever written
inside the (signed) app bundle.

## CI/CD

- `.github/workflows/ci.yml` runs on every pull request and on `main`: engine
  tests, frontend type check and build, Rust tests. `main` is protected; changes
  land through pull requests with these checks green.
- `.github/workflows/release.yml` runs when a `v*` tag on `main` is pushed. It
  builds macOS (Apple silicon and Intel `.dmg`s, on `macos-15` and
  `macos-15-intel`), Windows (`-setup.exe`) and Linux (`.AppImage`, `.deb`) in
  parallel and attaches them, with `SHA256SUMS.txt`, to a **draft** release.
  Tags with a hyphen (`v0.2.0-beta.1`) are marked as pre-releases.
- `.github/workflows/linux-appimage.yml` builds the AppImage on pull requests
  that touch the app or its packaging, checks it, smoke-tests it in clean Arch
  and Ubuntu 24.04 containers, and keeps it as an artifact for a week.
- `.github/workflows/installer-smoke.yml` does the same for the Windows
  installer and the macOS disk image (unsigned), so a packaging mistake is found
  on the pull request. Its artifacts are for testing by hand on machines CI does
  not have, such as a clean Windows 10 or 11 PC.

### From tag to public release

Three stages, no long-lived branches:

1. **Pull request.** Unsigned installers for all three platforms are built and
   smoke-tested on clean machines (`installer-smoke.yml`,
   `linux-appimage.yml`). Nothing is published. The artifacts stay on the
   workflow run for a week; this repository is public, so anyone signed in to
   GitHub who opens that run can download them, but nothing links to them.
2. **Draft release.** Pushing a `v*` tag on `main` builds, signs, notarizes and
   smoke-tests the real installers and attaches them to a draft. A draft is
   visible only to people who can push here. Install from it and try it,
   especially on machines CI does not have (a clean Windows 10 or 11 PC, a
   Linux desktop with a real GPU). To let someone without push access try a
   candidate, send them the file.
3. **Published.** `gh release edit v0.2.0-beta.1 --draft=false`, or "Publish
   release" on the draft's page. The files users get are the ones that were
   tested. To drop a candidate instead:
   `gh release delete v0.2.0-beta.1 --cleanup-tag`, fix it, and tag the next
   number.

Signed Windows builds exist only at stage 2: the `release` environment that
code signing trusts accepts `v*` tags and nothing else.

### Release smoke tests

Nothing is published until every installer has been installed on a machine that
did not build it and has charted a song there. The `smoke-*` jobs in
`release.yml` do that, and `publish` needs all of them:

| Installer | Where it is tested | Script |
|---|---|---|
| macOS `.dmg` (Apple silicon) | fresh `macos-15` and `macos-14` runners | `scripts/smoke_test.py` |
| macOS `.dmg` (Intel) | fresh `macos-15-intel` runner | `scripts/smoke_test.py` |
| Windows `-setup.exe` | fresh `windows-2022` and `windows-2025` runners | `scripts/smoke_test.py` |
| Linux `.AppImage` | clean `archlinux` and `ubuntu:24.04` containers | `scripts/smoke_linux.sh` |
| Linux `.deb` | clean `ubuntu:24.04` and `ubuntu:22.04` containers | `scripts/smoke_linux.sh` |

Each one runs the app's own self-test, `StepForge --headless selftest`, with a
new, empty profile — once at a plain path and once at one with spaces and
non-ASCII characters. The self-test (`python -m beatforge.selftest` underneath)
imports every engine dependency, onnxruntime among them, loads both models,
charts a 20-second synthesized clip through the real command line and reads the
`.ssc` back. On Linux it also checks that the GStreamer elements the Results
screen plays audio with can be created. It exits 0 only if all of that passed.

Three things the build checks before the smoke tests ever run
(`scripts/stage_runtime.py`):

- **Every native library the bundled runtime loads is in the bundle or part of
  the OS** (`scripts/native_audit.py`, which reads PE, ELF and Mach-O dependency
  tables). Beta 6 for Windows shipped onnxruntime without the Visual C++ runtime
  it imports (`msvcp140.dll`, `msvcp140_1.dll`); it only failed on PCs without
  the VC++ redistributable, which no build or test machine was.
- **On Windows the missing VC++ runtime DLLs are copied next to `python.exe`**
  (`bundle_vc_runtime`), from Visual Studio's redist folder on the build machine.
- **The engine's self-test passes with the staged interpreter**, and on Windows
  with `--self-contained`: it fails if any VC++ runtime DLL was loaded from
  System32 rather than from the app. CI runners have the redistributable
  installed, so "it imported" proves nothing there; where it was loaded from does.

What CI cannot cover, and still needs a person before a release that changes
packaging: a clean desktop Windows 10 and 11, real GPU sessions on Linux
(Wayland and X11, NVIDIA and Mesa), and audible playback.

The AppImage is built on Ubuntu 22.04. Three rules keep it working on newer
distributions, and `scripts/check_appimage.py` enforces them before a release
is published:

- GStreamer's plugins are bundled with it
  (`bundle.linux.appimage.bundleMediaFramework`). WebKitGTK links GStreamer, so
  the library is always in the AppImage; without its plugins the web view's
  process died as soon as the Results screen created its audio player, and the
  window stopped taking clicks and went black. On the build distribution the
  system's plugins are found instead, so this only shows on other distributions.
  The system's GStreamer cannot be used in its place: it needs a newer GLib than
  the one in the AppImage.

- Host-owned libraries (systemd, udev, D-Bus, Wayland, epoxy, nghttp2) are left
  out of the bundle (`LINUXDEPLOY_EXCLUDED_LIBRARIES`). An old bundled
  libwayland made Mesa abort with `EGL_BAD_PARAMETER` on Arch.
- The AppImage launcher exports `PYTHONHOME`, `LD_LIBRARY_PATH` and `PATH` into
  its own mount for the app. `engine::command` takes those back out for every
  program the app starts, and the bundled Python never inherits `PYTHONHOME`.
  The check charts a song through `AppRun --headless` to prove it.

On Linux the app also starts WebKitGTK with `WEBKIT_DISABLE_DMABUF_RENDERER=1`
(unless the user sets it), which fixes blank or black windows on NVIDIA and
similar setups.

```bash
git tag -a v0.1.0-beta.2 -m "StepForge 0.1.0 beta 2"
git push origin v0.1.0-beta.2
```

### Signing

| | How | Configured by |
|---|---|---|
| macOS | Developer ID certificate imported into a temporary keychain; `stage_runtime.py` signs every Mach-O file of the bundled Python (hardened runtime, `scripts/python.entitlements`); Tauri signs and notarizes the app with an App Store Connect API key; the workflow signs, notarizes and staples the `.dmg` | secrets `APPLE_CERT_P12`, `APPLE_CERT_PASSWORD`, `APPLE_TEAM_ID`, `ASC_KEY_P8`, `ASC_KEY_ID`, `ASC_ISSUER_ID` |
| Windows | Azure Trusted Signing through GitHub OIDC (no stored secret): `azure/login`, then Tauri's `signCommand` (`src-tauri/tauri.windows-sign.conf.json`) calls `scripts/sign-windows.ps1` for the app and the installer | variables `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`; the `release` environment |
| Linux | unsigned | |

`scripts/setup-ci.sh` (repo root) sets the secrets, variables, the `release`
environment and the branch protection. A missing credential makes that
platform's build unsigned with a warning rather than failing the release.

## When the engine does not start

The "Engine not ready" banner and Settings show the real cause, from
`src-tauri/src/diagnostics.rs`: the exception a module raised when imported, or
how Python ended if it never got that far (an exit code, a Windows crash status
such as `0xC0000135` for a missing DLL, a Unix signal). "The engine is older
than the model" is reported only when the engine ran and its argument parser
rejected `neural-status`; a failure to start is never described that way.

**Copy diagnostics** (the banner and Settings) runs the self-test and copies a
report: app version, OS, CPU and its instruction sets, the self-test result and
the last engine error. Folder names from the user's machine, the user name and
the names of songs in the library are replaced before anything is copied
(`Redaction`), and no audio is ever read. The same text is printed by
`StepForge --headless diagnostics`; `--headless status` prints what the GUI's
status checks return.

After an update the app replaces the workspace's engine code at startup.
Engine calls wait for that to finish (`install::wait_for_startup_refresh`): a
status check that started Python while `tools/` was being rewritten used to find
half an engine.

## The workflow the GUI implements

Each track has one obvious next step, derived from what actually exists on disk, and
the card's primary button is always that step:

```
import/convert → Generate → Results → Export
   (ffmpeg)      (analysis runs here when needed)   (review)  (song folder)
```

| Track state | Meaning | Next action |
|---|---|---|
| `Needs .ogg` | source isn't .ogg, engine can't open it | **Convert to .ogg** |
| `Ready` | .ogg present, no analysis yet | **Generate** |
| `Analyzed` | `analysis.json` exists | **Generate** |
| `Charted` | a song folder / qa.json exists | **Results** |

Notes on practicality:
- There is no separate Analyze step. `stepforge`/`saberforge` call
  `analyze_track` themselves, and its cache is keyed by the audio's SHA-256 (plus
  the analysis version), so a song is analyzed on its first run and again only
  if its audio changed. The Analyze screen is still there to *see* the tempo, grid
  and onsets: "View analysis" in the Library inspector, on the Generate screen,
  and in a card's right-click menu.
- **Bulk generation:** tick cards (Shift-click for a range, Cmd/Ctrl+A for all,
  Escape to clear), then *Generate N selected*. The run is sequential, survives
  navigation, can file the songs under a pack and install them to the game.
- The Generate screen lists the blockers preventing a run (no difficulty picked,
  rights not acknowledged, engine not ready) instead of a dead button.
- Every run streams the engine's real stdout/stderr; a failure shows the tail of
  the error rather than silently doing nothing.
- The rights checkbox is tied to SABERFORGE's `--i-have-rights` flag, so with Beat
  Saber hidden it does not appear; StepMania/native have no such CLI flag.
- Deterministic mode is genuinely offline — no model call, no cost, no critic.

**Long-running jobs survive navigation.** The child process is detached and job
state lives in the Pinia store, not in a component, so leaving Analyze or Generate
never interrupts a run. Views re-attach on mount via `engine.jobFor(kind, trackId)`,
and the sidebar lists everything currently running (click an entry to jump back to
it). Because nothing is disrupted, there is deliberately no "are you sure you want
to leave" prompt.

**Actions for one track are serialised.** `engine.busyFor(trackId)` gates the
buttons: while an analysis runs, *Generate charts* is disabled (and vice versa),
so a chart run can't race the analysis it depends on.

**Chart player.** Results has a *Player* / *Timeline* toggle. The player
(`NotePlayer.vue`) is a small ITGmania-style 4-lane notefield that auto-plays the
song with the chart scrolling to receptors — a quick "does this actually feel
right" check. Notes come from `get_chart_notes`, which parses the emitted
`.ssc`/`.sm` (`simfile.rs`), so it shows exactly what the game would play rather
than an approximation. It previews; it does not score. STEPFORGE only — the
native target emits beatmap JSON, not a simfile, so the toggle is disabled there.
Speed (px/sec) is adjustable.

**Interactions.** Double-clicking a library card opens it. Right-click menus
(`ContextMenu.vue`) exist on library cards (open / analyze now / generate /
results / convert / reveal / copy id) and in the Results main area (switch view /
export / reveal / regenerate). Choosing *Analyze* from the Library navigates with
`?run=1`, which starts the run on arrival instead of just landing on the screen.

**Desktop notifications** fire on job start and finish (success or failure) via
`tauri-plugin-notification`; permission is requested once at startup and a denial
just disables them.

**Assist ticks are a toggle, on by default.** The engine's `.assist.ogg` is the
song *already mixed* with tick transients (ffmpeg `amix` of track + clicks), not a
separate tick layer — so the player swaps its source rather than stacking a second
`<audio>`. One clock, no drift. The swap preserves position and play state, so
toggling mid-song is seamless. In *Timeline* mode the standalone assist transport
is shown instead, so only one source ever plays.

**Current song is global.** The sidebar shows the selected song (with a status
dot) and opens a filterable switcher; picking one keeps you on the same kind of
screen. Everything else in the app acts on that selection.

**Song manager** (Library inspector, once a song has StepMania output):
- **Packs** — assign a song to a pack; membership lives in `build/packs.json`
  (app-side metadata; the engine has no pack concept).
- **Artwork** — attach a banner (image) or background (image *or* video). Files are
  copied in as `<base>-banner.<ext>` / `<base>-bg.<ext>`, the names STEPFORGE's
  serializer already looks for, **and** the emitted `.ssc`/`.sm` gets its
  `#BANNER`/`#BACKGROUND` tag patched so existing charts pick it up without paying
  for a regeneration.
- **AI artwork** — with a Gemini/Vertex backend selected, generate from a prompt via
  the `assetforge` CLI (Imagen for stills, Veo for video); it streams like any other
  job and attaches on success.
- **Install to game** — copies the emitted folder to `<Songs>/<Pack>/<Song>/`.

**Game `Songs` directory** is resolved cross-platform (the `BEATFORGE_ITG_SONGS`
environment variable overrides it):

| OS | Default |
|---|---|
| macOS | `~/Library/Application Support/ITGmania/Songs` |
| Windows | `%APPDATA%\ITGmania\Songs` |
| Linux | `$XDG_DATA_HOME/Songs`, else `~/.itgmania/Songs` |

Settings also lists any StepMania 5/5.1 installs it detects, and the path is
overridable there.

**Cost is broken down per song** — spend (LLM vs compute), model calls, whether the
analysis was cached, and what it bought: per-difficulty note counts, meters, critic
scores and gate pass/fail, with a jump straight to that song's results.

## Performance — things not to regress

The UI was badly unresponsive; these are the causes and the fixes. Re-introducing
any of them will bring the sluggishness straight back.

1. **Tauri commands must not be synchronous.** A `#[tauri::command]` on a plain
   `fn` runs **on the main thread** and blocks the webview. `engine_status` alone
   spawns three subprocesses (~0.7s, one importing numpy/scipy/soundfile) and ran
   on every library refresh — a visible freeze each time. Every command is now
   `#[tauri::command(async)]`, which dispatches to Tauri's threadpool.
2. **`engine_status` is memoized** (20s TTL) because probing is expensive; pass
   `force: true` (Settings → *Re-check*, and after saving settings) to re-probe.
3. **Never deep-watch the analysis payload.** It holds hundreds of onsets plus
   `energy_curve`/`beats`. `AnalysisCanvas` previously had
   `watch([analysis, playhead], …, { deep: true })`, so every playhead tick
   walked the whole structure and repainted. Analysis is held in `shallowRef`
   and watched shallowly; `peaks` is a `shallowRef` so the Float32Array is never
   proxied.
4. **The canvas caches its static layers.** Waveform + bar lines + onsets render
   once into an offscreen canvas; each frame only blits it and draws the
   playhead, coalesced through `requestAnimationFrame`.
5. **Log lines are batched.** Engine output is buffered outside reactivity and
   flushed every 80ms (cap 400 retained, 200 rendered), so a chatty run costs a
   few re-renders per second instead of one per line.
6. **The icon font is subset** — 3.4 MB → ~84 KB (`python3 scripts/subset-icons.py`,
   re-run after adding icons). Total bundle went ~4.4 MB → ~1.1 MB.
7. Selecting library cards debounces the audio decode (180ms) so clicking through
   the grid doesn't queue a decode per card.

> Also note `npm run tauri dev` runs an **unoptimized debug** Rust binary; the
> release build is substantially faster. Compare perf against `npx tauri build`.

## The engine contract the GUI is built against

These are the exact rules `tools/beatforge` enforces. The UI is wired to them;
changing one on either side breaks the app silently, so they are worth knowing.

**Track addressing**
- A track is addressed by `--track <id>`. `analyze._audio_for` resolves it as
  `TRACK_CATALOGUE.get(id, id)` → `<base>` and then looks for **`<base>.ogg` only**,
  under `public/assets/tracks/` then `assets/tracks/`.
- Consequence: **an imported .mp3/.wav is unusable until transcoded.** Import in the
  app runs the same ffmpeg recipe as `batch_stepforge.transcode`
  (`-vn -map 0:a:0 -c:a libvorbis -q:a 6`, `-vn` so embedded album art doesn't become
  a theora stream soundfile can't decode) and writes `<base>.ogg` to *both* roots.
- Unregistered ids work fine (base == id). Registration only affects the
  **title/artist** baked into exports — the Library's metadata editor writes
  `TRACK_META['<id>'] = ('Title', 'Artist')` into `config.py`, the same convention as
  the existing Founders-Mix block.

**Difficulties** (argparse rejects anything else)
| Target | Command | Difficulties |
|---|---|---|
| Native | `chart --difficulty` (one per run) | `casual, standard, overdrive` |
| STEPFORGE | `stepforge --difficulties a,b` | `beginner, easy, medium, hard, challenge` (default `easy,medium,hard`) |
| SABERFORGE *(hidden)* | `saberforge --difficulties a,b` | `easy, normal, hard, expert, expertplus` |

**Beat Saber is hidden.** SABERFORGE works in the engine but the app does not
support it yet, so it is gated off end-to-end by `ENABLED_TARGETS` in
`commands.rs`: it is never offered for generation, an emitted Beat Saber song
folder is never reported as output, and a direct request is rejected. To
re-enable, add `"saberforge"` back to `ENABLED_TARGETS` and to `TARGETS` in
`Generate.vue` — the rights-acknowledgement gate (`--i-have-rights`) is already
wired for it. `saberforge_is_hidden_end_to_end` in the test suite pins this.

**Artifacts**
- DSP truth → `build/analysis/<base>.analysis.json`, keys **`bpm`, `offset`,
  `duration_s`, `onsets[]` (objects with `time`/`nearest_beat`/`strength`/`sustain`),
  `sustain_available` (a *count*), `sections[]` (bar-indexed, with `role_guess`)**.
- Native `chart` → `build/analysis/<base>.<diff>.qa.json`. The **critic and
  `design_notes` live inside this file** — there are no separate `.critic.json` /
  `.design.json` for the native path. Previews: `<base>.<diff>.preview.{png,ogg}`.
- STEPFORGE / SABERFORGE → a song folder `build/{stepmania,saberforge}/<base>/`
  containing `report.json` (per-difficulty `metrics`, `critic`, `previews`, note
  counts), `<diff>.design.json`, `<diff>.critic.json` (Gemini path only), and
  previews `<diff>.{notefield,grid}.png` + `<diff>.assist.ogg`.
- The app reads preview paths **out of the `previews` dict** rather than guessing
  filenames, because each adapter names them differently.
- Cost → `build/cost/cost-report.{json,md}` via `cost-report`.

**Model selection.** The designer/critic model is chosen in Settings (and on the
Generate screen) and exported to the engine as env vars — no engine edit needed:

| Setting | Env var | Values |
|---|---|---|
| Backend | `BEATFORGE_LLM_BACKEND` | `gemini` (Vertex), `google-genai` (AI Studio), `openai` |
| Model | `BEATFORGE_GEMINI_MODEL` | `gemini-3.8-flash` *(default)*, `gemini-3.5-flash`, `gemini-3.5-flash-lite` |
| Thinking | `BEATFORGE_THINKING_LEVEL` | `minimal` \| `low` \| `medium` \| `high` *(default)* |

The model applies to **both** Gemini paths. The trade-offs shown in the UI come
from `docs/model-comparison.md`: 3.6 Flash is the default because it beat 3.5
Flash on flow, density correlation, wall-clock *and* cost; 3.5 Flash Lite is ~86%
cheaper but produced ~6x the jumps, which hurts danceability.

**Critic scores are 0–10**, not 0–100; `CRITIC_SHIP_THRESHOLD = 7.0`. Deterministic
runs skip the critic entirely, so a null score is normal, not an error.

**Simfile note counts.** `report.json`'s `notes` counts *placements*; a jump is one
placement on two panels. Arrows == `notes + jumps`. `matches_real_stepforge_output`
asserts the parser against a real emission when one is present.

`cargo test -p beatforge-desktop` pins the parts of this contract that are easy to
regress (slug rules, config.py parsing, status derivation, difficulty sets).

