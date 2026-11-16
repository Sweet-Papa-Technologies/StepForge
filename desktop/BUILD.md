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
stamps `ENGINE_VERSION`, and smoke-tests that the engine sees its model.
`tauri.release.conf.json` bundles that folder as an app resource.

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
  builds macOS (Apple silicon `.dmg`), Windows (`-setup.exe`) and Linux
  (`.AppImage`, `.deb`) in parallel and publishes them, with `SHA256SUMS.txt`, to
  the Releases page. Tags with a hyphen (`v0.2.0-beta.1`) become pre-releases.

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

## The workflow the GUI implements

Each track has one obvious next step, derived from what actually exists on disk, and
the card's primary button is always that step:

```
import/convert → Analyze → Generate → Results → Export
   (ffmpeg)      (DSP)     (adapters)  (review)  (song folder)
```

| Track state | Meaning | Next action |
|---|---|---|
| `Needs .ogg` | source isn't .ogg, engine can't open it | **Convert to .ogg** |
| `Draft` | .ogg present, no analysis yet | **Analyze** |
| `Analyzed` | `analysis.json` exists | **Generate** |
| `Charted` | a song folder / qa.json exists | **Results** |

Notes on practicality:
- Analysis is **not** a manual prerequisite for Generate — `stepforge`/`saberforge`
  call `analyze_track` themselves. Analyze is there to *see and verify* the grid
  before spending model calls.
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

