# Contributing to StepForge

Thanks for helping! Bug reports, chart feedback and pull requests are all welcome.

## Reporting a bug or a bad chart

Open an [issue](https://github.com/Sweet-Papa-Technologies/StepForge/issues). For a
chart that plays badly, say which song (if you can share it), which difficulty,
and what felt wrong: off the beat, awkward to stand on, too dense, too sparse.
Screenshots or a short clip help a lot.

## Pull requests

`main` is protected: every change goes through a pull request, and the CI checks
(engine tests, desktop type check and build, Rust tests) must pass before it can
merge. Nobody pushes to `main` directly.

1. Fork, or create a branch.
2. Make the change, with tests where it changes behaviour.
3. Run the checks locally:

   ```bash
   python3 -m pip install -r requirements.txt -r requirements-optional.txt pytest "setuptools<81"
   (cd tools && python3 -m pytest beatforge/tests -q)
   cd desktop && npm ci && npm run build && (cd src-tauri && cargo test)
   ```

4. Open the pull request against `main` and describe what changed and why.

## Layout

| Path | What |
|---|---|
| `desktop/` | The app: Vue frontend (`src/`), Rust backend (`src-tauri/`), build scripts |
| `tools/beatforge/` | The charting engine (DSP, StepForge Flow inference, StepMania writer) |
| `tools/beatforge/adapters/stepmania/neural/` | StepForge Flow: features, codec, inference; `train/` holds the training code |
| `models/stepforge_flow/` | The model the app ships: weights, model card, credits, license |

## Releases

Maintainers cut releases by pushing a `v*` tag on `main`; see
[desktop/BUILD.md](desktop/BUILD.md#cicd).

## Conduct

Be kind and constructive. We are here to make dancing to more music possible.
