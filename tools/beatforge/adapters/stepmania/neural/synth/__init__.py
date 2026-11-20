"""neural.synth — offline data tools for training StepForge Flow: a 4-stem
cache, acoustic augmentation of the training songs, a groove synthesizer with
exact charts, chart thinning, and programmatic step-pattern generators.

Nothing here is imported by the runtime (`infer.py`, `pipeline.py`); the extra
packages some of it needs are listed in `requirements-synth.txt`.

Every tool writes shards in the corpus format (`train/corpus.py`), one folder
per source, each with its own `manifest.json`:

    <corpus-root>/real/            the compiled human charts (train + validation)
    <corpus-root>/aug_stem/        remixed / pitch-shifted / tempo-stretched views
    <corpus-root>/synth_groove/    synthesized tracks with ground-truth charts
    <corpus-root>/synth_thin/      Beginner / Easy charts thinned from human ones
    <corpus-root>/synth_style/     step patterns with no audio
    <corpus-root>/synth_style_grounded/   the same patterns laid over real audio
    <corpus-root>/synth_cf/        training songs' rhythms re-stepped in several
                                   footing styles (counterfactual.py)
    <corpus-root>/synth_cf_stream/ the same for synthetic stream rhythms, no audio

Only `real/` may hold validation songs. A derived shard (`aug_stem`,
`synth_thin`, `synth_style_grounded`, `synth_cf`) names its parent and inherits the
parent's `song_id`, and is only ever built from a training song.
"""
from __future__ import annotations

#: Source folders under a corpus root, and whether their shards carry audio.
SOURCES = ("real", "aug_stem", "synth_groove", "synth_thin", "synth_style",
           "synth_style_grounded", "synth_cf", "synth_cf_stream")
#: Sources that must never contribute a validation or test item.
TRAIN_ONLY = tuple(s for s in SOURCES if s != "real")
