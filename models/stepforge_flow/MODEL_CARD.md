# StepForge Flow v1.6 — model card

Released 2026-11-22. The two networks are v1.5's, trained 2026-11-20 and
unchanged byte for byte; what is new in v1.6 is in the decoder (see "What
changed from v1.5"). Codec 1.0. Same architecture, files, inputs and outputs as
v1.4 and v1.5: a drop-in replacement. License: `LICENSE.md` (CC BY-NC-SA 4.0);
credits: `CREDITS.csv`.

| | |
|---|---|
| Stage A — RhythmNet | 6-layer bidirectional Conformer, d=256, AdaLN-Zero conditioning, 9.64M params |
| Stage B — ChoreoNet | 8-layer masked discrete diffusion transformer, d=320, 15.22M params |
| Files | `stage_a_rhythm.onnx` (38.8 MB), `stage_b_choreo.onnx` (61.1 MB), int8 variants (13.7 + 16.2 MB), `phrase_index.npz` (style retrieval, 16,064 phrases), `CREDITS.csv` |
| Runtime | numpy + scipy + simfile + onnxruntime; no PyTorch |
| Stems | not trained with stems (`trained_with_stems: false`); the stem channels are zero |

## What changed from v1.5

**No retraining.** v1.6 uses v1.5's weights. Three things changed in how a
chart is decoded from them, and the notes of an ordinary chart are the same as
v1.5's, one for one.

- **Arrows come back when the music does** (`decoder.callbacks`). The engine
  compares every bar of the song with the bars before it (`structure.py`,
  about 1 ms per song). Where a bar sounds like an earlier one, each of its
  notes gets a bonus toward the arrow the step network chose for the matching
  note of the earlier bar, before the foot-path search runs. Only the arrows
  can change: the rhythm is not touched (its weight is 0), because v1.5's
  rhythm already repeats more than human charts do. On 60 held-out songs the
  same arrow returns on repeated passages 4.4 points more often than chance;
  without callbacks it is 1.3, and human charts are at about 4.9. Unrelated
  passages are unaffected, and it adds at most 8 ms per chart.
- **The footswitch slider works.** Above 3%, the decoder places footswitches
  where a player keeps reading them as footswitches (each followed by both
  feet stepping out to their own side), reads the chart back with the same
  annotator that labels human charts, and takes out every one that reads as a
  jack. Asked for 6%, a chart now shows 6.3% (v1.5: 1.7%).
- **Stream mode has its own speed limit**: 16th-note streams up to 200 BPM at
  Challenge and 170 BPM at Hard, where v1.5 dropped to 8ths above 150 and 120.
  Ordinary charts keep their tier's limit.

Two larger experiments were run for v1.6 and are **not** in it, because they
did not beat what is shipped:

- **A voice network** (a small network that marks drums, bass, vocals and
  melody for the rhythm network). Step placement with it tied v1.5 (0.687
  against 0.689 on the 180 songs) and Hard charts gained no 16th notes. Feeding
  the rhythm network vocal and harmonic channels measured directly on
  separated stems did not help either (0.694 and 0.693 against 0.698 on the 60
  tuning songs), so the limit is not what the rhythm network hears.
- **A choreography critic** that scores four candidate sets of arrows and keeps
  the best. It tells human charts from damaged ones (AUC 0.86–0.90), but
  picking by it made the charts *less* like human charts (97.2% of statistics
  inside the human range fell to 93.9–96.4%) and added 2.4 s per song. The
  code is in the engine and switched off; this model does not use it.

## What changed from v1.4 (v1.5's training)

**No new human charts, and no new audio.** v1.5 is trained on the same 254
songs as v1.4, split the same way (227 for training, 27 for validation).

- **The rhythm network** is retrained on those songs with one addition: a share
  of the training windows is fed half a beat out of phase, and the network has
  to mark the true beats anyway. That is what lets it tell which half of the
  beat a grid sits on. 16th- and triplet-position notes of Hard and Challenge
  charts weigh more in its loss (×1.45 / ×1.30).
- **The step network** is trained on the human charts plus *counterfactual
  footing variants*: for every Medium, Hard and Challenge chart of a training
  song (524 rhythms), and for 300 synthetic stream rhythms, the same notes
  stepped up to five ways — plain alternation, crossover-heavy,
  footswitch-heavy, jack-heavy and a blend (3,195 charts). The variants of one
  rhythm share their notes and their audio, so the only thing that tells them
  apart is the style input, and the network has to read it. Every variant was
  written by a search over the foot model, read back by the same annotator
  that labels human charts, labelled with what that reading measures, and
  dropped unless it was legal and really different from the plain one. They
  are training material only and carry the id of the song they were made
  from; nothing was made from a validation or benchmark song.
- **The decoder** (since v1.5) routes the stamina preset (or a stream slider
  of 45% or more at Hard/Challenge) through Stream mode, refuses a step that
  the benchmark's foot model would read as forbidden, and refuses a note that
  can only be read as an unforced double-step.

An earlier attempt (v1.5a) added Demucs stems, 6,650 re-mixed charts, 300
synthesized grooves and 2,400 programmatic patterns. None of it improved step
placement over this recipe, so none of it is in this model.

## Training data (human charts, unchanged)

254 songs (1,066 dance-single charts, about 8.2 hours of audio) from:

| Source | Songs | Charts |
|---|---:|---:|
| StepMix 1.0, 2, 3, 4 and 5 (community contest packs) | 235 | 979 |
| OutFox Serenity, Volumes 1–3 | 17 | 77 |
| StepMania 5 default songs | 2 | 10 |

Every song is listed in `CREDITS.csv` with its artist, step artists, pack,
audio and chart license, the attribution text and where it came from.

**Licenses vary by song.** 218 of the 254 are StepMix contest entries whose audio
and steps are both under the StepMix contest grant; the rest carry Creative Commons licenses (CC-BY,
CC-BY-SA, some non-commercial variants, CC Sampling Plus) or the OutFox
Serenity free-content terms. The songs were checked for redistribution
permission; most licenses do not address machine-learning training one way or
the other. The set contains no charts generated by StepForge itself.

The songs went through the engine's quality gates: a sync check, BPM
50–320, at most 8 stops, 4/4 only, notes on the 48-tick grid, and a foot-model
discomfort ceiling. 254 of the 387 audited songs passed. One song that shares
audio with a held-out benchmark song was removed.

## Results

Benchmark: 180 community songs with human charts that this model never saw,
charted blind from audio and compared with the human chart of the same song and
difficulty. "smlab" is the strongest open-source auto-charter we tested.

| 180 held-out songs | v1.6 | v1.6 int8 | v1.4 | smlab |
|---|---:|---:|---:|---:|
| Step-placement F1 (±20 ms) | 0.689 | 0.687 | 0.671 | 0.674 |
| … Beginner / Easy | 0.606 / 0.695 | 0.603 / 0.689 | 0.562 / 0.667 | 0.587 / 0.687 |
| … Medium | 0.713 | 0.713 | 0.705 | 0.689 |
| … Hard / Challenge | 0.715 / 0.730 | 0.715 / 0.731 | 0.714 / 0.732 | 0.703 / 0.712 |
| … on the 120 songs never used for tuning | 0.685 | 0.684 | 0.663 | 0.671 |
| Precision / recall | 0.697 / 0.705 | 0.695 / 0.704 | 0.680 / 0.685 | 0.616 / 0.777 |
| Notes written, relative to the human chart | 1.06× | 1.06× | 1.06× | 1.34× |
| Looks like a human chart (12 statistics) | 97.4% | 97.6% | 98.1% | 89.9% |
| Unforced double-steps (notes, of 228,000; smlab of 283,000) | 0 | 0 | 23 | 3,556 |
| Charts with a step the foot model rates forbidden | 0 of 900 | 0 | 6 | 144 |
| Density follows song energy (Spearman ρ) | 0.675 | 0.673 | 0.713 | 0.446 |
| 16th notes per Hard chart, mean / median (human: 36 / 20) | 8 / 1 | 8 / 1 | 5 / 0 | |
| 16th notes per Challenge chart, mean / median (human: 97 / 88) | 32 / 16 | 33 / 16 | 27 / 15 | |

**Step placement is v1.5's, chart for chart**: the callbacks move arrows, not
notes, so every placement number in this table equals v1.5's. Generated again
with the v1.6 engine for all 180 songs in fp32 and int8, without a failure.

**Arrow reuse on repeated passages** (60 held-out songs, 300 charts; share of
notes that get the same arrow as the matching note of an earlier bar that
sounds alike, against the share expected by chance for that chart):

| | same arrow | chance | above chance |
|---|---:|---:|---:|
| v1.6 | 30.8% | 26.4% | +4.4 |
| the same engine, callbacks off | 26.9% | 25.6% | +1.3 |
| human charts (the 254 training songs) | | | about +4.9 |

43% of those charts have at least one repeated passage to link. Placement, the
QA pass rate (99.3%), forbidden steps (0) and double-steps (0) are identical
with callbacks on and off.

**Placement is better than v1.4**, by more than chance on these songs: +0.018
overall (95% interval +0.010 to +0.027), +0.044 at Beginner (+0.026 to +0.065),
+0.028 at Easy (+0.011 to +0.046), +0.008 at Medium; level at Hard and
Challenge. On the 120 songs the decoder was never tuned on: +0.022 (+0.011 to
+0.035). Against smlab it is ahead by 0.015 overall, which these songs cannot
distinguish from level (−0.004 to +0.036); at Medium the lead is real.
(Paired bootstrap over songs.) The int8 model is 0.002 behind fp32 overall,
all of it at Easy.

**Half-beat grid.** On the 163 benchmark songs where it can be judged, the
half-beat check decides wrongly on 1 (v1.4, with its three-way vote: 7), and 1
final grid is half a beat off (v1.4: 5). On the 105 of those songs not used
for tuning: none (v1.4: 6). On the 27 validation songs the check was right on
every song in 96% of the epochs after the 20th and on 26 in the rest.

**Style sliders** (20 held-out songs, Hard and Challenge, one slider moved at
a time on a frozen rhythm, measured as a player would read the chart):

| slider | asked | steps of that kind | change | rises on |
|---|---|---|---:|---:|
| crossovers | 0 → 0.10 → 0.15 | 0.1% → 10.5% → 13.6% | +13.6 pts | 100% of charts |
| jacks | 0 → 0.05 → 0.09 | 0.4% → 2.4% → 6.4% | +6.0 pts | 100% |
| footswitches | 0 → 0.06 | 0.0% → 6.3% | +6.3 pts (v1.5: +1.7) | 100% |

In v1.4 none of the three moved by as much as one point. The rhythm is the
same at every slider position and every chart passes QA. The app clamps the
sliders to the tier's ceilings (crossovers 15%, jacks 8–10%, footswitches 6%
at Hard/Challenge); with the ceilings lifted crossovers reach 16.5% at 0.20
and footswitches 10.6% at 0.16.

**Presets** (the same 20 songs, four presets, Hard and Challenge): mean slider
error 0.048 (v1.5: 0.060, v1.4: 0.080). Jumps, freezes and footswitches are
exact to within 0.002. The stamina preset is a stream chart: 80% of its
Challenge notes are in 16th streams (asked 80%; v1.5: 48%) and 34% at Hard
(asked 45%). The tech preset shows 6.2% footswitches (asked 6%) and 11.3%
crossovers (asked 15%), with fewer jacks than it asks for (under 1% against
5–6%). 98.8% of the 160 charts pass every QA gate (int8: 98.1%), none has a
forbidden step, a hold-lock or an unforced double-step.

**Stream mode** (120 held-out and tuning songs, Hard and Challenge). Streams
are 16ths in every chart up to 150 BPM. Above it Challenge stays in 16ths in
41 of 43 charts, and Hard in 18 of 43; the rest are above Hard's 170 BPM limit
and run in 8ths. No chart has a forbidden step or a double-step. The rhythm
network's own answer fills the stream where it is sure enough
(`decoder.neural_stream`); a stream is still completed by rule where the model
leaves a gap.

Validation (27 songs): Stage A AP 0.789, precision at matched density 0.777,
F1 0.792, F1 on Hard charts' 16th positions 0.410, half-beat check 27 of 27
(epoch 66 of 95, chosen inside the low-learning-rate phase by 0.5·precision
at matched density + 0.25·F1 at Hard 16ths + 0.25·half-beat accuracy).
Stage B cross-entropy 1.410, 48% accuracy with half the notes masked (epoch 112
of 125; v1.4: 1.418).

### Decoder settings stored with the model

`model_meta.json` → `decoder`, fitted by
`python -m beatforge.adapters.stepmania.neural.calibrate_decoder --fit-energy-tilt`
on 60 of the 180 songs:

- `phase_rule: "margin"`, `phase_margin: 1.5` — the rhythm network alone
  decides the half beat (v1.4 needed a three-way vote).
- `density_scale` — 0.80 / 1.00 / 0.95 / 0.94 / 0.88: the engine's own, which
  already put the median chart within 5% of the human note count.
- `energy_tilt` — a longer ladder (up to 12) for Beginner and Easy. This
  rhythm network is surer of itself than v1.4's, and the engine's steepest
  lean toward the loud bars no longer moved its low-tier charts: 85% passed
  the QA gates, 98% with the ladder, at the same placement.
- `neural_stream: true` — see Stream mode above.

## Memorization check

The same two tests as for v1.4, exact matching on the 48-ticks-per-bar grid,
run again with the v1.6 engine. The numbers are v1.5's: same weights, same
notes.

- **Its own training songs.** v1.6 charted its 227 training songs (944 charts)
  and the 27 validation songs (116 charts). **It places steps better on the
  songs it trained on: F1 0.743 against 0.723** (difference +0.020, 95%
  interval −0.005 to +0.044; v1.4: −0.001). The gap is at the low tiers
  (Beginner 0.758 vs 0.710, Easy 0.738 vs 0.703; Medium to Challenge within
  0.011). It is the rhythm network knowing its training songs' beats a little
  better, not copying: where it steps on the same tick as the human it picks
  the same arrows no more often than chance (24.5%, chance 24.7%), 0.1% of its
  bars are identical to the human chart's on training and validation songs
  alike, and its longest exact run of a training chart is 8 steps.
- **New songs.** 0.008% of v1.6's 16-step sequences on the 180 benchmark songs
  appear anywhere in the 950 human training charts (human charts of the same
  songs: 0.26%); no 32-step sequence does. Longest shared run: 17 steps.

The benchmark numbers above are on songs from other packs entirely, so the fit
gap does not flatter them; it does mean a chart of a training song is not a
fair sample of what the model does.

## Speed and size (Apple Silicon laptop CPU, 4 threads)

| | fp32 | int8 |
|---|---:|---:|
| Both networks | 100.0 MB | 29.9 MB |
| One chart, median (Beginner … Challenge) | 0.57 … 1.02 s | 0.52 … 0.93 s |
| All five charts of a song, median | 3.8 s | 3.5 s |

Measured on 12 songs of two to three minutes with the features already
extracted (0.2–0.7 s per song), on a machine that was quiet but not idle; v1.5
measured in the same sitting took 3.7 s and 3.4 s. The callbacks add at most
8 ms per chart. The decoder, not the networks, is most of the time, which is
why int8 is barely faster.

## Known limits

- **Beginner is still the weak tier** (0.606). It gained most (+0.044), and is
  still where a chart shares the fewest notes with the human one.
- **Still few 16th notes at Hard.** The median Hard chart has one (human: 20).
  A decoder option that promotes 16th figures exists
  (`decoder.hard_sixteenth_boost`) and is off: reaching a median of 8 costs
  0.006 of placement F1, because a promoted 16th agrees with the human chart
  less than half as often as the note it displaces.
- **The jacks slider cannot go above the model's own rate.** It lowers jacks
  to zero and raises them to about 6%, and the tech preset, which asks for
  jacks and footswitches together, gets its footswitches and few jacks.
- **Callbacks need a song that repeats.** They act on bars that sound like an
  earlier bar; 43% of held-out charts had any. On a song without repeated
  passages a default v1.6 chart is the chart v1.5 writes.
- **Streams stop at the speed limit.** Above 200 BPM a Challenge stream (170
  at Hard) runs in 8ths. Jumpstream and tech presets are not routed to Stream
  mode (a stream has no jumps), so their stream slider still does nothing.
- **Charts follow the song's energy a little less** than v1.4's (ρ 0.675 vs
  0.713), and 97.4% of their statistics fall inside the human range against
  98.1%; the misses are at Medium and Hard, in the share of notes off the
  quarter grid.
- **Fits its training songs' low-tier charts better than new ones** (see the
  memorization check).
- Trained on dance-single 4/4 charts without gimmick timing; constant-BPM
  inference only.
- The phrase index uses the built-in feature embedder, not EmbeddingGemma 2.
