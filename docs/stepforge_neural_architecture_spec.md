> **Original design spec, kept for history.** It is the plan StepForge Flow
> started from, and parts of it were never built or were built differently
> (for example the EmbeddingGemma and Decisions API pieces and the latency
> target). What was actually built and shipped is described in the
> [model card](../models/stepforge_flow/MODEL_CARD.md).

# STEPFORGE_NEURAL.SPEC.MD — Custom Beat-Grid & Biomechanical Chart Model ("StepForge Flow")

**Spec version:** 1.0  
**Target repo:** BeatForge (`tools/beatforge/` + `tools/chartbench/` + Tauri Desktop GUI)  
**Depends on:** `STEPMANIA.SPEC.MD` (reuses BeatForge DSP analysis, `footflow.py` ergonomics, `garcia/simfile` serialization, `barrysir/stepmania-parsing-code` validation, and `qa.py` gates)  
**Conventions:** REQ IDs `REQ-SFN-\d+`; acceptance criteria labelled `Accept:` (compatible with repo `spec-lint` / `trace-gate`).

---

## 0. Agent Execution Rules & Guardrails (READ FIRST)

Before writing any code, the implementing coding agent MUST follow these rules:

1. **Inspect Existing Files First:** Read `tools/beatforge/dsp.py`, `tools/beatforge/analyze.py`, `tools/beatforge/adapters/base.py`, and `tools/beatforge/adapters/stepmania/{grammar.py,quantize.py,footflow.py,realize.py,serialize.py,qa.py}` to align exact data structures, constants, and types.
2. **Zero Regressions to Core & Existing Engines:** Do NOT break `core/`, the `--deterministic` path, or the legacy LLM backends (`gemini`, `google-genai`, `openai`, `inkling`). The new custom model ships as `--engine neural` (and becomes the default once trained weights exist on disk).
3. **Strict Dependency Split (Training vs. Runtime):**
   - **Runtime / Tauri Desktop App (`requirements.txt`):** Must ONLY require `numpy`, `scipy`, `simfile`, and `onnxruntime` (no `torch` import at inference time). Keep desktop bundling lightweight and fast ($<50\text{ ms}$ CPU inference).
   - **Offline Training (`requirements-train.txt`):** `torch>=2.3`, `torchaudio`, `safetensors`, `onnx`, `onnxscript`, `tqdm`. Guard all PyTorch imports behind training modules so `pytest` runtime tests pass even if only `onnxruntime` is installed (or skip PyTorch-only training tests cleanly via `pytest.importorskip("torch")`).
4. **Milestone-Driven Execution:** Execute §11 (Order of Work) sequentially from M1 to M6. Run `PYTHONPATH=tools python3 -m pytest tools/beatforge/tests` at the end of every milestone and verify zero failures before starting the next milestone.
5. **Song-Level Data Splits Only:** Never split training/validation/test sets by individual chart difficulty. Split strictly by **audio fingerprint (SHA-256 / spectral hash)** so Easy/Medium/Hard/Challenge charts of the same song never leak across splits.

---

## 1. Mission & Architectural Overview

Replace slow, expensive, schema-fragile LLM JSON authoring with **StepForge Flow**: a ~22M parameter, non-LLM, domain-native neural architecture trained on local StepMania / ITG `.sm` and `.ssc` packs.

### Core Design Pillars
1. **Beat-Synchronous Coordinate Warping:** Never force the neural network to learn tempo division from raw millisecond STFT frames. Warp audio features onto a $G = 48$ ticks-per-measure musical grid using BeatForge's measured float BPM/offset (at inference) or the simfile's ground-truth `#BPMS`/`#OFFSET` (at training), while passing explicit physical tick duration $\Delta t_k$ in milliseconds for stamina awareness.
2. **Foot-Relative Biomechanical Codec (Zero Symmetry Collapse):** Human packs map identical melodies starting on either the Left or Right foot with equal validity. Instead of predicting raw `{L, D, U, R}` columns (which causes $50\%/50\%$ cross-entropy averaging and double-steps), invert `footflow.py` during corpus ingestion to tokenize every chart into a **20-token mirror-invariant Foot-Relative Action Codec**.
3. **Two-Stage Generation + Exact Ergonomic Projection:**
   - **Stage A (`RhythmNet`, ~8M Bidirectional Conformer):** Predicts active ticks, jumps, holds/rolls, and mines on the $G = 48$ measure grid using **Focal BCE + Per-Measure Cardinality + 1D Wasserstein (CDF) Loss**.
   - **Stage B (`ChoreoNet`, ~14M Masked Discrete Diffusion + Linear-Chain CRF):** Runs *only* on the $N \approx 250\text{--}1500$ active notes selected by Stage A. Unmasks Foot-Relative choreography bidirectionally in $K = 10$ parallel steps, terminating in an **exact constrained Viterbi decode** over the `footflow.py` transition cost matrix and active hold-locks.
4. **Optional Multimodal Style Retrieval & Macro-Planning:** Clean plug-in interfaces for **EmbeddingGemma 2** (256D 4-bar audio phrase style RAG stored in a flat `numpy` index) and the **Decisions API** (single-pass zero-output-token dataset quality filtering / macro-style scoring).

### End-to-End Dataflow

```text
Audio (WAV/MP3/OGG)
  |
  +--> [BeatForge DSP Core (reused)] --> BPM, Offset, Beats, Sections, Stem/HPSS Audio
  |
  +--> [grid_features.py] -------------> Beat-Warped 48-Tick Tensor X in R^(T x 64) + dt_ms
  |
  +--> [style_rag.py (optional)] ------> 4-Bar 256D EmbeddingGemma 2 -> Top-3 Pack Style Vectors
  |
  v
[Stage A: RhythmNet ONNX (~8M Bi-Conformer + AdaLN-Zero)]
  |  Inputs:  X in R^(T x 64), Conditioning c in R^(10) (Meter + Style Sliders)
  |  Decode:  Subdivision Mask + Density-Calibrated Peak Picker
  |  Outputs: Active note ticks t_1..t_N + Note Kinds (Tap, Jump, Hold/Roll + Duration, Mine)
  v
[Stage B: ChoreoNet ONNX (~14M Masked Diffusion Transformer)]
  |  Inputs:  Active-note features E in R^(N x 320), Hold-Lock Masks, Conditioning c
  |  Decode:  10 Cosine Unmasking Steps + Exact Constrained Viterbi (footflow.py matrix)
  |  Outputs: Bijective Foot-Relative Tokens a_1..a_N -> Concrete {L, D, U, R} Simfile Rows
  v
[Existing StepMania Serializer & QA] --> garcia/simfile (.ssc/.sm) + barrysir Referee + Previews
```

---

## 2. Package & Directory Layout

Add the following files without modifying `tools/beatforge/core/` (if extracted) or breaking existing StepMania adapter modules:

```text
tools/
  beatforge/
    adapters/
      stepmania/
        neural/
          __init__.py
          codec.py            # 20-token Foot-Relative Biomechanical Codec + inverse footflow
          grid_features.py    # 48-tick beat-warped DSP feature extractor (Demucs + HPSS unified)
          conditioning.py     # 10D style descriptor computation, presets & slider normalization
          infer.py            # Pure numpy + onnxruntime inference, calibration, Viterbi & inpaint
          style_rag.py        # Optional EmbeddingGemma 2 4-bar numpy cosine style retriever
          decisions_hook.py   # Optional /v1/decisions macro-scorer & dataset quality filter
          train/              # PyTorch-only subpackage (never imported by infer.py!)
            __init__.py
            corpus.py         # Scan local Songs/ packs -> filter -> shard into .safetensors/.npz
            dataset.py        # Song-split PyTorch Dataset, mirror aug, stem/CFG dropout
            losses.py         # Focal BCE + 1D Wasserstein CDF + Cardinality + CRF NLL
            rhythm_net.py     # Stage A: 6-layer Bi-Conformer + AdaLN-Zero (~8M params)
            choreo_net.py     # Stage B: 8-layer RoPE Masked Diffusion + Neural CRF (~14M params)
            train_stage_a.py  # CLI trainer for Stage A + validation metrics
            train_stage_b.py  # CLI trainer for Stage B + ergonomic legality checks
            export_onnx.py    # Export Stage A & Stage B to ONNX + parity verification
    tests/
      test_neural_grid.py
      test_neural_codec.py
      test_neural_losses.py
      test_neural_infer.py
      test_neural_onnx_parity.py
models/
  stepforge_flow/             # Default location for exported .onnx weights + model_meta.json
requirements-train.txt        # PyTorch + training-only dependencies
```

**REQ-SFN-01 — Runtime dependency isolation.** Importing and executing `tools.beatforge.adapters.stepmania.neural.infer` MUST succeed in a Python environment where `torch` is NOT installed, relying solely on `numpy`, `scipy`, `onnxruntime`, `simfile`, and existing BeatForge modules.  
`Accept:` A unit test blocks `sys.modules["torch"] = None` and runs a complete dummy ONNX chart generation + `.ssc` serialization end-to-end.

---

## 3. Workstream N1 — Beat-Synchronous 48-Tick Feature Extraction (`grid_features.py`)

### 3.1 Musical Subdivision Grid
Every 4-beat measure is subdivided into $G = 48$ ticks ($12$ ticks per beat). This grid exactly represents 4th, 8th, 12th (triplet), 16th, 24th, and 48th notes, covering $>99.5\%$ of human ITG/DDR charts without rounding error. For rare 32nd or 64th bursts in training simfiles, snap to the nearest 48th tick during ingestion only if timing error $\le 15\text{ ms}$, flagging any measure with $>2$ snapped notes in metadata.

For tick index $k \in \{0, 1, \dots, T-1\}$, its musical beat is $b_k = \frac{k}{12}$. Using the timing map (from `#OFFSET` + `#BPMS` + `#STOPS` during training via `simfile.timing`, or from BeatForge's measured `offset` + `bpm` during inference), compute the exact timestamp $t(b_k)$ in seconds and instantaneous physical tick duration in milliseconds:

$$\Delta t_k = \left( t(b_{k+1}) - t(b_k) \right) \times 1000.0$$

Clamp $\Delta t_k$ to $[10.0, 500.0]\text{ ms}$ and encode both $\log(\Delta t_k / 40.0)$ and instantaneous $\text{BPM}_k / 150.0$.

### 3.2 Unified Stem + HPSS Audio Features ($D_{\text{in}} = 64$)
BeatForge supports both 4-stem Demucs GPU extraction and local CPU HPSS-only extraction (`stem_source: none`). To ensure **one single neural model** works natively in both modes:

Compute audio frames on a short hop ($10\text{ ms}$ at $22050\text{ Hz}$ mono) and linearly interpolate/pool onto tick timestamps $t(b_k)$ using a triangular window spanning $[t(b_{k-0.5}), t(b_{k+0.5})]$. Build a $64$-channel feature vector per tick:

| Channels | Dim | Description |
|---|---|---|
| `broadband_mel` | 20 | Log-Mel spectrogram ($20$ bands, $30\text{ Hz}\text{--}11\text{ kHz}$), per-track z-normalized |
| `hpss_onset_flux` | 6 | Harmonic & Percussive spectral flux across Low ($<150\text{ Hz}$), Mid ($150\text{--}2000\text{ Hz}$), High ($>2\text{ kHz}$) |
| `flux_delta` | 6 | First-order finite difference $F_k - F_{k-1}$ of `hpss_onset_flux` (sharp transient localization) |
| `stem_features` | 16 | 4 Demucs stems (`drums`, `bass`, `vocals`, `other`) $\times$ 4 bands (sub, low, mid, high RMS + flux); zero-filled when `has_stems == 0` |
| `chroma_contour` | 6 | 4 pitch-height summary quantiles + spectral centroid normalized + $\Delta$ spectral centroid (melodic sweep direction) |
| `metric_timing` | 10 | One-hot subdivision class `{4th, 8th, 12th, 16th, 24th, 48th}` (6) + $\sin/\cos(2\pi b_k / 16)$ 4-bar phrase phase (2) + $\log(\Delta t_k / 40.0)$ (1) + song progress $k / T$ (1) |

**Stem-Dropout Rule (Training):** When training on Demucs-cached shards, zero out `stem_features` (channels `32..47`) and set conditioning flag `has_stems = 0.0` with probability $p_{\text{drop\_stems}} = 0.50$. This guarantees local CPU Tauri runs without Demucs suffer minimal degradation while GPU/Demucs runs exploit full stem separation.

**REQ-SFN-02 — Beat-warped grid tensor.** `extract_grid_features()` produces `X` of shape `(T, 64)` (`float32`, zero `NaN`/`Inf`), exact tick timestamps `tick_times_sec` of shape `(T,)`, and `dt_ms` of shape `(T,)`.  
`Accept:` Unit tests verify exact alignment on synthetic click tracks at $120\text{ BPM}$ and $175\text{ BPM}$ (flux peaks land strictly on target grid ticks $k \equiv 0 \pmod{12}$) and identical tensor shapes with `stems=None` vs. `stems={...}`.

---

## 4. Workstream N2 — Inverse `footflow` & The 20-Token Biomechanical Codec (`codec.py`)

This module solves **Mirror Symmetry Collapse** by encoding all `dance-single` taps and jumps relative to the active physical foot rather than absolute pad columns.

### 4.1 Foot-Relative Panel Mapping
Let physical foot $f \in \{\text{LEFT}, \text{RIGHT}\}$ and foot-relative panel $z \in \{\text{HOME}, \text{DOWN}, \text{UP}, \text{CROSS}\}$. Define the deterministic physical panel decoder $\text{ panel}(f, z) \in \{\text{L}, \text{D}, \text{U}, \text{R}\}$:

- When $f = \text{LEFT}$: $\text{HOME} \mapsto \text{L}$, $\text{DOWN} \mapsto \text{D}$, $\text{UP} \mapsto \text{U}$, $\text{CROSS} \mapsto \text{R}$.
- When $f = \text{RIGHT}$: $\text{HOME} \mapsto \text{R}$, $\text{DOWN} \mapsto \text{D}$, $\text{UP} \mapsto \text{U}$, $\text{CROSS} \mapsto \text{L}$.

Let $\sigma$ denote horizontal mirror reflection: $\sigma(\text{L})=\text{R}$, $\sigma(\text{R})=\text{L}$, $\sigma(\text{D})=\text{D}$, $\sigma(\text{U})=\text{U}$, and $\sigma(\text{LEFT})=\text{RIGHT}$, $\sigma(\text{RIGHT})=\text{LEFT}$. By construction:

$$\text{panel}(\sigma(f), z) = \sigma(\text{panel}(f, z)) \quad \forall f, z$$

### 4.2 Unified 20-Token Action Vocabulary
At active note event $i \in \{0, \dots, N-1\}$, let $f_{\text{last}} \in \{\text{LEFT}, \text{RIGHT}\}$ be the foot that moved most recently prior to event $i$ (initialized at $i = 0$ by chart parity seed $f_{\text{init}} \in \{\text{LEFT}, \text{RIGHT}\}$). Define $f_{\text{alt}} = \sigma(f_{\text{last}})$ (the alternating foot) and $f_{\text{same}} = f_{\text{last}}$ (the repeating foot).

Every active event (1-panel step or 2-panel jump) is represented by a single integer token $a_i \in \{0, \dots, 19\}$:

1. **Tokens `0..3` — Alternating Single Step (`ALT`):** Active foot is $f_i = f_{\text{alt}}$, placed on $z \in \{\text{HOME}, \text{DOWN}, \text{UP}, \text{CROSS}\}$. Updates $f_{\text{last}} \leftarrow f_{\text{alt}}$, $p_{f_{\text{alt}}} \leftarrow \text{panel}(f_{\text{alt}}, z)$.
2. **Tokens `4..7` — Same-Foot Single Step (`SAME` / Jack / Double-Step):** Active foot is $f_i = f_{\text{same}}$, placed on $z \in \{\text{HOME}, \text{DOWN}, \text{UP}, \text{CROSS}\}$. Updates $f_{\text{last}} \leftarrow f_{\text{same}}$, $p_{f_{\text{same}}} \leftarrow \text{panel}(f_{\text{same}}, z)$.
3. **Tokens `8..19` — Two-Foot Jumps (`JUMP`):** Out of all $4 \times 4 = 16$ pairs $(z_{\text{alt}}, z_{\text{same}})$, exactly 4 pairs collide on the same physical column (`(DOWN, DOWN)`, `(UP, UP)`, `(HOME, CROSS)`, `(CROSS, HOME)`). The remaining **12 non-colliding pairs** ordered lexicographically form tokens `8..19`, placing $f_{\text{alt}}$ on $\text{panel}(f_{\text{alt}}, z_{\text{alt}})$ and $f_{\text{same}}$ on $\text{panel}(f_{\text{same}}, z_{\text{same}})$.
   - **Deterministic Post-Jump Parity Update:** Compare $(p'_{f_{\text{alt}}}, p'_{f_{\text{same}}})$ to $(p_{f_{\text{alt}}}, p_{f_{\text{same}}})$:
     - If $p'_{f_{\text{alt}}} \neq p_{f_{\text{alt}}}$ and $p'_{f_{\text{same}}} = p_{f_{\text{same}}}$, set $f_{\text{last}} \leftarrow f_{\text{alt}}$.
     - Else if $p'_{f_{\text{same}}} \neq p_{f_{\text{same}}}$ and $p'_{f_{\text{alt}}} = p_{f_{\text{alt}}}$, set $f_{\text{last}} \leftarrow f_{\text{same}}$.
     - Else (both feet moved or jump-jack where neither moved), set $f_{\text{last}} \leftarrow f_{\text{alt}}$.

*(Reserve token `20` as `[MASK]` for Masked Discrete Diffusion in Stage B.)*

### 4.3 Hold / Roll Attachment & Active Hold-Locks
Holds (`2`..`3`) and rolls (`4`..`3`) attach relative to `f_alt` and `f_same` so mirror invariance is preserved:
- `hold_flavor_i` $\in \{\text{NONE}=0, \text{HOLD\_ALT}=1, \text{HOLD\_SAME}=2, \text{HOLD\_BOTH}=3, \text{ROLL\_ALT}=4, \text{ROLL\_SAME}=5\}$ (for single-step tokens `0..3`, only `NONE`, `HOLD_ALT`, `ROLL_ALT` apply; for `4..7`, `HOLD_SAME`, `ROLL_SAME`).
- `hold_dur_ticks_i`: integer duration in 48th-note ticks, quantized during Stage A prediction to the 10-bucket vocabulary $\{3, 6, 9, 12, 18, 24, 36, 48, 72, 96\}$ ticks ($0.25$ to $8.0$ beats).
- **Hold-Lock Invariant:** If foot $f$ starts a hold at tick $t_i$ with tail at $t_i + d_i$, then for every intermediate event $j$ with $t_i < t_j \le t_i + d_i$, foot $f$ is **locked** on panel $p_f$: any action token $a_j$ that moves foot $f$ away from $p_f$ OR places the other foot onto $p_f$ receives transition score $-\infty$.

### 4.4 Inverse `footflow` Chart Annotator
To encode existing human charts from `.sm`/`.ssc` into ground-truth $(f_{\text{init}}, a_{0..N-1})$:
1. Extract all note rows (discarding 3+ panel hands/quads in v1 by either skipping charts with $>1\%$ hands or thinning the lowest-salience 3rd panel during ingestion).
2. Run a forward-backward Viterbi search using `footflow.py`'s existing ergonomic transition penalties (respecting hold-locks!) to infer the minimum-cost physical foot assignment $(p_{L,i}, p_{R,i}, f_{\text{last},i})$ for every row $i$.
3. Convert each inferred transition deterministically into $a_i \in \{0..19\}$.

**REQ-SFN-03 — Bijective & mirror-invariant codec.** Given any sequence of legal 1- and 2-panel `dance-single` events with holds/rolls:
1. `decode(encode(events))` reproduces the exact `{L, D, U, R}` taps, jumps, holds, and rolls bit-for-bit.
2. Horizontally mirroring the input chart (`L <-> R`) yields the **exact same token array** `a_0..a_{N-1}` and `hold_flavor_0..N-1`, differing *only* in the boolean initial parity bit `f_init`!  
`Accept:` `test_neural_codec.py` verifies 100% lossless round-trip and exact token identity under `L <-> R` reflection across synthetic patterns (streams, crossovers, candles, jacks, `D`/`U` footswitches, jumps, and freeze-arrow tap transitions) plus real `.ssc` fixtures.

---

## 5. Workstream N3 — Conditioning & Style Descriptors (`conditioning.py`)

Both Stage A and Stage B receive a $10$-dimensional continuous conditioning vector $c \in \mathbb{R}^{10}$:

| Index | Field | Range / Normalization | Meaning |
|---|---|---|---|
| `0` | `meter_norm` | $\text{clamp}(\text{meter}, 1, 15) / 15.0$ | Target ITG / StepMania integer rating |
| `1` | `nps_norm` | $\text{clamp}(\text{sustained\_nps}, 0.5, 16.0) / 16.0$ | 80th-percentile 4-second window NPS |
| `2` | `stream_share` | $[0.0, 1.0]$ | Share of steps inside contiguous $\ge 8$-note 16th+ runs |
| `3` | `jump_share` | $[0.0, 0.35]$ | Share of active events that are 2-panel jumps |
| `4` | `hold_share` | $[0.0, 0.35]$ | Share of active events carrying holds or rolls |
| `5` | `crossover_share` | $[0.0, 0.30]$ | Share of single steps using `CROSS` token |
| `6` | `jack_share` | $[0.0, 0.20]$ | Share of single steps using `SAME` on identical panel |
| `7` | `footswitch_share` | $[0.0, 0.20]$ | Share of `ALT` steps landing on the previous step's `D`/`U` panel |
| `8` | `mine_share` | $[0.0, 0.15]$ | Ratio of mines to active note events ($0.0$ default) |
| `9` | `has_stems` | $\{0.0, 1.0\}$ | $1.0$ if 4-stem Demucs channels are populated, else $0.0$ |

### Partial Conditioning & Style Presets
- During training, compute the exact empirical $c$ of each human chart. Apply **Conditioning Dropout**: independently mask indices `1..8` (replacing each dropped value with $-1.0$ alongside a binary availability mask concatenated to $c$, making total conditioning input $19$-D) with probability $0.25$, and drop *all* style indices `1..8` simultaneously with probability $0.25$.
- At inference time, the user can pass **only a difficulty tier / meter** (setting `1..8` to masked $-1.0$ so the model infers idiomatic style from the audio), OR select a built-in preset (`casual_flow`, `itg_stamina`, `tech_footswitch`, `jumpstream`), OR set individual sliders from the Tauri desktop GUI!
- All generated outputs still clamp against `adapters/stepmania/grammar.py::BUDGETS` per difficulty tier (`Beginner`, `Easy`, `Medium`, `Hard`, `Challenge`).

**REQ-SFN-04 — Empirical style round-trip & presets.** `compute_style_vector(chart)` extracts deterministic $c \in \mathbb{R}^{10}$ from any parsed chart; `PRESETS` maps `(difficulty, preset_name)` to valid vectors within `grammar.py::BUDGETS`.  
`Accept:` Unit tests confirm pure-stream, pure-jumpstream, and crossover-heavy fixtures score high on their respective indices and stay within `[0, 1]` bounds.

---

## 6. Workstream N4 — Neural Models & Training Losses (`neural/train/`)

### 6.1 Stage A: `RhythmNet` (~8M Parameters, `rhythm_net.py`)
- **Input:** `X` $\in \mathbb{R}^{B \times T \times 64}$ (padded to measure multiples; typical 3-min track $T \approx 4320$), conditioning `cond` $\in \mathbb{R}^{B \times 19}$.
- **Backbone:** Linear projection $64 \to 256$, followed by **6 Bidirectional Conformer blocks** ($d_{\text{model}} = 256$, $n_{\text{heads}} = 4$, FFN expansion $4\times$, depthwise conv kernel $= 7$, Rotary Position Embeddings `RoPE`, dropout $= 0.1$). Inject `cond` into every block via **Adaptive LayerNorm (`AdaLN-Zero`)**.
- **Output Heads (per tick $t \in \{0..T-1\}$):**
  1. `onset_logit` $\in \mathbb{R}^{B \times T}$ (active note probability $\hat{p}_t = \sigma(\text{logit}_t)$)
  2. `jump_logit` $\in \mathbb{R}^{B \times T}$ (2-panel jump probability given active event)
  3. `hold_type_logits` $\in \mathbb{R}^{B \times T \times 6}$ (`NONE`, `HOLD_ALT`, `HOLD_SAME`, `HOLD_BOTH`, `ROLL_ALT`, `ROLL_SAME`)
  4. `hold_dur_logits` $\in \mathbb{R}^{B \times T \times 10}$ (quantized duration buckets)
  5. `mine_logit` $\in \mathbb{R}^{B \times T}$ (mine probability)
  6. `hidden_state` $H \in \mathbb{R}^{B \times T \times 256}$ (passed to Stage B at active ticks!)

### 6.2 Stage A Loss Function (`losses.py`)
Standard BCE blurs sparse measure grids. Implement the composite rhythm loss over unpadded measures $m \in \{1..M\}$ (each of length $G = 48$ ticks):

$$\mathcal{L}_{\text{StageA}} = \mathcal{L}_{\text{Focal}}(\hat{p}, y) + \lambda_{\text{card}} \frac{1}{M} \sum_{m=1}^{M} \left| \sum_{t \in m} \hat{p}_t - \sum_{t \in m} y_t \right| + \lambda_{\text{W1}} \frac{1}{M} \sum_{m=1}^{M} \sum_{g=1}^{48} \left| \text{CDF}_m(\hat{p})_g - \text{CDF}_m(y)_g \right| + \mathcal{L}_{\text{aux}}$$

where:
- $\mathcal{L}_{\text{Focal}}$ is Focal Binary Cross-Entropy ($\alpha = 0.75, \gamma = 2.0$).
- $\text{CDF}_m(v)_g = \frac{\sum_{j=1}^{g} v_{m,j}}{\epsilon + \sum_{j=1}^{48} v_{m,j}}$ with $\epsilon = 10^{-6}$ (only evaluated on non-empty ground-truth measures where $\sum_{j=1}^{48} y_{m,j} > 0$).
- Default weights: $\lambda_{\text{card}} = 0.15$, $\lambda_{\text{W1}} = 0.50$.
- $\mathcal{L}_{\text{aux}}$ sums masked BCE on `jump_logit` (only at ground-truth active ticks $y_t = 1$), masked cross-entropy on `hold_type_logits` and `hold_dur_logits` (at $y_t = 1$), and Focal BCE on `mine_logit` ($0.25$ weight).

### 6.3 Stage A Inference Decoder & Density Calibration (`infer.py`)
1. Mask illegal tick subdivisions for the requested difficulty tier (e.g., `Easy` masks ticks not divisible by $6$, restricting to 8th notes; `Medium` masks ticks not divisible by $3$ unless high-salience triplet).
2. Enforce minimum physical inter-onset gap $\Delta t_{\min}$ per tier from `grammar.py::BUDGETS` via 1D local peak-picking.
3. **Monotonic Threshold Calibration:** Run 12 iterations of bisection search on probability threshold $\theta \in [0.15, 0.85]$ so the selected active note count matches the target sustained NPS budget (or model's predicted measure cardinality sum $\sum_t \hat{p}_t$ when NPS is unconditioned), weighted by local audio energy so BeatForge's Spearman density-vs-energy gate ($\rho \ge 0.55$) is naturally satisfied.

### 6.4 Stage B: `ChoreoNet` (~14M Parameters, `choreo_net.py`)
Stage B runs **exclusively on the $N$ active note events** ($t_{i_0} < t_{i_1} < \dots < t_{i_{N-1}}$), never on empty ticks!
- **Per-Event Input Embedding ($d_{\text{model}} = 320$):** Concatenate and project:
  1. Stage A backbone feature $H_{t_i} \in \mathbb{R}^{256}$ (detached or lightly fine-tuned; during Stage B standalone training, feed ground-truth tick gather of grid features + lightweight 2-layer conv stem or frozen Stage A encoder).
  2. Event-level timing & musical deltas: $\log(\Delta t_{\text{prev}})$, $\log(\Delta t_{\text{next}})$, beat phase, `is_jump` flag, `hold_flavor` embedding, `hold_dur` embedding, pitch-contour delta $\Delta \text{centroid}_i$, kick-to-snare band ratio, and active hold-lock indicator flags (`alt_locked_from_prev`, `same_locked_from_prev`).
  3. Token embedding of `masked_action_tokens` $\in \{0..20\}$ ($20 = \text{[MASK]}$).
- **Backbone:** **8 Bidirectional Transformer blocks** ($d_{\text{model}} = 320$, $n_{\text{heads}} = 8$, `RoPE` over event index $i$, `AdaLN-Zero` from `cond` $\in \mathbb{R}^{19}$). Output unary logits $\Psi \in \mathbb{R}^{B \times N \times 20}$.
- **Training Objective (Masked Discrete Diffusion + Ergonomic Transition Regularizer):**
  - Sample $u \sim \mathcal{U}(0, 1)$ per sequence and mask rate $\gamma = \cos(\frac{\pi}{2} u)$. Replace each ground-truth token $a_i$ independently with `[MASK]` (`20`) with probability $\gamma$ (also explicitly mask single-step tokens `0..7` vs jump tokens `8..19` while keeping `is_jump` known so jump vs. single-tap count never drifts from Stage A).
  - Compute cross-entropy loss on masked positions, PLUS an **expected ergonomic penalty** over adjacent token probabilities $\hat{P}_i = \text{softmax}(\Psi_i)$:
    $$\mathcal{L}_{\text{StageB}} = \mathcal{L}_{\text{CE}}(\Psi_{\text{masked}}, a_{\text{masked}}) + \lambda_{\text{ergo}} \frac{1}{N-1} \sum_{i=0}^{N-2} \hat{P}_i^\top \mathbf{C}_{\text{ergo}}(\Delta t_{i+1}, c) \hat{P}_{i+1}$$
    where $\mathbf{C}_{\text{ergo}} \in \mathbb{R}_{\ge 0}^{20 \times 20}$ encodes normalized biomechanical transition discomfort (double-step penalty, fast crossover twist penalty when $\Delta t < 110\text{ ms}$, low-meter jack/footswitch penalty, and collision penalty on locked holds).

### 6.5 Stage B Inference: Parallel Unmasking + Exact Viterbi Projection (`infer.py`)
1. Initialize all $N$ active events to `[MASK]` (`20`). Restrict candidate logits at event $i$ to `{0..7}` if Stage A marked event $i$ as a single step, or `{8..19}` if Stage A marked it as a jump.
2. **10 Cosine Unmasking Steps:** For step $s = 1 \dots 10$:
   - Run `choreo_net.onnx` once on current partially unmasked sequence to obtain unary logits $\Psi^{(s)} \in \mathbb{R}^{N \times 20}$.
   - Sample candidate tokens with Gumbel noise at temperature $\tau \cdot (1 - s/10)$ (default $\tau = 0.6$; $\tau = 0.0$ for deterministic runs), compute confidence scores, and keep the top $\lfloor N \cdot (1 - \cos(\frac{\pi s}{20})) \rfloor$ most confident tokens unmasked while re-masking the rest.
3. **Step 11 — Exact Constrained Viterbi Decode (100% Pad Legality):**
   - Take the final unary log-probabilities $\log \text{softmax}(\Psi^{(10)})$ and run **exact dynamic programming (Viterbi)** over the explicit physical state space $(p_L, p_R, f_{\text{last}}) \in 12 \times 2 = 24$ states using `footflow.py` transition costs, difficulty-tier hard masks (e.g., zero crossovers on `Easy`, no $>180^\circ$ spins ever, no moving a hold-locked foot ever).
   - Choose $f_{\text{init}} \in \{\text{LEFT}, \text{RIGHT}\}$ by selecting whichever initial parity gives the lower total Viterbi path cost (or honoring user mirror seed).
4. **Section Inpainting (`inpaint_section`):** Given an existing chart and `[start_bar, end_bar]`, clamp tokens outside `[start_bar, end_bar]` to their existing values, set tokens inside `[start_bar, end_bar]` to `[MASK]`, run the 10 unmasking steps + Viterbi pinned to the entry and exit foot states. Runs in $<15\text{ ms}$ for instant GUI re-rolling!

**REQ-SFN-05 — Mathematical loss & ergonomic Viterbi guarantees.**  
1. `wasserstein_1d_measure_loss` produces smooth, non-zero finite gradients when a single predicted spike is shifted by $1..10$ ticks from a ground-truth spike within the same measure.  
2. Final Stage B Viterbi output has **zero** hold-lock collisions, **zero** impossible $>180^\circ$ twists, and **zero** double-steps on `Easy`/`Medium` synthetic test inputs regardless of random unary logits.  
`Accept:` Verified in `test_neural_losses.py` and `test_neural_infer.py` using synthetic & random logits.

---

## 7. Workstream N5 — Corpus Compiler & Training CLI (`neural/train/corpus.py`)

Build a fast, multiprocessing corpus compiler that ingests local ITGmania / StepMania pack directories into clean training shards:

```bash
PYTHONPATH=tools python3 -m beatforge.adapters.stepmania.neural.train.corpus \
  --packs-dir "/path/to/ITGmania/Songs" \
  --out-dir "build/neural_corpus" \
  --workers 4 [--with-demucs]
```

### Corpus Filtering Rules (Deterministic Quality Gate)
Reject any chart (or simfile) that fails any of the following:
1. `STEPSTYPE` is not `dance-single`, audio file missing/corrupt, or playable duration $< 30\text{ s}$ or $> 600\text{ s}$.
2. Gimmick timing filters (v1): negative BPMs, `#WARPS`, $> 8$ `#STOPS` / `#DELAYS`, or BPM changes exceeding $[50, 320]\text{ BPM}$.
3. Pathological chart filters (reusing `chartbench` lessons): $< 25$ notes, sustained NPS $> 22$, hold share $> 0.40$, $> 2\%$ 3+ panel hands, or `footflow` inverse Viterbi average transition discomfort exceeding $3\times$ tier ceiling (catches keyboard-only dump files masquerading as pad files!).
4. Deduplicate audio tracks by acoustic fingerprint / SHA-256 of decoded downsampled PCM so re-packaged songs across packs share the **exact same `song_id`** and land in the same `train` ($85\%$), `val` ($7.5\%$), or `test` ($7.5\%$) split.

### Training Commands & Hyperparameters
- **Stage A (`train_stage_a.py`):** AdamW (`lr=3e-4`, `weight_decay=1e-2`, cosine decay, $1000$-step linear warmup), mixed precision `bf16` (or `fp16`), random 32-measure crops ($T = 1536$ ticks), batch size $32$, gradient clip $1.0$, $40\text{--}60$ epochs.
- **Stage B (`train_stage_b.py`):** AdamW (`lr=3e-4`, `weight_decay=1e-2`, cosine decay), crop up to $N = 512$ active notes (full chart when $N \le 512$), batch size $64$, horizontal mirror augmentation (free via codec invariance + Stage A hidden invariance check), $50\text{--}80$ epochs.
- **ONNX Export (`export_onnx.py`):** Export `stage_a_rhythm.onnx` (dynamic `batch`, `ticks`) and `stage_b_choreo.onnx` (dynamic `batch`, `events`) at opset $\ge 17$, plus `model_meta.json` storing feature normalization stats, codec version `"1.0"`, and training metrics. Optionally emit `int8` dynamic-quantized `.onnx` variants.

**REQ-SFN-06 — Leak-free corpus ingestion & ONNX parity.**  
1. Two `.ssc` files pointing to identical audio in different folders receive the same `song_id` and never cross `train`/`val` splits.  
2. `export_onnx.py` verifies `np.allclose(pytorch_out, ort_out, atol=1e-4)` before writing `model_meta.json`.  
`Accept:` Unit tests run `corpus.py` on a 3-song miniature fixture pack and run `export_onnx.py` on randomly initialized tiny (`d_model=32, n_layers=1`) Stage A + Stage B models, confirming parity in `test_neural_onnx_parity.py`.

---

## 8. Workstream N6 — Optional Multimodal Modules (`style_rag.py` & `decisions_hook.py`)

Both modules are strictly optional enhancements behind clean interfaces (`None`-safe when disabled):

### 8.1 EmbeddingGemma 2 Phrase Style RAG (`style_rag.py`)
- Slices corpus tracks into 4-bar audio windows, embeds via **EmbeddingGemma 2** (`300M` audio encoder, truncated via Matryoshka Representation Learning to $256\text{D}$ `float16` L2-normalized vectors), and saves `phrase_index.npz` (`vectors: (M, 256) float16`, `styles: (M, 9) float16`).
- **Zero-FAISS Lookup:** Even $50,000$ 4-bar phrases occupy only $\approx 25\text{ MB}$ in RAM. Retrieval uses a single `numpy` matrix multiply (`scores = index_vectors @ query_vec`), returning the empirical human style distribution of the top-$k$ ($k=5$) acoustically matching 4-bar phrases in $<2\text{ ms}$ to guide local section density & style sliders!

### 8.2 Decisions API Scorer (`decisions_hook.py`)
- Targets any `/v1/decisions`-compatible endpoint (e.g., `Gemini Decision 1` / OpenRouter `decisions`).
- Sends track audio + section boundaries once to answer typed `choice` and `score` questions in a single prefill pass ($0$ output tokens):
  1. **Offline Corpus Filter (`--decisions-filter`):** Scores unlabeled community charts for pad musicality (`0..10`) to exclude junk charts prior to training.
  2. **Optional Critic / Macro-Preset Selector:** Can fill unconditioned style sliders per section in a single $\sim 200\text{ ms}$ call if requested via `--macro-planner decisions`.

---

## 9. Workstream N7 — BeatForge CLI & Tauri Desktop Integration

### 9.1 CLI Integration (`tools/beatforge/__main__.py` & `adapters/stepmania/`)
Extend `stepforge` without breaking existing flags:

```bash
# Generate full pack / song using the local ONNX neural model (no API key, <100ms after DSP)
PYTHONPATH=tools python3 -m beatforge stepforge \
  --track overdrive \
  --difficulties beginner,easy,medium,hard,challenge \
  --engine neural \
  [--style-preset itg_stamina|casual_flow|tech_footswitch|jumpstream] \
  [--model-dir models/stepforge_flow]

# Inpaint / re-roll bars 16..24 of an existing neural chart draft
PYTHONPATH=tools python3 -m beatforge stepforge-inpaint \
  --ssc build/stepmania/overdrive/overdrive.ssc \
  --difficulty hard \
  --bars 16:24 \
  [--seed 42]
```

Engine resolution rules for `--engine`:
- `neural`: Uses `StepManiaNeuralEngine` (`infer.py`). Raises a clear, actionable error naming `models/stepforge_flow/` if `.onnx` weights are not yet present.
- `llm`: Existing Gemini / Inkling / OpenAI designer + DP realizer path (unchanged).
- `deterministic`: Existing DSP-only deterministic path (unchanged).
- `auto`: Uses `neural` if `models/stepforge_flow/model_meta.json` exists, else falls back to `llm` (or `deterministic` when `--deterministic` is passed).

### 9.2 Tauri Desktop GUI Contract (Backend JSON Bridge)
Expose clean JSON subcommands / IPC responses for the Tauri + Vue/TS desktop app (`Overview.MD`):
1. `beatforge neural-status --json`: Returns `{ "installed": bool, "model_dir": str, "quantization": str, "style_rag_available": bool, "presets": [...] }`.
2. Custom style overrides via `--style-json '{"jump_share":0.12,"crossover_share":0.15,"stream_share":0.60}'` mapped directly to Tauri UI sliders.
3. Every neural run emits the exact same `.ssc`, `.sm`, `.qa.json`, `.preview.png`, and click-track `.ogg` artifacts under `build/stepmania/<track>/` as the LLM pipeline so existing UI previews and `chartbench` work out of the box!

**REQ-SFN-07 — Full pipeline & `chartbench` compatibility.** Every chart emitted by `--engine neural` passes `garcia/simfile` byte-stable round-trip, `barrysir/stepmania-parsing-code` referee check (when installed), and runs through `tools/chartbench/score.py` without schema changes.  
`Accept:` End-to-end CLI smoke test emits all 5 difficulties (`Beginner` through `Challenge`) with monotonic meters (`Beginner < Easy < Medium < Hard < Challenge`), zero orphan holds, legal row quantizations, and valid `.qa.json` reports.

---

## 10. Testing & Verification Matrix

Map all `REQ-SFN-*` requirements in `TEST-REQS` (if present) and implement offline, fast (`<10s` total) pytest modules under `tools/beatforge/tests/`:

| Test File | Covers | Key Assertions |
|---|---|---|
| `test_neural_grid.py` | `REQ-SFN-02` | 48-tick beat warping, variable BPM/stop handling, $\Delta t_k$ accuracy, Demucs vs. HPSS fallback equivalence |
| `test_neural_codec.py` | `REQ-SFN-03`, `REQ-SFN-04` | Lossless encode/decode round-trip, exact `L <-> R` mirror token invariance, hold-lock tracking, empirical style vector bounds |
| `test_neural_losses.py` | `REQ-SFN-05` | 1D Wasserstein CDF non-zero shift gradient, cardinality loss zero at exact count, ergonomic matrix symmetry |
| `test_neural_onnx_parity.py` | `REQ-SFN-06` | Tiny Stage A + Stage B PyTorch -> ONNX export, dynamic sequence lengths, `atol=1e-4` numerical match, song-ID leak check |
| `test_neural_infer.py` | `REQ-SFN-01`, `REQ-SFN-05`, `REQ-SFN-07` | Torch-blocked import isolation, threshold NPS calibration, constrained Viterbi legality (0 hold collisions, 0 spins), bar-range inpainting boundary preservation, `.ssc` serialization round-trip |

---

## 11. Order of Work (Milestones for the AI Coder)

Execute strictly in this order, running `PYTHONPATH=tools python3 -m pytest tools/beatforge/tests` after each milestone:

- **Milestone 1 — Biomechanical Codec & Conditioning (`codec.py`, `conditioning.py`, `test_neural_codec.py`):**  
  Inspect `footflow.py` and `grammar.py`. Implement the 20-token Foot-Relative Codec, hold-lock tracker, inverse `footflow` Viterbi chart encoder, and 10D style descriptor. Prove 100% lossless round-trip and mirror symmetry invariance in pytest before touching any audio or neural code.
- **Milestone 2 — Beat-Warped 48-Tick Feature Extractor (`grid_features.py`, `test_neural_grid.py`):**  
  Implement 48-tick timing map interpolation and the 64-channel unified HPSS + optional Demucs stem extractor. Verify transient alignment on synthetic click tracks.
- **Milestone 3 — Corpus Compiler (`train/corpus.py`, `train/dataset.py`):**  
  Build the multiprocessing `.sm`/`.ssc` pack scanner with gimmick/dump filtering, audio-hash song deduplication, train/val/test song splitting, stem dropout, and conditioning dropout.
- **Milestone 4 — Stage A & Stage B PyTorch Models + Losses + ONNX Export (`train/*.py`, `test_neural_losses.py`, `test_neural_onnx_parity.py`):**  
  Implement `RhythmNet` (Bi-Conformer + `AdaLN-Zero`), `ChoreoNet` (RoPE Masked Diffusion Transformer), Focal + Cardinality + 1D Wasserstein CDF loss, training CLIs, and `export_onnx.py` with tiny-model ONNX parity unit tests.
- **Milestone 5 — Pure ONNX Runtime Inference, Viterbi & Inpainting (`infer.py`, `test_neural_infer.py`):**  
  Implement the `torch`-free inference engine: subdivision masking, bisection density calibration, 10-step cosine unmasking, exact `footflow` + hold-lock Viterbi projection, and `inpaint_section`. Verify runtime isolation with `torch` blocked.
- **Milestone 6 — CLI, Optional RAG/Decisions Hooks & Documentation (`__main__.py`, `style_rag.py`, `decisions_hook.py`, `README.md`):**  
  Wire `--engine neural`, `stepforge-inpaint`, and `neural-status --json` into the BeatForge CLI, add optional EmbeddingGemma 2 / Decisions API adapters, run the full existing + new `pytest` suite to verify zero regressions, and document training + Tauri usage in `README.md`.