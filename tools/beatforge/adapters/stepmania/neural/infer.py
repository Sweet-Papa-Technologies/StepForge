"""infer.py — StepForge Flow at runtime: numpy + onnxruntime only (spec §6.3,
§6.5, REQ-SFN-01/05/07). PyTorch is never imported here.

    features X ──► RhythmNet (ONNX) ──► decode_rhythm ──► which ticks carry a note,
                                                          jump / hold / mine kinds
    notes ──► ChoreoNet (ONNX), 10 unmasking steps ──► unary log-probs per token
          ──► constrained_viterbi ──► foot-relative tokens ──► {L, D, U, R}

The two networks propose; this module disposes. Everything a tier forbids —
subdivisions it may not use, notes-per-second it may not exceed, crossovers and
footswitches below Hard, any unforced double-step below Hard, moving a foot
that is holding a freeze, a 180-degree spin at any tier — is enforced here by
construction, in the decoders, not hoped for from the weights. A chart leaves
this module legal whatever the logits were.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..grammar import BUDGETS
from . import MODEL_DIR_NAME
from . import codec as C
from . import conditioning as cond
from . import event_features as EF
from . import grid_features as GF

META_FILE = "model_meta.json"
ENV_MODEL_DIR = "BEATFORGE_NEURAL_MODEL_DIR"
WINDOW_TICKS = 32 * C.G             # the crop both stages were trained on
WINDOW_HOP = WINDOW_TICKS // 2
EVENT_WINDOW = 512
EVENT_HOP = 256
UNMASK_STEPS = 10
DEFAULT_TAU = 0.6
THETA_RANGE = (0.15, 0.85)
THETA_MAX_TILTED = 0.995
BISECTION_ITERS = 12
LAMBDA_COST = 0.04                  # nats per unit of foot-model discomfort
#: How far, in nats, the Viterbi prefers the token the sampler committed over
#: ANY alternative at that note (`committed_unary`). 2.5 nats is ~62 units of
#: foot-model discomfort at LAMBDA_COST: a sampled crossover (4-8), jack (<=16)
#: or D/U footswitch (24-60) is kept; an unforced double-step (85-90) or a slow
#: side footswitch is overridden; hard rules override regardless.
#:
#: This replaced "add 1.5 to the sampled token's log-probability", which could
#: not do its job: a sampled crossover has log p around -4, so +1.5 still left
#: it far below the mode and the search reverted every rare token. Measured on
#: held-out Hard charts, crossovers came out at 0.0% against 4.4% in the human
#: charts — whatever the sampler or the sliders asked for.
COMMIT_BONUS = 2.5
#: How much a note's confidence counts in deciding WHEN it is committed during
#: unmasking, against pure chance. 0 = notes are committed in random order,
#: which is plain ancestral sampling of the masked-diffusion model and the only
#: setting that reproduces what the model learned:
#:
#:     share of single steps      HOME  DOWN  UP    CROSS | jack/double-step
#:     human charts (held out)    .388  .245  .220  .034  | .112
#:     random order (0.0)         .372  .249  .239  .042  | .098
#:     confidence-first (MaskGIT) .883  .059  .043  .000  | .015
#:
#: Most-confident-first is a selection effect, not a neutral schedule: a note
#: drawn as the common token is always more "confident" than one drawn as a
#: crossover, so the rare ones are re-masked and re-drawn until they come up
#: common, and a Hard chart collapses into left-right-left-right.
CONFIDENCE_WEIGHT = 0.0
FIRST_PLAYABLE_BEAT = 4.0
PLAYABLE_TAIL_S = 1.4
NEG = -1e9


class ModelNotInstalled(RuntimeError):
    """Raised when the neural engine is asked to run without usable weights."""


# --------------------------------------------------------------------------- #
# Model directory / status
# --------------------------------------------------------------------------- #
def default_model_dir() -> Path:
    from .... import config
    return config.REPO_ROOT / "models" / MODEL_DIR_NAME


def resolve_model_dir(model_dir=None) -> Path:
    """--model-dir, else $BEATFORGE_NEURAL_MODEL_DIR, else models/stepforge_flow."""
    if model_dir:
        return Path(model_dir).expanduser()
    env = os.environ.get(ENV_MODEL_DIR, "").strip()
    return Path(env).expanduser() if env else default_model_dir()


def is_installed(model_dir=None) -> bool:
    return (resolve_model_dir(model_dir) / META_FILE).exists()


def _onnxruntime_error() -> str | None:
    """None when onnxruntime imports, else the real failure ("ImportError: DLL
    load failed while importing onnxruntime_pybind11_state: …"). The reason is
    the useful part: "not importable" covers a missing package, a missing system
    library and a CPU the build cannot run on, which have nothing in common."""
    try:
        import onnxruntime  # noqa: F401
        return None
    except Exception as e:                       # noqa: BLE001 — reported, not raised
        msg = str(e).strip()
        return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def _have_onnxruntime() -> bool:
    return _onnxruntime_error() is None


STYLE_FIELDS = [
    {"key": "nps", "label": "Notes per second", "min": 0.5, "max": 16.0, "step": 0.25},
    {"key": "stream_share", "label": "Streams", "min": 0.0, "max": 1.0, "step": 0.01},
    {"key": "jump_share", "label": "Jumps", "min": 0.0, "max": 0.35, "step": 0.01},
    {"key": "hold_share", "label": "Holds", "min": 0.0, "max": 0.35, "step": 0.01},
    {"key": "crossover_share", "label": "Crossovers", "min": 0.0, "max": 0.30, "step": 0.01},
    {"key": "jack_share", "label": "Jacks", "min": 0.0, "max": 0.20, "step": 0.01},
    {"key": "footswitch_share", "label": "Footswitches", "min": 0.0, "max": 0.20,
     "step": 0.01},
    {"key": "mine_share", "label": "Mines", "min": 0.0, "max": 0.15, "step": 0.01},
]


def neural_status(model_dir=None) -> dict:
    """The `beatforge neural-status --json` payload. Never raises."""
    d = resolve_model_dir(model_dir)
    ort_error = _onnxruntime_error()
    # `installed` means "a neural run can start" (weights AND onnxruntime).
    # `weights_present` and `onnxruntime_error` say which half is missing, so a
    # complete model next to a broken onnxruntime is not reported as "no model".
    out = {"installed": False, "weights_present": False, "model_dir": str(d),
           "default_model_dir": str(default_model_dir()), "quantization": "none",
           "style_rag_available": False, "presets": list(cond.PRESET_NAMES),
           "codec_version": None, "trained_with_stems": False, "params": None,
           "version": None, "display_name": None,
           "metrics": {}, "style_fields": STYLE_FIELDS,
           "difficulties": list(BUDGETS), "onnxruntime": ort_error is None,
           "onnxruntime_error": ort_error, "error": None}
    try:
        from . import style_rag
        out["style_rag_available"] = style_rag.index_path(d).exists()
    except Exception:
        pass
    meta_path = d / META_FILE
    if not meta_path.exists():
        out["error"] = (f"StepForge Flow weights are not installed: {meta_path} not "
                        f"found. Train and export them (README, 'Neural engine') or "
                        f"point --model-dir / ${ENV_MODEL_DIR} at an exported model.")
        return out
    try:
        meta = json.loads(meta_path.read_text())
        missing = [f for f in (meta["files"]["stage_a"], meta["files"]["stage_b"])
                   if not (d / f).exists()]
        if missing:
            out["error"] = f"model_meta.json names missing file(s): {', '.join(missing)}"
            return out
        out["weights_present"] = True
        _check_compat(meta)
        out.update(quantization=meta.get("quantization", "fp32"),
                   codec_version=meta.get("codec_version"),
                   version=meta.get("version"), display_name=meta.get("display_name"),
                   trained_with_stems=bool(meta.get("trained_with_stems")),
                   params={"stage_a": meta["stage_a"]["params"],
                           "stage_b": meta["stage_b"]["params"]},
                   metrics=meta.get("metrics", {}))
        if not out["onnxruntime"]:
            out["error"] = ("The model files are present, but onnxruntime (which runs "
                            f"them) did not load: {ort_error}")
            return out
        out["installed"] = True
    except Exception as e:
        out["error"] = f"model_meta.json is unusable: {e}"
    return out


def _check_compat(meta: dict) -> None:
    """A model exported against a different codec or feature layout would run
    and produce confident nonsense. Refuse it instead."""
    if meta.get("codec_version") != C.CODEC_VERSION:
        raise ValueError(f"model codec {meta.get('codec_version')} != runtime "
                         f"{C.CODEC_VERSION}")
    f = meta.get("features", {})
    want = {"sr": GF.SR, "n_fft": GF.N_FFT, "hop": GF.HOP, "d_in": GF.D_IN,
            "ticks_per_measure": C.G, "ctx_k": EF.CTX_K, "n_scalars": EF.N_SCALARS,
            "cond_dim": cond.COND_DIM}
    bad = {k: (f.get(k), v) for k, v in want.items() if f.get(k) != v}
    if bad:
        raise ValueError(f"feature layout mismatch (model, runtime): {bad}")


class FlowModel:
    """The two ONNX sessions plus their metadata."""

    def __init__(self, model_dir=None, int8: bool = False, threads: int | None = None):
        self.dir = resolve_model_dir(model_dir)
        meta_path = self.dir / META_FILE
        if not meta_path.exists():
            raise ModelNotInstalled(
                f"The neural engine has no weights: {meta_path} does not exist.\n"
                f"Expected an exported StepForge Flow model in {self.dir}/ "
                f"(stage_a_rhythm.onnx, stage_b_choreo.onnx, {META_FILE}).\n"
                f"Train and export one (see README 'Neural engine'), pass "
                f"--model-dir, or use --engine llm / --engine deterministic.")
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ModelNotInstalled(
                "onnxruntime, which runs the model, did not load: "
                f"{type(e).__name__}: {e}") from e
        self.meta = json.loads(meta_path.read_text())
        _check_compat(self.meta)
        files = self.meta["files"]
        a = files.get("stage_a_int8") if int8 else None
        b = files.get("stage_b_int8") if int8 else None
        self.quantization = "int8" if (int8 and a and b) else "fp32"
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if threads:
            opts.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.sess_a = ort.InferenceSession(str(self.dir / (a or files["stage_a"])),
                                           opts, providers=prov)
        self.sess_b = ort.InferenceSession(str(self.dir / (b or files["stage_b"])),
                                           opts, providers=prov)
        self.trained_with_stems = bool(self.meta.get("trained_with_stems"))

    # -- Stage A ---------------------------------------------------------- #
    def rhythm(self, x: np.ndarray, cond19: np.ndarray) -> dict:
        """X (T, 64) -> per-tick logits, run in overlapping 32-measure windows
        (the length it was trained on) and cross-faded."""
        t = len(x)
        names = ("onset", "jump", "hold_type", "hold_dur", "mine")
        c = cond19[None, :].astype(np.float32)
        if t <= WINDOW_TICKS:
            out = self.sess_a.run(None, {"x": x[None].astype(np.float32), "cond": c})
            return {n: o[0] for n, o in zip(names, out)}
        starts = list(range(0, t - WINDOW_TICKS, WINDOW_HOP)) + [t - WINDOW_TICKS]
        ramp = np.minimum(np.arange(1, WINDOW_TICKS + 1), np.arange(WINDOW_TICKS, 0, -1))
        w = (ramp / ramp.max()).astype(np.float32) + 1e-3
        acc = None
        wsum = np.zeros(t, np.float32)
        for a in starts:
            out = self.sess_a.run(None, {"x": x[None, a:a + WINDOW_TICKS].astype(np.float32),
                                         "cond": c})
            if acc is None:
                acc = [np.zeros((t,) + o.shape[2:], np.float32) for o in out[:5]]
            for buf, o in zip(acc, out[:5]):
                buf[a:a + WINDOW_TICKS] += o[0] * (w if o.ndim == 2 else w[:, None])
            wsum[a:a + WINDOW_TICKS] += w
        return {n: buf / (wsum if buf.ndim == 1 else wsum[:, None])
                for n, buf in zip(names, acc)}

    # -- Stage B ---------------------------------------------------------- #
    def choreo(self, ctx: np.ndarray, scalars: np.ndarray, tokens: np.ndarray,
               cond19: np.ndarray) -> np.ndarray:
        """(N, K, 64), (N, 31), (N,) -> logits (N, 20), in overlapping windows
        of at most 512 notes."""
        n = len(tokens)
        c = cond19[None, :].astype(np.float32)

        def run(a, b):
            return self.sess_b.run(None, {
                "ctx": ctx[None, a:b].astype(np.float32),
                "scalars": scalars[None, a:b].astype(np.float32),
                "tokens": tokens[None, a:b].astype(np.int64), "cond": c})[0][0]

        if n <= EVENT_WINDOW:
            return run(0, n)
        starts = list(range(0, n - EVENT_WINDOW, EVENT_HOP)) + [n - EVENT_WINDOW]
        ramp = np.minimum(np.arange(1, EVENT_WINDOW + 1), np.arange(EVENT_WINDOW, 0, -1))
        w = (ramp / ramp.max()).astype(np.float32) + 1e-3
        acc = np.zeros((n, C.N_TOKENS), np.float32)
        wsum = np.zeros(n, np.float32)
        for a in starts:
            acc[a:a + EVENT_WINDOW] += run(a, a + EVENT_WINDOW) * w[:, None]
            wsum[a:a + EVENT_WINDOW] += w
        return acc / wsum[:, None]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def log_softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


# --------------------------------------------------------------------------- #
# Tier rules
# --------------------------------------------------------------------------- #
#: Ticks a tier may place a note on, as allowed `tick % 12` residues.
_SUBDIVISIONS = {
    "beginner": (0,),                               # 4ths
    "easy": (0, 6),                                 # 8ths
    "medium": (0, 6),                               # 8ths
    "hard": (0, 3, 4, 6, 8, 9),                     # 16ths + triplet 8ths
    "challenge": (0, 3, 4, 6, 8, 9),                # the same; 24ths are opt-in
}
#: 24th notes (two ticks apart) for Challenge. Off by default: mixed in with
#: 16ths they read as tight 32nd-like bursts in play.
SUBDIVISIONS_24THS = (0, 2, 3, 4, 6, 8, 9, 10)
ALLOW_24THS = False
_MAX_NOTES_IN_HOLD = {"beginner": 0, "easy": 1, "medium": 2, "hard": 4, "challenge": 8}
_JUMP_TICKS = {"downbeats": 48, "accents": 24, "free": 12}
_JUMP_MIN_SPACING_BEATS = {"downbeats": 4.0, "accents": 2.0, "free": 1.0}
_JUMP_MAX_FRAC = {"downbeats": 0.06, "accents": 0.09, "free": 0.12}
#: With the jump / freeze sliders left to the model, its own estimate of the
#: share is multiplied by these. v1.0 wrote 4.6% jumps and 4.2% freezes where
#: human charts of the same songs have 7.3% and 6.1%.
JUMP_SHARE_BOOST = 1.25
HOLD_SHARE_BOOST = 1.25
#: While one foot is pinned on a freeze the other plays every note inside it.
#: Two of those closer together than this is a one-footed jack or double-step,
#: so the freeze is released before the second.
PINNED_MIN_GAP_S = 0.45
MINE_CLEARANCE_TICKS = C.TICKS_PER_BEAT    # no mine within a beat of a 16th run
HOLD_RELEASE_TICKS = 3              # a freeze ends at least a 16th before a note
                                    # the holding foot must not coincide with


#: Tiers with no jack budget (Beginner, Easy) may still step on the same arrow
#: again when it is slow: never faster than FAST_REPEAT_S, freely from
#: SLOW_REPEAT_S (the same thresholds as footflow's JACK_FULL_S / JACK_FREE_S),
#: with REPEAT_PENALTY nats fading out in between, and at most SLOW_REPEAT_RUN
#: identical rows in a row.
FAST_REPEAT_S = 0.35
SLOW_REPEAT_S = 0.65
REPEAT_PENALTY = 3.0
SLOW_REPEAT_RUN = 3
#: Hard and Challenge may footswitch, and the model overdoes it: 0.9% / 1.3% of
#: steps where human charts have 0.1%, above the human range on half the
#: Challenge charts. Each footswitch quicker than FOOTSWITCH_SLOW_S costs this
#: many nats unless the footswitch slider asks for them.
FOOTSWITCH_PENALTY = 2.5
FOOTSWITCH_SLOW_S = 0.60


@dataclass(frozen=True)
class TierRules:
    difficulty: str
    no_cross: bool               # no step may end with the feet crossed
    no_footswitch: bool
    no_double_step: bool         # an unforced same-foot move to a new panel
    panel_run_limit: int         # longest run of identical consecutive rows
    jump_holds: bool             # may a jump carry a freeze
    cross_scale: float           # multiplier on crossover discomfort
    slow_repeats_only: bool = False   # no jack budget: an arrow may repeat only slowly

    @classmethod
    def for_difficulty(cls, difficulty: str) -> "TierRules":
        b = BUDGETS[difficulty]
        # No tier gets an unforced double-step. Human Hard/Challenge charts do
        # contain them (~0.5% of steps) and the model will sample them, but they
        # are what chart reviewers flag first, and on the benchmark they were the
        # one playability measure where the neural engine trailed DDC.
        slow = b.jack_limit <= 0
        return cls(difficulty, b.crossover == "none", not b.footswitch,
                   True, SLOW_REPEAT_RUN if slow else max(1, b.jack_limit),
                   difficulty in ("hard", "challenge"),
                   {"none": 1.0, "light": 3.0, "moderate": 1.0}.get(b.crossover, 1.0),
                   slow)


def min_gap_s(difficulty: str) -> float:
    """Shortest physical gap between notes for a tier: half the period of its
    peak sustained rate. Keeps a fast song from handing an Easy chart 8ths that
    are faster than a Hard chart's 16ths on a slow one."""
    return 1.0 / (2.0 * BUDGETS[difficulty].max_nps_4s)


# --------------------------------------------------------------------------- #
# Stage A decode
# --------------------------------------------------------------------------- #
@dataclass
class RhythmPlan:
    tick: np.ndarray            # (N,) int64 note ticks, ascending
    is_jump: np.ndarray         # (N,) bool
    flavor: np.ndarray          # (N,) int: 0 none, 1 HOLD_ALT .. 5 ROLL_SAME
    dur: np.ndarray             # (N,) int ticks (0 for taps)
    mine_tick: np.ndarray       # (M,) int64
    stats: dict = field(default_factory=dict)


def _nms(order: np.ndarray, times: np.ndarray, gap_s: float) -> np.ndarray:
    """Greedy pick in the given order, dropping anything closer than `gap_s`
    to an already-picked tick. Returns the survivors in pick order."""
    n = len(times)
    blocked = np.zeros(n, bool)
    kept = []
    for k in order:
        if blocked[k]:
            continue
        kept.append(k)
        lo = np.searchsorted(times, times[k] - gap_s + 1e-9, side="left")
        hi = np.searchsorted(times, times[k] + gap_s - 1e-9, side="right")
        blocked[lo:hi] = True
    return np.asarray(kept, np.int64)


def _thin_to_peak_nps(ticks, scores, times, max_nps: float, window_s: float = 4.0):
    """Drop the weakest note of any 4 s window that exceeds the tier's rate,
    until none does."""
    keep = np.ones(len(ticks), bool)
    limit = int(math.floor(max_nps * window_s + 1e-9))
    t = times[ticks]
    while True:
        idx = np.nonzero(keep)[0]
        tt = t[idx]
        hi = np.searchsorted(tt, tt + window_s, side="left")
        cnt = hi - np.arange(len(idx))
        over = np.nonzero(cnt > limit)[0]
        if len(over) == 0:
            return keep
        i = over[int(np.argmax(cnt[over]))]
        win = idx[i:hi[i]]
        weakest = win[scores[win] <= scores[win].min() + 1e-12]
        if len(weakest) > 1:
            # equal scores: take the most crowded one, so a run of ties is
            # thinned evenly instead of from one end
            pos = np.searchsorted(idx, weakest)
            before = tt[pos] - tt[np.maximum(pos - 1, 0)]
            after = tt[np.minimum(pos + 1, len(idx) - 1)] - tt[pos]
            before[pos == 0] = np.inf
            after[pos == len(idx) - 1] = np.inf
            weakest = weakest[[int(np.argmin(np.minimum(before, after)))]]
        keep[weakest[0]] = False


def _cap_runs(ticks, scores, keep, max_run: int):
    """Break runs of 8th-or-faster notes longer than `max_run` by removing the
    weakest note of each over-long stretch."""
    changed = True
    while changed:
        changed = False
        idx = np.nonzero(keep)[0]
        tk = ticks[idx]
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and tk[j + 1] - tk[j] <= 6:
                j += 1
            if j - i + 1 > max_run:
                seg = idx[i:i + max_run + 1]
                inner = seg[1:-1] if len(seg) > 2 else seg
                keep[inner[int(np.argmin(scores[inner]))]] = False
                changed = True
                break
            i = j + 1
    return keep


#: The off-beat cap may thin a chart, not erase it. If the beat grid is half a
#: beat out of phase (a DSP tempo fit can be), the music's on-beats land on
#: 8th ticks and nearly every note counts as "fine"; honouring the cap to the
#: letter would then delete the chart. It removes at most this share of notes.
MAX_FINE_CAP_LOSS = 0.4


def _cap_fine_share(ticks, scores, keep, fine_frac: float):
    """Pull the share of notes off the quarter grid toward `fine_frac` by
    dropping the weakest of them, removing at most MAX_FINE_CAP_LOSS of the
    chart. Returns (keep, met) — `met` is False if the cap could not be
    reached within that limit."""
    idx = np.nonzero(keep)[0]
    fine = idx[ticks[idx] % C.TICKS_PER_BEAT != 0]
    quarters = len(idx) - len(fine)
    allowed = (len(fine) if fine_frac >= 0.999
               else int(math.floor(fine_frac / (1 - fine_frac) * quarters + 1e-9)))
    excess = len(fine) - allowed
    if excess <= 0:
        return keep, True
    n_drop = min(excess, int(MAX_FINE_CAP_LOSS * len(idx)))
    drop = fine[np.argsort(scores[fine], kind="stable")[:n_drop]]
    keep[drop] = False
    return keep, n_drop == excess


def _one_grid_per_beat(ticks, scores, keep):
    """A beat is counted in 16ths or in triplets, not both. Left to the scores
    a beat can take a triplet 8th (tick 4) next to an 8th (tick 6): two ticks
    apart, a 24th-note gap in the middle of an otherwise 16th-note chart, which
    plays as a stray micro-burst. Keep whichever family carries more score in
    that beat and drop the other's notes."""
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return keep
    res = ticks[idx] % C.TICKS_PER_BEAT
    beat = ticks[idx] // C.TICKS_PER_BEAT
    duple = np.isin(res, (3, 6, 9))
    triple = np.isin(res, (2, 4, 8, 10))
    for b in np.unique(beat[duple]):
        m = beat == b
        if not (triple & m).any():
            continue
        d, t3 = idx[m & duple], idx[m & triple]
        keep[t3 if scores[d].sum() >= scores[t3].sum() else d] = False
    return keep


#: 16th-stream commitment (Challenge, and Hard when asked for meter 9+). Human
#: charts answer a song's climax with whole bars of 16ths; left to the scores,
#: Flow wrote runs with one to five 16ths missing and not a single full bar in
#: the v1.0 benchmark (humans: 0.4% of Hard/Challenge measures). A bar in the
#: loudest STREAM_COMMIT_ENERGY of the song that already has STREAM_COMMIT_MIN
#: of its sixteen 16ths gets the rest, where the model sees any onset at all.
STREAM_COMMIT_MIN = 12
STREAM_COMMIT_ENERGY = 0.80
STREAM_COMMIT_MIN_P = 0.03
STREAM_COMMIT_HARD_METER = 9


def _commit_streams(ticks: np.ndarray, p: np.ndarray, allowed: np.ndarray,
                    rank: np.ndarray, tick_times: np.ndarray, max_nps: float,
                    gap_s: float) -> tuple:
    """Fill the missing 16ths of near-complete bars in the loudest part of the
    song. Returns (ticks, bars filled)."""
    have = np.zeros(len(p), bool)
    have[ticks] = True
    filled = 0
    limit = int(math.floor(max_nps * 4.0 + 1e-9))
    for m in np.nonzero(rank >= STREAM_COMMIT_ENERGY)[0]:
        slots = m * C.G + 3 * np.arange(16)
        if slots[-1] >= len(p):
            continue
        active = have[slots]
        if not STREAM_COMMIT_MIN <= active.sum() < 16:
            continue
        missing = slots[~active]
        bar = np.arange(m * C.G, (m + 1) * C.G)
        if have[bar].sum() != active.sum():              # triplets in this bar
            continue
        if not allowed[missing].all() or (p[missing] < STREAM_COMMIT_MIN_P).any():
            continue
        if tick_times[slots[1]] - tick_times[slots[0]] < gap_s:   # too fast for the tier
            continue
        trial = have.copy()
        trial[missing] = True
        tk = np.nonzero(trial)[0]
        tt = tick_times[tk]
        lo = np.searchsorted(tt, tick_times[slots[0]] - 4.0, side="left")
        hi = np.searchsorted(tt, tick_times[slots[-1]] + 4.0, side="right")
        seg = tt[lo:hi]
        if (np.searchsorted(seg, seg + 4.0, side="left") - np.arange(len(seg))).max() > limit:
            continue                                     # would break the tier's peak rate
        have = trial
        filled += 1
    return np.nonzero(have)[0], filled


#: How hard the onset score leans toward the song's louder bars: per tier, the
#: tilt strengths tried in order (multipliers on the odds are exp(w * (rank -
#: 1/2)), rank being the bar's energy rank in the song) and the Spearman (notes
#: per bar vs bar energy) at which to stop. The project gate is 0.30. On the
#: dev songs these raised the correlation from 0.60 to 0.71 for 0.002 of
#: placement F1; pushing further (every tier at 3-6) reached 0.73 but began to
#: cost Hard and Challenge half a point of agreement with the human chart.
ENERGY_TILT = {
    "beginner": ((4.0, 5.0, 6.0), 0.75),
    "easy": ((4.0, 5.0, 6.0), 0.75),
    "medium": ((4.0, 5.0, 6.0), 0.75),
    "hard": ((3.0, 4.0, 5.0), 0.70),
    "challenge": ((2.2, 3.0, 4.0), 0.65),
}
ENERGY_WEIGHTS = (0.7, 1.4, 2.2)      # fallback for a tier not listed above
MIN_ENERGY_CORR = 0.36


#: At most this share of a chart's notes may sit on a tick with no detected
#: transient nearby, when the caller supplies `on_transient`. The project's QA
#: gate asks for >= 90% of notes within 35 ms of a DSP onset; 8% leaves margin.
MAX_OFF_TRANSIENT = 0.08
OFF_TRANSIENT_WEIGHT = 0.5          # score multiplier for ticks with no transient


def _cap_off_transient(ticks, scores, keep, on_transient):
    """Drop the weakest notes that have no transient under them until they are
    at most MAX_OFF_TRANSIENT of the chart (never more than MAX_FINE_CAP_LOSS
    of it). Returns (keep, met)."""
    idx = np.nonzero(keep)[0]
    off = idx[~on_transient[ticks[idx]]]
    on = len(idx) - len(off)
    allowed = int(math.floor(MAX_OFF_TRANSIENT / (1 - MAX_OFF_TRANSIENT) * on + 1e-9))
    excess = len(off) - allowed
    if excess <= 0:
        return keep, True
    n_drop = min(excess, int(MAX_FINE_CAP_LOSS * len(idx)))
    keep[off[np.argsort(scores[off], kind="stable")[:n_drop]]] = False
    return keep, n_drop == excess


#: Default density, as a multiple of the note count RhythmNet itself expects,
#: used when the caller leaves the NPS slider to the model. Left alone the model
#: wrote more notes than human charters do on the same song (v1.0 benchmark,
#: median 1.26x at Beginner); these bring the median to 1.0x.
DENSITY_SCALE = {"beginner": 0.80, "easy": 1.0, "medium": 0.95, "hard": 0.94,
                 "challenge": 0.88}

#: Metrical prior on the onset score for the two lowest tiers, as multipliers
#: for (beats 1 and 3, beats 2 and 4, 8th off-beats, anything finer).
METRIC_PRIOR = {}


def _scale_odds(p: np.ndarray, m) -> np.ndarray:
    """p with its odds multiplied by m: stays strictly inside (0, 1) and keeps
    the order of any two notes that share a multiplier."""
    pm = p * m
    return pm / (1.0 - p + pm)


def _metric_weight(n_ticks: int, difficulty: str) -> np.ndarray | None:
    w = METRIC_PRIOR.get(difficulty)
    if not w:
        return None
    phase = np.arange(n_ticks) % C.G
    out = np.full(n_ticks, float(w[3]))
    out[phase % 6 == 0] = w[2]
    out[phase % 12 == 0] = w[1]
    out[phase % 24 == 0] = w[0]
    return out


def decode_rhythm(logits: dict, tick_times: np.ndarray, difficulty: str,
                  style: np.ndarray, duration_s: float,
                  energy_per_bar: np.ndarray | None = None,
                  on_transient: np.ndarray | None = None) -> RhythmPlan:
    """RhythmNet logits -> the notes of the chart (spec §6.3).

    1. mask ticks the tier may not use, and anything outside the playable window
    2. peak-pick with the tier's minimum physical gap
    3. choose the probability threshold by 12 bisection steps so the sustained
       note rate matches the NPS slider — or, with the slider left to the
       model, so the note count matches the model's own expected count
    4. enforce the tier's peak rate, stream length and off-beat share
    5. choose which notes are jumps, freezes and (if asked for) mines, inside
       the tier's shares, and trim freezes so they can always be stood on

    `on_transient`, a bool per tick, marks ticks with a detected onset under
    them; given it, notes on other ticks are held to MAX_OFF_TRANSIENT of the
    chart, so the result is on the music rather than on the grid.
    """
    b = BUDGETS[difficulty]
    t = len(tick_times)
    k = np.arange(t)
    p = sigmoid(logits["onset"].astype(np.float64))
    subdiv = SUBDIVISIONS_24THS if (difficulty == "challenge" and ALLOW_24THS) \
        else _SUBDIVISIONS[difficulty]
    allowed = np.isin(k % C.TICKS_PER_BEAT, subdiv)
    allowed &= k >= int(FIRST_PLAYABLE_BEAT * C.TICKS_PER_BEAT)
    allowed &= tick_times <= duration_s - PLAYABLE_TAIL_S
    allowed &= tick_times >= 0.0

    lo0, hi0 = THETA_RANGE
    want_nps = float(style[1]) * 16.0 if style[1] >= 0 else None
    expected = float(p[allowed].sum()) * DENSITY_SCALE.get(difficulty, 1.0)
    metric = _metric_weight(t, difficulty)
    if want_nps is not None:
        want_nps = min(want_nps, b.max_nps_4s)
    playable_s = max(0.0, duration_s - PLAYABLE_TAIL_S
                     - float(tick_times[min(t - 1, int(FIRST_PLAYABLE_BEAT * 12))]))
    cand = np.nonzero(allowed)[0]
    energy = None
    if energy_per_bar is not None and len(energy_per_bar) > 2:
        energy = np.asarray(energy_per_bar, np.float64)
        rank = np.argsort(np.argsort(energy)) / max(1, len(energy) - 1)     # 0..1
        bar = np.minimum(k // C.G, len(energy) - 1)

    def select(weight: float):
        """Peak-pick and calibrate with the onset probability tilted toward
        the song's louder bars by exp(weight * (rank - 1/2))."""
        # The tilt multiplies the ODDS, not the probability. Multiplying the
        # probability and clipping at 1 made every confident note of a loud bar
        # score exactly 1.0, and the limits below then had nothing to rank
        # them by: thinning to the tier's rate deleted them by position.
        score = p if energy is None else _scale_odds(
            p, np.exp(weight * (rank[bar] - 0.5)))
        if metric is not None:
            score = _scale_odds(score, metric)
        if on_transient is not None:
            # prefer ticks with a transient under them, so the cap below rarely
            # has to delete anything
            score = np.where(on_transient, score, OFF_TRANSIENT_WEIGHT * score)
        score = np.where(allowed, score, 0.0)
        order = cand[np.argsort(-score[cand], kind="stable")]
        picked = _nms(order, tick_times, min_gap_s(difficulty))   # by falling score
        pscore = score[picked]

        def chosen(theta: float):
            """The chart at threshold `theta`, AFTER the tier's own limits
            (peak rate, off-beat share, stream length, transients).
            Calibrating on the capped chart is what makes the density target
            mean the density that ships: calibrating first and capping
            afterwards left Hard and Challenge 15% under the rate asked for."""
            tk = np.sort(picked[pscore >= theta])
            if len(tk) == 0:
                return tk, True
            keep = _thin_to_peak_nps(tk, score[tk], tick_times, b.max_nps_4s)
            keep = _one_grid_per_beat(tk, score[tk], keep)
            keep, met = _cap_fine_share(tk, score[tk], keep, b.fine_frac)
            keep = _cap_runs(tk, score[tk], keep, b.max_run)
            if on_transient is not None:
                keep, on_met = _cap_off_transient(tk, score[tk], keep, on_transient)
                met = met and on_met
            return tk[keep], met

        def miss(theta: float) -> float:                   # > 0: too dense
            tk, _ = chosen(theta)
            if want_nps is not None:
                return cond.sustained_nps(tick_times[tk]) - want_nps
            return float(len(tk)) - expected

        # The range is a statement about the model's own probability; a score
        # tilted toward the loud bars routinely exceeds its top end, and the
        # density target must still be reachable, so the ceiling opens up.
        lo, hi = lo0, (THETA_MAX_TILTED if energy is not None or metric is not None
                       else hi0)
        if miss(lo) <= 0:
            theta = lo
        elif miss(hi) >= 0:
            theta = hi
        else:
            for _ in range(BISECTION_ITERS):
                mid = 0.5 * (lo + hi)
                if miss(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            theta = 0.5 * (lo + hi)
        tk, met = chosen(theta)
        # A chart with next to nothing in it is not a sparse chart, it is a
        # failed one: if even the bottom of the threshold range selects almost
        # no notes (weights that barely fire on this audio), take the strongest
        # peaks down to a quarter of a note per second and say so in the stats.
        floor = min(len(picked), max(8, int(0.25 * playable_s)))
        below = len(tk) < floor
        if below:
            tk = np.sort(picked[:floor])
            theta = float(pscore[floor - 1]) if floor else theta
        return tk, theta, met, below

    def energy_corr(tk) -> float:
        """Spearman of notes-per-bar against bar energy — the QA gate's own
        statistic (qa._density_energy)."""
        from scipy.stats import spearmanr
        counts = np.bincount(np.minimum(tk // C.G, len(energy) - 1), minlength=len(energy))
        rho = spearmanr(counts, energy)[0]
        return float(rho) if rho == rho else 1.0

    # Tilt gently by default; lean harder only if the chart would otherwise not
    # follow the song's energy (the project's density-vs-energy gate).
    best = None
    weights, enough = ENERGY_TILT.get(difficulty, (ENERGY_WEIGHTS, MIN_ENERGY_CORR))
    for weight in weights if energy is not None else (0.0,):
        tk, theta, met, below = select(weight)
        corr = energy_corr(tk) if energy is not None and len(tk) else None
        if best is None or (corr is not None and corr > best[4]):
            best = (tk, theta, met, below, corr if corr is not None else 1.0, weight)
        if corr is None or corr >= enough:
            break
    ticks, theta, fine_met, below_range, corr, weight = best
    n_calibrated = len(ticks)
    stream_bars = 0
    meter = float(style[0]) * 15.0
    if energy is not None and len(ticks) and (
            difficulty == "challenge"
            or (difficulty == "hard" and meter >= STREAM_COMMIT_HARD_METER - 1e-6)):
        bar_rank = np.zeros(t // C.G + 2)
        bar_rank[:len(rank)] = rank
        ticks, stream_bars = _commit_streams(ticks, p, allowed, bar_rank, tick_times,
                                             b.max_nps_4s, min_gap_s(difficulty))
    n = len(ticks)
    stats = {"theta": round(float(theta), 4), "calibrated": n_calibrated, "notes": n,
             "below_threshold_range": bool(below_range), "fine_share_met": bool(fine_met),
             "target_nps": want_nps, "expected_count": round(expected, 1),
             "energy_weight": weight, "stream_bars": stream_bars,
             "energy_corr": None if energy is None else round(corr, 3)}
    if n == 0:
        z = np.zeros(0, np.int64)
        return RhythmPlan(z, z.astype(bool), z, z, z, stats)

    # ---- jumps ----------------------------------------------------------- #
    pj = sigmoid(logits["jump"][ticks].astype(np.float64))
    share = float(style[3]) if style[3] >= 0 else JUMP_SHARE_BOOST * float(pj.mean())
    stats["jump_share_model"] = round(float(pj.mean()), 4)
    n_jump = int(math.floor(min(share, _JUMP_MAX_FRAC[b.jumps]) * n + 1e-9))
    is_jump = np.zeros(n, bool)
    if n_jump > 0:
        ok = ticks % _JUMP_TICKS[b.jumps] == 0
        spacing = _JUMP_MIN_SPACING_BEATS[b.jumps] * C.TICKS_PER_BEAT
        taken = []
        for i in np.argsort(-pj, kind="stable"):
            if len(taken) >= n_jump:
                break
            if ok[i] and all(abs(ticks[i] - ticks[j]) >= spacing for j in taken):
                taken.append(i)
        is_jump[taken] = True

    # ---- freezes --------------------------------------------------------- #
    ht = log_softmax(logits["hold_type"][ticks].astype(np.float64))
    p_hold = 1.0 - np.exp(ht[:, 0])
    share = float(style[4]) if style[4] >= 0 else HOLD_SHARE_BOOST * float(p_hold.mean())
    stats["hold_share_model"] = round(float(p_hold.mean()), 4)
    # floor, not round: 8 freezes in 95 notes is 8.4%, over an 8% ceiling. The
    # tier's share is the only limit: its fixed count (max_holds, written for
    # the LLM designer) held a 400-note Hard chart to 3% freezes.
    n_hold = int(math.floor(min(share, b.hold_share[1]) * n + 1e-9))
    flavor = np.zeros(n, np.int64)
    dur = np.zeros(n, np.int64)
    if n_hold > 0:
        kinds = 1 + ht[:, 1:].argmax(1)
        buckets = np.asarray(C.HOLD_DUR_BUCKETS)[logits["hold_dur"][ticks].argmax(1)]
        lo_t = int(round(b.hold_len_beats[0] * C.TICKS_PER_BEAT))
        hi_t = int(round(b.hold_len_beats[1] * C.TICKS_PER_BEAT))
        max_inside = _MAX_NOTES_IN_HOLD[difficulty]
        placed = 0
        # strongest candidates first, never overlapping an earlier freeze; keep
        # going down the ranking until the asked-for number has been placed
        spans = []
        for i in np.argsort(-p_hold, kind="stable"):
            if placed >= n_hold or p_hold[i] < 0.02:
                break
            kind = int(kinds[i])
            if is_jump[i] and not TierRules.for_difficulty(difficulty).jump_holds:
                continue
            start = int(ticks[i])
            if any(s <= start <= e for s, e in spans):     # starts inside a freeze
                continue
            d = int(np.clip(buckets[i], lo_t, hi_t))
            # a freeze may enclose a few notes for the other foot — never a
            # jump, and never the start of another freeze
            limit = max_inside if kind != C.HOLD_BOTH or not is_jump[i] else 0
            inside = 0
            end = start + d
            last_inside = None
            for j in range(i + 1, n):
                if ticks[j] > end:
                    break
                # the free foot plays everything inside the freeze: a second
                # note too soon after the first would be a one-footed jack
                rushed = (last_inside is not None and
                          tick_times[ticks[j]] - tick_times[last_inside] < PINNED_MIN_GAP_S)
                stop = (is_jump[j] or flavor[j] > 0 or inside >= limit or rushed)
                if stop:
                    end = int(ticks[j]) - HOLD_RELEASE_TICKS
                    break
                inside += 1
                last_inside = int(ticks[j])
            if any(start < s <= end for s, e in spans):    # would swallow a later one
                end = min(s for s, e in spans if s > start) - HOLD_RELEASE_TICKS
            if end - start < lo_t:
                continue
            if not is_jump[i]:
                kind = C.HOLD_ALT if kind in (C.HOLD_ALT, C.HOLD_SAME, C.HOLD_BOTH) \
                    else C.ROLL_ALT
            flavor[i], dur[i] = kind, end - start
            spans.append((start, end))
            placed += 1

    # ---- mines ----------------------------------------------------------- #
    mine_tick = np.zeros(0, np.int64)
    ceil_mine = cond.tier_ceilings(difficulty)["mine"]
    want_mines = float(style[8]) if style[8] >= 0 else 0.0
    n_mine = int(round(min(want_mines, ceil_mine) * n))
    if n_mine > 0:
        pm = sigmoid(logits["mine"].astype(np.float64))
        free = allowed.copy()
        free[ticks] = False
        for s, d_ in zip(ticks[flavor > 0], dur[flavor > 0]):
            free[s:s + d_ + 1] = False                    # not under a freeze
        # ...and not within a beat of a 16th run: a mine beside a burst is a
        # trap for a foot that is already committed
        gap = np.diff(ticks)
        fast = np.zeros(n, bool)
        fast[:-1] |= gap <= 3
        fast[1:] |= gap <= 3
        for s in ticks[fast]:
            free[max(0, s - MINE_CLEARANCE_TICKS):s + MINE_CLEARANCE_TICKS + 1] = False
        cand = np.nonzero(free)[0]
        cand = cand[np.argsort(-pm[cand], kind="stable")]
        mine_tick = np.sort(_nms(cand, tick_times, 2 * min_gap_s(difficulty))[:n_mine])
    stats.update(jumps=int(is_jump.sum()), holds=int((flavor > 0).sum()),
                 mines=int(len(mine_tick)))
    return RhythmPlan(ticks.astype(np.int64), is_jump, flavor, dur, mine_tick, stats)


# --------------------------------------------------------------------------- #
# Constrained Viterbi over the foot-state machine
# --------------------------------------------------------------------------- #
_GEN = None


def _gen_tables() -> dict:
    """Static per-(state, token) facts the generator's masks are built from."""
    global _GEN
    if _GEN is not None:
        return _GEN
    t = C.tables()
    s_n, a_n = C.N_STATES, C.N_TOKENS
    crossed = np.zeros((s_n, a_n), bool)
    fsw = np.zeros((s_n, a_n), bool)
    dstep = np.zeros((s_n, a_n), bool)
    repeat = np.zeros((s_n, a_n), bool)
    repeat_any = np.zeros((s_n, a_n), bool)   # ...when the previous row was a jump
    panels = np.zeros((s_n, a_n, 4), np.float64)   # which panels a token presses
    onto_l = np.zeros((s_n, a_n), bool)       # single lands on the LEFT foot's panel
    onto_r = np.zeros((s_n, a_n), bool)
    # hold kind -> absolute foot that holds. Kinds 1-4 are foot-relative (jump
    # alt / jump same / both / the single's own foot); 5 + P means "whichever
    # foot presses absolute panel P", for notes whose panels are already fixed.
    holder = np.zeros((9, s_n, a_n), np.int8)
    for s in range(s_n):
        pl, pr, fl = C.state_tuple(s)
        pos = (pl, pr)
        for a in range(a_n):
            npl, npr, _, pa, ps = C.apply_token(pl, pr, fl, a)
            crossed[s, a] = C.PANEL_X[npl] > C.PANEL_X[npr]
            if a < 8:
                foot = (1 - fl) if a < 4 else fl
                panel = pa if a < 4 else ps
                other = 1 - foot
                fsw[s, a] = panel == pos[other] and panel != pos[foot]
                dstep[s, a] = a >= 4 and panel != pos[foot]
                repeat[s, a] = panel == pos[fl]
                repeat_any[s, a] = panel in pos
                panels[s, a, panel] = 1.0
                onto_l[s, a] = foot == C.RIGHT and panel == pl
                onto_r[s, a] = foot == C.LEFT and panel == pr
                holder[4, s, a] = foot
            else:
                repeat[s, a] = repeat_any[s, a] = (npl, npr) == (pl, pr)
                panels[s, a, pa] = panels[s, a, ps] = 1.0
            holder[1, s, a] = holder[3, s, a] = 1 - fl
            holder[2, s, a] = fl
            for panel in range(4):
                if pa == panel:
                    holder[5 + panel, s, a] = 1 - fl
                elif ps == panel:
                    holder[5 + panel, s, a] = fl
    # lock legality by lock bits (bit0 = left pinned, bit1 = right pinned)
    lock_bad = np.zeros((4, s_n, a_n), bool)
    for lb in range(4):
        ll, lr = bool(lb & 1), bool(lb & 2)
        bad = np.zeros((s_n, a_n), bool)
        if ll:
            bad |= t["moves_l"] | onto_l
        if lr:
            bad |= t["moves_r"] | onto_r
        lock_bad[lb] = bad
    _GEN = {"crossed": crossed, "fsw": fsw, "dstep": dstep, "repeat": repeat,
            "repeat_any": repeat_any, "panels": panels,
            "holder": holder, "lock_bad": lock_bad,
            "jump_tok": np.arange(a_n) >= 8}
    return _GEN


def lock_schedule(ticks: np.ndarray, is_jump: np.ndarray, flavor: np.ndarray,
                  dur: np.ndarray):
    """From Stage A's freezes, per note: which lock slots are down when the
    note is played (bit 0 / bit 1), and how this note starts a freeze
    (0 none, 1 jump-alt, 2 jump-same, 3 jump-both, 4 the single's own foot).

    A slot is down for notes with start < tick <= start + dur. `decode_rhythm`
    never overlaps two separate freezes, so slot 0 carries every one-foot
    freeze and slot 1 only the second foot of a both-feet jump.
    """
    n = len(ticks)
    slots = np.zeros(n, np.int64)
    kind = np.zeros(n, np.int64)
    for i in np.nonzero((flavor > 0) & (dur > 0))[0]:
        j1 = int(np.searchsorted(ticks, ticks[i] + dur[i], side="right"))
        f = int(flavor[i])
        if is_jump[i]:
            kind[i] = {C.HOLD_ALT: 1, C.ROLL_ALT: 1, C.HOLD_SAME: 2, C.ROLL_SAME: 2,
                       C.HOLD_BOTH: 3}.get(f, 1)
            slots[i + 1:j1] |= 3 if kind[i] == 3 else 1
        else:
            kind[i] = 4
            slots[i + 1:j1] |= 1
    return slots, kind


@dataclass
class ViterbiResult:
    token: np.ndarray
    f_init: int
    states: np.ndarray          # (N,) state index (32) before each note
    score: float
    relaxed: int                # notes whose hard rules had to be dropped


def constrained_viterbi(unary: np.ndarray, dtb: np.ndarray, is_jump: np.ndarray,
                        slots: np.ndarray, hold_kind: np.ndarray, rules: TierRules,
                        *, lam: float = LAMBDA_COST, f_init: int | None = None,
                        start_state: int | None = None,
                        clamp: np.ndarray | None = None,
                        no_repeat: np.ndarray | None = None,
                        ban: np.ndarray | None = None,
                        panel_bias: np.ndarray | None = None,
                        repeat_pen: np.ndarray | None = None,
                        fsw_pen: np.ndarray | None = None) -> ViterbiResult:
    """Exact dynamic programme over (pL, pR, f_last) x lock ownership (spec
    §6.5 step 11).

    `unary` is (N, 20) log-probabilities. Every transition is scored
    unary - lam * discomfort and hard-masked by the physical rules (no spin,
    never move or step onto a foot that is holding) and the tier's (no
    crossover / footswitch / unforced double-step below Hard, a limit on
    repeated rows). The best of the two initial parities is returned unless
    `f_init` or `start_state` pins it.

    `clamp`, (N, 32, 20) bool, restricts individual notes to given tokens per
    state — used by inpainting to hold everything outside the section to its
    existing panels. Clamped notes are scored on discomfort alone and keep
    only the physical rules.

    `ban`, (N, 20) bool, removes individual tokens at individual notes (the
    QA veto loop); `panel_bias`, (4,), adds that many nats per press of each
    panel (the panel-balance loop); `repeat_pen` and `fsw_pen`, (N,), cost that
    many nats for repeating the previous row / footswitching at that note.
    None of them applies to clamped notes.

    If the rules ever leave a note with no legal token (they cannot for a plan
    from `decode_rhythm`; they can for a hand-built one), that note's tier
    rules are relaxed and the search repeated; `relaxed` counts them.
    """
    n = len(unary)
    if n == 0:
        return ViterbiResult(np.zeros(0, np.uint8), C.LEFT if f_init is None else f_init,
                             np.zeros(0, np.int64), 0.0, 0)
    t, g = C.tables(), _gen_tables()
    s_n, a_n = C.N_STATES, C.N_TOKENS
    sq_s = np.repeat(np.arange(s_n), 2)                    # (64,) state of each sq
    sq_q = np.tile(np.arange(2), s_n)                      # (64,) owner of slot 0
    sq_fl = sq_s & 1
    unary = np.asarray(unary, np.float64)
    dtb = np.asarray(dtb, np.int64)
    soft = np.zeros(n, bool)
    slots_next = np.concatenate([slots[1:], [0]])
    if no_repeat is None:
        no_repeat = np.full(n, rules.panel_run_limit <= 1)
    cross_cost = np.asarray(C.CROSS_COST, np.float64) * (rules.cross_scale - 1.0)
    bias_tab = None if panel_bias is None else g["panels"] @ np.asarray(panel_bias, float)

    def solve():
        score = np.full(2 * s_n, NEG)
        if start_state is not None:
            score[2 * start_state] = 0.0
        else:
            for f in ((C.LEFT, C.RIGHT) if f_init is None else (f_init,)):
                score[2 * C.start_state(f)] = 0.0
        back = np.full((n, 2 * s_n), -1, np.int32)
        for j in range(n):
            sa = int(slots[j])
            lb = ((sa & 1) << sq_q) | (((sa >> 1) & 1) << (1 - sq_q))      # (64,)
            cost = t["cost"][dtb[j], lb, sq_s].astype(np.float64)          # (64, 20)
            bad = g["lock_bad"][lb, sq_s] | (cost >= C.HARD_COST)
            clamped = clamp is not None and clamp[j].any()
            if clamped:
                ok = clamp[j][sq_s] & ~bad
                gain = -lam * cost
            else:
                ok = (g["jump_tok"][None, :] == bool(is_jump[j])) & ~bad
                if is_jump[j]:
                    ok &= ~g["crossed"][sq_s]                 # no crossed jumps, ever
                if not soft[j]:
                    if rules.no_cross:
                        ok &= ~g["crossed"][sq_s]
                    if rules.no_footswitch:
                        ok &= ~g["fsw"][sq_s]
                    if rules.no_double_step:
                        free_other = ((lb >> (1 - sq_fl)) & 1) == 0
                        ok &= ~(g["dstep"][sq_s] & free_other[:, None])
                    if no_repeat[j] and j > 0:
                        # after a jump either of its panels counts as a repeat
                        ok &= ~g["repeat_any" if is_jump[j - 1] else "repeat"][sq_s]
                    if ban is not None:
                        ok &= ~ban[j][None, :]
                gain = unary[j][None, :] - lam * (
                    cost + g["crossed"][sq_s] * cross_cost[dtb[j]])
                if repeat_pen is not None and repeat_pen[j] and j > 0:
                    gain = gain - repeat_pen[j] * g[
                        "repeat_any" if is_jump[j - 1] else "repeat"][sq_s]
                if fsw_pen is not None and fsw_pen[j]:
                    gain = gain - fsw_pen[j] * g["fsw"][sq_s]
                if bias_tab is not None:
                    gain = gain + bias_tab[sq_s]
            cand = np.where(ok, score[:, None] + gain, NEG)
            # lock ownership after this note
            hk = int(hold_kind[j])
            if hk:
                nq = g["holder"][hk][sq_s].astype(np.int64)
            else:
                nq = np.repeat(sq_q[:, None], a_n, 1)
            if slots_next[j] == 0:
                nq = np.zeros_like(nq)
            tgt = (t["next"][sq_s].astype(np.int64) * 2 + nq).ravel()
            flat = cand.ravel()
            order = np.argsort(flat, kind="stable")
            new = np.full(2 * s_n, NEG)
            new[tgt[order]] = flat[order]
            back[j, tgt[order]] = order
            if new.max() <= NEG / 2:
                return None, j
            score = new
        end = int(score.argmax())
        toks = np.empty(n, np.uint8)
        states = np.empty(n, np.int64)
        sq = end
        for j in range(n - 1, -1, -1):
            src = int(back[j, sq])
            toks[j] = src % a_n
            sq = src // a_n
            states[j] = sq // 2
        return (toks, states, float(score[end]), sq // 2), None

    relaxed = 0
    while True:
        res, stuck = solve()
        if res is not None:
            break
        if soft[stuck]:
            # already relaxed and still stuck: the PHYSICAL rules (spin, moving
            # a holding foot) cannot be met here, and those are never waived
            raise RuntimeError(f"no physically legal foot path at note {stuck}")
        soft[stuck] = True
        relaxed += 1
    toks, states, total, s0 = res
    return ViterbiResult(toks, int(s0 & 1), states, total, relaxed)


def timing_penalties(rules: TierRules, style: np.ndarray, dt_s: np.ndarray) -> tuple:
    """The time-dependent parts of a tier's rules, as extra arguments for
    `constrained_viterbi` plus the notes that may not repeat the previous row:

    - a tier with no jack budget never repeats an arrow faster than
      FAST_REPEAT_S, pays a fading cost up to SLOW_REPEAT_S, and repeats
      freely after that — by then it is just standing on the arrow
    - a tier that may footswitch pays FOOTSWITCH_PENALTY for each one quicker
      than FOOTSWITCH_SLOW_S, less as the footswitch slider is raised (at the
      tier's ceiling the slider means "footswitches wanted": no penalty)
    """
    dt_s = np.asarray(dt_s, np.float64)
    extra, no_repeat = {}, None
    if rules.slow_repeats_only:
        no_repeat = dt_s < FAST_REPEAT_S
        extra["repeat_pen"] = REPEAT_PENALTY * np.clip(
            (SLOW_REPEAT_S - dt_s) / (SLOW_REPEAT_S - FAST_REPEAT_S), 0.0, 1.0)
    if not rules.no_footswitch and FOOTSWITCH_PENALTY > 0:
        asked = float(style[7])
        ceil = cond.tier_ceilings(rules.difficulty)["fsw"] or 1.0
        scale = 1.0 if asked < 0 else max(0.0, 1.0 - asked / ceil)
        scale = 0.0 if scale < 1e-6 else scale
        extra["fsw_pen"] = FOOTSWITCH_PENALTY * scale * (dt_s < FOOTSWITCH_SLOW_S)
    return extra, no_repeat


def enforce_run_limit(vit_args: dict, ticks: np.ndarray, rules: TierRules,
                      max_rounds: int = 6,
                      no_repeat: np.ndarray | None = None) -> ViterbiResult:
    """Run the Viterbi, then forbid a repeat wherever a run of identical rows
    outgrew the tier's limit, and re-solve until none does. `no_repeat` marks
    notes that may not repeat the previous row to begin with."""
    n = len(ticks)
    no_repeat = (np.full(n, rules.panel_run_limit <= 1) if no_repeat is None
                 else np.asarray(no_repeat, bool).copy())
    res = constrained_viterbi(no_repeat=no_repeat, **vit_args)
    if rules.panel_run_limit <= 1:
        return res
    for _ in range(max_rounds):
        rows = token_rows(res.token, res.f_init)
        run, changed = 1, False
        for i in range(1, n):
            run = run + 1 if rows[i] == rows[i - 1] else 1
            if run > rules.panel_run_limit and not no_repeat[i]:
                no_repeat[i] = True
                changed = True
                run = 1
        if not changed:
            break
        res = constrained_viterbi(no_repeat=no_repeat, **vit_args)
    return res


INPAINT_CONTEXT = 128               # notes of fixed context each side of a re-roll
MAX_REFINE_ROUNDS = 8
MAX_VETO_ROUNDS = 8                 # re-solves per balance round for the caller's referee
MAX_BALANCE_ROUNDS = 12
MIN_NOTES_FOR_BALANCE = 40
BALANCE_BAND = (0.16, 0.34)         # inside the QA gate's 0.15-0.35
BALANCE_STEP = 1.0                  # nats of bias per round at a fully missing panel


def panel_shares(ev: C.ChartEvents) -> np.ndarray:
    """Share of all arrows on each of L, D, U, R."""
    cells = np.concatenate([ev.p0, ev.p1[ev.p1 >= 0]]).astype(np.int64)
    counts = np.bincount(cells, minlength=4).astype(np.float64)
    return counts / max(1.0, counts.sum())


def token_rows(tokens: np.ndarray, f_init: int) -> list:
    """Tokens -> the set of panels pressed at each note."""
    pl, pr, fl = C.L, C.R, int(f_init)
    rows = []
    for tok in tokens:
        pl, pr, fl, pa, ps = C.apply_token(pl, pr, fl, int(tok))
        rows.append(tuple(sorted(p for p in (pa, ps) if p >= 0)))
    return rows


# --------------------------------------------------------------------------- #
# Stage B decode
# --------------------------------------------------------------------------- #
def committed_unary(logp: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    """Unaries for the Viterbi from a finished unmasking pass: 0 for the
    committed token, log p - COMMIT_BONUS for every other. Alternatives are
    still ranked by the model, but all sit at least COMMIT_BONUS below the
    choice that was actually sampled."""
    unary = logp - COMMIT_BONUS
    unary[np.arange(len(sampled)), sampled] = 0.0
    return unary


def unmask(model: FlowModel, ctx, scalars, cond19, is_jump: np.ndarray,
           tokens: np.ndarray | None = None, frozen: np.ndarray | None = None,
           steps: int = UNMASK_STEPS, tau: float = DEFAULT_TAU,
           rng: np.random.Generator | None = None):
    """Parallel cosine unmasking (spec §6.5 steps 1-2).

    Starts from `tokens` (default: all [MASK]); positions marked `frozen` are
    known context and never change. Returns (committed tokens, unary
    log-probabilities from a final pass with every note committed).

    `tau` is the sampling temperature: `DEFAULT_TAU` draws from the model's
    own distribution, 0 is argmax with most-confident-first commitment (fully
    deterministic). The spec's literal schedule — Gumbel noise on the token
    choice annealed to zero, most confident tokens kept — was measured on
    held-out songs and is mode-seeking to the point of uselessness: crossovers
    went from 3.4% of human steps to 0.0% of generated ones whatever the
    sliders asked for. So each candidate is drawn at a fixed temperature and
    notes are committed in random order (see CONFIDENCE_WEIGHT).
    """
    n = len(is_jump)
    rng = rng or np.random.default_rng(0)
    legal = (np.arange(C.N_TOKENS) >= 8)[None, :] == np.asarray(is_jump, bool)[:, None]
    frozen = np.zeros(n, bool) if frozen is None else np.asarray(frozen, bool)
    cur = np.full(n, C.MASK, np.int64) if tokens is None else np.asarray(tokens, np.int64).copy()
    cur[~frozen] = C.MASK
    free = ~frozen
    n_free = int(free.sum())
    for s in range(1, steps + 1):
        if n_free == 0:
            break
        logits = np.where(legal, model.choreo(ctx, scalars, cur, cond19), NEG)
        if tau > 0:
            # draw each candidate from the model's own distribution...
            t_tok = tau / DEFAULT_TAU
            gumbel = -np.log(-np.log(rng.uniform(1e-9, 1.0, logits.shape)))
            pick = (logits / t_tok + gumbel).argmax(1)
        else:
            pick = logits.argmax(1)
        conf = log_softmax(logits)[np.arange(n), pick]
        if tau > 0:
            # ...and commit notes in (near-)random order: see CONFIDENCE_WEIGHT
            conf = CONFIDENCE_WEIGHT * conf - np.log(-np.log(rng.uniform(1e-9, 1.0, n)))
        known = cur != C.MASK
        conf = np.where(known, 1e9, conf)
        conf[frozen] = np.inf
        keep = n_free if s == steps else int(math.floor(
            n_free * (1 - math.cos(math.pi * s / (2 * steps)))))
        order = np.argsort(-conf, kind="stable")
        commit = np.zeros(n, bool)
        commit[order[:int(frozen.sum()) + keep]] = True
        nxt = np.where(known, cur, pick)
        cur = np.where(commit, nxt, C.MASK)
        cur[frozen] = tokens[frozen] if tokens is not None else cur[frozen]
    final = np.where(legal, model.choreo(ctx, scalars, cur, cond19), NEG)
    return cur, log_softmax(final)


@dataclass
class NeuralChart:
    events: C.ChartEvents
    encoded: C.EncodedChart
    stats: dict


def _encoded_from(plan: RhythmPlan, tokens: np.ndarray, f_init: int,
                  mines: np.ndarray) -> C.EncodedChart:
    """Attach Stage A's freezes to the feet Stage B chose."""
    n = len(plan.tick)
    flavor = np.zeros(n, np.uint8)
    da = np.zeros(n, np.int32)
    ds = np.zeros(n, np.int32)
    for i in np.nonzero(plan.flavor > 0)[0]:
        f, d, tok = int(plan.flavor[i]), int(plan.dur[i]), int(tokens[i])
        roll = f in (C.ROLL_ALT, C.ROLL_SAME)
        if tok < 4:                                   # single, alternating foot
            flavor[i], da[i] = (C.ROLL_ALT if roll else C.HOLD_ALT), d
        elif tok < 8:                                 # single, repeating foot
            flavor[i], ds[i] = (C.ROLL_SAME if roll else C.HOLD_SAME), d
        else:
            flavor[i] = f
            if f in (C.HOLD_ALT, C.ROLL_ALT, C.HOLD_BOTH):
                da[i] = d
            if f in (C.HOLD_SAME, C.ROLL_SAME, C.HOLD_BOTH):
                ds[i] = d
    return C.EncodedChart(int(f_init), plan.tick.astype(np.int32),
                          np.asarray(tokens, np.uint8), flavor, da, ds, mines)


def place_mines(mine_ticks: np.ndarray, enc: C.EncodedChart, seed: int) -> np.ndarray:
    """Give each mine a column neither foot is standing on."""
    if len(mine_ticks) == 0:
        return np.zeros((0, 2), np.int32)
    rng = np.random.default_rng(seed)
    before, _, _ = C.foot_path(enc)
    out = []
    for mt in mine_ticks:
        i = int(np.searchsorted(enc.tick, mt))
        s = before[i] if i < len(before) else (before[-1] if len(before) else 6)
        pl, pr, _ = C.state_tuple(int(s))
        free = [c for c in range(4) if c not in (pl, pr)]
        out.append((int(mt), int(rng.choice(free)) if free else 0))
    return np.asarray(out, np.int32).reshape(-1, 2)


class StepManiaNeuralEngine:
    """Audio features in, a legal dance-single chart out."""

    def __init__(self, model_dir=None, int8: bool = False, threads: int | None = None):
        self.model = FlowModel(model_dir, int8=int8, threads=threads)

    def cond_for(self, style: np.ndarray, has_stems: bool = False) -> np.ndarray:
        s = np.asarray(style, np.float32).copy()
        s[9] = 1.0 if (has_stems and self.model.trained_with_stems) else 0.0
        return cond.cond19(s)

    def generate(self, x: np.ndarray, tick_times: np.ndarray, difficulty: str,
                 style: np.ndarray, duration_s: float, *, seed: int = 0,
                 tau: float = DEFAULT_TAU, energy_per_bar=None,
                 f_init: int | None = None, has_stems: bool = False,
                 progress=None, veto=None, balance: bool = True,
                 on_transient: np.ndarray | None = None) -> NeuralChart:
        """One chart for one difficulty. `style` is the 10-D vector from
        conditioning.resolve_style (masked fields = let the model decide).

        `veto(chart) -> note indices` lets a caller's own referee reject
        individual notes; each is banned at that note and the foot path
        re-solved (the StepMania pipeline passes the project's flow-ceiling
        gate). `balance` nudges the path until every panel carries 15-35% of
        the arrows, the project's panel-balance gate."""
        rules = TierRules.for_difficulty(difficulty)
        cv = self.cond_for(style, has_stems)
        if progress:
            progress("rhythm")
        plan = decode_rhythm(self.model.rhythm(x, cv), tick_times, difficulty, style,
                             duration_s, energy_per_bar, on_transient)
        n = len(plan.tick)
        stats = {"rhythm": plan.stats, "difficulty": difficulty, "seed": seed,
                 "tau": tau, "quantization": self.model.quantization}
        if n == 0:
            enc = _encoded_from(plan, np.zeros(0, np.uint8), C.LEFT,
                                np.zeros((0, 2), np.int32))
            return NeuralChart(C.decode(enc), enc, stats)
        if progress:
            progress("choreography")
        times_ms = tick_times[plan.tick] * 1000.0
        ctx = EF.event_context(x, plan.tick)
        scalars = EF.event_scalars(plan.tick, times_ms, plan.is_jump, plan.flavor,
                                   plan.dur, np.zeros(n, np.int64))
        rng = np.random.default_rng(seed)
        sampled, logp = unmask(self.model, ctx, scalars, cv, plan.is_jump, tau=tau,
                               rng=rng)
        unary = committed_unary(logp, sampled)
        dt = np.concatenate([[1e9], np.diff(times_ms)])
        dt_s = dt / 1000.0
        slots, hold_kind = lock_schedule(plan.tick, plan.is_jump, plan.flavor, plan.dur)
        args = dict(unary=unary, dtb=C.dt_bucket(dt), is_jump=plan.is_jump, slots=slots,
                    hold_kind=hold_kind, rules=rules, f_init=f_init)
        extra, base_no_repeat = timing_penalties(rules, style, dt_s)
        args.update(extra)
        ban = np.zeros((n, C.N_TOKENS), bool)
        bias = np.zeros(4)
        rounds, best = 0, None
        want_balance = balance and n >= MIN_NOTES_FOR_BALANCE
        for _ in range(MAX_BALANCE_ROUNDS + 1):
            # inner loop: re-solve until the caller's referee has no objection
            clean = True
            for _ in range(MAX_VETO_ROUNDS):
                res = enforce_run_limit(dict(args, ban=ban, panel_bias=bias),
                                        plan.tick, rules, no_repeat=base_no_repeat)
                enc = _encoded_from(plan, res.token, res.f_init,
                                    np.zeros((0, 2), np.int32))
                chart = NeuralChart(C.decode(enc), enc, stats)
                rounds += 1
                bad = [i for i in (veto(chart) if veto else ())
                       if not ban[i, res.token[i]]]
                clean = not bad
                if clean:
                    break
                for i in bad:
                    ban[i, res.token[i]] = True
            share = panel_shares(chart.events)
            excess = max(0.0, BALANCE_BAND[0] - share.min(), share.max() - BALANCE_BAND[1])
            key = (not clean, excess if want_balance else 0.0)
            if best is None or key < best[0]:
                best = (key, res, enc, chart, bias.copy())
            if not want_balance or excess == 0.0:
                break
            # nudge only the panels that are out of band, and keep the starting
            # foot fixed: the bias is in absolute panels, and a free parity lets
            # the path dodge it by mirroring itself instead of rebalancing
            out = (share < BALANCE_BAND[0]) | (share > BALANCE_BAND[1])
            bias += BALANCE_STEP * np.where(out, (0.25 - share) / 0.25, 0.0)
            args["f_init"] = res.f_init
        _, res, enc, chart, bias = best
        enc.mines = place_mines(plan.mine_tick, enc, seed)
        stats["choreo"] = {"viterbi_score": round(res.score, 2), "relaxed": res.relaxed,
                           "changed_by_viterbi": int((res.token != sampled).sum()),
                           "f_init": res.f_init, "refine_rounds": rounds,
                           "vetoed": int(ban.sum()),
                           "panel_bias": [round(float(v), 2) for v in bias]}
        return NeuralChart(C.decode(enc), enc, stats)

    def inpaint_section(self, events: C.ChartEvents, x: np.ndarray,
                        tick_times: np.ndarray, difficulty: str, style: np.ndarray,
                        start_bar: int, end_bar: int, *, seed: int = 0,
                        tau: float = DEFAULT_TAU, has_stems: bool = False,
                        veto=None) -> NeuralChart:
        """Re-roll the choreography of bars [start_bar, end_bar) of an existing
        chart (spec §6.5 step 4). The rhythm — which ticks carry notes, and
        every freeze — is kept. Every note outside the section keeps its exact
        panels: outside notes are clamped at the PANEL level, so the result is
        identical there bit for bit even though its foot annotation is free to
        re-settle around the new section.
        """
        rules = TierRules.for_difficulty(difficulty)
        n = len(events)
        times_ms = tick_times[np.minimum(events.tick, len(tick_times) - 1)] * 1000.0
        base, _ = C.encode(events, times_ms)
        inside = (events.tick >= start_bar * C.G) & (events.tick < end_bar * C.G)
        stats = {"difficulty": difficulty, "seed": seed, "bars": [start_bar, end_bar],
                 "notes_in_section": int(inside.sum())}
        if not inside.any():
            return NeuralChart(events, base, {**stats, "changed": 0})
        is_jump = events.p1 >= 0
        cv = self.cond_for(style, has_stems)
        ctx = EF.event_context(x, events.tick)
        scalars = EF.event_scalars(events.tick, times_ms, is_jump, base.flavor,
                                   base.dur_alt, base.dur_same)
        rng = np.random.default_rng(seed)
        # The network only has to see the section and its surroundings: notes
        # further out are fixed context it would never be asked about, so the
        # cost of a re-roll does not grow with the length of the chart.
        idx = np.nonzero(inside)[0]
        lo = max(0, int(idx[0]) - INPAINT_CONTEXT)
        hi = min(n, int(idx[-1]) + 1 + INPAINT_CONTEXT)
        sampled, logp = unmask(self.model, ctx[lo:hi], scalars[lo:hi], cv, is_jump[lo:hi],
                               tokens=base.token[lo:hi].astype(np.int64),
                               frozen=~inside[lo:hi], tau=tau, rng=rng)
        unary = np.zeros((n, C.N_TOKENS))                 # clamped notes ignore it
        unary[lo:hi] = committed_unary(logp, sampled)

        # outside the section: only tokens that reproduce the existing panels
        t = C.tables()
        p0 = events.p0.astype(np.int64)
        p1 = events.p1.astype(np.int64)
        pa, ps = t["p_alt"][None], t["p_same"][None]                  # (1, S, A)
        lo = np.minimum(np.where(pa < 0, ps, pa), np.where(ps < 0, pa, ps))
        hi = np.maximum(pa, ps)
        single = (p1 < 0)[:, None, None]
        same_rows = np.where(single,
                             ((pa >= 0) ^ (ps >= 0)) & (np.maximum(pa, ps) == p0[:, None, None]),
                             (pa >= 0) & (ps >= 0) & (lo == p0[:, None, None])
                             & (hi == p1[:, None, None]))
        clamp = same_rows & (~inside)[:, None, None]

        # The lock schedule comes from the chart's own freezes. Inside the
        # section a jump's freeze follows the foot role it was written for
        # (alt / same); outside, panels are fixed, so the freeze is pinned to
        # its absolute panel and held by whichever foot the path puts there.
        flavor6 = np.where(base.flavor > 0, np.where(is_jump, C.flavor6(base.flavor),
                                                     C.HOLD_ALT), 0)
        dur = np.maximum(base.dur_alt, base.dur_same)
        slots, hold_kind = lock_schedule(events.tick.astype(np.int64), is_jump,
                                         flavor6, dur)
        one_held = is_jump & ~inside & ((events.k0 > 0) ^ (events.k1 > 0))
        hold_kind = np.where(one_held, 5 + np.where(events.k0 > 0, events.p0, events.p1),
                             hold_kind)
        dt = np.concatenate([[1e9], np.diff(times_ms)])
        ban = np.zeros((n, C.N_TOKENS), bool)
        rounds = 0
        while True:
            res = constrained_viterbi(unary, C.dt_bucket(dt), is_jump, slots, hold_kind,
                                      rules, clamp=clamp, ban=ban)
            new = _rebuild_events(events, base, res, inside)
            enc = encoded_from_path(new, res)
            if not C.decode(enc).equals(new):
                raise RuntimeError("inpainting produced an inconsistent foot path")
            chart = NeuralChart(new, enc, stats)
            rounds += 1
            again = False
            if veto and rounds <= MAX_REFINE_ROUNDS:
                for i in veto(chart):                 # only notes we may change
                    if inside[i] and not ban[i, res.token[i]]:
                        ban[i, res.token[i]] = again = True
            if not again:
                break
        changed = int(sum(a != b for a, b in zip(new.rows(), events.rows())))
        chart.stats = {**stats, "changed": changed, "relaxed": res.relaxed,
                       "refine_rounds": rounds, "vetoed": int(ban.sum())}
        return chart


def encoded_from_path(ev: C.ChartEvents, res: ViterbiResult) -> C.EncodedChart:
    """The encoding of `ev` along the foot path the Viterbi actually solved
    (rather than a fresh annotation, which is free to stand differently)."""
    t = C.tables()
    n = len(ev)
    pa = t["p_alt"][res.states, res.token.astype(np.int64)]
    ps = t["p_same"][res.states, res.token.astype(np.int64)]
    alt0, same0 = pa == ev.p0, ps == ev.p0
    k_alt = np.where(pa < 0, 0, np.where(alt0, ev.k0, ev.k1))
    k_same = np.where(ps < 0, 0, np.where(same0, ev.k0, ev.k1))
    d_alt = np.where(pa < 0, 0, np.where(alt0, ev.d0, ev.d1))
    d_same = np.where(ps < 0, 0, np.where(same0, ev.d0, ev.d1))
    flavor = np.array([C._FLAVOR_OF[(int(a), int(b))] for a, b in zip(k_alt, k_same)],
                      np.uint8).reshape(n)
    return C.EncodedChart(int(res.f_init), ev.tick.astype(np.int32),
                          res.token.astype(np.uint8), flavor, d_alt.astype(np.int32),
                          d_same.astype(np.int32), ev.mines.copy())


def _rebuild_events(old: C.ChartEvents, base: C.EncodedChart, res: ViterbiResult,
                    inside: np.ndarray) -> C.ChartEvents:
    """The chart after inpainting: notes outside the section are the old rows,
    untouched; notes inside get their panels from the solved tokens, with each
    freeze on the foot role (alt / same) it had."""
    t = C.tables()
    old_rows = old.rows()
    out = []
    for i, (tick, cells) in enumerate(old_rows):
        if not inside[i]:
            out.append((tick, cells))
            continue
        s, tok = int(res.states[i]), int(res.token[i])
        pa, ps = int(t["p_alt"][s, tok]), int(t["p_same"][s, tok])
        if old.p1[i] < 0:                                  # single: its one freeze
            out.append((tick, [(pa if pa >= 0 else ps, int(old.k0[i]), int(old.d0[i]))]))
        else:
            ka, ks = C.FLAVOR_KINDS[int(base.flavor[i])]
            out.append((tick, [(pa, ka, int(base.dur_alt[i]) if ka else 0),
                               (ps, ks, int(base.dur_same[i]) if ks else 0)]))
    return C.ChartEvents.from_rows(out, [tuple(m) for m in old.mines.tolist()])


# --------------------------------------------------------------------------- #
# Chart -> existing BeatForge structures
# --------------------------------------------------------------------------- #
def to_placements(chart: NeuralChart) -> list:
    """ChartEvents -> the Placement list the StepMania serializer and QA take.

    A jump whose two panels share one freeze (or have none) is one Placement.
    A jump with a freeze on only one panel, or two different ones, becomes two
    Placements on the same beat, because a Placement has a single hold length.
    Jump panels are ordered (left foot, right foot) so QA scores the stance the
    chart was actually written for.
    """
    from ..quantize import Placement
    ev, enc = chart.events, chart.encoded
    before, pa, ps = C.foot_path(enc)
    kind_name = {C.TAP: "tap", C.HOLD: "hold", C.ROLL: "roll"}
    out = []
    for i in range(len(ev)):
        beat = int(ev.tick[i]) / C.TICKS_PER_BEAT
        fl = int(before[i]) & 1
        tok = int(enc.token[i])
        if ev.p1[i] < 0:
            foot = (1 - fl) if tok < 4 else fl
            k, d = int(ev.k0[i]), int(ev.d0[i])
            out.append(Placement(beat, (int(ev.p0[i]),), kind_name[k],
                                 d / C.TICKS_PER_BEAT if k else None,
                                 {"foot": foot, "engine": "neural"}))
            continue
        alt = 1 - fl
        left_p = int(pa[i]) if alt == C.LEFT else int(ps[i])
        right_p = int(ps[i]) if alt == C.LEFT else int(pa[i])
        cells = {int(ev.p0[i]): (int(ev.k0[i]), int(ev.d0[i])),
                 int(ev.p1[i]): (int(ev.k1[i]), int(ev.d1[i]))}
        (kl, dl), (kr, dr) = cells[left_p], cells[right_p]
        if (kl, dl) == (kr, dr):
            out.append(Placement(beat, (left_p, right_p), kind_name[kl],
                                 dl / C.TICKS_PER_BEAT if kl else None,
                                 {"engine": "neural"}))
        else:
            for foot, p, k, d in ((C.LEFT, left_p, kl, dl), (C.RIGHT, right_p, kr, dr)):
                out.append(Placement(beat, (p,), kind_name[k],
                                     d / C.TICKS_PER_BEAT if k else None,
                                     {"foot": foot, "engine": "neural", "jump_part": True}))
    for mt, col in ev.mines.tolist():
        out.append(Placement(mt / C.TICKS_PER_BEAT, (int(col),), "mine", None,
                             {"engine": "neural"}))
    out.sort(key=lambda p: (p.beat, p.panels))
    return out


def chart_stats(chart: NeuralChart, tick_times: np.ndarray) -> dict:
    """Foot-model facts about a generated chart, for the QA report and tests."""
    enc = chart.encoded
    n = len(enc)
    if n == 0:
        return {"notes": 0}
    times_ms = tick_times[np.minimum(enc.tick, len(tick_times) - 1)] * 1000.0
    dtb = C.dt_bucket(np.concatenate([[1e9], np.diff(times_ms)]))
    before, _, _ = C.foot_path(enc)
    g = _gen_tables()
    tok = enc.token.astype(np.int64)
    held = C._held_masks(chart.events, inclusive=True)   # the generator's lock rule
    lock = C.tables()["lockbits"][held, before]
    cost = C.tables()["cost"][dtb, lock, before, tok]
    fl = before & 1
    other_locked = ((lock >> (1 - fl)) & 1) == 1
    return {"notes": n,
            "spins": int(((cost % C.LOCK_VIOLATION) >= C.SPIN_COST).sum()),
            "lock_violations": int((cost >= C.LOCK_VIOLATION).sum()),
            "double_steps": int((g["dstep"][before, tok] & ~other_locked).sum()),
            "crossed": int(g["crossed"][before, tok].sum()),
            "footswitches": int(g["fsw"][before, tok].sum()),
            "mean_cost": round(float(np.minimum(cost, C.HARD_COST).mean()), 3)}
