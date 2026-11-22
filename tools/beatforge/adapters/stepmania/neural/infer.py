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
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..grammar import BUDGETS
from . import MODEL_DIR_NAME
from . import codec as C
from . import conditioning as cond
from . import critic as CR
from . import event_features as EF
from . import grid_features as GF
from . import structure as ST

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
        # v1.6: a voice network (Stage 0) and the structure analysis fill the
        # sixteen channels X[:, 32:48] for the rhythm network. A model without
        # one (v1.4, v1.5) leaves them zero, as before.
        self.feature_layout = str(self.meta.get("feature_layout_version") or "")
        self.sess_0 = None
        v = (files.get("stage_0_int8") if int8 else None) or files.get("stage_0")
        if self.feature_layout == VOICE_LAYOUT and v:
            self.sess_0 = ort.InferenceSession(str(self.dir / v), opts, providers=prov)
        self._voice_key, self._voice_x = None, None
        # `stage_0.live: false` silences the voice channels (the structure
        # channels stay): how a rhythm network trained without them is run,
        # and what one trained with them falls back to.
        self.voice_live = bool((self.meta.get("stage_0") or {}).get("live", True))
        # v1.6 Phase 3: a critic that picks among several samples of the step
        # network on one rhythm (`decoder.critic`; see `critic.py`).
        self.sess_c = None
        c = files.get("stage_3")
        if c and (self.dir / c).exists():
            self.sess_c = CR.Critic(self.dir / c)

    def density_scale(self, difficulty: str) -> float:
        """The tier's density scale for THESE weights. DENSITY_SCALE was fitted
        with the first full-corpus model; a model trained on other data is
        calibrated differently and carries its own as `decoder.density_scale`
        in model_meta.json (as it may carry `decoder.phase_margin`)."""
        own = (self.meta.get("decoder") or {}).get("density_scale") or {}
        v = own.get(difficulty)
        return float(v) if isinstance(v, (int, float)) else DENSITY_SCALE.get(difficulty, 1.0)

    def sixteenth_boost(self, difficulty: str) -> float | None:
        """Odds multiplier for 16th-note figures at this tier, for THESE
        weights (`decoder.hard_sixteenth_boost` in model_meta.json; see
        `_promote_sixteenths`). None when the model declares none, which is
        the decoder as it always was."""
        if difficulty not in SIXTEENTH_BURST_TIERS:
            return None
        v = (self.meta.get("decoder") or {}).get("hard_sixteenth_boost")
        return float(v) if isinstance(v, (int, float)) and v > 1.0 else None

    def energy_tilt(self, difficulty: str) -> tuple | None:
        """The ladder of energy-tilt weights for this tier, for THESE weights
        (`decoder.energy_tilt` in model_meta.json: {tier: [weights]}), or None
        for the engine's own (`ENERGY_TILT`). A rhythm network trained longer
        is surer of itself, and the engine's steepest tilt then no longer
        bends a Beginner chart toward the loud bars; such a model carries a
        longer ladder, fitted by `calibrate_decoder --fit-energy-tilt`."""
        own = (self.meta.get("decoder") or {}).get("energy_tilt") or {}
        v = own.get(difficulty)
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
            return tuple(float(x) for x in v)
        return None

    @property
    def neural_stream(self) -> bool:
        """Were these weights trained to write stream charts when asked
        (`decoder.neural_stream` in model_meta.json)? Without it Stream mode
        always uses the rule fill."""
        return bool((self.meta.get("decoder") or {}).get("neural_stream"))

    def callbacks(self) -> dict | None:
        """The model's callback settings (`decoder.callbacks` in
        model_meta.json), or None when it declares none or switches them off:
        {'tau_sim', 'rhythm_weight', 'choreo_bonus'} — see CALLBACK_DEFAULTS."""
        cb = (self.meta.get("decoder") or {}).get("callbacks")
        if not isinstance(cb, dict) or not cb.get("enabled", True):
            return None
        out = dict(CALLBACK_DEFAULTS)
        for k in out:
            try:
                out[k] = float(cb.get(k, out[k]))
            except (TypeError, ValueError):
                pass
        if not 0.0 < out["tau_sim"] < 1.0 or out["rhythm_weight"] < 0 or out["choreo_bonus"] < 0:
            return None
        out["rhythm_weight"] = min(out["rhythm_weight"], 1.0)
        return out if (out["rhythm_weight"] > 0 or out["choreo_bonus"] > 0) else None

    def critic(self) -> dict | None:
        """The model's reranking settings (`decoder.critic` in
        model_meta.json), or None when it has no critic or switches it off —
        see CRITIC_DEFAULTS."""
        cr = (self.meta.get("decoder") or {}).get("critic")
        if self.sess_c is None or not isinstance(cr, dict) or not cr.get("enabled", True):
            return None
        out = dict(CRITIC_DEFAULTS)
        try:
            out["k"] = int(cr.get("k", out["k"]))
            out["source"] = str(cr.get("source", out["source"]))
            out["steps"] = int(cr.get("steps", out["steps"]))
            out["remask"] = float(cr.get("remask", out["remask"]))
            out["tau_sim"] = float(cr.get("tau_sim", out["tau_sim"]))
            for key in ("bonus_scale", "tau_scale"):
                v = cr.get(key)
                if isinstance(v, (list, tuple)) and v:
                    out[key] = tuple(float(q) for q in v)
        except (TypeError, ValueError):
            return None
        if out["k"] < 2 or out["steps"] < 1 or not 0.0 < out["remask"] <= 1.0 \
                or out["source"] not in ("sample", "logits"):
            return None
        return out

    # -- Stage 0 ---------------------------------------------------------- #
    @property
    def has_voice(self) -> bool:
        return self.sess_0 is not None

    def voice(self, x: np.ndarray) -> np.ndarray:
        """X (T, 64) -> the voice network's 14 channels (T, 14), run in
        windows with a short overlap and cross-faded over it."""
        t = len(x)
        win = int((self.meta.get("stage_0") or {}).get("window_ticks") or VOICE_WINDOW_TICKS)
        xv = np.ascontiguousarray(x[:, VOICE_COLUMNS], np.float32)
        if t <= win:
            return self.sess_0.run(None, {"x_voice": xv[None]})[0][0]
        hop = win - VOICE_OVERLAP_TICKS
        starts = list(range(0, t - win, hop)) + [t - win]
        out = np.stack(self.sess_0.run(None, {"x_voice": np.stack([xv[a:a + win] for a in starts])})[0])
        edge = np.minimum(np.arange(1, win + 1), np.arange(win, 0, -1)).astype(np.float32)
        w = np.minimum(edge / VOICE_OVERLAP_TICKS, 1.0) + 1e-3
        acc = np.zeros((t, out.shape[2]), np.float32)
        wsum = np.zeros(t, np.float32)
        for a, o in zip(starts, out):
            acc[a:a + win] += o * w[:, None]
            wsum[a:a + win] += w
        return acc / wsum[:, None]

    def with_voice(self, x: np.ndarray) -> np.ndarray:
        """X as the rhythm network of a v1.6 model wants it: channels 32..45
        from the voice network, 46..47 from the structure analysis. The last
        song's result is kept, since a song is charted once per difficulty."""
        key = (x.shape, float(x[::37, :32].sum()), float(x[-1, :32].sum()) if len(x) else 0.0)
        if self._voice_key == key and self._voice_x is not None:
            return self._voice_x
        out = np.array(x, np.float32, copy=True)
        if len(out):
            v = self.voice(out)
            out[:, VOICE_SLICE] = v
            out[:, STRUCTURE_SLICE] = ST.analyze_structure(x, v).channels()
        self._voice_key, self._voice_x = key, out
        return out

    # -- Stage A ---------------------------------------------------------- #
    def rhythm(self, x: np.ndarray, cond19: np.ndarray) -> dict:
        """X (T, 64) -> per-tick logits, run in overlapping 32-measure windows
        (the length it was trained on) and cross-faded. A v1.6 model first has
        its voice and structure channels filled in (`with_voice`)."""
        if self.has_voice:
            x = self.with_voice(x)
            if self.voice_live:
                cond19 = np.array(cond19, np.float32, copy=True)
                cond19[cond.INDEX["has_stems"]] = 1.0   # "channels 32..45 are live"
            else:
                x = np.array(x, np.float32, copy=True)
                x[:, VOICE_SLICE] = 0.0
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

    def choreo_many(self, ctx: np.ndarray, scalars: np.ndarray, tokens: np.ndarray,
                    cond19: np.ndarray) -> np.ndarray:
        """`choreo` for K token arrays on the same notes in one batched call:
        (N, K, 64), (N, 31), (K, N) -> logits (K, N, 20)."""
        k, n = tokens.shape
        c = np.repeat(cond19[None, :].astype(np.float32), k, axis=0)

        def run(a, b):
            m = b - a
            return self.sess_b.run(None, {
                "ctx": np.ascontiguousarray(np.broadcast_to(
                    ctx[None, a:b].astype(np.float32), (k, m) + ctx.shape[1:])),
                "scalars": np.ascontiguousarray(np.broadcast_to(
                    scalars[None, a:b].astype(np.float32), (k, m, scalars.shape[1]))),
                "tokens": np.ascontiguousarray(tokens[:, a:b].astype(np.int64)), "cond": c})[0]

        if n <= EVENT_WINDOW:
            return run(0, n)
        starts = list(range(0, n - EVENT_WINDOW, EVENT_HOP)) + [n - EVENT_WINDOW]
        ramp = np.minimum(np.arange(1, EVENT_WINDOW + 1), np.arange(EVENT_WINDOW, 0, -1))
        w = (ramp / ramp.max()).astype(np.float32) + 1e-3
        acc = np.zeros((k, n, C.N_TOKENS), np.float32)
        wsum = np.zeros(n, np.float32)
        for a in starts:
            acc[:, a:a + EVENT_WINDOW] += run(a, a + EVENT_WINDOW) * w[None, :, None]
            wsum[a:a + EVENT_WINDOW] += w
        return acc / wsum[None, :, None]


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
#: "Max subdivision": the ticks a chart may use when the user caps it at 4ths,
#: 8ths or 16ths. A cap is strict about what it names: 8ths means no triplets
#: either (a triplet 8th is closer to its neighbour than an 8th is), and 16ths
#: means the duple grid only. It can only coarsen a tier, never refine it.
SUBDIVISION_RESIDUES = {4: (0,), 8: (0, 6), 16: (0, 3, 6, 9)}
SUBDIVISION_TICKS = {4: 12, 8: 6, 16: 3}
#: The finest duple subdivision each tier writes.
_FINEST = {"beginner": 4, "easy": 8, "medium": 8, "hard": 16, "challenge": 16}
#: Longest run of notes an 8th apart or closer, per tier. grammar.BUDGETS has
#: its own (4 / 8 / 16 at Medium / Hard / Challenge), drafted for the LLM
#: designer. Applied here it cut every stream at that length: on the 180-song
#: benchmark the longest run on the median chart was 3 / 7 / 15 notes where the
#: human chart of the same song has 5 / 15 / 18 (and 9 / 26 / 33 at the 75th
#: percentile), and half of a Hard chart's intervals sat in runs of exactly
#: that shape. These sit at the human median. Longer still (8 / 32 / 48) scored
#: the same on placement but put a quarter of a Hard chart in runs of 16+
#: intervals, where human charts have 8%. Unbroken streams are Stream mode's
#: job (`DecodeOptions.stream`), not the default chart's.
RUN_LIMIT = {"beginner": 1, "easy": 2, "medium": 6, "hard": 16, "challenge": 24}


#: v1.6 feature layout: a model that names it carries a voice network.
VOICE_LAYOUT = "1.6_voicenet"
VOICE_COLUMNS = np.r_[0:32, 48:64]          # what the voice network reads of X
VOICE_SLICE = slice(32, 46)                 # ...and what it fills
STRUCTURE_SLICE = slice(46, 48)
VOICE_WINDOW_TICKS = 8 * C.G
VOICE_OVERLAP_TICKS = C.G


#: Callbacks (v1.6). Where a passage of the song comes back (`structure.py`
#: links a bar to the earlier bar it repeats), the chart may answer it with
#: what it wrote the first time:
#:   rhythm_weight   how far the rhythm network's onset logits of the later bar
#:                   are moved toward the earlier bar's, at a perfect match
#:                   (scaled down to nothing at tau_sim)
#:   choreo_bonus    nats, times the similarity, in favour of the codec token
#:                   the earlier bar has on the same tick. The token is
#:                   relative to the feet, so it brings back the same figure or
#:                   its left-right mirror, whichever the feet are placed for.
#: Both are soft: a fill or a variation the networks are sure of still wins,
#: and every hard rule of the foot-path solver and the referees applies as
#: before. Off unless the model declares `decoder.callbacks`.
CALLBACK_DEFAULTS = {"tau_sim": ST.TAU_SIM, "rhythm_weight": 0.0, "choreo_bonus": 0.0}


def echo_onset_logits(onset: np.ndarray, links: list, tau_sim: float,
                      weight: float) -> np.ndarray:
    """Onset logits with every linked bar blended toward the bar it repeats.
    Always reads the network's own logits, so the order of the links does not
    matter and a chain of repeats cannot compound."""
    if weight <= 0 or not links:
        return onset
    src = np.asarray(onset)
    out = src.astype(np.float32, copy=True)
    t = len(src)
    for b, j, sim, _ in links:
        a = weight * float(np.clip((sim - tau_sim) / max(1e-6, 1.0 - tau_sim), 0.0, 1.0))
        lo, lo2 = b * C.G, j * C.G
        n = min(C.G, t - lo, t - lo2)
        if a <= 0 or n <= 0:
            continue
        out[lo:lo + n] = (1.0 - a) * src[lo:lo + n] + a * src[lo2:lo2 + n]
    return out


def callback_bonus(ticks: np.ndarray, is_jump: np.ndarray, tokens: np.ndarray,
                   links: list, bonus: float) -> tuple:
    """(N, 20) unary bonus toward the token the linked earlier bar has on the
    same tick of the bar, and how many notes got one. A single is only ever
    pointed at a single's token and a jump at a jump's."""
    n = len(ticks)
    out = np.zeros((n, C.N_TOKENS), np.float64)
    if bonus <= 0 or not links or n == 0:
        return out, 0
    index = {int(t): i for i, t in enumerate(ticks)}
    hit = 0
    for b, j, sim, _ in links:
        lo, hi = np.searchsorted(ticks, [b * C.G, (b + 1) * C.G])
        for i in range(lo, hi):
            k = index.get(int(ticks[i]) - (b - j) * C.G)
            if k is None or bool(is_jump[i]) != bool(is_jump[k]):
                continue
            out[i, int(tokens[k])] += bonus * float(sim)
            hit += 1
    return out, hit


#: Best-of-K (v1.6 Phase 3). The rhythm is decided once; the step network is
#: then sampled K times on those ticks and `critic.py` keeps the sample it
#: scores highest among those the referees pass. Sample 0 is always the chart
#: the model writes without a critic.
#:   k            samples, the first included
#:   source       where the further samples come from. "sample": the step
#:                network is run again for each (K - 1 more unmasking passes,
#:                batched). "logits": they are drawn from the note-by-note
#:                probabilities the first pass already ended with — no further
#:                network call, only one more foot-path solve each
#:   steps        unmasking steps of each further sample ("sample" only)
#:   remask       share of the first sample's arrows a further sample redraws
#:                (1.0 = an independent sample; less = a variation, in fewer steps)
#:   bonus_scale  per sample, a factor on the callback bonus
#:   tau_scale    per sample, a factor on the sampling temperature
#:   tau_sim      similarity from which bars are linked when the model
#:                declares no callbacks of its own
#: Off unless the model ships `stage_3` and declares `decoder.critic`.
CRITIC_DEFAULTS = {"k": 4, "source": "sample", "steps": UNMASK_STEPS, "remask": 1.0,
                   "bonus_scale": (1.0, 1.0, 2.0, 1.0), "tau_scale": (1.0, 1.0, 1.0, 0.75),
                   "tau_sim": 0.70}
CRITIC_MIN_NOTES = 24


STREAM_FILLS = ("neural", "rule")
#: A model-written stream chart is kept when the model itself puts at least
#: this share of the stream's notes above even odds; below it the model has
#: not written a stream and the rule fill is used instead.
NEURAL_STREAM_MIN_SHARE = 0.60


@dataclass(frozen=True)
class DecodeOptions:
    """What the user asked of the decoder beyond the style sliders. Everything
    here is enforced by construction, like the tier rules.

    max_subdivision  4, 8 or 16: no note finer than that (None: the tier's own)
    stream           Stream mode: unbroken runs at one subdivision with real
                     breaks between them, instead of note-by-note selection
    stream_coverage  share of the playable bars that are stream (Stream mode)
    min_break_bars   a break is never shorter than this many bars (Stream mode)
    stream_fill      Stream mode: 'neural' asks the model itself for a stream
                     chart (see `stream_style`) and falls back to 'rule' when
                     the model does not write one; 'rule' lays the streams out
                     over an ordinary chart's probabilities, as v1.4 did
    stream_share     Stream mode: the share of the chart's NOTES that should
                     sit in 16th streams (the style vector's `stream_share`,
                     what the Stream slider means). The coverage in bars is
                     then solved for it and `stream_coverage` is ignored.
                     Set by `route_stream`; None: `stream_coverage` as given
    crossovers       False: no step may end with the feet crossed
    density          multiplier on the note count the model itself expects
                     (used when the NPS slider is left to the model)
    """
    max_subdivision: int | None = None
    stream: bool = False
    stream_coverage: float = 0.75
    min_break_bars: int = 2
    crossovers: bool = True
    density: float = 1.0
    stream_fill: str = "neural"
    stream_share: float | None = None

    def __post_init__(self):
        if self.stream_share is not None and not 0.0 < float(self.stream_share) <= 1.0:
            raise ValueError("stream_share must be in (0, 1]")
        if self.stream_fill not in STREAM_FILLS:
            raise ValueError(f"stream_fill must be one of {', '.join(STREAM_FILLS)}, "
                             f"not {self.stream_fill!r}")
        if self.max_subdivision not in (None, 4, 8, 16):
            raise ValueError(f"max_subdivision must be 4, 8 or 16, not {self.max_subdivision!r}")
        if not 0.0 < float(self.stream_coverage) <= 1.0:
            raise ValueError("stream_coverage must be in (0, 1]")
        if int(self.min_break_bars) < 1:
            raise ValueError("min_break_bars must be at least 1")
        if not 0.25 <= float(self.density) <= 2.0:
            raise ValueError("density must be between 0.25 and 2.0")

    @classmethod
    def from_dict(cls, d: dict | None) -> "DecodeOptions":
        """From the plain mapping the CLI and the app pass (and report.json
        records). Unknown keys are an error: a typo must not silently do
        nothing."""
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(f"unknown decoder option(s): {', '.join(bad)} "
                             f"(choose from {', '.join(sorted(known))})")
        return cls(**d)

    def to_dict(self) -> dict:
        """Only what differs from the defaults (plus the Stream settings when
        Stream mode is on), for report.json."""
        base = DecodeOptions()
        out = {k: getattr(self, k) for k in self.__dataclass_fields__
               if getattr(self, k) != getattr(base, k)}
        if self.stream:
            out.update(stream_coverage=self.stream_coverage,
                       min_break_bars=self.min_break_bars)
        else:
            out.pop("stream_coverage", None)
            out.pop("min_break_bars", None)
            out.pop("stream_fill", None)
            out.pop("stream_share", None)
        return out


def subdivision_residues(difficulty: str, max_subdivision: int | None = None) -> tuple:
    """`tick % 12` residues a chart of `difficulty` may use under the cap."""
    tier = SUBDIVISIONS_24THS if (difficulty == "challenge" and ALLOW_24THS) \
        else _SUBDIVISIONS[difficulty]
    if max_subdivision is None:
        return tier
    return tuple(r for r in SUBDIVISION_RESIDUES[max_subdivision] if r in tier)
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
#: Footswitch placement. A footswitch on disk is one repeated arrow, and whoever
#: reads the chart decides which foot plays it. Read the other way it is a jack
#: with the feet swapped, and that reading is the cheaper one unless the steps
#: after it punish it: with the v1.5b step network, 12% of a chart's steps were
#: footswitches in the decoder's own footing when 16% was asked, and 2.7% in
#: the annotator's reading — the rest were read as jacks. What makes the
#: reading stick is the two steps that follow: each foot out to its own side
#: panel. Swapped, that is a spin, so the footswitch is the only way to stand
#: on it. When the footswitch slider asks for more than FSW_PLACE_MIN_SHARE the
#: decoder therefore places them: a bonus on a D/U footswitch at chosen notes
#: and on that continuation, the chart read back as a player would read it
#: (`codec.encode`), and every footswitch that was not read as one taken out
#: again, until the READ share is what was asked. Below the threshold, and with
#: the slider left to the model, nothing here runs.
FSW_PLACE_MIN_SHARE = 0.03
FSW_MAX_GAP_S = 0.48               # slower than this a repeated arrow is a jack to anyone
FSW_BONUS = 8.0                    # nats on a D/U footswitch at a chosen note
FSW_FOLLOW_BONUS = 4.0             # ...and on each of the two steps after it going home
FSW_BLOCK = 50.0                   # nats against a footswitch that reads as a jack
FSW_SPACING = 4                    # notes between two placed footswitches
FSW_ROUNDS = 6
FSW_OVERSHOOT = 1.3                # place this many times what is still missing


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
    def for_difficulty(cls, difficulty: str, crossovers: bool = True) -> "TierRules":
        """`crossovers=False` forbids them at tiers that would allow them."""
        b = BUDGETS[difficulty]
        # No tier gets an unforced double-step. Human Hard/Challenge charts do
        # contain them (~0.5% of steps) and the model will sample them, but they
        # are what chart reviewers flag first, and on the benchmark they were the
        # one playability measure where the neural engine trailed DDC.
        slow = b.jack_limit <= 0
        return cls(difficulty, b.crossover == "none" or not crossovers, not b.footswitch,
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


#: Stream mode. A stream shorter than this many bars is not worth breaking the
#: song up for and is given back to the break around it.
STREAM_MIN_BARS = 2
#: How much a bar's energy rank counts beside the model's own onset probability
#: when deciding which bars are stream and which are break.
STREAM_ENERGY_WEIGHT = 0.25


#: Stream mode's own speed limit, as the shortest gap between two notes of a
#: stream. A chart that is asked to be a stream chart (Stream mode, the stamina
#: preset, a Stream slider of STREAM_ROUTE_MIN_SHARE or more) is not held to
#: the peak rate of an ordinary chart of its tier: that rate (8 / 10 notes a
#: second) turned every stream above 120 / 150 BPM into 8ths, and the stamina
#: preset then delivered 48% of the chart as 16th stream where 80% was asked.
#: These keep 16ths up to 170 BPM at Hard and 200 BPM at Challenge. Nothing
#: outside Stream mode reads them.
STREAM_MIN_GAP_S = {"hard": 0.088, "challenge": 0.075}


def stream_rate_limit(difficulty: str) -> float:
    """Fastest notes-per-second a Stream-mode stream may run at for a tier."""
    limit = float(BUDGETS[difficulty].max_nps_4s)
    gap = STREAM_MIN_GAP_S.get(difficulty)
    return max(limit, 1.0 / gap) if gap else limit


def stream_subdivision(difficulty: str, max_subdivision: int | None,
                       beat_s: float) -> int:
    """The subdivision (4, 8 or 16) a Stream-mode chart runs at.

    A subdivision the user chose is taken as given, up to the finest the tier
    writes at all — they asked for 16ths, they get 16ths, whatever rate that
    is at this tempo. Left open, it is the finest one whose rate stays inside
    Stream mode's speed limit for the tier (`stream_rate_limit`)."""
    finest = _FINEST[difficulty]
    if max_subdivision is not None:
        return min(int(max_subdivision), finest)
    limit = stream_rate_limit(difficulty)
    for sub in (16, 8, 4):
        if sub <= finest and (sub / 4.0) / beat_s <= limit + 1e-6:
            return sub
    return 4


def stream_style(style: np.ndarray, difficulty: str, opts: DecodeOptions,
                 beat_s: float) -> np.ndarray:
    """The style a Stream-mode chart is described by, for asking the model to
    write one: the density of the stream's own subdivision at this tempo, and
    — when that subdivision is 16ths, the only runs the stream slider counts —
    the share of the chart the user wants as stream. Sliders the user set
    stay; the rest stay with the model. Deliberately not clamped to the tier:
    Stream mode already exceeds the tier's run length by design."""
    out = np.asarray(style, np.float32).copy()
    sub = stream_subdivision(difficulty, opts.max_subdivision, beat_s)
    rate = float(np.clip((sub / 4.0) / beat_s, 0.5, 16.0))
    out[cond.INDEX["meter_norm"]] = cond.effective_meter(rate) / 15.0
    out[cond.INDEX["nps_norm"]] = rate / 16.0
    asked = opts.stream_coverage if opts.stream_share is None else opts.stream_share
    out[cond.INDEX["stream_share"]] = float(asked) if sub == 16 else 0.0
    return out


#: Stream routing. Outside Stream mode the tier's limits on speed and run
#: length decide how much of a chart can be stream, whatever the Stream slider
#: says: on 20 held-out songs a preset asking for 80 % stream got 3 %. So a
#: chart that is ASKED to be a stream chart is made in Stream mode: the
#: `itg_stamina` preset, or a Stream slider the caller set to at least
#: STREAM_ROUTE_MIN_SHARE, at the tiers that write streams at all. The other
#: presets carry a stream value too (jumpstream asks 56 % at Challenge), but
#: Stream mode writes no jumps inside a stream, so they stay note-by-note.
STREAM_ROUTE_MIN_SHARE = 0.45
STREAM_ROUTE_TIERS = ("hard", "challenge")
STREAM_ROUTE_PRESETS = ("itg_stamina",)
#: Narrowest stream coverage the solver for `DecodeOptions.stream_share` tries.
STREAM_COVERAGE_MIN = 0.05
STREAM_SHARE_ITERS = 7
#: In a routed chart the jump and freeze sliders are shares of the whole chart,
#: but only break notes can carry them: at most this share of the break notes.
STREAM_BREAK_SHARE_MAX = 0.5


def route_stream(difficulty: str, style: np.ndarray, preset: str | None,
                 overrides: dict | None, options: "DecodeOptions | None") -> tuple:
    """Should this chart be made in Stream mode although the caller did not
    switch it on? -> (options to decode with, why or None).

    `overrides` are the sliders the caller set by hand (not ones filled in by
    style retrieval). A chart that is not routed gets `options` back untouched,
    so nothing about an ordinary chart changes."""
    if difficulty not in STREAM_ROUTE_TIERS or (options is not None and options.stream):
        return options, None
    asked = float(style[cond.INDEX["stream_share"]])
    if asked < STREAM_ROUTE_MIN_SHARE - 1e-4:
        return options, None
    if preset in STREAM_ROUTE_PRESETS:
        why = f"preset {preset}"
    elif overrides and "stream_share" in overrides:
        why = "stream_share slider"
    else:
        return options, None
    routed = dataclasses.replace(options or DecodeOptions(), stream=True,
                                 stream_share=min(1.0, asked))
    return routed, why


def _solve_stream_share(decode, opts: DecodeOptions) -> "RhythmPlan":
    """Stream mode with `opts.stream_share` set: find the coverage (in bars) at
    which that share of the chart's notes is in 16th streams, by bisection.
    `decode(options) -> RhythmPlan`.

    A stream note is not a bar: a bar of 16ths is sixteen notes and a bar of
    break two to four, so 80 % of the bars as stream would be 95 % of the
    notes. Where the tier's rate makes the stream run in 8ths at this tempo the
    slider's statistic (runs of 16ths) cannot see it at all; the share asked
    for is then taken as the share of bars."""
    target = float(opts.stream_share)

    def at(c: float):
        plan = decode(dataclasses.replace(opts, stream_coverage=float(c)))
        share = cond.stream_share(plan.tick, plan.is_jump) if len(plan.tick) else 0.0
        return plan, float(share)

    c = float(np.clip(target, STREAM_COVERAGE_MIN, 1.0))
    plan, share = at(c)
    best = (abs(share - target), c, plan, share)
    if plan.stats.get("stream", {}).get("subdivision") == 16:
        lo, hi = (STREAM_COVERAGE_MIN, c) if share > target else (c, 1.0)
        for _ in range(STREAM_SHARE_ITERS):
            if hi - lo < 1e-3:
                break
            c = 0.5 * (lo + hi)
            plan, share = at(c)
            if abs(share - target) < best[0]:
                best = (abs(share - target), c, plan, share)
            if share > target:
                hi = c
            else:
                lo = c
    _, c, plan, share = best
    if "stream" in plan.stats:
        plan.stats["stream"].update(coverage=round(c, 3), share_asked=round(target, 3),
                                    share=round(share, 3))
    return plan


def _stream_layout(p: np.ndarray, allowed: np.ndarray, tick_times: np.ndarray,
                   difficulty: str, opts: DecodeOptions,
                   rank: np.ndarray | None) -> dict:
    """Which bars are stream and which are break, and the notes of the streams.

    Every playable bar is scored by how much the model wants notes on the
    stream's own grid there (plus a little of the bar's energy rank). The
    weakest bars become breaks until `1 - stream_coverage` of the playable bars
    are break — never in pieces shorter than `min_break_bars`, so a break is a
    real rest and not a dropped note. A stream fills every slot of its bars at
    the one subdivision and lands on the downbeat that follows it.

    Returns {'ticks': stream note ticks, 'break_allowed': the ticks the breaks
    may still use (quarter notes at most, so nothing in a break is as fast as
    the stream), 'info': for the report}.
    """
    t = len(p)
    n_bars = t // C.G
    beat_s = float(np.median(np.diff(tick_times))) * C.TICKS_PER_BEAT if t > 1 else 0.5
    sub = stream_subdivision(difficulty, opts.max_subdivision, beat_s)
    step = SUBDIVISION_TICKS[sub]
    per_bar = C.G // step
    slots = (np.arange(n_bars)[:, None] * C.G + step * np.arange(per_bar)[None, :])
    # a bar can be stream only if the chart may use every one of its slots (the
    # stream's grid is never finer than what `allowed` already admits)
    playable = allowed[slots].all(axis=1) if n_bars else np.zeros(0, bool)
    act = p[slots].mean(axis=1) if n_bars else np.zeros(0)
    if rank is not None and n_bars:
        r = np.zeros(n_bars)
        m = min(n_bars, len(rank))
        r[:m] = rank[:m]
        act = act + STREAM_ENERGY_WEIGHT * r
    brk = ~playable
    n_play = int(playable.sum())
    need = int(round((1.0 - float(opts.stream_coverage)) * n_play))
    mb = int(opts.min_break_bars)
    while need > 0:
        best = None                                   # (mean score, bars)
        free = playable & ~brk
        for m in np.nonzero(free)[0]:
            # lengthen a break that is already there by one bar...
            if (m > 0 and brk[m - 1]) or (m + 1 < n_bars and brk[m + 1]):
                if best is None or act[m] < best[0]:
                    best = (float(act[m]), [int(m)])
        for m in range(0, n_bars - mb + 1):
            # ...or open a new one at its minimum length
            if free[m:m + mb].all():
                c = float(act[m:m + mb].mean())
                if best is None or c < best[0]:
                    best = (c, list(range(m, m + mb)))
        if best is None:
            break
        brk[best[1]] = True
        need -= len(best[1])
    # a sliver of stream between two breaks belongs to the break
    m = 0
    while m < n_bars:
        if brk[m]:
            m += 1
            continue
        e = m
        while e < n_bars and not brk[e]:
            e += 1
        if e - m < STREAM_MIN_BARS:
            brk[m:e] = True
        m = e
    ticks, segments = [], []
    m = 0
    while m < n_bars:
        if brk[m]:
            m += 1
            continue
        e = m
        while e < n_bars and not brk[e]:
            e += 1
        ticks.append(np.arange(m * C.G, e * C.G, step))
        if e * C.G < t and allowed[e * C.G]:
            ticks.append(np.array([e * C.G]))         # land on the next downbeat
        segments.append([int(m), int(e)])
        m = e
    ticks = np.concatenate(ticks).astype(np.int64) if ticks else np.zeros(0, np.int64)
    # breaks: quarter notes at most (half notes under a stream of 4ths), so
    # nothing in a break is as fast as the stream, and nothing on top of it
    break_step = max(C.TICKS_PER_BEAT, 2 * step)
    kk = np.arange(t)
    break_allowed = allowed & (kk % break_step == 0)
    in_break_bar = np.zeros(t, bool)
    for m in np.nonzero(brk)[0]:
        in_break_bar[m * C.G:(m + 1) * C.G] = True
    in_break_bar[n_bars * C.G:] = True
    break_allowed &= in_break_bar
    break_allowed[ticks] = False
    nps = (sub / 4.0) / beat_s
    info = {"subdivision": sub, "segments": segments,
            "stream_bars": int(sum(e - s for s, e in segments)),
            "playable_bars": n_play, "nps": round(nps, 2),
            "over_tier_rate": bool(nps > BUDGETS[difficulty].max_nps_4s + 1e-6),
            # how much of the stream the model would have written unprompted
            "model_share": round(float((p[ticks] >= 0.5).mean()), 3) if len(ticks) else 0.0}
    return {"ticks": ticks, "break_allowed": break_allowed, "info": info}


#: How hard the onset score leans toward the song's louder bars: per tier, the
#: tilt strengths tried in order (multipliers on the odds are exp(w * (rank -
#: 1/2)), rank being the bar's energy rank in the song) and the Spearman (notes
#: per bar vs bar energy) at which to stop. The project gate is 0.30. On the
#: dev songs these raised the correlation from 0.60 to 0.71 for 0.002 of
#: placement F1; pushing further (every tier at 3-6) reached 0.73 but began to
#: cost Hard and Challenge half a point of agreement with the human chart.
#:
#: Beginner and Easy lean much less. Those charts have few notes to place, and
#: a strong tilt spends them all on the loud bars: with the note count brought
#: down to a human chart's (see `FlowModel.density_scale`), the old 4-6 tilt
#: cost Beginner 0.03 of placement F1 on the 60 tuning songs and the gentle one
#: none, at a correlation (0.67) still far above the gate and above human
#: charts' own (0.42).
ENERGY_TILT = {
    "beginner": ((1.5, 3.0), 0.50),
    "easy": ((1.5, 3.0), 0.50),
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


#: 16th-note figures at Hard. The median human Hard chart has about twenty
#: 16th notes; the model's has none, because a 16th rarely gets enough
#: probability to pass the threshold on its own (v1.5: mean 7, median 0). A
#: model can carry an odds multiplier for them (`decoder.hard_sixteenth_boost`,
#: fitted by calibrate_decoder.py). It is not a blanket boost: a 16th is only
#: promoted BETWEEN two notes the chart already has, with a transient under
#: it — which makes every promoted note part of a figure that starts and ends
#: on an 8th or a quarter (8th-16th-8th is three notes; two in a row make
#: five, four make nine) and never an isolated 16th.
SIXTEENTH_BURST_TIERS = ("hard",)
SIXTEENTH_BURST_MAX_RUN = 9         # notes a 16th apart in one figure


def _promote_sixteenths(tk: np.ndarray, score: np.ndarray, theta: float, boost: float,
                        allowed: np.ndarray, on_transient: np.ndarray | None,
                        tick_times: np.ndarray, max_nps: float, fine_frac: float,
                        run_limit: int, gap_s: float) -> tuple:
    """Add 16ths whose score, with its odds multiplied by `boost`, reaches the
    chart's threshold, where both neighbouring 8th positions already carry a
    note. Strongest first, and only while the tier's own limits still hold
    (peak rate, off-beat share, run length, physical gap). Returns (ticks,
    how many were added)."""
    t = len(score)
    if len(tk) < 2 or boost <= 1.0:
        return tk, 0
    have = np.zeros(t, bool)
    have[tk] = True
    k = np.arange(3, t - 3)
    k = k[(k % 6 == 3) & allowed[k] & ~have[k] & have[k - 3] & have[k + 3]]
    if on_transient is not None:
        k = k[on_transient[k]]
    if len(k) == 0:
        return tk, 0
    boosted = _scale_odds(score[k], boost)
    ok = boosted >= theta
    k, boosted = k[ok], boosted[ok]
    # no 16th in a beat that is counted in triplets (see _one_grid_per_beat)
    beat0 = (k // C.TICKS_PER_BEAT) * C.TICKS_PER_BEAT
    triplet = np.zeros(len(k), bool)
    for r in (2, 4, 8, 10):
        idx = beat0 + r
        triplet |= have[np.minimum(idx, t - 1)] & (idx < t)
    k, boosted = k[~triplet], boosted[~triplet]
    if len(k) == 0:
        return tk, 0
    limit = int(math.floor(max_nps * 4.0 + 1e-9))
    quarters = int((tk % C.TICKS_PER_BEAT == 0).sum())
    fine = len(tk) - quarters
    fine_max = (10 ** 9 if fine_frac >= 0.999
                else int(math.floor(fine_frac / (1 - fine_frac) * quarters + 1e-9)))
    added = {}
    for i in np.argsort(-boosted, kind="stable"):
        if fine + len(added) + 1 > fine_max:
            break
        c = int(k[i])
        if (tick_times[c] - tick_times[c - 3] < gap_s - 1e-9
                or tick_times[c + 3] - tick_times[c] < gap_s - 1e-9):
            continue
        # the figure this note would be part of: notes a 16th apart
        a = c
        while a - 3 >= 0 and have[a - 3]:
            a -= 3
        z = c
        while z + 3 < t and have[z + 3]:
            z += 3
        if (z - a) // 3 + 1 > SIXTEENTH_BURST_MAX_RUN:
            continue
        # the run of 8ths-or-faster it sits in
        idx = np.nonzero(have)[0]
        pos = int(np.searchsorted(idx, c))
        lo = pos - 1
        while lo > 0 and idx[lo] - idx[lo - 1] <= 6:
            lo -= 1
        hi = pos
        while hi + 1 < len(idx) and idx[hi + 1] - idx[hi] <= 6:
            hi += 1
        if hi - lo + 2 > run_limit:
            continue
        # the tier's peak rate, in every 4 s window that contains the note
        tt = tick_times[idx]
        first = int(np.searchsorted(tt, tick_times[c] - 4.0, side="right"))
        ends = np.searchsorted(tt, tt[first:pos] + 4.0, side="left") - np.arange(first, pos)
        tail = int(np.searchsorted(tt, tick_times[c] + 4.0, side="left")) - pos
        if (len(ends) and int(ends.max()) + 1 > limit) or tail + 1 > limit:
            continue
        have[c] = True
        added[c] = float(boosted[i])
    # figures of three, five or nine notes: seven loses its weakest promoted 16th
    for c in sorted(added):
        if not have[c]:
            continue
        a = c
        while a - 3 >= 0 and have[a - 3]:
            a -= 3
        z = c
        while z + 3 < t and have[z + 3]:
            z += 3
        if (z - a) // 3 + 1 == 7:
            mine = [m for m in range(a, z + 1, 3) if m in added and have[m]]
            if mine:
                have[min(mine, key=lambda m: added[m])] = False
    n_added = int(sum(have[c] for c in added))
    return np.nonzero(have)[0].astype(np.int64), n_added


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
                  on_transient: np.ndarray | None = None, *,
                  options: DecodeOptions | None = None,
                  density_scale: float | None = None,
                  window: tuple | None = None,
                  blocked: np.ndarray | None = None,
                  sixteenth_boost: float | None = None,
                  energy_tilt: tuple | None = None) -> RhythmPlan:
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

    `options` carries the user's decoder settings (see `DecodeOptions`): a cap
    on the subdivision, Stream mode, a density multiplier. `density_scale`
    replaces the tier's DENSITY_SCALE (a model may carry its own).

    `window`, (first tick, end tick), decodes only that part of the song — for
    re-generating a bar range of an existing chart — and `blocked`, a bool per
    tick, removes individual ticks from it (a freeze of the surrounding chart
    is being held there). In a window the density target is always the model's
    own expectation for those ticks times `options.density`: the NPS slider
    describes a whole chart, not eight bars of one.

    `sixteenth_boost`, above 1, promotes 16th-note figures at the tiers in
    SIXTEENTH_BURST_TIERS (see `_promote_sixteenths`); the note count is still
    calibrated on the chart that results, so the figures replace the chart's
    weakest notes rather than adding to it.
    """
    b = BUDGETS[difficulty]
    opts = options or DecodeOptions()
    t = len(tick_times)
    k = np.arange(t)
    p = sigmoid(logits["onset"].astype(np.float64))
    subdiv = subdivision_residues(difficulty, opts.max_subdivision)
    allowed = np.isin(k % C.TICKS_PER_BEAT, subdiv)
    allowed &= k >= int(FIRST_PLAYABLE_BEAT * C.TICKS_PER_BEAT)
    allowed &= tick_times <= duration_s - PLAYABLE_TAIL_S
    allowed &= tick_times >= 0.0
    if window is not None:
        allowed &= (k >= int(window[0])) & (k < int(window[1]))
    if blocked is not None:
        allowed &= ~np.asarray(blocked, bool)

    rank = bar = None
    if energy_per_bar is not None and len(energy_per_bar) > 2:
        e_ = np.asarray(energy_per_bar, np.float64)
        rank = np.argsort(np.argsort(e_)) / max(1, len(e_) - 1)             # 0..1
        bar = np.minimum(k // C.G, len(e_) - 1)

    # Stream mode lays the streams out first; what is selected note by note
    # below is then only the breaks between them.
    stream = None
    if opts.stream:
        stream = _stream_layout(p, allowed, tick_times, difficulty, opts, rank)
        allowed = stream["break_allowed"]

    lo0, hi0 = THETA_RANGE
    local = window is not None or stream is not None
    want_nps = float(style[1]) * 16.0 if (style[1] >= 0 and not local) else None
    scale = DENSITY_SCALE.get(difficulty, 1.0) if density_scale is None else float(density_scale)
    expected = float(p[allowed].sum()) * scale * float(opts.density)
    metric = _metric_weight(t, difficulty)
    if want_nps is not None:
        want_nps = min(want_nps, b.max_nps_4s)
    playable_s = max(0.0, duration_s - PLAYABLE_TAIL_S
                     - float(tick_times[min(t - 1, int(FIRST_PLAYABLE_BEAT * 12))]))
    cand = np.nonzero(allowed)[0]
    energy = None if rank is None else np.asarray(energy_per_bar, np.float64)
    burst = (float(sixteenth_boost) if sixteenth_boost and sixteenth_boost > 1.0
             and difficulty in SIXTEENTH_BURST_TIERS and 3 in subdiv else None)
    promoted = {"n": 0}

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
            keep = _cap_runs(tk, score[tk], keep, RUN_LIMIT.get(difficulty, b.max_run))
            if on_transient is not None:
                keep, on_met = _cap_off_transient(tk, score[tk], keep, on_transient)
                met = met and on_met
            out = tk[keep]
            if burst is not None:
                out, promoted["n"] = _promote_sixteenths(
                    out, score, theta, burst, allowed, on_transient, tick_times,
                    b.max_nps_4s, b.fine_frac, RUN_LIMIT.get(difficulty, b.max_run),
                    min_gap_s(difficulty))
            return out, met

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
        floor = 0 if local else min(len(picked), max(8, int(0.25 * playable_s)))
        below = len(tk) < floor
        if below:
            tk = np.sort(picked[:floor])
            theta = float(pscore[floor - 1]) if floor else theta
        return tk, theta, met, below, (0 if below else promoted["n"])

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
    if energy_tilt:
        weights = tuple(energy_tilt)       # the model's own ladder (FlowModel.energy_tilt)
    if local:
        # the correlation is a whole-chart statistic; a part of the song takes
        # the gentlest tilt and is not searched
        weights = weights[:1]
    for weight in weights if energy is not None else (0.0,):
        tk, theta, met, below, n_promoted = select(weight)
        corr = energy_corr(tk) if energy is not None and len(tk) and not local else None
        if best is None or (corr is not None and corr > best[4]):
            best = (tk, theta, met, below, corr if corr is not None else 1.0, weight,
                    n_promoted)
        if corr is None or corr >= enough:
            break
    ticks, theta, fine_met, below_range, corr, weight, n_promoted = best
    n_calibrated = len(ticks)
    stream_bars = 0
    meter = float(style[0]) * 15.0
    if energy is not None and len(ticks) and not local and (
            difficulty == "challenge"
            or (difficulty == "hard" and meter >= STREAM_COMMIT_HARD_METER - 1e-6)):
        bar_rank = np.zeros(t // C.G + 2)
        bar_rank[:len(rank)] = rank
        ticks, stream_bars = _commit_streams(ticks, p, allowed, bar_rank, tick_times,
                                             b.max_nps_4s, min_gap_s(difficulty))
    in_stream = np.zeros(0, bool)
    if stream is not None:
        ticks = np.union1d(ticks, stream["ticks"]).astype(np.int64)
        in_stream = np.isin(ticks, stream["ticks"])
    else:
        in_stream = np.zeros(len(ticks), bool)
    n = len(ticks)
    stats = {"theta": round(float(theta), 4), "calibrated": n_calibrated, "notes": n,
             "below_threshold_range": bool(below_range), "fine_share_met": bool(fine_met),
             "target_nps": want_nps, "expected_count": round(expected, 1),
             "energy_weight": weight, "stream_bars": stream_bars,
             "energy_corr": None if (energy is None or local) else round(corr, 3)}
    if stream is not None:
        stats["stream"] = stream["info"]
    if burst is not None:
        stats["sixteenth_boost"] = {"boost": burst, "promoted": int(n_promoted)}
    if opts.to_dict():
        stats["options"] = opts.to_dict()
    if n == 0:
        z = np.zeros(0, np.int64)
        return RhythmPlan(z, z.astype(bool), z, z, z, stats)

    # ---- jumps ----------------------------------------------------------- #
    pj = sigmoid(logits["jump"][ticks].astype(np.float64))
    share = float(style[3]) if style[3] >= 0 else JUMP_SHARE_BOOST * float(pj.mean())
    stats["jump_share_model"] = round(float(pj.mean()), 4)
    # shares are of the notes that can carry a jump or a freeze at all: in
    # Stream mode that is the breaks, and the whole chart's share spent there
    # would turn every break note into a jump
    n_open = int((~in_stream).sum())
    # ...unless the chart was routed here by its Stream slider: then the other
    # sliders describe the whole chart too, and the breaks carry what was
    # asked for, up to STREAM_BREAK_SHARE_MAX of their notes
    whole = stream is not None and opts.stream_share is not None
    n_break_max = int(math.floor(STREAM_BREAK_SHARE_MAX * n_open + 1e-9))
    n_jump = int(math.floor(min(share, _JUMP_MAX_FRAC[b.jumps]) * n_open + 1e-9))
    if whole:
        n_jump = min(n_break_max,
                     int(math.floor(min(share, _JUMP_MAX_FRAC[b.jumps]) * n + 1e-9)))
    is_jump = np.zeros(n, bool)
    if n_jump > 0:
        ok = (ticks % _JUMP_TICKS[b.jumps] == 0) & ~in_stream
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
    n_hold = int(math.floor(min(share, b.hold_share[1]) * n_open + 1e-9))
    if whole:
        n_hold = min(n_break_max,
                     int(math.floor(min(share, b.hold_share[1]) * n + 1e-9)))
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
            if in_stream[i]:
                continue
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
                stop = (is_jump[j] or flavor[j] > 0 or inside >= limit or rushed
                        or in_stream[j])
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
        fast = in_stream.copy()
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
    fsw_du = np.zeros((s_n, a_n), bool)       # the footswitch the style vector counts
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
                fsw_du[s, a] = a < 4 and fsw[s, a] and panel in (C.D, C.U)
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
    _GEN = {"crossed": crossed, "fsw": fsw, "fsw_du": fsw_du, "dstep": dstep, "repeat": repeat,
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
                        fsw_pen: np.ndarray | None = None,
                        fsw_bonus: np.ndarray | None = None) -> ViterbiResult:
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
    the physical rules plus the ban on unforced double-steps.

    `ban`, (N, 20) bool, removes individual tokens at individual notes (the
    QA veto loop); `panel_bias`, (4,), adds that many nats per press of each
    panel (the panel-balance loop); `repeat_pen` and `fsw_pen`, (N,), cost that
    many nats for repeating the previous row / footswitching at that note;
    `fsw_bonus`, (N,), pays that many for a D/U footswitch there (see
    FSW_PLACE_MIN_SHARE). None of them applies to clamped notes.

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
                if rules.no_double_step and not soft[j]:
                    # the one tier rule that is about feet and not panels: the
                    # kept part of a chart was written without an unforced
                    # double-step, and stays readable without one
                    free_other = ((lb >> (1 - sq_fl)) & 1) == 0
                    ok &= ~(g["dstep"][sq_s] & free_other[:, None])
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
                if fsw_bonus is not None and fsw_bonus[j]:
                    gain = gain + fsw_bonus[j] * g["fsw_du"][sq_s]
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
#: Most crossover steps a Stream-mode chart may have, as a share of its single
#: steps, unless the crossover slider asks for more. Left alone, one stream of
#: a real song came out at 15% (Challenge, 8ths); the human charts the model
#: was trained on that are mostly long runs have 4% at the median and 9-11% at
#: the 90th percentile. Over the cap, the path is solved again with crossovers
#: costing more — by these steps, gently first, so the chart lands under the
#: cap rather than at none at all.
STREAM_MAX_CROSS = 0.10
STREAM_CROSS_STEPS = (0.5, 1.0, 2.0, 4.0, 8.0)


def _cross_share(tokens: np.ndarray) -> float:
    """Share of single steps that are crossover steps (the style vector's
    `crossover_share`)."""
    tok = np.asarray(tokens, np.int64)
    single = tok < 8
    return float((single & ((tok & 3) == C.CROSS)).sum() / max(1, int(single.sum())))


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


def resample(logp: np.ndarray, k: int, tau: float, tau_scale, seed: int):
    """`k` further token arrays drawn note by note from `logp`, the step
    network's probabilities with every other note of the first sample
    committed (a temperature scale of 0 takes the likeliest token). No network
    call: each differs from the first sample where the network was undecided.
    Returns (tokens (k, N), log-probs (k, N, 20)), the log-probs being `logp`."""
    n = len(logp)
    out = np.empty((k, n), np.int64)
    for j in range(k):
        t = (tau / DEFAULT_TAU) * float(tau_scale[j % len(tau_scale)])
        if t <= 0:
            out[j] = logp.argmax(1)
            continue
        rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, 7919, j + 1])
        gumbel = -np.log(-np.log(rng.uniform(1e-9, 1.0, logp.shape)))
        out[j] = (logp / t + gumbel).argmax(1)
    return out, np.broadcast_to(logp, (k,) + logp.shape)


def unmask_many(model: FlowModel, ctx, scalars, cond19, is_jump: np.ndarray,
                base: np.ndarray, k: int, steps: int, tau: float, tau_scale,
                remask: float, seed: int):
    """`unmask` for `k` further samples at once (one batched call per step).
    Each starts from `base` with a `remask` share of its notes masked again
    (all of them at 1.0) and has its own random stream, so the samples do not
    depend on how many are drawn. Returns (tokens (k, N), log-probs (k, N, 20))."""
    n = len(is_jump)
    legal = (np.arange(C.N_TOKENS) >= 8)[None, :] == np.asarray(is_jump, bool)[:, None]
    rngs = [np.random.default_rng([int(seed) & 0x7FFFFFFF, 7919, j + 1]) for j in range(k)]
    cur = np.full((k, n), C.MASK, np.int64)
    frozen = np.zeros((k, n), bool)
    if remask < 1.0:
        for j in range(k):
            frozen[j] = rngs[j].random(n) >= remask
            cur[j, frozen[j]] = base[frozen[j]]
    n_free = (~frozen).sum(axis=1)
    rows = np.arange(n)
    for s in range(1, steps + 1):
        logits = np.where(legal[None], model.choreo_many(ctx, scalars, cur, cond19), NEG)
        for j in range(k):
            if n_free[j] == 0:
                continue
            rng = rngs[j]
            t_tok = (tau / DEFAULT_TAU) * float(tau_scale[j % len(tau_scale)])
            gumbel = -np.log(-np.log(rng.uniform(1e-9, 1.0, logits[j].shape)))
            pick = (logits[j] / max(t_tok, 1e-6) + gumbel).argmax(1)
            conf = CONFIDENCE_WEIGHT * log_softmax(logits[j])[rows, pick] \
                - np.log(-np.log(rng.uniform(1e-9, 1.0, n)))
            known = cur[j] != C.MASK
            conf = np.where(known, 1e9, conf)
            conf[frozen[j]] = np.inf
            keep = int(n_free[j]) if s == steps else int(math.floor(
                n_free[j] * (1 - math.cos(math.pi * s / (2 * steps)))))
            order = np.argsort(-conf, kind="stable")
            commit = np.zeros(n, bool)
            commit[order[:int(frozen[j].sum()) + keep]] = True
            nxt = np.where(known, cur[j], pick)
            cur[j] = np.where(commit, nxt, C.MASK)
    final = np.where(legal[None], model.choreo_many(ctx, scalars, cur, cond19), NEG)
    return cur, log_softmax(final)


@dataclass
class NeuralChart:
    events: C.ChartEvents
    encoded: C.EncodedChart
    stats: dict


def read_footswitches(events: C.ChartEvents, times_ms) -> np.ndarray:
    """Per note: is it a D/U footswitch in the annotator's own reading of the
    chart (`codec.encode`) — what the style vector's `footswitch_share` counts,
    and what someone reading the arrows without foot labels would play."""
    enc, _ = C.encode(events, np.asarray(times_ms, np.float64))
    return cond.footswitch_flags(enc)


def place_footswitches(attempt, first: tuple, unary: np.ndarray, plan: RhythmPlan,
                       times_ms: np.ndarray, slots: np.ndarray, hold_kind: np.ndarray,
                       asked: float, seed: int) -> tuple:
    """Bring the share of READ footswitches up to `asked` (see
    FSW_PLACE_MIN_SHARE). `attempt(un=, fsw_bonus=, fsw_block=)` solves the
    foot path as `generate` does, referee and panel balance included, and
    `first` is its result without any placement.

    Each round: read the chart back; block every footswitch — placed here or
    the step network's own — that the reading turned into a jack; if the read
    share is still short, choose more notes for one (fast enough, inside a run
    of single steps, clear of freezes and of each other, where the model
    itself likes a D/U step best) and solve again. Returns (the attempt whose
    read share is closest to `asked`, a summary for the report)."""
    n = len(plan.tick)
    g = _gen_tables()
    rng = np.random.default_rng((int(seed), 0xF5))
    single = ~np.asarray(plan.is_jump, bool)
    n_single = max(1, int(single.sum()))
    target = int(round(asked * n_single))
    dt_s = np.concatenate([[1e9], np.diff(times_ms)]) / 1000.0
    # a note can carry a placed footswitch if it and the two after it are
    # single steps of one run, with no freeze starting or held anywhere near
    busy = ~single | (slots > 0) | (hold_kind > 0)
    fast = dt_s < FSW_MAX_GAP_S
    can = np.zeros(n, bool)
    for j in range(2, n - 2):
        can[j] = (not busy[j - 1:j + 3].any()) and bool(fast[j:j + 3].all())
    noise = rng.gumbel(0.0, 0.5, n)
    bonus = np.zeros(n)
    block = np.zeros(n, bool)
    cur = first
    extra_rounds = 0
    best = None

    def unaries() -> np.ndarray:
        """The step network's unaries with the placement's wishes written in:
        at a chosen note every alternating step is as good as the one the
        network sampled (so the bonus alone decides, however sure the network
        was of something else), and each of the two steps after it prefers the
        stepping foot's own side panel to anything else."""
        un = unary.copy()
        for j in np.nonzero(bonus > 0)[0]:
            un[j, :4] = np.maximum(un[j, :4], 0.0)
            for k in (j + 1, j + 2):
                un[k, 0] = max(float(un[k].max()), 0.0) + FSW_FOLLOW_BONUS
        return un

    un = unary

    for rnd in range(FSW_ROUNDS + 1):
        res, _, chart = cur[0], cur[1], cur[2]
        read = read_footswitches(chart.events, times_ms)
        have = int(read.sum())
        states = np.asarray(res.states, np.int64)
        own = g["fsw"][states, res.token.astype(np.int64)]
        # closest to what was asked; among charts that are close enough, the
        # one with the fewest footswitches that read as jacks
        off = abs(have - target)
        key = (off > max(1, int(0.1 * target)), int((own & ~read).sum()), off, rnd)
        if best is None or key < best[0]:
            best = (key, cur, have)
        if rnd == FSW_ROUNDS:
            break
        changed = False
        # a footswitch that is read as something else is a jack: take it out
        for j in np.nonzero(own & ~read & ~block)[0]:
            block[j] = True
            changed = True
            bonus[j] = 0.0
        if have > target + max(1, int(0.15 * target)):
            # too many: give up the ones the model itself did not ask for first
            extra = [int(j) for j in np.nonzero(read)[0]]
            extra.sort(key=lambda j: (bonus[j] > 0, noise[j]))
            for j in extra[:have - target]:
                block[j] = True
                changed = True
                bonus[j] = 0.0
        elif have < target:
            taken = read | (bonus > 0)
            near = np.zeros(n + 2 * FSW_SPACING + 1, bool)
            for j in np.nonzero(taken)[0]:
                near[j:j + 2 * FSW_SPACING + 1] = True
            # where the path stands now, which notes could be a D/U footswitch
            able = g["fsw_du"][states].any(axis=1)
            cand = [int(j) for j in np.nonzero(can & able & ~block & ~taken)[0]]
            score = {j: float(np.max(np.where(g["fsw_du"][states[j]], un[j], NEG))) + noise[j]
                     for j in cand}
            cand.sort(key=lambda j: -score[j])
            want = int(math.ceil((target - have) * FSW_OVERSHOOT))
            for j in cand:
                if want <= 0:
                    break
                if near[j + FSW_SPACING]:
                    continue
                near[j:j + 2 * FSW_SPACING + 1] = True
                bonus[j] = FSW_BONUS
                want -= 1
                changed = True
        if not changed:
            break
        un = unaries()
        cur = attempt(un=un, fsw_bonus=bonus, fsw_block=block)
        extra_rounds += cur[4]
    _, cur, have = best
    info = {"asked": round(asked, 4), "read": have, "read_share": round(have / n_single, 4),
            "placed": int((bonus > 0).sum()), "blocked": int(block.sum())}
    return (cur[0], cur[1], cur[2], cur[3], extra_rounds, cur[5]), info


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
                 on_transient: np.ndarray | None = None,
                 options: DecodeOptions | None = None) -> NeuralChart:
        """One chart for one difficulty. `style` is the 10-D vector from
        conditioning.resolve_style (masked fields = let the model decide).

        `veto(chart) -> note indices` lets a caller's own referee reject
        individual notes; each is banned at that note and the foot path
        re-solved (the StepMania pipeline passes the project's flow-ceiling
        gate). `balance` nudges the path until every panel carries 15-35% of
        the arrows, the project's panel-balance gate. `options` are the
        user's decoder settings (subdivision cap, Stream mode, no crossovers)."""
        opts = options or DecodeOptions()
        rules = TierRules.for_difficulty(difficulty, crossovers=opts.crossovers)
        cv = self.cond_for(style, has_stems)
        if progress:
            progress("rhythm")

        cb = self.model.callbacks()
        cr = self.model.critic() if tau > 0 else None
        links, novelty, cr_links = [], None, []
        if cb is not None or cr is not None:
            st = ST.analyze_structure(x, tau_sim=cb["tau_sim"] if cb else cr["tau_sim"])
            novelty, cr_links = st.novelty, st.motif_links
            if cb is not None:
                links = st.motif_links

        def rhythm(cond_vec):
            logits = self.model.rhythm(x, cond_vec)
            if links and cb["rhythm_weight"] > 0:
                logits = dict(logits, onset=echo_onset_logits(
                    logits["onset"], links, cb["tau_sim"], cb["rhythm_weight"]))

            def decode(o):
                return decode_rhythm(logits, tick_times, difficulty,
                                     style, duration_s, energy_per_bar, on_transient,
                                     options=o,
                                     density_scale=self.model.density_scale(difficulty),
                                     sixteenth_boost=self.model.sixteenth_boost(difficulty),
                                     energy_tilt=self.model.energy_tilt(difficulty))
            if opts.stream and opts.stream_share is not None:
                return _solve_stream_share(decode, opts)
            return decode(opts)

        fill = None
        if opts.stream:
            fill = "rule"
            if opts.stream_fill == "neural" and self.model.neural_stream and len(tick_times) > 1:
                # ask the model for a stream chart, and keep its answer if it
                # wrote one; otherwise fall back to laying the streams out
                beat_s = float(np.median(np.diff(tick_times))) * C.TICKS_PER_BEAT
                asked = stream_style(style, difficulty, opts, beat_s)
                cv_stream = self.cond_for(asked, has_stems)
                plan = rhythm(cv_stream)
                share = plan.stats.get("stream", {}).get("model_share", 0.0)
                if share >= NEURAL_STREAM_MIN_SHARE:
                    fill, cv, style = "neural", cv_stream, asked
                else:
                    fill = f"rule (the model wrote {share:.0%} of the stream)"
        if fill != "neural":
            plan = rhythm(cv)
        if fill is not None and "stream" in plan.stats:
            plan.stats["stream"]["fill"] = fill
        n = len(plan.tick)
        stats = {"rhythm": plan.stats, "difficulty": difficulty, "seed": seed,
                 "tau": tau, "quantization": self.model.quantization}
        if opts.to_dict():
            stats["options"] = opts.to_dict()
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
        last_key = [None]               # (referee objects, panels out of band) of the last solve

        def attempt(rules: TierRules, un=None, fsw_bonus=None, fsw_block=None):
            """Solve the foot path under `rules`: re-solve until the caller's
            referee has no objection, and nudge until the panels balance.
            `un`, `fsw_bonus` and `fsw_block` are the footswitch placement's
            (`place_footswitches`): its own unaries, the notes it wants a
            footswitch at, and the notes that may not have one."""
            args = dict(unary=unary if un is None else un, dtb=C.dt_bucket(dt),
                        is_jump=plan.is_jump, slots=slots,
                        hold_kind=hold_kind, rules=rules, f_init=f_init)
            extra, base_no_repeat = timing_penalties(rules, style, dt_s)
            args.update(extra)
            if fsw_bonus is not None:
                args["fsw_bonus"] = fsw_bonus
            if fsw_block is not None and fsw_block.any():
                args["fsw_pen"] = args.get("fsw_pen", np.zeros(n)) + FSW_BLOCK * fsw_block
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
            last_key[0], res, enc, chart, bias = best
            return res, enc, chart, bias, rounds, int(ban.sum())

        cb_info = None
        if links and cb["choreo_bonus"] > 0:
            # answer a repeated passage with the figure it got the first time:
            # a bonus toward the token the step network itself chose on the
            # same tick of the earlier bar. It is part of the unaries from the
            # first solve on, so it costs no extra pass and every re-solve
            # (referee, panel balance, footswitch placement) keeps it.
            extra, anchored = callback_bonus(plan.tick, plan.is_jump, sampled, links,
                                             cb["choreo_bonus"])
            if anchored:
                unary = unary + extra
            cb_info = {"links": len(links), "anchored": int(anchored)}
        elif links:
            cb_info = {"links": len(links), "anchored": 0}
        res, enc, chart, bias, rounds, vetoed = attempt(rules)
        cr_info = None
        if cr is not None and n >= CRITIC_MIN_NOTES:
            # Best-of-K: further samples of the step network on the same notes,
            # each solved like the first; the critic's favourite among those the
            # referees and the panel balance treat no worse than the first wins.
            k_more = cr["k"] - 1
            scales = cr["tau_scale"][1:] or (1.0,)
            if cr["source"] == "logits":
                more_tok, more_logp = resample(logp, k_more, tau, scales, seed)
            else:
                more_tok, more_logp = unmask_many(
                    self.model, ctx, scalars, cv, plan.is_jump, sampled, k_more, cr["steps"],
                    tau, scales, cr["remask"], seed)
            cands = [(sampled, unary, (res, enc, chart, bias, rounds, vetoed), last_key[0])]
            for j in range(k_more):
                un = committed_unary(more_logp[j], more_tok[j])
                if links and cb["choreo_bonus"] > 0:
                    scale = cr["bonus_scale"][(j + 1) % len(cr["bonus_scale"])]
                    un = un + callback_bonus(plan.tick, plan.is_jump, more_tok[j], links,
                                             cb["choreo_bonus"] * scale)[0]
                solved = attempt(rules, un=un)
                cands.append((more_tok[j], un, solved, last_key[0]))
            prep = CR.prepare(x, plan.tick, times_ms, plan.is_jump, difficulty, cr_links,
                              novelty, slots > 0)
            scores = self.model.sess_c.score(np.stack(
                [CR.chart_features(prep, c[2][0].token, c[2][0].f_init) for c in cands]))
            ok = [c[3] <= cands[0][3] for c in cands]
            pick = max((j for j in range(len(cands)) if ok[j]), key=lambda j: (scores[j], -j))
            cr_info = {"k": len(cands), "picked": int(pick),
                       "scores": [round(float(v), 3) for v in scores],
                       "eligible": [bool(v) for v in ok]}
            if pick != 0:
                sampled, unary = cands[pick][0], cands[pick][1]
                res, enc, chart, bias, _, vetoed = cands[pick][2]
            rounds = sum(c[2][4] for c in cands)
        cross_scale = rules.cross_scale
        if opts.stream and not rules.no_cross:
            # A stream is one long run at one speed, and the sampler's taste for
            # crossovers compounds over it: see STREAM_MAX_CROSS.
            asked = float(style[5]) if style[5] >= 0 else 0.0
            cap = max(STREAM_MAX_CROSS, asked)
            for step in STREAM_CROSS_STEPS:
                if _cross_share(res.token) <= cap:
                    break
                cross_scale = rules.cross_scale + step
                r2 = attempt(dataclasses.replace(rules, cross_scale=cross_scale))
                rounds += r2[4]
                if _cross_share(r2[0].token) < _cross_share(res.token):
                    res, enc, chart, bias, vetoed = r2[0], r2[1], r2[2], r2[3], r2[5]
        fsw_info = None
        asked_fsw = float(style[cond.INDEX["footswitch_share"]])
        if asked_fsw > FSW_PLACE_MIN_SHARE and not rules.no_footswitch and n >= 8:
            placed = place_footswitches(
                lambda **kw: attempt(dataclasses.replace(rules, cross_scale=cross_scale), **kw),
                (res, enc, chart, bias, 0, vetoed), unary, plan, times_ms, slots, hold_kind,
                asked_fsw, seed)
            (res, enc, chart, bias, more, vetoed), fsw_info = placed
            rounds += more
        enc.mines = place_mines(plan.mine_tick, enc, seed)
        stats["choreo"] = {"viterbi_score": round(res.score, 2), "relaxed": res.relaxed,
                           "changed_by_viterbi": int((res.token != sampled).sum()),
                           "f_init": res.f_init, "refine_rounds": rounds,
                           "vetoed": vetoed,
                           "panel_bias": [round(float(v), 2) for v in bias]}
        if cross_scale != rules.cross_scale:
            stats["choreo"]["cross_scale"] = cross_scale
        if fsw_info is not None:
            stats["choreo"]["footswitches"] = fsw_info
        if cb_info is not None:
            stats["choreo"]["callbacks"] = cb_info
        if cr_info is not None:
            stats["choreo"]["critic"] = cr_info
        return NeuralChart(C.decode(enc), enc, stats)

    def inpaint_section(self, events: C.ChartEvents, x: np.ndarray,
                        tick_times: np.ndarray, difficulty: str, style: np.ndarray,
                        start_bar: int, end_bar: int, *, seed: int = 0,
                        tau: float = DEFAULT_TAU, has_stems: bool = False,
                        veto=None, crossovers: bool = True) -> NeuralChart:
        """Re-roll the choreography of bars [start_bar, end_bar) of an existing
        chart (spec §6.5 step 4). The rhythm — which ticks carry notes, and
        every freeze — is kept. Every note outside the section keeps its exact
        panels: outside notes are clamped at the PANEL level, so the result is
        identical there bit for bit even though its foot annotation is free to
        re-settle around the new section.
        """
        inside = (events.tick >= start_bar * C.G) & (events.tick < end_bar * C.G)
        stats = {"difficulty": difficulty, "seed": seed, "bars": [start_bar, end_bar],
                 "notes_in_section": int(inside.sum())}
        chart = self._solve_inside(events, inside, x, tick_times, difficulty, style,
                                   TierRules.for_difficulty(difficulty, crossovers=crossovers),
                                   seed=seed, tau=tau,
                                   has_stems=has_stems, veto=veto)
        changed = int(sum(a != b for a, b in zip(chart.events.rows(), events.rows())))
        chart.stats = {**stats, "changed": changed, **chart.stats}
        return chart

    def regenerate_section(self, events: C.ChartEvents, x: np.ndarray,
                           tick_times: np.ndarray, difficulty: str, style: np.ndarray,
                           start_bar: int, end_bar: int, duration_s: float, *,
                           options: DecodeOptions | None = None, seed: int = 0,
                           tau: float = DEFAULT_TAU, has_stems: bool = False,
                           energy_per_bar=None, on_transient: np.ndarray | None = None,
                           veto=None) -> NeuralChart:
        """Write bars [start_bar, end_bar) of an existing chart again — rhythm
        and arrows — with their own settings, and leave the rest alone.

        Where `inpaint_section` keeps the notes and re-rolls only which panels
        they are on, this decodes the range from the audio as `generate` would
        a whole chart, under `style` and `options` (a subdivision cap, Stream
        mode, more or fewer notes, no crossovers, a jump share). Every note
        outside the range keeps its tick, panels and freezes bit for bit; the
        foot path is solved across both boundaries, so the chart is as legal
        going into and out of the range as anywhere else.

        Freezes are the one place the two parts touch. A freeze of the outside
        chart that is still held inside the range keeps its ticks clear of new
        notes; a new freeze is released before the first note after the range.
        """
        opts = options or DecodeOptions()
        rules = TierRules.for_difficulty(difficulty, crossovers=opts.crossovers)
        lo_t, hi_t = int(start_bar) * C.G, int(end_bar) * C.G
        n_ticks = len(tick_times)
        old_inside = (events.tick >= lo_t) & (events.tick < hi_t)
        keep = ~old_inside
        gap_ticks = _min_gap_ticks(tick_times, difficulty)

        # ticks the range may not use: under an outside freeze, or too close to
        # the notes on either side of it
        blocked = np.zeros(n_ticks, bool)
        for p_, k_, d_ in ((events.p0, events.k0, events.d0), (events.p1, events.k1, events.d1)):
            for i in np.nonzero(keep & (k_ > 0) & (d_ > 0))[0]:
                a, b = int(events.tick[i]), int(events.tick[i] + d_[i]) + HOLD_RELEASE_TICKS
                blocked[max(0, a):min(n_ticks, b + 1)] = True
        for tk in events.tick[keep]:
            a, b = int(tk) - gap_ticks + 1, int(tk) + gap_ticks
            blocked[max(0, a):min(n_ticks, max(a, b))] = True

        cv = self.cond_for(style, has_stems)
        plan = decode_rhythm(self.model.rhythm(x, cv), tick_times, difficulty, style,
                             duration_s, energy_per_bar, on_transient, options=opts,
                             density_scale=self.model.density_scale(difficulty),
                             window=(lo_t, hi_t), blocked=blocked,
                             sixteenth_boost=self.model.sixteenth_boost(difficulty),
                             energy_tilt=self.model.energy_tilt(difficulty))

        # a new freeze ends before the chart that follows the range resumes
        after = events.tick[keep & (events.tick >= hi_t)]
        limit = (int(after[0]) if len(after) else n_ticks) - HOLD_RELEASE_TICKS
        min_hold = int(round(BUDGETS[difficulty].hold_len_beats[0] * C.TICKS_PER_BEAT))
        rows = [r for r, kp in zip(events.rows(), keep) if kp]
        for i in range(len(plan.tick)):
            tick, f, d = int(plan.tick[i]), int(plan.flavor[i]), int(plan.dur[i])
            if f and tick + d > limit:
                d = limit - tick
                if d < min_hold:
                    f, d = 0, 0
            # placeholder panels: the solver below chooses the real ones
            if plan.is_jump[i]:
                ka, ks = C.FLAVOR_KINDS[f]
                rows.append((tick, [(C.L, ka, d if ka else 0), (C.R, ks, d if ks else 0)]))
            else:
                kind = 0 if not f else (C.ROLL if f in (C.ROLL_ALT, C.ROLL_SAME) else C.HOLD)
                rows.append((tick, [(C.L if i % 2 == 0 else C.R, kind, d if kind else 0)]))
        mines = [tuple(m) for m in events.mines.tolist() if not lo_t <= m[0] < hi_t]
        draft = C.ChartEvents.from_rows(rows, mines)
        inside = (draft.tick >= lo_t) & (draft.tick < hi_t)
        chart = self._solve_inside(draft, inside, x, tick_times, difficulty, style, rules,
                                   seed=seed, tau=tau, has_stems=has_stems, veto=veto)
        if len(plan.mine_tick):
            chart.encoded.mines = np.asarray(sorted(
                mines + [tuple(m) for m in place_mines(plan.mine_tick, chart.encoded, seed).tolist()]),
                np.int32).reshape(-1, 2)
            chart.events = C.decode(chart.encoded)
        untouched = [r for r, kp in zip(events.rows(), keep) if kp]
        now_outside = [r for r, ins in zip(chart.events.rows(), inside) if not ins]
        if untouched != now_outside:
            raise RuntimeError("re-generating the range changed a note outside it")
        chart.stats = {"difficulty": difficulty, "seed": seed, "bars": [start_bar, end_bar],
                       "notes_before": int(old_inside.sum()), "notes_in_section": int(inside.sum()),
                       "rhythm": plan.stats, "options": opts.to_dict(), **chart.stats}
        return chart

    def _solve_inside(self, events: C.ChartEvents, inside: np.ndarray, x: np.ndarray,
                      tick_times: np.ndarray, difficulty: str, style: np.ndarray,
                      rules: TierRules, *, seed: int = 0, tau: float = DEFAULT_TAU,
                      has_stems: bool = False, veto=None) -> NeuralChart:
        """Choose the panels of the notes marked `inside`, holding every other
        note of `events` to the panels it has. Returns the chart with stats
        {'relaxed', 'refine_rounds', 'vetoed'}."""
        n = len(events)
        times_ms = tick_times[np.minimum(events.tick, len(tick_times) - 1)] * 1000.0
        base, _ = C.encode(events, times_ms)
        if not inside.any():
            return NeuralChart(events, base, {"relaxed": 0, "refine_rounds": 0, "vetoed": 0})
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
        # the tier's time-dependent rules (no fast repeated arrow below Medium,
        # the footswitch cost) hold inside the section as in a whole chart
        extra, no_repeat = timing_penalties(rules, style, dt / 1000.0)
        ban = np.zeros((n, C.N_TOKENS), bool)
        rounds = 0
        while True:
            res = constrained_viterbi(unary, C.dt_bucket(dt), is_jump, slots, hold_kind,
                                      rules, clamp=clamp, ban=ban, no_repeat=no_repeat,
                                      **extra)
            new = _rebuild_events(events, base, res, inside)
            enc = encoded_from_path(new, res)
            if not C.decode(enc).equals(new):
                raise RuntimeError("inpainting produced an inconsistent foot path")
            chart = NeuralChart(new, enc, {})
            rounds += 1
            again = False
            if veto and rounds <= MAX_REFINE_ROUNDS:
                for i in veto(chart):                 # only notes we may change
                    if inside[i] and not ban[i, res.token[i]]:
                        ban[i, res.token[i]] = again = True
            if not again:
                break
        chart.stats = {"relaxed": res.relaxed, "refine_rounds": rounds,
                       "vetoed": int(ban.sum())}
        return chart


def _min_gap_ticks(tick_times: np.ndarray, difficulty: str) -> int:
    """The tier's shortest gap between two notes, in ticks at this tempo."""
    tick_s = float(np.median(np.diff(tick_times))) if len(tick_times) > 1 else 0.04
    return max(1, int(math.ceil(min_gap_s(difficulty) / max(tick_s, 1e-6) - 1e-6)))


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
