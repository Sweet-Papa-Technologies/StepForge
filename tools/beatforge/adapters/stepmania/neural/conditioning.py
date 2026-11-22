"""conditioning.py — the 10-D style descriptor, its 19-D network encoding,
presets and slider handling (spec §5, REQ-SFN-04).

    0 meter_norm        clamp(meter, 1, 15) / 15
    1 nps_norm          clamp(80th-percentile 4 s NPS, 0.5, 16) / 16
    2 stream_share      steps inside >=8-note runs of 16ths or faster   [0, 1]
    3 jump_share        2-panel events / events                         [0, .35]
    4 hold_share        events carrying a hold or roll / events         [0, .35]
    5 crossover_share   single steps that land CROSS                    [0, .30]
    6 jack_share        single steps repeating the same foot + panel    [0, .20]
    7 footswitch_share  ALT steps landing on the other foot's D/U panel [0, .20]
    8 mine_share        mines / events                                  [0, .15]
    9 has_stems         1 when the Demucs stem channels are populated

The network sees 19 numbers: the ten values, then an availability bit for each
of indices 0..8. A masked value is -1.0 with its bit at 0, which is how "only a
difficulty was given, infer the style from the audio" is expressed.
"""
from __future__ import annotations

import numpy as np

from ..grammar import BUDGETS
from . import codec as C

STYLE_DIM = 10
COND_DIM = 19
FIELDS = ("meter_norm", "nps_norm", "stream_share", "jump_share", "hold_share",
          "crossover_share", "jack_share", "footswitch_share", "mine_share",
          "has_stems")
INDEX = {f: i for i, f in enumerate(FIELDS)}
#: Upper bound of each field (lower bound is 0, or 0.5/16 for nps_norm).
MAXIMA = np.array([1.0, 1.0, 1.0, 0.35, 0.35, 0.30, 0.20, 0.20, 0.15, 1.0], np.float32)
STYLE_SLICE = slice(1, 9)          # the eight droppable style sliders
MASKED = -1.0

NPS_WINDOW_S = 4.0
NPS_PERCENTILE = 80.0
STREAM_MIN_RUN = 8
STREAM_MAX_GAP_TICKS = 3           # a 16th note on the 48-tick grid


def sustained_nps(times_s: np.ndarray) -> float:
    """80th percentile of note counts in sliding 4-second windows (1 s hop)."""
    t = np.sort(np.asarray(times_s, np.float64))
    if len(t) == 0:
        return 0.0
    span = t[-1] - t[0]
    if span <= NPS_WINDOW_S:
        return len(t) / NPS_WINDOW_S
    starts = np.arange(t[0], t[-1] - NPS_WINDOW_S + 1e-9, 1.0)
    counts = (np.searchsorted(t, starts + NPS_WINDOW_S, side="left")
              - np.searchsorted(t, starts, side="left"))
    return float(np.percentile(counts, NPS_PERCENTILE) / NPS_WINDOW_S)


def peak_nps(times_s: np.ndarray, window_s: float = 4.0) -> float:
    """Max notes/second over any `window_s` window starting at a note."""
    t = np.sort(np.asarray(times_s, np.float64))
    if len(t) == 0:
        return 0.0
    hi = np.searchsorted(t, t + window_s, side="left")
    return float((hi - np.arange(len(t))).max() / window_s)


def stream_share(ticks: np.ndarray, is_jump: np.ndarray) -> float:
    """Share of events inside contiguous runs of >= STREAM_MIN_RUN single steps
    spaced a 16th or closer."""
    n = len(ticks)
    if n < STREAM_MIN_RUN:
        return 0.0
    ok = (np.diff(ticks) <= STREAM_MAX_GAP_TICKS) & ~is_jump[1:] & ~is_jump[:-1]
    in_run = np.zeros(n, bool)
    i = 0
    while i < n - 1:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n - 1 and ok[j]:
            j += 1
        if j - i + 1 >= STREAM_MIN_RUN:
            in_run[i:j + 1] = True
        i = j
    return float(in_run.mean())


def effective_meter(nps: float) -> float:
    """A density-derived meter on the classic 1-10+ ITG/DDR scale.

    The number printed on a chart is not one scale. Measured over 19,806 human
    charts, Hard charts rated 9 sustain 4.8 notes/s and those rated 10-12
    sustain 4.2-4.5: the corpus mixes the classic scale with the wider DDR-X
    one, so the label alone would tell the model contradictory things. Where
    the scales agree (classic ratings 1-10) sustained NPS grows by about 0.5
    per meter step, and that line is what this inverts. Training conditions on
    this value; a tier's default meter maps back to a sustained rate through
    `nps_for_meter`.
    """
    return float(np.clip(2.0 * nps - 0.7, 1.0, 15.0))


def nps_for_meter(meter: float) -> float:
    """Inverse of `effective_meter`: the sustained notes/s a meter stands for."""
    return float(np.clip((meter + 0.7) / 2.0, 0.5, 16.0))


def training_style(style: np.ndarray) -> np.ndarray:
    """A corpus chart's stored style vector as the networks should see it:
    index 0 replaced by the density-derived meter (see `effective_meter`)."""
    out = np.asarray(style, np.float32).copy()
    out[0] = effective_meter(float(out[1]) * 16.0) / 15.0
    return out


def footswitch_flags(enc: C.EncodedChart) -> np.ndarray:
    """Per note: a footswitch as the style vector counts them — an ALT token
    landing on the D/U panel the other foot rests on."""
    tok = np.asarray(enc.token, np.int64)
    if len(tok) == 0:
        return np.zeros(0, bool)
    before, _, _ = C.foot_path(enc)
    pl, pr, fl = (before >> 1) // 4, (before >> 1) % 4, before & 1
    alt = tok < 4
    alt_foot = 1 - fl
    alt_panel = np.where(alt_foot == C.LEFT, tok & 3,
                         np.asarray(C.PANEL_OF[C.RIGHT])[tok & 3])
    other_pos = np.where(alt_foot == C.LEFT, pr, pl)
    fsw = alt & (alt_panel == other_pos) & ((alt_panel == C.D) | (alt_panel == C.U))
    fsw[0] = False                       # nothing precedes the first step
    return fsw


def compute_style_vector(enc: C.EncodedChart, times_s, meter: float,
                         has_stems: bool = False) -> np.ndarray:
    """Empirical style of an encoded chart -> c in R^10, every field inside its
    documented range. `times_s` are the events' times in seconds."""
    c = np.zeros(STYLE_DIM, np.float32)
    c[0] = np.clip(meter, 1, 15) / 15.0
    c[9] = 1.0 if has_stems else 0.0
    n = len(enc)
    if n == 0:
        c[1] = 0.5 / 16.0
        return c
    tok = np.asarray(enc.token, np.int64)
    is_jump = tok >= 8
    single = ~is_jump
    n_single = max(1, int(single.sum()))
    before, _, _ = C.foot_path(enc)
    pl, pr, fl = (before >> 1) // 4, (before >> 1) % 4, before & 1

    c[1] = np.clip(sustained_nps(times_s), 0.5, 16.0) / 16.0
    c[2] = stream_share(np.asarray(enc.tick), is_jump)
    c[3] = is_jump.mean()
    c[4] = (np.asarray(enc.flavor) > 0).mean()
    c[5] = (single & ((tok & 3) == C.CROSS)).sum() / n_single

    # jack: SAME token landing where that foot already stands
    same = (tok >= 4) & (tok < 8)
    same_pos = np.where(fl == C.LEFT, pl, pr)
    same_panel = np.where(fl == C.LEFT, tok & 3, np.asarray(C.PANEL_OF[C.RIGHT])[tok & 3])
    c[6] = (same & (same_panel == same_pos)).sum() / n_single

    c[7] = footswitch_flags(enc).sum() / n_single

    c[8] = len(enc.mines) / n
    lo = np.zeros(STYLE_DIM, np.float32)
    lo[1] = 0.5 / 16.0
    return np.clip(c, lo, MAXIMA)


def cond19(style, drop=None) -> np.ndarray:
    """Encode a style vector for the networks: 10 values + 9 availability bits
    (indices 0..8). `drop` is an iterable of indices to mask; a value already
    equal to MASKED (or NaN) is treated as dropped too."""
    c = np.asarray(style, np.float32).copy()
    avail = np.ones(9, np.float32)
    for i in (drop or ()):
        if 0 <= i < 9:
            avail[i] = 0.0
    gone = ~np.isfinite(c[:9]) | (c[:9] < 0)
    avail[gone] = 0.0
    c[:9] = np.where(avail > 0, c[:9], MASKED)
    return np.concatenate([c, avail]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Presets & sliders
# --------------------------------------------------------------------------- #
PRESET_NAMES = ("casual_flow", "itg_stamina", "tech_footswitch", "jumpstream")

# Style sliders per preset as a fraction of what the TIER allows, so one recipe
# yields a legal vector at every difficulty: `nps` scales the tier's max_nps_4s,
# the rest scale that field's tier ceiling (see `tier_ceilings`).
_PRESET_RECIPES = {
    "casual_flow":     dict(nps=0.70, stream=0.20, jump=0.50, hold=0.60,
                            cross=0.15, jack=0.20, fsw=0.00, mine=0.0),
    "itg_stamina":     dict(nps=0.95, stream=1.00, jump=0.15, hold=0.15,
                            cross=0.50, jack=0.05, fsw=0.00, mine=0.0),
    "tech_footswitch": dict(nps=0.85, stream=0.60, jump=0.30, hold=0.40,
                            cross=1.00, jack=0.60, fsw=1.00, mine=0.4),
    "jumpstream":      dict(nps=0.90, stream=0.70, jump=1.00, hold=0.20,
                            cross=0.25, jack=0.30, fsw=0.00, mine=0.0),
}

_JUMP_CEIL = {"downbeats": 0.03, "accents": 0.07, "free": 0.12}
_CROSS_CEIL = {"none": 0.0, "light": 0.06, "moderate": 0.15}
_STREAM_CEIL = {1: 0.0, 2: 0.0, 4: 0.10, 8: 0.45, 16: 0.80}


def tier_ceilings(difficulty: str) -> dict:
    """The most each slider may ask for at a difficulty, from grammar.BUDGETS."""
    b = BUDGETS[difficulty]
    return {
        "nps": b.max_nps_4s,
        "stream": _STREAM_CEIL.get(b.max_run, 0.45),
        "jump": _JUMP_CEIL.get(b.jumps, 0.07),
        "hold": b.hold_share[1],
        "cross": _CROSS_CEIL.get(b.crossover, 0.06),
        "jack": 0.02 * b.jack_limit,
        "fsw": 0.06 if b.footswitch else 0.0,
        "mine": 0.05 if b.name in ("hard", "challenge") else 0.0,
    }


def default_meter(difficulty: str) -> int:
    lo, hi = BUDGETS[difficulty].meter_range
    return int(round((lo + hi) / 2))


def clamp_to_budget(style, difficulty: str) -> np.ndarray:
    """Clamp a style vector into what `difficulty` permits. Masked fields stay
    masked — the tier limits are then enforced on the decoded chart instead."""
    c = np.asarray(style, np.float32).copy()
    ceil = tier_ceilings(difficulty)
    lo_m, hi_m = BUDGETS[difficulty].meter_range
    if c[0] >= 0:
        c[0] = np.clip(c[0], lo_m / 15.0, hi_m / 15.0)
    tops = (ceil["nps"] / 16.0, ceil["stream"], ceil["jump"], ceil["hold"],
            ceil["cross"], ceil["jack"], ceil["fsw"], ceil["mine"])
    for i, top in zip(range(1, 9), tops):
        if c[i] >= 0:
            c[i] = np.clip(c[i], 0.5 / 16.0 if i == 1 else 0.0, min(top, MAXIMA[i]))
    return c


def preset_vector(difficulty: str, preset: str, has_stems: bool = False) -> np.ndarray:
    r = _PRESET_RECIPES[preset]
    ceil = tier_ceilings(difficulty)
    c = np.zeros(STYLE_DIM, np.float32)
    c[0] = default_meter(difficulty) / 15.0
    c[1] = max(0.5, r["nps"] * ceil["nps"]) / 16.0
    for i, k in zip(range(2, 9), ("stream", "jump", "hold", "cross", "jack", "fsw", "mine")):
        c[i] = r[k] * ceil[k]
    c[9] = 1.0 if has_stems else 0.0
    return clamp_to_budget(c, difficulty)


#: (difficulty, preset_name) -> style vector, every one inside grammar.BUDGETS.
PRESETS = {(d, p): preset_vector(d, p) for d in BUDGETS for p in PRESET_NAMES}


def tier_only_vector(difficulty: str, meter: int | None = None,
                     has_stems: bool = False) -> np.ndarray:
    """Difficulty given, style left to the model: sliders 1..8 masked."""
    c = np.full(STYLE_DIM, MASKED, np.float32)
    c[0] = (meter if meter is not None else default_meter(difficulty)) / 15.0
    c[9] = 1.0 if has_stems else 0.0
    return clamp_to_budget(c, difficulty)


def resolve_style(difficulty: str, preset: str | None = None,
                  overrides: dict | None = None, meter: int | None = None,
                  has_stems: bool = False) -> np.ndarray:
    """What the CLI / GUI hand in -> a clamped style vector.

    Start from the preset if named, else from a tier-only (masked) vector; then
    apply `overrides` ({"jump_share": 0.12, ...}, the Tauri sliders). Unknown
    keys are an error, since a typo would otherwise silently do nothing.
    """
    if preset:
        if preset not in _PRESET_RECIPES:
            raise ValueError(f"unknown style preset '{preset}' "
                             f"(choose from {', '.join(PRESET_NAMES)})")
        c = preset_vector(difficulty, preset, has_stems)
    else:
        c = tier_only_vector(difficulty, None, has_stems)
    if meter is not None:
        c[0] = np.clip(meter, 1, 15) / 15.0
    for key, val in (overrides or {}).items():
        if key == "meter":
            c[0] = np.clip(float(val), 1, 15) / 15.0
        elif key == "nps":
            c[1] = np.clip(float(val), 0.5, 16.0) / 16.0
        elif key in INDEX and key != "has_stems":
            c[INDEX[key]] = float(val)
        else:
            raise ValueError(f"unknown style field '{key}' "
                             f"(choose from meter, nps, {', '.join(FIELDS[1:9])})")
    return clamp_to_budget(c, difficulty)
