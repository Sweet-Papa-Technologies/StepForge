"""event_features.py — the per-note inputs of Stage B (ChoreoNet).

Stage B never sees empty ticks: it runs on the N active notes Stage A selected.
For each note it gets a short window of the beat-warped audio tensor around
the note's tick, plus a handful of scalars describing the note itself. Both are
built here, in numpy, and used unchanged by the trainer and by the ONNX runtime
path — so the two cannot drift apart.

Nothing here depends on which FOOT takes a note. That is Stage B's output, so
it cannot be its input: the hold information is "how many freezes are in
progress at this note" (a fact of the rhythm), not "is the left foot pinned".
"""
from __future__ import annotations

import numpy as np

from . import codec as C

CTX_HALF = 8
CTX_K = 2 * CTX_HALF + 1            # 17 ticks: a little over a quarter note
N_SCALARS = 31


def event_context(x: np.ndarray, ticks: np.ndarray) -> np.ndarray:
    """X (T, D) and note ticks (N,) -> (N, CTX_K, D), zero-padded at the ends."""
    t, d = x.shape
    idx = np.asarray(ticks, np.int64)[:, None] + np.arange(-CTX_HALF, CTX_HALF + 1)[None, :]
    ok = (idx >= 0) & (idx < t)
    out = x[np.clip(idx, 0, t - 1)]
    out[~ok] = 0.0
    return out.astype(np.float32, copy=False)


def active_holds(ticks: np.ndarray, flavor: np.ndarray, dur_alt: np.ndarray,
                 dur_same: np.ndarray) -> np.ndarray:
    """Per note, how many freezes started EARLIER are still down (0, 1 or 2).
    A freeze started at tick t with length d covers notes with t < tick <= t+d."""
    ticks = np.asarray(ticks, np.int64)
    n = len(ticks)
    diff = np.zeros(n + 1, np.int64)
    kinds = np.asarray(C.FLAVOR_KINDS)[np.asarray(flavor, np.int64)]      # (N, 2)
    for col, dur in ((0, dur_alt), (1, dur_same)):
        dur = np.asarray(dur, np.int64)
        for i in np.nonzero((kinds[:, col] > 0) & (dur > 0))[0]:
            j1 = int(np.searchsorted(ticks, ticks[i] + dur[i], side="right"))
            if j1 > i + 1:
                diff[i + 1] += 1
                diff[j1] -= 1
    return np.minimum(np.cumsum(diff[:n]), 2)


def event_scalars(ticks: np.ndarray, times_ms: np.ndarray, is_jump: np.ndarray,
                  flavor: np.ndarray, dur_alt: np.ndarray, dur_same: np.ndarray
                  ) -> np.ndarray:
    """(N, 31) float32:
        0..1   log gap to the previous / next note (ms / 100), clipped
        2..3   position in the measure, sin / cos
        4..9   subdivision class one-hot (4th .. 48th)
        10     is a jump
        11..16 hold flavor one-hot (the six Stage A classes)
        17..27 hold length one-hot: none + the ten duration buckets
        28..30 freezes already in progress: 0 / 1 / 2
    """
    ticks = np.asarray(ticks, np.int64)
    n = len(ticks)
    f = np.zeros((n, N_SCALARS), np.float32)
    if n == 0:
        return f
    t = np.asarray(times_ms, np.float64)
    gap = np.diff(t)
    prev = np.concatenate([[2000.0], gap])
    nxt = np.concatenate([gap, [2000.0]])
    f[:, 0] = np.clip(np.log(np.maximum(prev, 1.0) / 100.0), -3.0, 3.0)
    f[:, 1] = np.clip(np.log(np.maximum(nxt, 1.0) / 100.0), -3.0, 3.0)
    phase = 2.0 * np.pi * (ticks % C.G) / C.G
    f[:, 2], f[:, 3] = np.sin(phase), np.cos(phase)
    r = ticks % C.TICKS_PER_BEAT
    cls = np.full(n, 5)
    cls[r % 2 == 0] = 4
    cls[r % 3 == 0] = 3
    cls[r % 4 == 0] = 2
    cls[r % 6 == 0] = 1
    cls[r == 0] = 0
    rows = np.arange(n)
    f[rows, 4 + cls] = 1.0
    f[:, 10] = np.asarray(is_jump, np.float32)
    fl6 = C.flavor6(flavor).astype(np.int64)
    f[rows, 11 + fl6] = 1.0
    dur = np.maximum(np.asarray(dur_alt, np.int64), np.asarray(dur_same, np.int64))
    bucket = np.where((fl6 > 0) & (dur > 0), 1 + C.dur_bucket(np.maximum(dur, 1)), 0)
    f[rows, 17 + bucket] = 1.0
    f[rows, 28 + active_holds(ticks, flavor, dur_alt, dur_same)] = 1.0
    return f
