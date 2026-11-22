"""structure.py — where a song repeats itself, from its grid features alone.

No weights, numpy only, a few milliseconds per song. Three things come out of
`analyze_structure`:

* `section_novelty` per tick, 0..1: how strongly the music changes at the bar
  line that opens this bar (verse -> chorus, a drop, a bridge). A checkerboard
  kernel slid down the diagonal of the bar-by-bar self-similarity matrix.
* `prior_repeat_sim` per tick, 0..1: how much this bar sounds like the most
  similar bar at least four bars earlier — "we have been here before".
* `motif_links`: for a bar that repeats an earlier passage, which earlier bar
  it repeats, so the decoder can answer a repeated passage with the steps it
  wrote the first time (`infer` callbacks).

How two bars are compared. A bar is described by its rhythm (the six spectral
flux bands, pooled to the sixteen 16th-note slots of the bar), its timbre (the
mean of the twenty mel bands) and its harmony (the mean chroma contour) — and,
once a voice network fills them in, by the voice channels pooled the same way.
Each dimension is standardized over the bars of the song before the vectors
are normalized: two bars of one song share most of their spectrum, and the raw
cosine would call everything alike. After centring, unrelated bars sit near 0
and a repeated bar near 1.

A single bar is a weak witness (any four-on-the-floor bar resembles any
other). A link therefore needs the 4-bar block around both bars to match:
the mean similarity along the diagonal of the block.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import codec as C
from . import grid_features as GF

#: Ticks of one descriptor slot: a 16th note.
SLOT_TICKS = 3
SLOTS_PER_BAR = C.G // SLOT_TICKS
#: Bars in the block that has to match for a link (and in the audit's pairs).
BLOCK_BARS = 4
#: A bar's own neighbourhood is not a repeat: how far back a repeat must be.
MIN_REPEAT_BARS = 4
#: ...and how far back a motif link must reach.
MIN_LINK_BARS = 8
#: A link connects bars a whole number of these apart: repeats come in phrases.
LINK_STEP_BARS = 4
#: Block similarity from which a bar is linked to an earlier one.
TAU_SIM = 0.85
#: Among earlier blocks this close to the best one, the EARLIEST is linked: a
#: third chorus answers the first, not the second, so every repeat of a
#: passage points at one statement of it.
EARLIEST_WITHIN = 0.03
#: Half-width, in bars, of the checkerboard kernel (the kernel is 2x this).
NOVELTY_HALF_BARS = 4
#: Relative weight of the three descriptor groups (rhythm, timbre, harmony)
#: and of the voice channels when given.
GROUP_WEIGHTS = {"rhythm": 1.0, "timbre": 0.6, "harmony": 0.6, "voice": 1.0}
CHROMA = slice(48, 53)


@dataclass
class Structure:
    n_bars: int
    ssm: np.ndarray                 # (M, M) bar cosine similarity, -1..1
    block_sim: np.ndarray           # (M, M) mean of the 4-bar diagonal starting at (i, j)
    novelty: np.ndarray             # (M,) 0..1, change at the bar line opening bar i
    repeat_sim: np.ndarray          # (M,) 0..1, best match at least MIN_REPEAT_BARS back
    section_novelty: np.ndarray     # (T,) float32, per tick
    prior_repeat_sim: np.ndarray    # (T,) float32, per tick
    motif_links: list = field(default_factory=list)   # (bar, earlier bar, sim, offset in block)

    def channels(self) -> np.ndarray:
        """(T, 2): what goes into X[:, 46:48]."""
        return np.stack([self.section_novelty, self.prior_repeat_sim], axis=1)

    def link_of(self) -> dict:
        """bar -> (earlier bar, similarity)."""
        return {int(b): (int(j), float(s)) for b, j, s, _ in self.motif_links}


def bar_descriptors(x: np.ndarray, voice_feats: np.ndarray | None = None) -> np.ndarray:
    """(T, >=54) grid features -> (M, D) unit vectors, one per whole bar.

    `voice_feats`, (T, K), are extra per-tick channels (a voice network's
    onsets and activities) pooled like the flux bands."""
    x = np.asarray(x, np.float32)
    m = len(x) // C.G
    if m == 0:
        return np.zeros((0, 1), np.float32)
    t = m * C.G

    def slots(a: np.ndarray, how: str) -> np.ndarray:
        a = a[:t].reshape(m, SLOTS_PER_BAR, SLOT_TICKS, a.shape[1])
        a = a.max(axis=2) if how == "max" else a.mean(axis=2)
        return a.reshape(m, -1)

    groups = [("rhythm", slots(x[:, GF.FLUX], "max")),
              ("timbre", x[:t, GF.MEL].reshape(m, C.G, -1).mean(axis=1)),
              ("harmony", x[:t, CHROMA].reshape(m, C.G, -1).mean(axis=1))]
    if voice_feats is not None and np.asarray(voice_feats).size:
        groups.append(("voice", slots(np.asarray(voice_feats, np.float32), "max")))
    parts = []
    for name, g in groups:
        g = g - g.mean(axis=0, keepdims=True)
        sd = g.std(axis=0, keepdims=True)
        g = g / np.maximum(sd, 1e-6) * (sd > 1e-6)
        norm = np.linalg.norm(g, axis=1, keepdims=True)
        # each group contributes its weight to the squared length of the vector
        parts.append(g / np.maximum(norm, 1e-9) * np.sqrt(GROUP_WEIGHTS[name]))
    d = np.concatenate(parts, axis=1)
    return (d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)).astype(np.float32)


def block_similarity(ssm: np.ndarray, bars: int = BLOCK_BARS) -> np.ndarray:
    """B[i, j] = mean of ssm[i + k, j + k] for k < bars: how alike the `bars`-bar
    blocks starting at bars i and j are. Rows and columns too close to the end
    for a whole block are -1."""
    m = len(ssm)
    out = np.full((m, m), -1.0, np.float32)
    if m < bars:
        return out
    n = m - bars + 1
    acc = np.zeros((n, n), np.float32)
    for k in range(bars):
        acc += ssm[k:k + n, k:k + n]
    out[:n, :n] = acc / bars
    return out


def _novelty(ssm: np.ndarray, half: int = NOVELTY_HALF_BARS) -> np.ndarray:
    m = len(ssm)
    nov = np.zeros(m, np.float32)
    if m < 2 * half:
        return nov
    sign = np.ones((2 * half, 2 * half), np.float32)
    sign[:half, half:] = -1.0
    sign[half:, :half] = -1.0
    taper = np.hanning(2 * half + 2)[1:-1]
    kernel = sign * np.outer(taper, taper)
    for b in range(half, m - half + 1):
        nov[b] = float((ssm[b - half:b + half, b - half:b + half] * kernel).sum())
    nov = np.maximum(nov, 0.0)
    top = float(nov.max())
    return nov / top if top > 0 else nov


def analyze_structure(x_acoustics: np.ndarray, voice_feats: np.ndarray | None = None,
                      tau_sim: float = TAU_SIM) -> Structure:
    """Grid features of a whole song -> its `Structure` (see the module text).

    `x_acoustics` is X as the networks see it, (T, 64); only its acoustic
    channels are read. `voice_feats` are a voice network's channels, used as a
    fourth descriptor group when given."""
    t = len(x_acoustics)
    d = bar_descriptors(x_acoustics, voice_feats)
    m = len(d)
    # (Apple's BLAS raises spurious floating-point flags on this product of
    # unit vectors; the result is finite — the tests check it)
    with np.errstate(all="ignore"):
        ssm = (d @ d.T).astype(np.float32) if m else np.zeros((0, 0), np.float32)
    blk = block_similarity(ssm)
    nov = _novelty(ssm)
    rep = np.zeros(m, np.float32)
    if m > MIN_REPEAT_BARS:
        low = np.tril(np.clip(ssm, 0.0, 1.0), k=-MIN_REPEAT_BARS)
        rep = low.max(axis=1).astype(np.float32)

    links = []
    for b in range(MIN_LINK_BARS, m):
        cands = []                                      # (sim, earlier bar, offset)
        for o in range(BLOCK_BARS):
            i = b - o                                   # the block containing b starts here
            if i < 0 or i + BLOCK_BARS > m:
                continue
            j = np.arange(i % LINK_STEP_BARS, i - MIN_LINK_BARS + 1, LINK_STEP_BARS)
            if len(j) == 0:
                continue
            s = blk[i, j]
            cands += [(float(v), int(jj) + o, o) for v, jj in zip(s, j) if v >= tau_sim]
        if cands:
            top = max(c[0] for c in cands)
            sim, earlier, o = min((c for c in cands if c[0] >= top - EARLIEST_WITHIN),
                                  key=lambda c: (c[1], -c[0]))
            links.append((b, earlier, round(sim, 4), o))

    def per_tick(v: np.ndarray) -> np.ndarray:
        out = np.zeros(t, np.float32)
        if m:
            out[:m * C.G] = np.repeat(v, C.G)
            out[m * C.G:] = v[-1]
        return out

    return Structure(m, ssm, blk, nov, rep, per_tick(nov), per_tick(rep), links)
