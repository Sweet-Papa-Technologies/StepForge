"""critic.py — a small referee for choreography (v1.6 Phase 3).

The rhythm network decides WHEN a chart steps; the step network decides which
arrows. For one rhythm the step network can be sampled several times, and the
samples differ only in their arrows. This module scores such a sample on what
can differ between them, and nothing else:

* callbacks       where the music repeats (`structure.motif_links`), do the
                  arrows come back too
* variety         entropy of the 4- and 8-note figures, short loops
* contour         do the arrows move when the pitch moves, and does the
                  footwork change where the song changes section
* ergonomics      discomfort of the foot path, crossovers, jacks, panel balance

`chart_features` turns one sample into `N_FEATURES` numbers (numpy only, about
a millisecond); `stage_3_critic.onnx` (a two-layer MLP, tens of kilobytes,
trained by `train/train_critic.py` to tell human charts from the same charts
with their arrows damaged) turns them into one score. `infer` generates K
samples on the same ticks and keeps the best-scored one that passes every
hard rule (`decoder.critic` in model_meta.json; off unless declared).

Everything here reads tokens, so a chart and its left-right mirror score the
same up to the two features that name a side, and those use absolute values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import codec as C
from . import grid_features as GF

DIFFICULTIES = ("beginner", "easy", "medium", "hard", "challenge")
#: Pitch height (the mean of the two middle quantiles) and spectral centroid.
PITCH_COLS = (GF.CHROMA.start + 1, GF.CHROMA.start + 2)
CENTROID_COL = GF.CHROMA.start + 4
#: Up is "higher" than Down; Left and Right are level.
PANEL_H = (0, -1, 1, 0)
#: Notes in the window local panel balance is measured in, and its hop.
LOCAL_WINDOW = 16
LOCAL_HOP = 8
#: A bar line is a section boundary from this novelty (0..1) when it is also
#: the largest within two bars either side.
SECTION_NOVELTY = 0.30
SECTION_BARS = 4
#: Discomfort is capped here (a double-step at speed): one impossible step
#: should not outweigh a chart.
COST_CAP = 100.0
HARD_STEP = 60.0
LOOP_CAP = 64

FEATURES = (
    # ergonomics and balance
    "cost_mean", "cost_p90", "hard_share", "cross_share", "jack_share", "dstep_share",
    "fsw_share", "crossed_share", "panel_lr", "panel_lr_diff", "panel_du_diff",
    "panel_max", "panel_min", "local_max_mean", "local_max_p90", "foot_balance",
    # variety
    "tok4_entropy", "tok8_entropy", "panel4_entropy", "top4_share", "distinct8",
    "period2", "period3", "period4", "loop_run", "bar_distinct",
    # callbacks
    "cb_coverage", "cb_token", "cb_panel", "cb_chance", "cb_lift", "cb_bar",
    # contour and sections
    "contour_x", "contour_h", "centroid_x", "centroid_h", "contour_travel",
    "section_change", "bar_change", "section_contrast",
    # what the chart is (the same for every sample of one rhythm)
    "has_links", "has_sections", "is_beginner", "is_easy", "is_medium", "is_hard",
    "is_challenge", "notes_per_bar", "jump_share", "log_notes",
)
N_FEATURES = len(FEATURES)
INDEX = {f: i for i, f in enumerate(FEATURES)}
#: The features that cannot differ between two samples of one rhythm.
CONTEXT = slice(INDEX["has_links"], N_FEATURES)


@dataclass
class Prepared:
    """What `chart_features` needs that is the same for every sample on one
    set of ticks (`prepare`)."""
    n: int
    difficulty: int
    is_jump: np.ndarray             # (N,) bool
    bar: np.ndarray                 # (N,) int
    in_bar: np.ndarray              # (N,) tick within the bar
    dtb: np.ndarray                 # (N,) gap bucket for the cost table
    free: np.ndarray                # (N,) bool: no freeze is down at this note
    link_later: np.ndarray          # (P,) note index in the repeating bar
    link_earlier: np.ndarray        # (P,) ...and of the same tick of the bar it repeats
    link_bar: np.ndarray            # (P,) the repeating bar
    d_pitch: np.ndarray             # (N,) pitch height at this note minus at the one before
    d_centroid: np.ndarray          # (N,)
    near: np.ndarray                # (N,) bool: within two beats of the note before
    boundaries: np.ndarray          # bars that open a section
    others: np.ndarray              # 4-bar lines that do not
    n_bars: int


def prepare(x: np.ndarray, ticks: np.ndarray, times_ms: np.ndarray, is_jump: np.ndarray,
            difficulty: str, links=(), novelty: np.ndarray | None = None,
            held: np.ndarray | None = None) -> Prepared:
    """`x` (T, 64) grid features, the chart's note ticks and times, the bars
    `structure.analyze_structure` links (`motif_links`) and its per-bar
    `novelty`. `held` marks notes played while a freeze is down: their
    discomfort is the freeze's doing and is left out."""
    ticks = np.asarray(ticks, np.int64)
    n = len(ticks)
    is_jump = np.asarray(is_jump, bool)
    bar, in_bar = ticks // C.G, ticks % C.G
    dt = np.concatenate([[1e9], np.diff(np.asarray(times_ms, np.float64))]) if n else np.zeros(0)
    index = {int(t): i for i, t in enumerate(ticks)}
    later, earlier, lbar = [], [], []
    for b, j, _sim, _o in links:
        lo, hi = np.searchsorted(ticks, [b * C.G, (b + 1) * C.G])
        for i in range(lo, hi):
            k = index.get(int(ticks[i]) - (b - j) * C.G)
            if k is not None and is_jump[i] == is_jump[k]:
                later.append(i)
                earlier.append(k)
                lbar.append(b)
    x = np.asarray(x, np.float32)
    at = np.clip(ticks, 0, max(0, len(x) - 1))
    if len(x) and n:
        # a beat of smoothing: the contour of the phrase, not of one transient
        kern = np.ones(C.TICKS_PER_BEAT, np.float32) / C.TICKS_PER_BEAT
        pitch = np.convolve(x[:, PITCH_COLS[0]] + x[:, PITCH_COLS[1]], kern, "same")[at]
        cent = np.convolve(x[:, CENTROID_COL], kern, "same")[at]
    else:
        pitch = cent = np.zeros(n, np.float32)
    d_pitch = np.concatenate([[0.0], np.diff(pitch)]) if n else np.zeros(0)
    d_cent = np.concatenate([[0.0], np.diff(cent)]) if n else np.zeros(0)
    near = np.concatenate([[False], np.diff(ticks) <= 2 * C.TICKS_PER_BEAT]) if n else np.zeros(0, bool)
    n_bars = int(bar.max()) + 1 if n else 0
    bounds, others = [], []
    if novelty is not None and len(novelty):
        nov = np.asarray(novelty, np.float32)
        for b in range(SECTION_BARS, min(len(nov), n_bars) - SECTION_BARS + 1):
            if nov[b] >= SECTION_NOVELTY and nov[b] >= nov[max(0, b - 2):b + 3].max():
                bounds.append(b)
    taken = set(bounds)
    for b in range(SECTION_BARS, n_bars - SECTION_BARS + 1, SECTION_BARS):
        if not any(abs(b - q) < SECTION_BARS for q in taken):
            others.append(b)
    return Prepared(
        n=n, difficulty=DIFFICULTIES.index(difficulty), is_jump=is_jump, bar=bar, in_bar=in_bar,
        dtb=np.asarray(C.dt_bucket(dt), np.int64),
        free=np.ones(n, bool) if held is None else ~np.asarray(held, bool),
        link_later=np.asarray(later, np.int64), link_earlier=np.asarray(earlier, np.int64),
        link_bar=np.asarray(lbar, np.int64), d_pitch=d_pitch.astype(np.float64),
        d_centroid=d_cent.astype(np.float64), near=near,
        boundaries=np.asarray(bounds, np.int64), others=np.asarray(others, np.int64),
        n_bars=n_bars)


def _entropy(codes: np.ndarray, k: int, base: int) -> tuple:
    """Normalized entropy of the k-grams of `codes`, the share of the most
    common one, and distinct / total."""
    m = len(codes) - k + 1
    if m < 2:
        return 0.0, 1.0, 0.0
    key = np.zeros(m, np.int64)
    for j in range(k):
        key = key * base + codes[j:j + m]
    _, cnt = np.unique(key, return_counts=True)
    p = cnt / m
    return float(-(p * np.log(p)).sum() / np.log(m)), float(p.max()), float(len(cnt) / m)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 8:
        return 0.0
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _longest_run(flags: np.ndarray) -> int:
    if not len(flags) or not flags.any():
        return 0
    edge = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return int((np.nonzero(edge == -1)[0] - np.nonzero(edge == 1)[0]).max())


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p / max(1e-9, p.sum()), q / max(1e-9, q.sum())
    m = 0.5 * (p + q)

    def kl(a):
        ok = a > 0
        return float((a[ok] * np.log(a[ok] / m[ok])).sum())
    return 0.5 * (kl(p) + kl(q)) / np.log(2.0)


def chart_features(prep: Prepared, tokens: np.ndarray, f_init: int) -> np.ndarray:
    """One sample's arrows on the prepared ticks -> (N_FEATURES,) float32."""
    out = np.zeros(N_FEATURES, np.float32)
    n = prep.n
    out[INDEX["is_beginner"] + prep.difficulty] = 1.0
    out[INDEX["log_notes"]] = np.log(max(1, n)) / 8.0
    if n == 0:
        return out
    tok = np.asarray(tokens, np.int64)
    t = C.tables()
    nxt = t["next"]
    before = np.empty(n, np.int64)
    s = C.start_state(int(f_init))
    for i in range(n):
        before[i] = s
        s = nxt[s, tok[i]]
    after = nxt[before, tok]
    pa, ps = t["p_alt"][before, tok].astype(np.int64), t["p_same"][before, tok].astype(np.int64)
    single = tok < 8
    n_single = max(1, int(single.sum()))
    panel = np.where(tok < 4, pa, ps)                     # a single's panel
    pl_b, pr_b, fl_b = (before >> 1) // 4, (before >> 1) % 4, before & 1
    pl_a, pr_a = (after >> 1) // 4, (after >> 1) % 4
    foot = np.where(tok < 4, 1 - fl_b, fl_b)              # the foot a single is taken with
    own_before = np.where(foot == C.LEFT, pl_b, pr_b)
    other_before = np.where(foot == C.LEFT, pr_b, pl_b)

    # -- ergonomics and balance ------------------------------------------- #
    cost = np.minimum(t["cost"][prep.dtb, 0, before, tok].astype(np.float64), COST_CAP)
    cost[0] = 0.0
    free = prep.free
    fc = cost[free] if free.any() else cost
    out[INDEX["cost_mean"]] = fc.mean() / COST_CAP
    out[INDEX["cost_p90"]] = np.percentile(fc, 90) / COST_CAP
    out[INDEX["hard_share"]] = (fc >= HARD_STEP).mean()
    same = single & (tok >= 4)
    out[INDEX["cross_share"]] = (single & ((tok & 3) == C.CROSS)).sum() / n_single
    out[INDEX["jack_share"]] = (same & (panel == own_before)).sum() / n_single
    out[INDEX["dstep_share"]] = (same & (panel != own_before) & free).sum() / n_single
    fsw = single & (tok < 4) & (panel == other_before) & ((panel == C.D) | (panel == C.U))
    fsw[0] = False
    out[INDEX["fsw_share"]] = fsw.sum() / n_single
    px = np.asarray(C.PANEL_X)
    out[INDEX["crossed_share"]] = (px[pl_a] > px[pr_a]).mean()
    counts = np.bincount(np.concatenate([pa[pa >= 0], ps[ps >= 0]]), minlength=4).astype(np.float64)
    share = counts / max(1.0, counts.sum())
    out[INDEX["panel_lr"]] = share[C.L] + share[C.R]
    out[INDEX["panel_lr_diff"]] = abs(share[C.L] - share[C.R])
    out[INDEX["panel_du_diff"]] = abs(share[C.D] - share[C.U])
    out[INDEX["panel_max"]], out[INDEX["panel_min"]] = share.max(), share.min()
    onehot = np.zeros((n, 4), np.float64)
    for p in (pa, ps):
        ok = p >= 0
        onehot[np.nonzero(ok)[0], p[ok]] += 1.0
    if n >= LOCAL_WINDOW:
        cs = np.concatenate([np.zeros((1, 4)), np.cumsum(onehot, axis=0)])
        a = np.arange(0, n - LOCAL_WINDOW + 1, LOCAL_HOP)
        win = cs[a + LOCAL_WINDOW] - cs[a]
        top = win.max(axis=1) / np.maximum(1.0, win.sum(axis=1))
        out[INDEX["local_max_mean"]] = top.mean()
        out[INDEX["local_max_p90"]] = np.percentile(top, 90)
    else:
        out[INDEX["local_max_mean"]] = out[INDEX["local_max_p90"]] = share.max()
    out[INDEX["foot_balance"]] = abs((foot[single] == C.LEFT).mean() - 0.5) if single.any() else 0.0

    # -- variety ---------------------------------------------------------- #
    # what the eye reads: the panel of a single, the pair of a jump
    lo, hi = np.minimum(pa, ps), np.maximum(pa, ps)
    code = np.where(single, panel, 4 + lo * 4 + hi)
    out[INDEX["tok4_entropy"]] = _entropy(tok, 4, C.N_TOKENS)[0]
    out[INDEX["tok8_entropy"]] = _entropy(tok, 8, C.N_TOKENS)[0]
    e4, top4, _ = _entropy(code, 4, 20)
    out[INDEX["panel4_entropy"]], out[INDEX["top4_share"]] = e4, top4
    out[INDEX["distinct8"]] = _entropy(code, 8, 20)[2]
    run = 0
    for p in (1, 2, 3, 4):
        eq = code[p:] == code[:-p] if n > p else np.zeros(0, bool)
        if p >= 2:
            out[INDEX[f"period{p}"]] = eq.mean() if len(eq) else 0.0
        run = max(run, _longest_run(eq))
    out[INDEX["loop_run"]] = min(run, LOOP_CAP) / LOOP_CAP
    bars_with = np.unique(prep.bar, return_counts=True)
    multi = bars_with[0][bars_with[1] >= 2]
    if len(multi):
        key = prep.in_bar * 20 + code
        seen = set()
        for b in multi:
            seen.add(key[prep.bar == b].tobytes())
        out[INDEX["bar_distinct"]] = len(seen) / len(multi)

    # -- callbacks -------------------------------------------------------- #
    q = np.bincount(tok, minlength=C.N_TOKENS) / n
    out[INDEX["cb_chance"]] = float((q * q).sum())
    if len(prep.link_later):
        a, b = prep.link_later, prep.link_earlier
        agree = tok[a] == tok[b]
        out[INDEX["has_links"]] = 1.0
        out[INDEX["cb_coverage"]] = len(a) / n
        out[INDEX["cb_token"]] = agree.mean()
        out[INDEX["cb_panel"]] = (code[a] == code[b]).mean()
        out[INDEX["cb_lift"]] = agree.mean() - out[INDEX["cb_chance"]]
        bars, inv = np.unique(prep.link_bar, return_inverse=True)
        tot = np.bincount(inv)
        hit = np.bincount(inv, weights=agree)
        big = tot >= 2
        out[INDEX["cb_bar"]] = float((hit[big] == tot[big]).mean()) if big.any() else 0.0

    # -- contour and sections --------------------------------------------- #
    use = single & np.concatenate([[False], single[:-1]]) & prep.near
    if use.sum() >= 8:
        prev = np.concatenate([[0], panel[:-1]])
        dx = (px[panel] - px[prev])[use].astype(np.float64)
        ph = np.asarray(PANEL_H)
        dh = (ph[panel] - ph[prev])[use].astype(np.float64)
        dp, dc = prep.d_pitch[use], prep.d_centroid[use]
        out[INDEX["contour_x"]] = abs(_corr(dp, dx))
        out[INDEX["contour_h"]] = _corr(dp, dh)
        out[INDEX["centroid_x"]] = abs(_corr(dc, dx))
        out[INDEX["centroid_h"]] = _corr(dc, dh)
        out[INDEX["contour_travel"]] = _corr(np.abs(dp), np.abs(dx) + np.abs(dh))

    def change(lines: np.ndarray) -> float:
        vals = []
        for b in lines:
            left = (prep.bar >= b - SECTION_BARS) & (prep.bar < b)
            right = (prep.bar >= b) & (prep.bar < b + SECTION_BARS)
            if left.sum() >= 4 and right.sum() >= 4:
                vals.append(_js(np.bincount(code[left], minlength=20).astype(np.float64),
                                np.bincount(code[right], minlength=20).astype(np.float64)))
        return float(np.mean(vals)) if vals else -1.0
    sec, oth = change(prep.boundaries), change(prep.others)
    out[INDEX["bar_change"]] = max(oth, 0.0)
    if sec >= 0:
        out[INDEX["has_sections"]] = 1.0
        out[INDEX["section_change"]] = sec
        out[INDEX["section_contrast"]] = sec - max(oth, 0.0)

    # -- context ---------------------------------------------------------- #
    out[INDEX["notes_per_bar"]] = np.log1p(n / max(1, prep.n_bars)) / 3.0
    out[INDEX["jump_share"]] = float((~single).mean())
    return out


class Critic:
    """`stage_3_critic.onnx`: (B, N_FEATURES) -> (B,) score, higher is better."""

    def __init__(self, path, threads: int = 1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def score(self, feats: np.ndarray) -> np.ndarray:
        f = np.atleast_2d(np.asarray(feats, np.float32))
        return np.asarray(self.sess.run(None, {"features": f})[0], np.float64).reshape(-1)
