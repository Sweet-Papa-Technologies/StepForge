"""counterfactual.py — the same rhythm, stepped in several footing styles.

    python -m beatforge.adapters.stepmania.neural.synth.counterfactual \
        --corpus-root build/v15b_corpus [--streams 300] [--workers 8]

Why. The step network (ChoreoNet) is told a style — how many crossovers,
footswitches and jacks a chart has — and is supposed to follow it. Trained on
human charts and on the v1.5 pattern archetypes it does not: each style came
with a rhythm of its own, so the rhythm alone told the network which style it
was looking at and the style inputs carried nothing extra. Moving one slider
moved the output by under half a percent.

Here the rhythm is held fixed. For one base rhythm — when the notes fall,
which of them are jumps, which carry a freeze and for how long — up to five
different sets of arrows are written:

    clean_alt          plain alternation: no crossover, footswitch or jack
    crossover_heavy    crossovers on 10-24 % of the single steps
    footswitch_heavy   D/U footswitches on 8-18 %
    jack_heavy         same-foot repeats on 5-14 %, where the gap is comfortable
    blended_tech       crossovers and footswitches together, moderately

All of them sit on the same notes and, for a real song, on the same audio. The
only thing that tells them apart is the style vector, so the network has to
read it to predict the arrows.

Base rhythms (training material only):
  * every Medium, Hard and Challenge chart of a TRAINING song in `real/`
    (written to `synth_cf/`, attached to that song and carrying its song id);
  * `--streams` synthetic 16th-note stream and jumpstream rhythms, 16-48 bars
    at 120-195 BPM, which the human charts do not contain (written to
    `synth_cf_stream/`, no audio).

How a variant is made. A dynamic programme walks the codec's own foot state
machine over the fixed notes. Each step is scored by how human charters step
from that foot position at that speed (a table counted from the training
charts), plus random noise so no two rhythms come out alike, plus a bonus on
the move the variant is about, minus the foot model's discomfort. The bonus is
bisected until the chart lands in the variant's range.

What is stored is never what the walk intended. The arrows are handed to
`codec.encode`, which reads them the cheapest way a player could — a repeated
arrow meant as a footswitch is a jack unless the steps around it say
otherwise — and the label is `compute_style_vector` of THAT reading, exactly
as for a human chart. A variant is kept only if that reading has no forbidden
transition, no hold-lock violation, no fast unforced double-step, no 8-step
pattern covering more than a fifth of it, and is really apart from the clean
variant on its own axis (crossovers +0.06, footswitches +0.05, jacks +0.03).

Every chart row carries `rhythm_group` (shared by all variants of one base
rhythm) and `variant`; the shard carries the parent song's id.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import codec as C
from .. import conditioning as cond
from ..train import corpus
from . import shards
from .style_patterns import MAX_NGRAM_SHARE, _check, top_ngram_share

SOURCE_REAL = "synth_cf"
SOURCE_STREAM = "synth_cf_stream"
VARIANTS = ("clean_alt", "crossover_heavy", "footswitch_heavy", "jack_heavy", "blended_tech")
BASE_SLOTS = ("medium", "hard", "challenge")
DEFAULT_SEED = 20261120

CROSS, JACK, FSW = "cross", "jack", "fsw"
AXES = (CROSS, JACK, FSW)
_STYLE_INDEX = {CROSS: cond.INDEX["crossover_share"], JACK: cond.INDEX["jack_share"],
                FSW: cond.INDEX["footswitch_share"]}
#: The measured range each variant aims for on its own axis (or axes).
TARGETS = {
    "clean_alt": {},
    "crossover_heavy": {CROSS: (0.10, 0.24)},
    "footswitch_heavy": {FSW: (0.08, 0.18)},
    "jack_heavy": {JACK: (0.05, 0.14)},
    "blended_tech": {CROSS: (0.08, 0.14), FSW: (0.06, 0.12)},
}
#: How far above the clean variant of the same rhythm a kept variant must be.
SEPARATION = {
    "crossover_heavy": {CROSS: 0.06},
    "footswitch_heavy": {FSW: 0.05},
    "jack_heavy": {JACK: 0.03},
    "blended_tech": {CROSS: 0.04, FSW: 0.03},
}
#: ...and how far it may drift on the axes it is not about.
OFF_AXIS_SLACK = 0.04
#: The clean variant itself: at most this much of anything.
CLEAN_MAX = 0.03

#: Bonus (nats) on the move a variant is about, tried in this order.
BONUS_LADDER = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
#: ...starting here: human charters already jack a tenth of their steps, so
#: the jack variant starts below zero; a crossover needs a real push.
LADDER_START = {"jack": -2.0, "cross": 1.0}
#: Nats per unit of foot-model discomfort. Three times the decoder's: a walk
#: that is close to the cheapest way of standing on its own arrows is one the
#: annotator reads back the way it was meant.
LAMBDA_COST = 0.12
#: Half-width of the window around the point a variant aims at.
AIM_HALF_WIDTH = 0.025
#: Footswitches are edited in (see `add_footswitches`): only on steps faster
#: than this, at least this many notes apart, in at most this many rounds.
FSW_MAX_GAP_MS = 150.0
FSW_SPACING = 3
FSW_ROUNDS = 6
#: Penalty on a move a variant is NOT about.
OFF_AXIS_PENALTY = 6.0
#: Shortest gap (ms) at which a jack is comfortable at a tier.
JACK_MIN_GAP_MS = {"beginner": 600.0, "easy": 450.0, "medium": 300.0, "hard": 200.0,
                   "challenge": 150.0, "edit": 150.0}
#: A footswitch or a crossover is an idiom of a run: no bonus after a gap of
#: this many ms or more.
RUN_GAP_MS = {FSW: 300.0, CROSS: 480.0}
#: A double-step faster than this is never written.
SLOW_DSTEP_BUCKET = 4

NEG = -1e9
MIN_NOTES = 24
MAX_NOTES = 1400
PRIOR_FLOOR = 1e-3


# --------------------------------------------------------------------------- #
# Tables over (state, token)
# --------------------------------------------------------------------------- #
_TAB = None


def _tables() -> dict:
    global _TAB
    if _TAB is not None:
        return _TAB
    t = C.tables()
    s_n, a_n = C.N_STATES, C.N_TOKENS
    crossed = np.zeros((s_n, a_n), bool)      # the feet end crossed
    cross_tok = np.zeros((s_n, a_n), bool)    # a single step onto the far side
    jack = np.zeros((s_n, a_n), bool)         # same foot, same panel
    fsw_du = np.zeros((s_n, a_n), bool)       # other foot onto the resting D/U panel
    fsw_any = np.zeros((s_n, a_n), bool)
    dstep = np.zeros((s_n, a_n), bool)        # same foot, new panel
    onto_l = np.zeros((s_n, a_n), bool)
    onto_r = np.zeros((s_n, a_n), bool)
    holder = np.zeros((5, s_n, a_n), np.int8)
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
                cross_tok[s, a] = (a & 3) == C.CROSS
                jack[s, a] = a >= 4 and panel == pos[foot]
                fsw_any[s, a] = panel == pos[other] and panel != pos[foot]
                fsw_du[s, a] = a < 4 and fsw_any[s, a] and panel in (C.D, C.U)
                dstep[s, a] = a >= 4 and panel != pos[foot]
                onto_l[s, a] = foot == C.RIGHT and panel == pl
                onto_r[s, a] = foot == C.LEFT and panel == pr
                holder[4, s, a] = foot
            holder[1, s, a] = holder[3, s, a] = 1 - fl
            holder[2, s, a] = fl
    lock_bad = np.zeros((4, s_n, a_n), bool)
    for lb in range(4):
        if lb & 1:
            lock_bad[lb] |= t["moves_l"] | onto_l
        if lb & 2:
            lock_bad[lb] |= t["moves_r"] | onto_r
    _TAB = {"crossed": crossed, CROSS: cross_tok, JACK: jack, FSW: fsw_du,
            "fsw_any": fsw_any, "dstep": dstep, "holder": holder, "lock_bad": lock_bad,
            "jump_tok": np.arange(a_n) >= 8}
    return _TAB


# --------------------------------------------------------------------------- #
# The rhythm skeleton
# --------------------------------------------------------------------------- #
@dataclass
class Skeleton:
    """What all variants of one rhythm share. For a single step the freeze (if
    any) is in `k_a` / `d_a`; a jump keeps one per foot."""
    tick: np.ndarray
    times_ms: np.ndarray
    is_jump: np.ndarray
    k_a: np.ndarray
    k_s: np.ndarray
    d_a: np.ndarray
    d_s: np.ndarray

    def __len__(self) -> int:
        return len(self.tick)

    def hold_length(self) -> np.ndarray:
        return np.maximum(np.where(self.k_a > 0, self.d_a, 0),
                          np.where(self.k_s > 0, self.d_s, 0))


def skeleton_of(tick, tok, flav, da, ds, times_ms) -> Skeleton:
    """The rhythm of an encoded chart, made playable by the generator's
    freeze rule: one freeze (or one both-feet jump freeze) at a time, and
    always a free foot for every note. A freeze that would break that is cut
    short, or made a tap if nothing of it is left."""
    tick = np.asarray(tick, np.int64)
    n = len(tick)
    is_jump = np.asarray(tok) >= 8
    kinds = np.array([C.FLAVOR_KINDS[int(f)] for f in np.asarray(flav)], np.int64).reshape(n, 2)
    k_a, k_s = kinds[:, 0].copy(), kinds[:, 1].copy()
    d_a, d_s = np.asarray(da, np.int64).copy(), np.asarray(ds, np.int64).copy()
    single = ~is_jump
    # a single's freeze, whichever foot the source chart gave it to
    k_a[single] = np.maximum(k_a[single], k_s[single])
    d_a[single] = np.maximum(d_a[single], d_s[single])
    k_s[single] = 0
    d_s[single] = 0
    d_a[k_a == 0] = 0
    d_s[k_s == 0] = 0

    def drop(i):
        k_a[i] = k_s[i] = 0
        d_a[i] = d_s[i] = 0

    def cut(i, at_tick):
        room = int(at_tick - tick[i]) - C.TICKS_PER_BEAT // 4
        if room < C.TICKS_PER_BEAT // 4:
            drop(i)
            return
        d_a[i] = min(d_a[i], room) if k_a[i] else 0
        d_s[i] = min(d_s[i], room) if k_s[i] else 0

    # A freeze lets go a 16th before the next note (the decoder's own rule,
    # infer.HOLD_RELEASE_TICKS). Human charts often end one exactly on a note;
    # the annotator then counts that foot as free and the generator does not,
    # so the two would read the chart differently.
    release = C.TICKS_PER_BEAT // 4
    for i in np.nonzero((k_a > 0) | (k_s > 0))[0]:
        end = tick[i] + max(d_a[i], d_s[i])
        nxt = int(np.searchsorted(tick, end - release + 1))
        if nxt < n and nxt > i and tick[nxt] - end < release:
            cut(i, tick[nxt] + release - C.TICKS_PER_BEAT // 4)

    active = None                              # (note index, end tick, feet pinned)
    for i in range(n):
        if active is not None and tick[i] > active[1]:
            active = None
        if active is not None:
            feet_free = 2 - active[2]
            if feet_free < (2 if is_jump[i] else 1):
                cut(active[0], tick[i])        # the earlier freeze makes room
                active = None
            elif k_a[i] or k_s[i]:
                drop(i)                        # no second freeze under the first
        if active is None and (k_a[i] or k_s[i]):
            if single[i]:
                active = (i, tick[i] + d_a[i], 1)
            else:
                both = bool(k_a[i]) and bool(k_s[i])
                active = (i, tick[i] + max(d_a[i], d_s[i]), 2 if both else 1)
    return Skeleton(tick, np.asarray(times_ms, np.float64), is_jump, k_a, k_s, d_a, d_s)


def _lock_schedule(sk: Skeleton) -> tuple:
    """Per note: which lock slots are down when it is played (bit 0 / bit 1)
    and how the note starts a freeze (0 none, 1 jump alt foot, 2 jump same
    foot, 3 both, 4 the single's own foot)."""
    n = len(sk)
    slots = np.zeros(n, np.int64)
    kind = np.zeros(n, np.int64)
    length = sk.hold_length()
    for i in np.nonzero(length > 0)[0]:
        j1 = int(np.searchsorted(sk.tick, sk.tick[i] + length[i], side="right"))
        if sk.is_jump[i]:
            kind[i] = 3 if (sk.k_a[i] and sk.k_s[i]) else (1 if sk.k_a[i] else 2)
            slots[i + 1:j1] |= 3 if kind[i] == 3 else 1
        else:
            kind[i] = 4
            slots[i + 1:j1] |= 1
    return slots, kind


# --------------------------------------------------------------------------- #
# How human charters step: log P(token | foot state, gap)
# --------------------------------------------------------------------------- #
def build_prior(real_dir, log=print) -> np.ndarray:
    """(gap bucket, state, token) log-probabilities counted from the kept
    charts of the TRAINING songs, smoothed toward the per-gap average."""
    real_dir = Path(real_dir)
    counts = np.zeros((C.N_DT, C.N_STATES, C.N_TOKENS), np.float64)
    n_charts = 0
    for key, row in shards.training_parents(real_dir).items():
        with np.load(real_dir / "songs" / f"{key}.npz", allow_pickle=False) as z:
            for c in row["charts"]:
                if not c["keep"]:
                    continue
                j = c["j"]
                tok = z[f"c{j}_tok"].astype(np.int64)
                tms = z[f"c{j}_tms"].astype(np.float64)
                meta_f = 0
                enc = C.EncodedChart(meta_f, z[f"c{j}_tick"], tok.astype(np.uint8),
                                     z[f"c{j}_flav"], z[f"c{j}_da"], z[f"c{j}_ds"])
                # the parity bit only mirrors the states; the table is made
                # mirror-symmetric below, so either start gives the same counts
                before, _, _ = C.foot_path(enc)
                dtb = C.dt_bucket(np.concatenate([[1e9], np.diff(tms)]))
                np.add.at(counts, (dtb, before, tok), 1.0)
                n_charts += 1
    mirror = np.array([C.mirror_state(s) for s in range(C.N_STATES)])
    counts = counts + counts[:, mirror, :]
    prior = np.empty_like(counts)
    for is_jump in (False, True):
        cls = (np.arange(C.N_TOKENS) >= 8) == is_jump
        sub = counts[:, :, cls]
        per_gap = sub.sum(1, keepdims=True)
        per_gap = (per_gap + 1.0) / (per_gap.sum(2, keepdims=True) + cls.sum())
        smooth = sub + 20.0 * per_gap                  # 20 pseudo-notes of the average
        prior[:, :, cls] = smooth / smooth.sum(2, keepdims=True)
    log(f"[cf] step prior from {n_charts} training charts, {int(counts.sum() / 2)} steps")
    return np.log(np.maximum(prior, PRIOR_FLOOR))


def uniform_prior() -> np.ndarray:
    return np.zeros((C.N_DT, C.N_STATES, C.N_TOKENS), np.float64)


# --------------------------------------------------------------------------- #
# One realization
# --------------------------------------------------------------------------- #
def realize(sk: Skeleton, prior: np.ndarray, noise: np.ndarray, bonus: dict,
            slot: str, ban: tuple = ()) -> tuple | None:
    """The best foot path over the skeleton -> (tokens, f_init), or None when
    the rhythm cannot be stood on under the rules.

    `bonus` maps an axis (cross / jack / fsw) to nats added to each such move
    where it is in place; `ban` lists axes that may not occur at all."""
    n = len(sk)
    t, g = C.tables(), _tables()
    s_n, a_n = C.N_STATES, C.N_TOKENS
    sq_s = np.repeat(np.arange(s_n), 2)
    sq_q = np.tile(np.arange(2), s_n)
    sq_fl = sq_s & 1
    dt = np.concatenate([[1e9], np.diff(sk.times_ms)])
    dtb = C.dt_bucket(dt)
    slots, hold_kind = _lock_schedule(sk)
    slots_next = np.concatenate([slots[1:], [0]])
    ok_at = {JACK: dt >= JACK_MIN_GAP_MS.get(slot, 200.0),
             FSW: dt < RUN_GAP_MS[FSW], CROSS: dt < RUN_GAP_MS[CROSS]}
    ok_at[JACK][0] = ok_at[FSW][0] = ok_at[CROSS][0] = False
    feat = {a: g[a][sq_s] for a in AXES}
    crossed, fsw_any, dstep = g["crossed"][sq_s], g["fsw_any"][sq_s], g["dstep"][sq_s]
    score = np.full(2 * s_n, NEG)
    for f in (C.LEFT, C.RIGHT):
        score[2 * C.start_state(f)] = 0.0
    back = np.full((n, 2 * s_n), -1, np.int32)
    for j in range(n):
        sa = int(slots[j])
        lb = ((sa & 1) << sq_q) | (((sa >> 1) & 1) << (1 - sq_q))
        cost = t["cost"][dtb[j], lb, sq_s].astype(np.float64)
        ok = (g["jump_tok"][None, :] == bool(sk.is_jump[j])) & ~g["lock_bad"][lb, sq_s]
        ok &= cost < C.HARD_COST
        if sk.is_jump[j]:
            ok &= ~crossed
        if dtb[j] < SLOW_DSTEP_BUCKET:
            free_other = ((lb >> (1 - sq_fl)) & 1) == 0
            ok &= ~(dstep & free_other[:, None])
        # a footswitch is written on purpose or not at all: on D/U, in a run
        ok &= ~(fsw_any & ~feat[FSW])
        if CROSS in ban:
            ok &= ~crossed
        gain = prior[dtb[j]][sq_s] + noise[j][None, :] - LAMBDA_COST * cost
        for axis in AXES:
            if axis in ban or not ok_at[axis][j]:
                if axis == CROSS:
                    gain = gain - OFF_AXIS_PENALTY * feat[axis]
                else:
                    ok &= ~feat[axis]
            else:
                gain = gain + bonus.get(axis, -OFF_AXIS_PENALTY) * feat[axis]
        cand = np.where(ok, score[:, None] + gain, NEG)
        hk = int(hold_kind[j])
        nq = (g["holder"][hk][sq_s].astype(np.int64) if hk
              else np.repeat(sq_q[:, None], a_n, 1))
        if slots_next[j] == 0:
            nq = np.zeros_like(nq)
        tgt = (t["next"][sq_s].astype(np.int64) * 2 + nq).ravel()
        flat = cand.ravel()
        order = np.argsort(flat, kind="stable")
        new = np.full(2 * s_n, NEG)
        new[tgt[order]] = flat[order]
        back[j, tgt[order]] = order
        if new.max() <= NEG / 2:
            return None
        score = new
    sq = int(score.argmax())
    toks = np.empty(n, np.uint8)
    for j in range(n - 1, -1, -1):
        src = int(back[j, sq])
        toks[j] = src % a_n
        sq = src // a_n
    return toks, int((sq // 2) & 1)


def events_of(sk: Skeleton, toks: np.ndarray, f_init: int) -> C.ChartEvents:
    """The concrete chart of a realization: the skeleton's notes and freezes
    on the panels the foot path presses."""
    single_same = (toks >= 4) & (toks < 8)
    k_a = np.where(single_same, 0, sk.k_a)
    k_s = np.where(single_same, sk.k_a, sk.k_s)
    d_a = np.where(single_same, 0, sk.d_a)
    d_s = np.where(single_same, sk.d_a, sk.d_s)
    flavor = np.array([C._FLAVOR_OF[(int(a), int(b))] for a, b in zip(k_a, k_s)], np.uint8)
    enc = C.EncodedChart(int(f_init), sk.tick.astype(np.int32), toks.astype(np.uint8),
                         flavor, d_a.astype(np.int32), d_s.astype(np.int32))
    return C.decode(enc)


def measure(ev: C.ChartEvents, times_ms, slot: str, meter: int) -> dict:
    """The annotator's reading of a chart: its three footing shares and why
    it may not be kept ([] when it may)."""
    enc, info = C.encode(ev, times_ms)
    style = cond.compute_style_vector(enc, np.asarray(times_ms) / 1000.0, meter)
    reasons = [r for r in _check(ev, times_ms, slot, enc, info) if r != "discomfort"]
    return {**{a: float(style[_STYLE_INDEX[a]]) for a in AXES},
            "reasons": reasons, "mean_cost": float(info.mean_cost),
            "loop": top_ngram_share(enc.token)}


def footing_flags(enc: C.EncodedChart) -> dict:
    """Per note, is it a crossover / jack / footswitch — the same definitions
    `conditioning.compute_style_vector` counts."""
    tok = np.asarray(enc.token, np.int64)
    before, _, _ = C.foot_path(enc)
    pl, pr, fl = (before >> 1) // 4, (before >> 1) % 4, before & 1
    single = tok < 8
    same = (tok >= 4) & single
    right = np.asarray(C.PANEL_OF[C.RIGHT])[tok & 3]
    same_panel = np.where(fl == C.LEFT, tok & 3, right)
    alt = tok < 4
    alt_foot = 1 - fl
    alt_panel = np.where(alt_foot == C.LEFT, tok & 3, right)
    fsw = alt & (alt_panel == np.where(alt_foot == C.LEFT, pr, pl)) \
        & ((alt_panel == C.D) | (alt_panel == C.U))
    if len(fsw):
        fsw[0] = False
    return {CROSS: single & ((tok & 3) == C.CROSS),
            JACK: same & (same_panel == np.where(fl == C.LEFT, pl, pr)), FSW: fsw,
            "before": before}


def _distance(m: dict, target: dict) -> float:
    """0 inside every target band and legal; otherwise how far outside."""
    d = sum(max(0.0, band[0] - m[a]) + max(0.0, m[a] - band[1])
            for a, band in target.items())
    return d + (10.0 if m["reasons"] else 0.0)


def _search(sk: Skeleton, prior, noise, slot: str, meter: int, target: dict,
            ban: tuple) -> tuple | None:
    """Walk the bonus up its ladder and keep the realization whose READING is
    closest to the target. (Not a bisection: past some bonus the walk crosses
    so often that the annotator finds a cheaper way to stand on the same
    arrows, and the measured share falls again.)"""
    best = None
    start = max(LADDER_START.get(a, 0.0) for a in target)
    for bonus in (b for b in BONUS_LADDER if b >= start):
        res = realize(sk, prior, noise, {a: bonus for a in target}, slot, ban=ban)
        if res is None:
            continue
        ev = events_of(sk, *res)
        m = measure(ev, sk.times_ms, slot, meter)
        if best is None or _distance(m, target) < _distance(best[2], target):
            best = (res, ev, m)
        if _distance(m, target) == 0.0:
            break
        if all(m[a] > band[1] for a, band in target.items()):
            break                                      # past the band: more will not help
    return best


def add_footswitches(sk: Skeleton, toks: np.ndarray, f_init: int, band: tuple,
                     rng) -> np.ndarray:
    """Turn steps of a foot path into D/U footswitches until the READING has
    a share of them inside `band` (or no more can be placed).

    A footswitch is one changed arrow: the stepping foot lands on the D/U
    panel the other foot is resting on. Whether the annotator calls it a
    footswitch depends on the steps around it — read the other way it is a
    jack, with the feet swapped until the next repeated arrow, and that costs
    a crossover for every step taken with the feet apart. So the edit is made,
    the chart is read back, and every edit that was not read as a footswitch
    is undone. Only fast steps qualify: a jack is cheap once there is time
    for it, and nothing around it can outweigh that."""
    toks = toks.copy()
    n = len(sk)
    dt = np.concatenate([[1e9], np.diff(sk.times_ms)])
    slots, hold_kind = _lock_schedule(sk)
    n_single = max(1, int((~sk.is_jump).sum()))
    done = np.zeros(n, bool)
    tried = np.zeros(n, bool)

    def read():
        ev = events_of(sk, toks, f_init)
        enc, _ = C.encode(ev, sk.times_ms)
        return footing_flags(enc)[FSW]

    def path():
        z = np.zeros(n, np.int32)
        enc = C.EncodedChart(int(f_init), sk.tick.astype(np.int32), toks, z.astype(np.uint8), z, z)
        before, _, _ = C.foot_path(enc)
        return before

    for _ in range(FSW_ROUNDS):
        have = int(done.sum())
        want = int(round(0.5 * (band[0] + band[1]) * n_single))
        if have >= int(np.ceil(band[0] * n_single)) and have >= want:
            break
        before = path()
        pl, pr, fl = (before >> 1) // 4, (before >> 1) % 4, before & 1
        cand = []
        for j in range(2, n - 2):
            if tried[j] or toks[j] >= 4 or dt[j] >= FSW_MAX_GAP_MS or dt[j + 1] >= RUN_GAP_MS[FSW]:
                continue
            if slots[j] or slots[j + 1] or hold_kind[j] or hold_kind[j - 1]:
                continue
            if sk.is_jump[j - 1] or sk.is_jump[j + 1] or toks[j + 1] >= 4:
                continue
            foot = 1 - fl[j]
            rest = pr[j] if foot == C.LEFT else pl[j]          # the other foot's panel
            own = pl[j] if foot == C.LEFT else pr[j]
            if rest not in (C.D, C.U) or own == rest:
                continue
            nxt_panel = C.PANEL_OF[1 - foot][toks[j + 1] & 3]  # where the other foot goes
            if nxt_panel == rest:
                continue
            cand.append((j, C.Z_OF[foot][rest]))
        if not cand:
            break
        order = rng.permutation(len(cand))
        placed = []
        near = np.zeros(n + 2 * FSW_SPACING, bool)
        near_idx = np.nonzero(done)[0]
        for k in near_idx:
            near[k:k + 2 * FSW_SPACING + 1] = True
        for i in order:
            if len(placed) >= int(np.ceil((want - have) * 1.4)) + 1:
                break
            j, z = cand[i]
            if near[j + FSW_SPACING]:
                continue
            near[j:j + 2 * FSW_SPACING + 1] = True
            placed.append((j, int(toks[j])))
            toks[j] = z
            tried[j] = True
        if not placed:
            break
        got = read()
        for j, old in placed:
            if got[j]:
                done[j] = True
            else:
                toks[j] = old
    return toks


def _fit(sk: Skeleton, prior, noise, slot: str, meter: int, variant: str, rng) -> tuple | None:
    """One variant of a rhythm -> (events, measured) of the realization whose
    reading is closest to the variant's target, or None."""
    # Aim at a random point of the variant's range, not at "anywhere in it":
    # the search stops at the first hit, which would put every chart at the
    # low edge and leave the sliders' upper halves without examples.
    target = {}
    for a, (lo, hi) in TARGETS[variant].items():
        aim = float(rng.uniform(lo, hi))
        target[a] = (max(lo, aim - AIM_HALF_WIDTH), min(hi, aim + AIM_HALF_WIDTH))
    off = tuple(a for a in AXES if a not in target)
    walk = {a: band for a, band in target.items() if a != FSW}
    if walk:
        got = _search(sk, prior, noise, slot, meter, walk, ban=off + ((FSW,) if FSW in target else ()))
        if got is None:
            return None
        res, ev, m = got
    else:
        res = realize(sk, prior, noise, {}, slot, ban=AXES)
        if res is None:
            return None
        ev = events_of(sk, *res)
        m = measure(ev, sk.times_ms, slot, meter)
    if FSW in target and not m["reasons"]:
        toks = add_footswitches(sk, res[0], res[1], target[FSW], rng)
        ev = events_of(sk, toks, res[1])
        m = measure(ev, sk.times_ms, slot, meter)
    return ev, m


def accept(variant: str, m: dict, clean: dict | None) -> str | None:
    """Why a measured realization is not kept; None when it is."""
    if m["reasons"]:
        return m["reasons"][0]
    if m["loop"] > MAX_NGRAM_SHARE:
        return "loop"
    if variant == "clean_alt":
        return "not_clean" if max(m[a] for a in AXES) > CLEAN_MAX else None
    if clean is None:
        return "no_clean_variant"
    for a, gap in SEPARATION[variant].items():
        if m[a] < clean[a] + gap:
            return f"not_apart_{a}"
    for a in AXES:
        if a not in TARGETS[variant] and m[a] > clean[a] + OFF_AXIS_SLACK:
            return f"drift_{a}"
    return None


def realize_group(sk: Skeleton, prior, slot: str, meter: int, seed: int) -> tuple:
    """All kept variants of one rhythm -> ({variant: (events, measured)},
    {variant: reason rejected})."""
    rng = np.random.default_rng(seed)
    kept, lost = {}, {}
    if not MIN_NOTES <= len(sk) <= MAX_NOTES:
        return kept, {v: "length" for v in VARIANTS}
    for variant in VARIANTS:
        noise = rng.gumbel(size=(len(sk), C.N_TOKENS))
        got = _fit(sk, prior, noise, slot, meter, variant, rng)
        if got is None:
            lost[variant] = "no_path"
            continue
        why = accept(variant, got[1], kept["clean_alt"][1] if "clean_alt" in kept else None)
        if why:
            lost[variant] = why
        else:
            kept[variant] = got
    if set(kept) == {"clean_alt"}:            # nothing to contrast it with
        lost["clean_alt"] = "alone"
        kept = {}
    return kept, lost


def same_rhythm(a: C.ChartEvents, b: C.ChartEvents) -> bool:
    """Do two charts have the same notes, jumps and freezes (arrows aside)?"""
    if len(a) != len(b) or not np.array_equal(a.tick, b.tick):
        return False
    if not np.array_equal(a.p1 >= 0, b.p1 >= 0):
        return False
    la = np.maximum(np.where(a.k0 > 0, a.d0, 0), np.where(a.k1 > 0, a.d1, 0))
    lb = np.maximum(np.where(b.k0 > 0, b.d0, 0), np.where(b.k1 > 0, b.d1, 0))
    return bool(np.array_equal(la, lb)
                and np.array_equal((a.k0 > 0).astype(int) + (a.k1 > 0),
                                   (b.k0 > 0).astype(int) + (b.k1 > 0)))


def _pack(arrays: dict, kept: dict, sk: Skeleton, slot: str, meter: int, group: str,
          extra: dict, j0: int = 0) -> list:
    """Write a group's variants into `arrays` as charts j0, j0+1, ..."""
    rows = []
    for j, variant in enumerate((v for v in VARIANTS if v in kept), j0):
        ev, m = kept[variant]
        row = corpus.pack_chart(arrays, j, ev, sk.times_ms, meter, slot, description=variant)
        row.update(rhythm_group=group, variant=variant,
                   measured={a: round(m[a], 4) for a in AXES}, **extra)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Source 1: the rhythms of the training songs
# --------------------------------------------------------------------------- #
def _real_song(args) -> dict:
    real_dir, out_dir, key, song_id, prior, seed = args
    meta, arrays = corpus.load_song(Path(real_dir) / "songs" / f"{key}.npz")
    out: dict = {}
    rows, lost, n_groups = [], {}, 0
    kept_j = _kept_charts(real_dir, key)
    for c in meta["charts"]:
        if c["difficulty"] not in BASE_SLOTS or c["j"] not in kept_j:
            continue
        j = c["j"]
        sk = skeleton_of(arrays[f"c{j}_tick"], arrays[f"c{j}_tok"], arrays[f"c{j}_flav"],
                         arrays[f"c{j}_da"], arrays[f"c{j}_ds"], arrays[f"c{j}_tms"])
        kept, miss = realize_group(sk, prior, c["difficulty"], c["meter"],
                                   zlib.crc32(f"{seed}:{key}:{j}".encode()))
        for v, why in miss.items():
            lost[f"{v}:{why}"] = lost.get(f"{v}:{why}", 0) + 1
        if not kept:
            continue
        n_groups += 1
        rows.extend(_pack(out, kept, sk, c["difficulty"], c["meter"], f"{song_id}:{j}",
                          {"source_j": j}, j0=len(rows)))
    if not rows:
        return {"key": key, "groups": 0, "charts": 0, "lost": lost, "rows": []}
    m = shards.base_meta(f"{key}__cf", song_id, SOURCE_REAL, title=meta.get("title", ""),
                         artist=meta.get("artist", ""), n_ticks=meta["n_ticks"],
                         duration_s=meta["duration_s"], has_audio=False,
                         parent={"source": "real", "key": key})
    m["charts"] = rows
    corpus.save_shard(Path(out_dir) / "songs" / f"{key}__cf.npz", out, m)
    return {"key": key, "groups": n_groups, "charts": len(rows), "lost": lost,
            "rows": [(r["variant"], r["measured"]) for r in rows]}


_KEEP = {}


def _kept_charts(real_dir, key: str) -> set:
    """The chart indices of a real song that the manifest kept."""
    real_dir = str(real_dir)
    if real_dir not in _KEEP:
        man = json.loads((Path(real_dir) / "manifest.json").read_text())
        _KEEP[real_dir] = {s["key"]: {c["j"] for c in s["charts"] if c["keep"]}
                           for s in man["songs"]}
    return _KEEP[real_dir].get(key, set())


# --------------------------------------------------------------------------- #
# Source 2: stream and jumpstream rhythms no training song has
# --------------------------------------------------------------------------- #
S16, S8, BEAT, BAR = 3, 6, C.TICKS_PER_BEAT, C.G


def stream_rhythm(rng, jumpstream: bool) -> tuple:
    """(ticks, is_jump, bpm, n_bars): runs of 16ths of 1-12 bars with breaks
    between them and, in a jumpstream, jumps on some of the beats."""
    bpm = float(rng.uniform(120.0, 195.0))
    n_bars = int(rng.integers(16, 49))
    jump_p = float(rng.uniform(0.25, 0.6)) if jumpstream else 0.0
    ticks, jumps = [], []
    t = BAR                                            # a bar of lead-in
    end = (n_bars - 1) * BAR
    while t < end:
        run = int(rng.integers(1, 13)) * BAR if rng.random() < 0.8 else \
            int(rng.integers(2, 9)) * BEAT
        stop = min(end, t + run)
        last_jump = -10 ** 9
        while t < stop:
            beat = t % BEAT == 0
            off = t % BEAT == S8
            is_j = (jumpstream and (beat or (off and rng.random() < 0.3))
                    and rng.random() < jump_p and t - last_jump >= BEAT)
            if is_j:
                last_jump = t
            ticks.append(t)
            jumps.append(bool(is_j))
            t += S16
        ticks.append(t)                                # land on the beat after the run
        jumps.append(bool(jumpstream and rng.random() < 0.5))
        kind = rng.random()
        if kind < 0.45:
            t += int(rng.choice((BEAT, 2 * BEAT, BAR)))
        else:                                          # a few 8ths or quarters as a break
            step = S8 if kind < 0.8 else BEAT
            for _ in range(int(rng.integers(2, 9))):
                t += step
                if t >= end:
                    break
                ticks.append(t)
                jumps.append(False)
            t += step
    ticks = np.asarray(ticks, np.int64)
    keep = np.concatenate([[True], np.diff(ticks) > 0])
    return ticks[keep], np.asarray(jumps, bool)[keep], bpm, n_bars


def _stream(args) -> dict:
    out_dir, index, prior, seed = args
    rng = np.random.default_rng((seed, index))
    for attempt in range(8):
        jumpstream = bool(index % 2)
        ticks, is_jump, bpm, n_bars = stream_rhythm(rng, jumpstream)
        if len(ticks) > MAX_NOTES:
            continue
        times = ticks * (5000.0 / bpm)
        z = np.zeros(len(ticks), np.int64)
        sk = Skeleton(ticks, times, is_jump, z, z.copy(), z.copy(), z.copy())
        nps = cond.sustained_nps(times / 1000.0)
        meter = int(np.clip(round(cond.effective_meter(nps)), 9, 15))
        key = f"synth_style_cf_{index:05d}"
        kept, lost = realize_group(sk, prior, "challenge", meter,
                                   zlib.crc32(f"{seed}:{key}:{attempt}".encode()))
        if not kept:
            continue
        arrays: dict = {}
        rows = _pack(arrays, kept, sk, "challenge", meter, key,
                     {"bpm": round(bpm, 3), "jumpstream": jumpstream})
        m = shards.base_meta(key, key, SOURCE_STREAM,
                             title=("jumpstream " if jumpstream else "stream ") + str(index),
                             artist="StepForge synth", n_ticks=n_bars * BAR,
                             duration_s=n_bars * BAR * 5.0 / bpm, has_audio=False,
                             bpm=round(bpm, 3), seed=int(seed))
        m["charts"] = rows
        corpus.save_shard(Path(out_dir) / "songs" / f"{key}.npz", arrays, m)
        return {"key": key, "groups": 1, "charts": len(rows),
                "lost": {f"{v}:{w}": 1 for v, w in lost.items()},
                "rows": [(r["variant"], r["measured"]) for r in rows]}
    return {"key": f"synth_style_cf_{index:05d}", "groups": 0, "charts": 0, "lost": {},
            "rows": []}


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
def _summary(results: list) -> dict:
    by: dict = {}
    lost: dict = {}
    for r in results:
        for k, v in r["lost"].items():
            lost[k] = lost.get(k, 0) + v
        for variant, m in r["rows"]:
            by.setdefault(variant, []).append([m[a] for a in AXES])
    out = {"groups": sum(r["groups"] for r in results),
           "charts": sum(r["charts"] for r in results), "variants": {},
           "rejected": dict(sorted(lost.items()))}
    for variant, rows in by.items():
        a = np.asarray(rows)
        out["variants"][variant] = {
            "charts": len(rows),
            **{axis: {"median": round(float(np.median(a[:, i])), 4),
                      "p10": round(float(np.percentile(a[:, i], 10)), 4),
                      "p90": round(float(np.percentile(a[:, i], 90)), 4)}
               for i, axis in enumerate(AXES)}}
    return out


def generate(corpus_root, streams: int = 300, seed: int = DEFAULT_SEED, workers: int = 4,
             limit: int | None = None, log=print) -> dict:
    """Write `synth_cf/` and `synth_cf_stream/` under a corpus root that has
    `real/`. Returns the two summaries."""
    root = Path(corpus_root)
    real = root / "real"
    prior = build_prior(real, log)
    out = {}
    jobs = {
        SOURCE_REAL: (_real_song, [(str(real), str(root / SOURCE_REAL), key, row["song_id"],
                                    prior, seed)
                                   for key, row in sorted(shards.training_parents(real).items())]),
        SOURCE_STREAM: (_stream, [(str(root / SOURCE_STREAM), i, prior, seed)
                                  for i in range(streams)]),
    }
    for source, (fn, tasks) in jobs.items():
        d = root / source
        (d / "songs").mkdir(parents=True, exist_ok=True)
        for old in shards.shard_files(d):
            old.unlink()
        tasks = tasks[:limit] if limit else tasks
        t0 = time.time()
        if workers > 1 and len(tasks) > 1:
            with multiprocessing.get_context("spawn").Pool(workers) as pool:
                results = list(pool.imap_unordered(fn, tasks, chunksize=2))
        else:
            results = [fn(t) for t in tasks]
        shards.write_source_manifest(d, source, log)
        s = _summary(results)
        s["seconds"] = round(time.time() - t0, 1)
        (d / "profile.json").write_text(json.dumps(s, indent=1))
        log(f"[cf] {source}: {s['groups']} rhythms, {s['charts']} charts in "
            f"{s['seconds']:.0f}s; " + "; ".join(
                f"{v} x{d_['charts']}" for v, d_ in s["variants"].items()))
        out[source] = s
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--streams", type=int, default=300,
                    help="synthetic stream / jumpstream rhythms to write")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N rhythms of each source")
    a = ap.parse_args(argv)
    generate(a.corpus_root, a.streams, a.seed, a.workers, a.limit)


if __name__ == "__main__":
    main()
