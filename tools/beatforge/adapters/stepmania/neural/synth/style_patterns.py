"""style_patterns.py — programmatic step sequences in six style archetypes
(REQ-SFV15-06), written as symbolic `synth_style` shards.

    python -m beatforge.adapters.stepmania.neural.synth.style_patterns \
        --out-dir build/neural_corpus/synth_style --total-charts 2400

ChoreoNet is conditioned on a measured style vector, and the human corpus
hardly covers the ends of it: 38 of 1,066 charts hold two bars of continuous
16ths, none is a stamina chart, and footswitches and crossovers are rare. So
the style sliders have little to learn from. This module writes the missing
styles directly: `itg_stamina`, `jumpstream`, `crossover_flow`,
`tech_footswitch`, `modern_hard_burst` and `casual_ddr_flow`.

A sequence is built as an explicit walk through the codec's own foot state
machine (`_Walker`), so every step it takes is one whose cost is known, on a
rhythm skeleton of its own. What is stored, though, is never what the walk
intended. The concrete panels are handed to `codec.encode`, which picks the
cheapest reading of them — a repeated arrow meant as a footswitch is read as a
jack unless the surrounding steps make the jack reading dearer — and then:

* `verify` rejects the chart unless that reading has no spin, no hold-lock
  violation, no forbidden transition, no unforced fast double-step, a mean
  discomfort within the tier's plain ceiling, and no 8-token pattern covering
  more than a fifth of it;
* the style label is `compute_style_vector` of that reading (it is what
  `corpus.pack_chart` stores), and the sequence is kept only when the label
  falls inside `PROFILES[archetype]`.

A rejected sequence is rebuilt from the next sub-seed, so a sequence is a pure
function of (seed, index).

`PROFILES` is the measured range each archetype is held to. It is the design
table's, except where the measurement itself puts part of a range out of reach:

* `jumpstream` is narrower (stream 0.55-0.78, jumps 0.14-0.24, against
  0.55-0.85 and 0.14-0.30). `stream_share` only counts runs of eight single
  steps, so a jump on every beat or half-beat scores zero however continuous
  the 16ths are. With jumps on the 8th grid of a 16th stream, a counted run
  between two jumps is at least 9 steps and two jumps with no run between
  them have a step between them, so
  jump_share <= 0.5 - (4/9) * stream_share: at most 0.256 jumps at the
  table's lowest stream share, at most 0.81 stream at its lowest jump share.
  The builder mixes open runs with accented passages just inside that line.
* `tech_footswitch` keeps the table's range, but a footswitch costs 24 at
  16th spacing and the top ceiling (Challenge) is 3.0 a transition, so lone
  switches carry it only to about 0.11 of the steps. Above that the share is
  made of switch drills — three or four alternating-foot notes on one arrow,
  which the annotator reads as two or three switches for the price of one —
  and the slot is always Challenge, whatever the density.
* `crossover_flow` keeps the table's range, but every crossover is at least
  two crossed steps, at 4 each on 8ths and 6-8 on 16ths, against a Hard
  ceiling of 2.0. The crossovers therefore sit mostly in the 8th-note
  phrases, and no chart is both half stream and a quarter crossovers.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...grammar import BUDGETS
from .. import codec as C
from .. import conditioning as cond
from ..train import corpus
from . import shards

SOURCE = "synth_style"
ARCHETYPES = ("itg_stamina", "jumpstream", "crossover_flow", "tech_footswitch",
              "modern_hard_burst", "casual_ddr_flow")
#: Sequences per archetype in the default 2,400; other totals scale these.
DEFAULT_COUNTS = {"itg_stamina": 500, "jumpstream": 450, "crossover_flow": 400,
                  "tech_footswitch": 400, "modern_hard_burst": 350,
                  "casual_ddr_flow": 300}
DEFAULT_TOTAL = sum(DEFAULT_COUNTS.values())
DEFAULT_SEED = 20261119

#: Tempo of a sequence, drawn as a whole BPM inside the archetype's range.
BPM_RANGES = {"itg_stamina": (140, 200), "jumpstream": (120, 175),
              "crossover_flow": (105, 160), "tech_footswitch": (110, 165),
              "modern_hard_burst": (115, 170), "casual_ddr_flow": (95, 150)}
#: Measured style ranges (`conditioning.FIELDS`) every emitted sequence is
#: inside: the design table's, but for `jumpstream` (see the module docstring).
PROFILES = {
    "itg_stamina": {"stream_share": (0.70, 0.98), "jump_share": (0.0, 0.03),
                    "crossover_share": (0.0, 0.04)},
    "jumpstream": {"stream_share": (0.55, 0.78), "jump_share": (0.14, 0.24)},
    "crossover_flow": {"crossover_share": (0.10, 0.24), "stream_share": (0.15, 0.55)},
    "tech_footswitch": {"footswitch_share": (0.08, 0.18), "jack_share": (0.04, 0.12),
                        "hold_share": (0.08, 0.22)},
    "modern_hard_burst": {"stream_share": (0.15, 0.45), "jump_share": (0.06, 0.16)},
    "casual_ddr_flow": {"stream_share": (0.0, 0.10), "crossover_share": (0.0, 0.0),
                        "jack_share": (0.0, 0.0)},
}
#: Difficulty slots an archetype may carry, easiest first; the sequence takes
#: the first one whose peak sustained rate (grammar.BUDGETS) covers its own.
SLOTS = {"itg_stamina": ("hard", "challenge"), "jumpstream": ("hard", "challenge"),
         "crossover_flow": ("hard", "challenge"), "tech_footswitch": ("challenge",),
         "modern_hard_burst": ("hard", "challenge"),
         "casual_ddr_flow": ("easy", "medium")}

MIN_EVENTS, MAX_EVENTS = 128, 768
MIN_BARS, MAX_BARS = 16, 64
NGRAM = 8
MAX_NGRAM_SHARE = 0.20
#: A double-step is "fast" in the gap buckets where it costs at least this.
FAST_DSTEP_COST = 60
#: Share of a tech chart's notes its idioms (switch runs, jacks, hold-taps) may
#: claim between them; the rest is the 8th-note flow that joins them. The
#: notes one unit of each share takes are estimates, so this is a dial.
_TECH_FILL = 0.9
MAX_ATTEMPTS = 200

S16, S8, BEAT, BAR = 3, 6, C.TICKS_PER_BEAT, C.G
HOME, DOWN, UP, CROSS = C.HOME, C.DOWN, C.UP, C.CROSS
_MIDS = (DOWN, UP)


@dataclass
class Sequence:
    seq_id: str
    archetype: str
    bpm: float
    difficulty: str
    meter: int
    n_ticks: int
    events: C.ChartEvents
    times_ms: np.ndarray
    seed: int
    attempts: int

    @property
    def index(self) -> int:
        return int(self.seq_id.rsplit("_", 1)[1])


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
def top_ngram_share(tokens, n: int = NGRAM) -> float:
    """Share of the sequence's `n`-token windows (overlapping) taken by its
    single most frequent one."""
    tok = np.asarray(tokens, np.int64)
    windows = len(tok) - n + 1
    if windows <= 0:
        return 0.0
    key = np.zeros(windows, np.int64)
    for k in range(n):
        key = key * (C.N_TOKENS + 1) + tok[k:k + windows]
    return float(np.unique(key, return_counts=True)[1].max() / windows)


def _gap_buckets(times_ms) -> np.ndarray:
    return C.dt_bucket(np.concatenate([[1e9], np.diff(np.asarray(times_ms, np.float64))]))


def _check(ev: C.ChartEvents, times_ms, slot: str, enc: C.EncodedChart,
           info: C.EncodeInfo) -> list[str]:
    reasons = []
    t = C.tables()
    tok = enc.token.astype(np.int64)
    before, _, p_same = C.foot_path(enc)
    dtb = _gap_buckets(times_ms)

    # The generator's hold rule pins the tail tick too (codec._held_masks).
    pinned = C._held_masks(ev, inclusive=True)
    on_held = (pinned >> ev.p0) & 1
    on_held |= np.where(ev.p1 >= 0, (pinned >> np.maximum(ev.p1, 0)) & 1, 0)
    if on_held.any():
        reasons.append("note_on_held_panel")
    pinned_cost = t["cost"][dtb, t["lockbits"][pinned, before], before, tok]
    if info.lock_violations or (pinned_cost >= C.LOCK_VIOLATION).any():
        reasons.append("hold_lock")
    if info.spins:
        reasons.append("spin")
    if (info.cost >= C.HARD_COST).any():
        reasons.append("forbidden_transition")

    last = before & 1
    stood = np.where(last == C.LEFT, (before >> 1) // 4, (before >> 1) % 4)
    lock = t["lockbits"][C._held_masks(ev), before]
    other_free = ((lock >> (1 - last)) & 1) == 0
    fast = np.asarray(C.DSTEP_COST)[dtb] >= FAST_DSTEP_COST
    if ((tok >= 4) & (tok < 8) & (p_same != stood) & other_free & fast).any():
        reasons.append("double_step")

    if info.mean_cost > corpus.DISCOMFORT_CEILING[slot]:
        reasons.append("discomfort")
    if top_ngram_share(tok) > MAX_NGRAM_SHARE:
        reasons.append("loop")
    return reasons


def verify(ev: C.ChartEvents, times_ms, slot: str) -> list[str]:
    """Why a chart may not be emitted at difficulty `slot`; [] when it may.

    Judged on the annotator's own reading of the panels (`codec.encode`):
    `spin`, `hold_lock` (a holding foot moves before its release),
    `note_on_held_panel`, `forbidden_transition` (any step at or above
    codec.HARD_COST), `double_step` (the same foot onto a new panel at a fast
    gap while the other foot is free), `discomfort` (mean cost above the
    tier's ceiling) and `loop` (one 8-token pattern over a fifth of the chart).
    """
    if len(ev) == 0:
        return ["empty"]
    enc, info = C.encode(ev, np.asarray(times_ms, np.float64))
    return _check(ev, times_ms, slot, enc, info)


def profile_misses(archetype: str, style) -> list[str]:
    """Fields of a measured style vector that fall outside the archetype's
    profile."""
    return [f for f, (lo, hi) in PROFILES[archetype].items()
            if not lo <= float(style[cond.INDEX[f]]) <= hi]


# --------------------------------------------------------------------------- #
# The walk: explicit feet on the codec's state machine
# --------------------------------------------------------------------------- #
class _Walker:
    """Lays notes down as codec tokens from a tracked (pL, pR, last foot)
    state, at a tick cursor `t`. Plain steps alternate feet and never land on
    the resting foot's panel, so they cost nothing; the idioms that do cost
    (crossovers, footswitches, jacks) are separate, counted calls.

    The counters describe the INTENDED walk and only steer the builders. The
    stream, jump and hold shares are exact, since they do not depend on the
    foot assignment; the rest is whatever the annotator later says it is.
    """

    def __init__(self, rng: np.random.Generator, bpm: float):
        self.rng = rng
        self.bpm = bpm
        self.pos = [C.L, C.R]
        self.last = int(rng.integers(2))
        self.t = 0
        self.rows: list = []
        self.queue: list = []          # z's left to play of the running motif
        self.weights: dict = {}
        self.cross_target = 0.0        # share of single steps that should cross
        self.cross_ok = False
        self.weaves = (1,)             # crossed steps per weave to draw from
        self._recover = False          # the last motif was a weave: step out first
        self.n_single = self.n_jump = self.n_hold = 0
        self.n_cross = self.n_fsw = self.n_jack = 0
        self._stream = self._run = 0
        self.prev = (-BAR, False)      # (tick, was a single) of the last note
        self._last_jump = ()
        self._off = ()                 # panels the next plain step stays off

    # ---- bookkeeping ------------------------------------------------------ #
    @property
    def n(self) -> int:
        return len(self.rows)

    def stream_share(self) -> float:
        run = self._run if self._run >= cond.STREAM_MIN_RUN else 0
        return (self._stream + run) / max(1, self.n)

    def jump_share(self) -> float:
        return self.n_jump / max(1, self.n)

    def hold_share(self) -> float:
        return self.n_hold / max(1, self.n)

    def gap_bucket(self, ticks: int) -> int:
        return int(C.dt_bucket(ticks * 5000.0 / self.bpm))

    def crossed(self) -> bool:
        return C.PANEL_X[self.pos[C.LEFT]] > C.PANEL_X[self.pos[C.RIGHT]]

    def _emit(self, tok: int, gap: int, hold: int = 0) -> None:
        pl, pr, self.last, pa, ps = C.apply_token(self.pos[0], self.pos[1], self.last, tok)
        self.pos = [pl, pr]
        kind = C.HOLD if hold else C.TAP
        self._off = ()
        self.rows.append((self.t, [(int(p), kind, hold) for p in (pa, ps) if p >= 0]))
        single = tok < 8
        joined = single and self.prev[1] and self.t - self.prev[0] <= cond.STREAM_MAX_GAP_TICKS
        if not joined:
            if self._run >= cond.STREAM_MIN_RUN:
                self._stream += self._run
            self._run = 0
        self._run += single
        self.prev = (self.t, single)
        self.n_single += single
        self.n_jump += not single
        self.n_hold += hold > 0
        self.t += gap

    def rest(self, ticks: int) -> None:
        self.t += ticks

    # ---- plain steps ------------------------------------------------------ #
    def free(self, z: int) -> bool:
        """May the stepping foot take its own-side panel `z` at no cost?"""
        return C.PANEL_OF[1 - self.last][z] != self.pos[self.last]

    def _motif(self) -> list:
        """A new run of own-side z's for the feet in turn, stepping foot first:
        a staircase, sweep, anchor, candle or trill — or, while the walk is
        short of its crossover share, a weave (L D R D: the box, and longer)."""
        rng = self.rng
        alt, rest = 1 - self.last, self.last
        mids = [z for z in _MIDS if self.free(z)]
        if not self.free(HOME):                        # the other foot is crossed onto it
            return [mids[rng.integers(len(mids))]]
        want = self.cross_ok and self.n_cross < self.cross_target * (self.n_single + 4)
        if want and self._recover:                     # one step back out, then again
            self._recover = False
            return [int(rng.choice([HOME] + mids))]
        self._recover = want
        if want:
            # only a foot standing on D/U can be crossed over
            lead = [] if self.pos[rest] in (C.D, C.U) else [HOME, int(rng.choice(_MIDS))]
            k = int(rng.choice(self.weaves))
            return lead + [z for _ in range(k) for z in (CROSS, int(rng.choice(_MIDS)))]
        a = mids[rng.integers(len(mids))]
        b = DOWN + UP - a
        x = int(rng.choice(_MIDS))
        y = DOWN + UP - x
        names = list(self.weights)
        name = names[rng.choice(len(names), p=list(self.weights.values()))]
        if name == "stair":                            # L D U R
            return [HOME, x, y, HOME]
        if name == "sweep":                            # L D U R U D (and back to L)
            return [HOME, x, y, HOME, y, x]
        if name == "anchor":                           # L D L U: one foot pinned at home
            return [HOME, x, HOME, y] * int(rng.integers(1, 3))
        if name == "pivot":                            # D R D U: one foot pinned mid-pad
            return [a, HOME, a, b] * int(rng.integers(1, 3))
        if name == "candle":                           # D L U: one foot across the pad
            return [a, HOME, b]
        if name == "trill":
            za = int(rng.choice([HOME, a]))
            zb = [z for z in (HOME, DOWN, UP) if C.PANEL_OF[rest][z] != C.PANEL_OF[alt][za]]
            return [za, int(rng.choice(zb))] * int(rng.integers(2, 4))
        return [int(rng.choice([HOME] + mids))]        # "walk"

    def flow(self, gap: int, hold: int = 0) -> None:
        """The next alternating step of the running motif — or, straight after
        a jump, any step that does not repeat one of the jump's arrows."""
        if self._off:
            ok = [z for z in (HOME, DOWN, UP) if self.free(z)]
            clear = [z for z in ok if C.PANEL_OF[1 - self.last][z] not in self._off]
            self.queue = [int(self.rng.choice(clear or ok))]
        elif not self.queue:
            self.queue = self._motif()
        z = self.queue.pop(0)
        self.n_cross += z == CROSS
        self._emit(z, gap, hold)

    def drop_weave(self) -> None:
        """Forget the crossed steps still queued (the foot simply returns)."""
        if CROSS in self.queue:
            del self.queue[self.queue.index(CROSS):]

    def step(self, gap: int, z: int, hold: int = 0) -> None:
        """An alternating step onto an explicit own-side panel."""
        self.queue = []
        self._emit(z, gap, hold)

    def same(self, gap: int, z: int) -> None:
        """The foot that just stepped steps again (a hold-tap or a jack)."""
        self.queue = []
        self.n_jack += C.PANEL_OF[self.last][z] == self.pos[self.last]
        self._emit(4 + z, gap)

    def jack(self, gap: int) -> None:
        self.same(gap, C.Z_OF[self.last][self.pos[self.last]])

    def switch(self, gap: int) -> None:
        """The other foot lands on the D/U panel the last foot rests on."""
        self.n_fsw += 1
        self.step(gap, C.Z_OF[1 - self.last][self.pos[self.last]])

    def jump(self, gap: int) -> None:
        """A two-panel accent with both feet on their own side, never on the
        arrow of the step just before it nor the same pair as the last jump; a
        crossed stance steps out of the crossover first instead."""
        if self.crossed() or CROSS in self.queue:
            self.flow(gap)
            return
        alt, same = 1 - self.last, self.last
        taken = self.pos[same] if self.prev[1] else -1
        cands, wts = [], []
        for za in (HOME, DOWN, UP):
            for zs in (HOME, DOWN, UP):
                pa, ps = C.PANEL_OF[alt][za], C.PANEL_OF[same][zs]
                if pa == ps or taken in (pa, ps) or {pa, ps} == set(self._last_jump):
                    continue
                moved = (pa != self.pos[alt]) + (ps != self.pos[same])
                cands.append((za, zs))
                wts.append((0.5, 3.0, 1.5)[moved])
        za, zs = cands[self.rng.choice(len(cands), p=np.asarray(wts) / sum(wts))]
        self._last_jump = (C.PANEL_OF[alt][za], C.PANEL_OF[same][zs])
        self.queue = []
        self._emit(8 + C.JUMP_PAIRS.index((za, zs)), gap)
        self._off = self._last_jump

    def accent(self, gap: int, jump: bool) -> None:
        if jump:
            self.jump(gap)
        else:
            self.flow(gap)


def _mix(rng, **base) -> dict:
    """Motif weights of one sequence: the archetype's, each scaled at random
    so two sequences of an archetype do not share a vocabulary balance."""
    w = {k: v * rng.uniform(0.4, 1.6) for k, v in base.items() if v > 0}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def _bars_for(rng, per_bar: float) -> int:
    """A random bar count whose notes, at about `per_bar` a bar, stay inside
    the event limits."""
    lo = max(MIN_BARS, int(np.ceil(1.2 * MIN_EVENTS / per_bar)))
    hi = min(MAX_BARS, int(0.95 * MAX_EVENTS / per_bar))
    return int(rng.integers(lo, max(lo, hi) + 1))


# --------------------------------------------------------------------------- #
# Archetype builders: each fills the walker with one sequence
# --------------------------------------------------------------------------- #
def _groove_bar(w: _Walker, count: int, jump: bool) -> None:
    """One bar of `count` quarter/8th notes, always with its downbeat."""
    rng = w.rng
    weight = np.array([0, 1, 3, 1, 3, 1, 3, 1], np.float64)
    slots = [0] + sorted(rng.choice(8, size=min(count, 8) - 1, replace=False,
                                    p=weight / weight.sum()).tolist())
    start = w.t
    for i, s in enumerate(slots):
        w.t = start + s * S8
        w.accent(0, jump and i == 0)
    w.t = start + BAR


def _itg_stamina(w: _Walker) -> None:
    rng = w.rng
    w.weights = _mix(rng, stair=3, sweep=2.5, anchor=2.5, pivot=1, candle=1,
                     trill=0.5, walk=1.5)
    w.cross_target = rng.uniform(0.004, 0.034) if rng.random() < 0.7 else 0.0
    budget = int(rng.choice((16, 24, 32, 40)))
    sections: list = []
    while sum(sections) < budget:
        sections.append(int(rng.choice([b for b in (8, 16, 32)
                                        if sum(sections) + b <= budget])))
    # every section is 16ths up to a 2-beat recovery break at its end
    n_stream = sum(16 * b - 8 for b in sections)
    share = rng.uniform(max(0.715, n_stream / 750.0), 0.972)
    n_groove = max(2, int(round(n_stream * (1.0 / share - 1.0))))
    per_bar = int(rng.integers(max(4, -(-n_groove // (MAX_BARS - budget))), 9))
    counts = [per_bar] * (n_groove // per_bar)
    if n_groove % per_bar:
        counts.append(n_groove % per_bar)
    # a few accent jumps, on the downbeats that open a bar or a stream
    n_jumps = int(rng.uniform(0.0, 0.029) * (n_stream + n_groove))
    jump_p = 1.3 * n_jumps / (len(counts) + len(sections))
    where = rng.integers(0, len(sections) + 1, size=len(counts))
    for i in range(len(sections) + 1):
        for count in (c for c, at in zip(counts, where) if at == i):
            _groove_bar(w, count, w.n_jump < n_jumps and rng.random() < jump_p)
        if i == len(sections):
            break
        end = w.t + sections[i] * BAR
        w.accent(S16, w.n_jump < n_jumps and rng.random() < jump_p)
        while w.t < end - 2 * BEAT:
            w.cross_ok = w.t < end - 5 * BEAT          # leave room to step back out
            w.flow(S16)
        w.cross_ok = False
        w.t = end


def _jumpstream(w: _Walker) -> None:
    rng = w.rng
    w.weights = _mix(rng, stair=3, sweep=2, anchor=2, pivot=1, candle=1.5,
                     trill=0.4, walk=2)
    # the corner of the table the measurement allows: jumps <= 0.5 - (4/9) stream
    stream = rng.uniform(0.57, 0.76)
    jumps = rng.uniform(0.15, max(0.153, min(0.245, 0.488 - 0.444 * stream)))
    end = _bars_for(rng, 14.5) * BAR
    while w.t < end and w.n < MAX_EVENTS - 20:
        if w.n and rng.random() < 0.10:
            w.rest(int(rng.choice((S8, BEAT, 2 * BEAT))))
            continue
        left = (end - w.t) // S16
        if left < 4:
            break
        want_jump = w.jump_share() < jumps
        if w.stream_share() < stream:
            # an open run: an odd count of steps keeps the next jump on the 8th grid
            if want_jump:
                w.jump(S16)
            for _ in range(min(left, int(rng.choice((9, 11) if want_jump else (10, 12, 16))))):
                w.flow(S16)
        else:
            for _ in range(min(left // 4, int(rng.integers(2, 7)))):
                eighths = w.jump_share() < jumps
                w.jump(S16)
                w.flow(S16)
                w.accent(S16, eighths)
                w.flow(S16)


def _crossover_flow(w: _Walker) -> None:
    rng = w.rng
    w.weights = _mix(rng, stair=2, sweep=2, anchor=1, pivot=1, candle=3, trill=0.3,
                     walk=1.5)
    stream = rng.uniform(0.18, 0.52)
    w.cross_target = rng.uniform(0.11, min(0.235, 0.30 * (1.0 - stream)))
    w.weaves = (1, 1, 2, 2, 3)
    # a weave inside a 16th run costs 2 * c16 per crossover against 8 at 8th
    # spacing; let through only the share of them a Hard ceiling can carry
    c16 = 2 * C.CROSS_COST[w.gap_bucket(S16)]
    in_runs = float(np.clip((1.7 / w.cross_target - 8.0) / (c16 - 8.0), 0.0, 0.5))
    jumps = rng.uniform(0.0, 0.04)
    end = _bars_for(rng, 14.5 / (2.0 - stream)) * BAR
    while w.t < end - 2 * BAR and w.n < MAX_EVENTS - 20:
        mid_weave = w.crossed() or CROSS in w.queue
        if w.n and not mid_weave and rng.random() < 0.15:
            w.rest(int(rng.choice((S8, BEAT))))
        if w.stream_share() < stream:
            w.cross_ok = rng.random() < in_runs
            if not w.cross_ok:
                w.drop_weave()
            for _ in range(int(rng.choice((8, 12, 16)))):
                w.flow(S16)
        else:
            w.cross_ok = True
            w.accent(S8, w.jump_share() < jumps and w.t % BEAT == 0)
            for _ in range(int(rng.integers(3, 12))):
                w.flow(S8)


def _pinned_flow(w: _Walker, pins: int, land_mid: bool) -> None:
    """16ths that put `pins` steps on L/R — the notes that fix which foot is
    which, so that reading a nearby footswitch as a jack would turn every one
    of them into a crossover — and, with `land_mid`, end on D or U."""
    rng = w.rng
    got = 0
    while got < pins or (land_mid and w.pos[w.last] not in (C.D, C.U)):
        mids = [z for z in _MIDS if w.free(z)]
        z = HOME if got < pins and rng.random() < 0.8 else mids[rng.integers(len(mids))]
        got += z == HOME
        w.step(S16, z)


def _hold_taps(w: _Walker, gap: int, taps: int, jack: bool) -> None:
    """A freeze on one foot while the other keeps stepping (with `jack`, twice
    on one arrow), then one empty slot on which the freeze ends before its
    foot moves again."""
    rng = w.rng
    w.flow(gap, hold=(taps + 1) * gap)
    held = w.pos[w.last]
    foot = 1 - w.last
    repeat = int(rng.integers(1, taps)) if jack and taps > 1 else 0
    for i in range(taps):
        if i and i == repeat:
            w.jack(gap)
            continue
        stay_off = (held,) if i == 0 else (held, w.pos[foot])
        z = int(rng.choice([z for z in (HOME, DOWN, UP)
                            if C.PANEL_OF[foot][z] not in stay_off]))
        if i == 0:
            w.step(gap, z)
        else:
            w.same(gap, z)
    w.rest(gap)


def _tech_footswitch(w: _Walker) -> None:
    rng = w.rng
    w.weights = _mix(rng, stair=2.5, sweep=1.5, anchor=1.5, pivot=1, candle=2,
                     trill=0.3, walk=2)
    b16 = w.gap_bucket(S16)
    # A lone switch beats the jack reading only if flipping the feet on either
    # side of it costs more than the difference, FSW - JACK, in crossed steps:
    # one L/R step where a 16th is under 110 ms, two where it is not.
    pins_end, pins_mid = (1, 2) if b16 == 0 else (2, 3)
    fast_jacks = rng.uniform(0.0, 0.5)
    # Targets for the (switch, jack, hold) shares. Half the sequences lean on
    # one of the three, so each range is covered up to its top; the others
    # keep the switch share low, where lone switches can carry it.
    lo, span = np.array([0.085, 0.045, 0.09]), np.array([0.09, 0.07, 0.12])
    x = rng.uniform(0.0, 1.0, 3)
    x[0] **= 1.7
    lead = int(rng.integers(6))
    if lead < 3:
        x *= 0.5
        x[lead] = rng.uniform(0.6, 1.0)
    spare = rng.uniform(0.0, 0.1)
    lone, drilled = 3.0 + pins_mid, 2.4            # notes one switch takes, each way
    for _ in range(2):
        switches, jacks, holds = lo + span * x
        # Lone switches as far as the Challenge ceiling lets them go; drills
        # (two or three switches for the cost of one, in fewer notes) beyond
        # that, and where a chart leaning on jacks or holds has no room.
        jack_cost = jacks * (C.JACK_COST[2]
                             + fast_jacks * (C.JACK_COST[b16] - C.JACK_COST[2]))
        room = ((0.95 * corpus.DISCOMFORT_CEILING["challenge"] - jack_cost)
                / (C.FSW_DU_COST[b16] * switches))
        drills = 2.0 * (1.0 - room) + spare
        other = 2.4 * jacks + 2.6 * holds
        if lead in (1, 2):
            drills = max(drills, (lone - (_TECH_FILL - other) / switches) / (lone - drilled))
        drills = float(np.clip(drills, 0.0, 1.0))
        # ...and all three pulled back together if the notes still do not fit
        per = np.array([(1.0 - drills) * lone + drills * drilled, 2.4, 2.6])
        x *= min(1.0, max(0.0, _TECH_FILL - (per * lo).sum()) / (per * span * x).sum())
    switches, jacks, holds = lo + span * x
    drill_p = drills / (2.0 - drills)
    end = _bars_for(rng, 9.0) * BAR
    while w.t < end - 2 * BAR and w.n < MAX_EVENTS - 30:
        w.t += -w.t % S8
        if w.n and rng.random() < 0.12:
            w.rest(int(rng.choice((S8, BEAT, 2 * BEAT))))
        need = {"switch": 1.0 - (w.n_fsw / max(1, w.n_single)) / switches,
                "jack": 1.0 - (w.n_jack / max(1, w.n_single)) / jacks,
                "hold": 1.0 - w.hold_share() / holds}
        kind = max(need, key=need.get)
        if need[kind] <= 0:
            for _ in range(int(rng.integers(2, 6))):
                w.flow(S8)
        elif kind == "switch":
            pins = pins_end
            for _ in range(int(rng.integers(1, 4))):
                _pinned_flow(w, pins, land_mid=True)
                w.switch(S16)
                if rng.random() < drill_p:
                    for _ in range(int(rng.choice((1, 1, 2)))):
                        w.switch(S16)
                pins = pins_mid
            _pinned_flow(w, pins_end, land_mid=False)
        elif kind == "jack":
            gap = S16 if rng.random() < fast_jacks else S8
            for _ in range(int(rng.integers(1, 3))):
                w.flow(gap)
            for _ in range(1 if gap == S16 else int(rng.choice((1, 1, 2)))):
                w.jack(gap)
            if rng.random() < 0.5:
                w.flow(gap)
        else:
            # one foot cannot walk the pad in 16ths: a quick freeze gets one tap
            gap = int(rng.choice((S16, S8, S8)))
            _hold_taps(w, gap, 1 if gap == S16 else int(rng.integers(1, 4)),
                       jack=need["jack"] > 0 and rng.random() < 0.5)


def _modern_hard_burst(w: _Walker) -> None:
    rng = w.rng
    w.weights = _mix(rng, stair=3, sweep=2, anchor=1.5, pivot=1, candle=2, trill=0.4,
                     walk=2)
    stream = rng.uniform(0.165, 0.435)
    jumps = rng.uniform(0.066, 0.154)
    end = _bars_for(rng, 6.5 + 5.0 * stream) * BAR
    while w.t < end - 3 * BAR and w.n < MAX_EVENTS - 24:
        if w.n and rng.random() < 0.2:
            w.rest(int(rng.choice((BEAT, 2 * BEAT))))
        w.t += -w.t % BEAT
        w.accent(S8, w.jump_share() < jumps and rng.random() < 0.7)
        for _ in range(int(rng.integers(1, 8))):
            w.flow(S8)
        w.t += -w.t % BEAT
        long = w.stream_share() < stream
        k = int(rng.choice((8, 12) if long else (4, 5)))
        # the burst's last note is its accent and lands on the beat; an 8-note
        # burst keeps a freeze there so that all eight count as a run
        to_jump = k != 8 and w.jump_share() < jumps
        if k % 2 == 0:
            w.rest(S16)
        for _ in range(k - 1):
            w.flow(S16)
        if to_jump:
            w.jump(int(rng.choice((S8, BEAT))))
        else:
            hold = int(rng.choice((S8, S8 + S16, BEAT)))
            w.flow(BEAT + S8 if hold == BEAT else BEAT, hold=hold)


def _casual_ddr_flow(w: _Walker) -> None:
    rng = w.rng
    w.weights = _mix(rng, stair=3, sweep=3, anchor=1, pivot=0.5, candle=2, walk=1.5)
    if rng.random() < 0.5:                             # Easy: quarters, a few 8th pairs
        pair, skip = rng.uniform(0.0, 0.15), rng.uniform(0.05, 0.25)
    else:                                              # Medium: 8th phrases
        pair, skip = rng.uniform(0.5, 0.95), rng.uniform(0.0, 0.1)
    jumps = rng.uniform(0.03, 0.09)
    phrase = int(rng.choice((4, 8)))
    bars = _bars_for(rng, 3.8 * (1.0 - skip) * (1.0 + pair))
    for bar in range(bars):
        for beat in range(4):
            w.t = bar * BAR + beat * BEAT
            breath = bar % phrase == phrase - 1 and beat == 3
            if breath or (beat and rng.random() < skip):
                continue
            # a jump only where the 8th before it is empty, on beats 1 and 3
            clear = w.t - w.prev[0] >= BEAT
            if beat % 2 == 0 and clear and w.jump_share() < jumps and rng.random() < 0.5:
                w.jump(S8)
            else:
                w.flow(S8)
            if beat < 3 and rng.random() < pair:
                w.flow(S8)


_BUILDERS = {"itg_stamina": _itg_stamina, "jumpstream": _jumpstream,
             "crossover_flow": _crossover_flow, "tech_footswitch": _tech_footswitch,
             "modern_hard_burst": _modern_hard_burst, "casual_ddr_flow": _casual_ddr_flow}


# --------------------------------------------------------------------------- #
# Sequences
# --------------------------------------------------------------------------- #
def allocation(total: int) -> list:
    """The archetype of each index for a run of `total` sequences: the default
    counts scaled to `total` (largest remainders), in archetype blocks."""
    exact = {a: DEFAULT_COUNTS[a] * total / DEFAULT_TOTAL for a in ARCHETYPES}
    counts = {a: int(exact[a]) for a in ARCHETYPES}
    for a in sorted(ARCHETYPES, key=lambda a: exact[a] - counts[a], reverse=True):
        if sum(counts.values()) == total:
            break
        counts[a] += 1
    return [a for a in ARCHETYPES for _ in range(counts[a])]


def _slot(archetype: str, nps: float) -> str:
    slots = SLOTS[archetype]
    return next((s for s in slots if nps <= BUDGETS[s].max_nps_4s), slots[-1])


def generate_sequence(index: int, seed: int, archetype: str | None = None) -> Sequence:
    """Sequence `index` of the run seeded `seed`: verified, inside its
    archetype's profile, and a pure function of its arguments. With no
    `archetype` it is the one `index` has in the default 2,400."""
    if archetype is None:
        archetype = allocation(DEFAULT_TOTAL)[index % DEFAULT_TOTAL]
    lo, hi = BPM_RANGES[archetype]
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.default_rng([seed, index, attempt])
        bpm = float(rng.integers(lo, hi + 1))
        w = _Walker(rng, bpm)
        _BUILDERS[archetype](w)
        if not MIN_EVENTS <= w.n <= MAX_EVENTS:
            continue
        ev = C.ChartEvents.from_rows(w.rows)
        reach = int((ev.tick + np.maximum(ev.d0, ev.d1)).max()) + 1
        n_bars = max(MIN_BARS, -(-reach // BAR))
        if n_bars > MAX_BARS:
            continue
        times_ms = ev.tick * (5000.0 / bpm)
        nps = cond.sustained_nps(times_ms / 1000.0)
        slot = _slot(archetype, nps)
        meter = int(np.clip(round(cond.effective_meter(nps)), 1, 15))
        enc, info = C.encode(ev, times_ms)
        if _check(ev, times_ms, slot, enc, info):
            continue
        if profile_misses(archetype, cond.compute_style_vector(enc, times_ms / 1000.0, meter)):
            continue
        return Sequence(f"{SOURCE}_{index:05d}", archetype, bpm, slot, meter,
                        n_bars * BAR, ev, times_ms, int(seed), attempt + 1)
    raise RuntimeError(f"{archetype} #{index}: no sequence passed in {MAX_ATTEMPTS} attempts")


def sequence_shard(seq: Sequence) -> tuple[dict, dict]:
    """(arrays, meta) of a sequence's symbolic shard: one chart, no audio."""
    arrays: dict = {}
    row = corpus.pack_chart(arrays, 0, seq.events, seq.times_ms, seq.meter, seq.difficulty)
    meta = shards.base_meta(seq.seq_id, seq.seq_id, SOURCE,
                            title=f"{seq.archetype} {seq.index}", artist="StepForge synth",
                            n_ticks=seq.n_ticks, duration_s=seq.n_ticks * 5.0 / seq.bpm,
                            has_audio=False, archetype=seq.archetype, bpm=seq.bpm,
                            seed=seq.seed)
    meta["charts"] = [row]
    return arrays, meta


# --------------------------------------------------------------------------- #
# Batch generation
# --------------------------------------------------------------------------- #
def _work(args) -> dict:
    index, seed, archetype, songs_dir = args
    seq = generate_sequence(index, seed, archetype)
    arrays, meta = sequence_shard(seq)
    corpus.save_shard(Path(songs_dir) / f"{seq.seq_id}.npz", arrays, meta)
    return {"key": seq.seq_id, "archetype": archetype, "attempts": seq.attempts,
            "bpm": seq.bpm, "n_events": len(seq.events),
            "style": arrays["c0_style"].tolist()}


def _spread(values) -> dict:
    v = np.asarray(values, np.float64)
    return {"min": round(float(v.min()), 4), "median": round(float(np.median(v)), 4),
            "max": round(float(v.max()), 4)}


def profile_summary(rows: list) -> dict:
    """Per archetype: count, mean attempts per accepted sequence, and the
    min / median / max of every measured style field."""
    out = {}
    for a in ARCHETYPES:
        mine = [r for r in rows if r["archetype"] == a]
        if not mine:
            continue
        style = np.asarray([r["style"] for r in mine])
        out[a] = {"count": len(mine),
                  "mean_attempts": round(float(np.mean([r["attempts"] for r in mine])), 3),
                  "bpm": _spread([r["bpm"] for r in mine]),
                  "n_events": _spread([r["n_events"] for r in mine]),
                  "fields": {f: _spread(style[:, i]) for i, f in enumerate(cond.FIELDS)}}
    return out


def generate(out_dir, total_charts: int = DEFAULT_TOTAL, seed: int = DEFAULT_SEED,
             workers: int = 4, log=print) -> dict:
    """Write `total_charts` shards to `out_dir/songs`, the source manifest and
    `profile.json`. Returns the profile summary."""
    out_dir = Path(out_dir)
    songs = out_dir / "songs"
    songs.mkdir(parents=True, exist_ok=True)
    jobs = [(i, seed, a, str(songs)) for i, a in enumerate(allocation(total_charts))]
    keep = {f"{SOURCE}_{i:05d}.npz" for i in range(total_charts)}
    for stale in songs.glob(f"{SOURCE}_*.npz"):        # a larger earlier run
        if stale.name not in keep:
            stale.unlink()
    t0 = time.time()
    rows = []
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            for row in pool.imap_unordered(_work, jobs, chunksize=8):
                rows.append(row)
                if len(rows) % 400 == 0:
                    log(f"[{SOURCE}] {len(rows)}/{total_charts}")
    else:
        rows = [_work(j) for j in jobs]
    shards.write_source_manifest(out_dir, SOURCE, log=log)
    summary = {"source_type": SOURCE, "seed": int(seed), "total_charts": int(total_charts),
               "profiles": {a: {f: list(r) for f, r in p.items()} for a, p in PROFILES.items()},
               "archetypes": profile_summary(rows)}
    (out_dir / "profile.json").write_text(json.dumps(summary, indent=1))
    log(f"[{SOURCE}] {len(rows)} sequences in {time.time() - t0:.1f}s; "
        + ", ".join(f"{a} x{s['count']} ({s['mean_attempts']} tries)"
                    for a, s in summary["archetypes"].items()))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--total-charts", type=int, default=DEFAULT_TOTAL)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    generate(args.out_dir, args.total_charts, args.seed, args.workers)


if __name__ == "__main__":
    main()
