"""codec.py — the 20-token foot-relative biomechanical codec and the inverse
footflow annotator (spec §4, REQ-SFN-03).

A dance-single chart is rewritten from absolute pad columns {L, D, U, R} into
actions relative to the foot that takes them:

    tokens 0..3    ALT  + z   the OTHER foot from last time steps on z
    tokens 4..7    SAME + z   the SAME foot steps again (jack / double-step)
    tokens 8..19   JUMP (z_alt, z_same)   both feet, the 12 non-colliding pairs
    token  20      [MASK]     (Stage B diffusion only; never in an encoded chart)

with z in {HOME, DOWN, UP, CROSS} measured from the acting foot's own side.
Mirroring a chart left<->right swaps which foot is which and nothing else, so a
chart and its mirror encode to the SAME token array and differ only in the
initial parity bit `f_init`. That is the point of the codec: a model that
predicts these tokens cannot average "start left" with "start right".

Decoding needs only `f_init` and the tokens, so `decode(encode(x)) == x` holds
for ANY foot assignment. The assignment itself is a convention, chosen by a
Viterbi search over an ergonomic cost model (`transition_cost`), and the same
cost model later constrains generation — so what the model learns to call
comfortable and what the decoder enforces as comfortable are one definition.

Two deliberate departures from the spec text, both needed for REQ-SFN-03:

* The state space is 32 = 4x4 positions x 2, not the spec's 24. A footswitch
  lands one foot on the panel the other foot was resting on, and jacks on that
  panel keep it there; excluding same-panel states would make those charts
  unencodable, and the requirement is a lossless round trip for any legal chart.
* `hold_flavor` has three codes past the spec's six (6 ROLL_BOTH, 7 HOLD_ALT +
  ROLL_SAME, 8 ROLL_ALT + HOLD_SAME) and durations are kept per foot. A jump
  whose two panels carry different freeze kinds or lengths is legal and does
  occur; six codes and one duration cannot reproduce it bit-for-bit.
  `flavor6()` projects onto the spec's six classes for the Stage A head.

The cost numbers are NOT footflow.py's. footflow ranks a jack as the worst
transition (double-step + jack = 14) because the realizer should never choose
one; read backwards onto human charts that ranking would explain "L L L L" as
three footswitches. The ordering used here is the one players and the ITG
parity tools use: alternate by default, a repeated arrow is the same foot, a
crossover is cheap, a double-step is expensive unless the other foot is pinned
by a hold or the gap is long, and a 180-degree spin is all but forbidden.
Integer costs keep ties exact, which mirror invariance depends on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CODEC_VERSION = "1.0"

G = 48                      # ticks per 4-beat measure
TICKS_PER_BEAT = 12

LEFT, RIGHT = 0, 1
L, D, U, R = 0, 1, 2, 3
HOME, DOWN, UP, CROSS = 0, 1, 2, 3

#: panel(foot, z). Its own inverse: PANEL_OF[f][PANEL_OF[f][z]] == z.
PANEL_OF = ((L, D, U, R), (R, D, U, L))
Z_OF = PANEL_OF
MIRROR_PANEL = (R, D, U, L)
PANEL_X = (-1, 0, 0, 1)

N_TOKENS = 20
MASK = 20
ALT_TOKENS = (0, 1, 2, 3)
SAME_TOKENS = (4, 5, 6, 7)
SINGLE_TOKENS = tuple(range(8))
JUMP_TOKENS = tuple(range(8, 20))
#: (z_alt, z_same) for tokens 8..19 — every pair that lands on two different
#: panels, in lexicographic order. The four excluded pairs collide.
JUMP_PAIRS = tuple((za, zs) for za in range(4) for zs in range(4)
                   if PANEL_OF[LEFT][za] != PANEL_OF[RIGHT][zs])
assert len(JUMP_PAIRS) == 12
_JUMP_INDEX = {p: i for i, p in enumerate(JUMP_PAIRS)}

# hold flavors (spec §4.3) + the three lossless extensions
NONE, HOLD_ALT, HOLD_SAME, HOLD_BOTH, ROLL_ALT, ROLL_SAME = 0, 1, 2, 3, 4, 5
ROLL_BOTH, HOLD_ALT_ROLL_SAME, ROLL_ALT_HOLD_SAME = 6, 7, 8
N_FLAVORS = 6
TAP, HOLD, ROLL = 0, 1, 2
#: flavor -> (kind on the alt foot's panel, kind on the same foot's panel)
FLAVOR_KINDS = ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 2), (1, 2), (2, 1))
_FLAVOR_OF = {k: i for i, k in enumerate(FLAVOR_KINDS)}
_FLAVOR6 = np.array([0, 1, 2, 3, 4, 5, 3, 3, 3], dtype=np.uint8)

#: Stage A hold-duration vocabulary, in ticks (0.25 .. 8 beats).
HOLD_DUR_BUCKETS = (3, 6, 9, 12, 18, 24, 36, 48, 72, 96)

N_STATES = 32               # (pL, pR, f_last)


def flavor6(flavor):
    """Project lossless flavors onto the six classes Stage A predicts."""
    return _FLAVOR6[np.asarray(flavor, dtype=np.int64)]


def dur_bucket(ticks):
    """Nearest HOLD_DUR_BUCKETS index for a duration in ticks."""
    b = np.asarray(HOLD_DUR_BUCKETS, dtype=np.float64)
    t = np.asarray(ticks, dtype=np.float64)
    return np.abs(np.log(np.maximum(t, 1.0))[..., None] - np.log(b)).argmin(-1)


def state_index(pl: int, pr: int, f_last: int) -> int:
    return (pl * 4 + pr) * 2 + f_last


def state_tuple(s: int) -> tuple:
    return (s >> 1) // 4, (s >> 1) % 4, s & 1


def start_state(f_init: int) -> int:
    """Feet at rest on their own side; `f_init` is the foot that 'moved last'
    before the chart begins, so the first ALT step is taken by the other one."""
    return state_index(L, R, f_init)


def apply_token(pl: int, pr: int, f_last: int, tok: int):
    """One codec step. Returns (pl, pr, f_last, p_alt, p_same) where p_alt /
    p_same are the panels pressed by the alternating / repeating foot at this
    event (-1 when that foot does not press)."""
    alt, same = 1 - f_last, f_last
    pos = [pl, pr]
    if tok < 4:                                   # ALT
        p = PANEL_OF[alt][tok]
        pos[alt] = p
        return pos[0], pos[1], alt, p, -1
    if tok < 8:                                   # SAME
        p = PANEL_OF[same][tok - 4]
        pos[same] = p
        return pos[0], pos[1], same, -1, p
    za, zs = JUMP_PAIRS[tok - 8]
    pa, ps = PANEL_OF[alt][za], PANEL_OF[same][zs]
    alt_moved, same_moved = pa != pos[alt], ps != pos[same]
    pos[alt], pos[same] = pa, ps
    # deterministic post-jump parity (spec §4.2): the foot that alone moved was
    # last; if both or neither moved, the alternating foot is.
    nf = same if (same_moved and not alt_moved) else alt
    return pos[0], pos[1], nf, pa, ps


# --------------------------------------------------------------------------- #
# Ergonomic cost model
# --------------------------------------------------------------------------- #
#: Inter-onset gap buckets in ms. Index via `dt_bucket`.
DT_EDGES_MS = (110.0, 150.0, 300.0, 480.0, 960.0)
N_DT = len(DT_EDGES_MS) + 1

JACK_COST = (16, 12, 4, 2, 1, 0)         # same foot, same panel
DSTEP_COST = (90, 90, 85, 60, 20, 2)     # same foot, new panel, other foot free
FSW_DU_COST = (24, 24, 30, 45, 60, 60)   # other foot onto the resting D/U panel
FSW_LR_COST = (50, 50, 60, 80, 90, 90)   # ...onto the resting L/R panel
# A step that ends with the feet crossed. Cheap inside a run, where it is the
# idiom; dear after a long gap, where a player has time to square up and a
# crossed landing (above all a crossed FIRST step) is not how anyone stands.
CROSS_COST = (8, 6, 4, 6, 12, 30)
CROSSED_JUMP_COST = 60
SPIN_COST = 200                          # left foot on R with right foot on L
LOCK_VIOLATION = 500                     # moving a foot that is holding a freeze
#: Any transition at or above this is treated as physically illegal by the
#: generator; the annotator still pays it so a human chart always encodes.
HARD_COST = SPIN_COST


def dt_bucket(dt_ms):
    return np.searchsorted(np.asarray(DT_EDGES_MS), np.asarray(dt_ms), side="right")


def transition_cost(pl: int, pr: int, f_last: int, tok: int, dtb: int,
                    lock_l: bool = False, lock_r: bool = False) -> int:
    """Discomfort of taking `tok` from state (pl, pr, f_last) after a gap in
    bucket `dtb`, with the given feet pinned by an active hold."""
    alt, same = 1 - f_last, f_last
    pos = [pl, pr]
    locked = (lock_l, lock_r)
    c = 0
    if tok < 8:
        foot = alt if tok < 4 else same
        other = 1 - foot
        panel = PANEL_OF[foot][tok & 3]
        if locked[foot]:
            c += LOCK_VIOLATION
        if tok >= 4:
            if panel == pos[foot]:
                c += JACK_COST[dtb]
            elif not locked[other]:
                c += DSTEP_COST[dtb]
        # Arriving on the panel the other foot rests on is a footswitch whichever
        # foot does it. (Charging it only on ALT let a chart open with the right
        # foot on L, on top of the left foot, for the price of a slow double-step.)
        if panel == pos[other] and panel != pos[foot]:
            c += FSW_DU_COST[dtb] if panel in (D, U) else FSW_LR_COST[dtb]
        new = list(pos)
        new[foot] = panel
        crossed = CROSS_COST[dtb]
    else:
        za, zs = JUMP_PAIRS[tok - 8]
        new = list(pos)
        new[alt], new[same] = PANEL_OF[alt][za], PANEL_OF[same][zs]
        if lock_l or lock_r:
            c += LOCK_VIOLATION
        c += (new[0] != pos[0]) + (new[1] != pos[1])
        crossed = CROSSED_JUMP_COST
    if PANEL_X[new[0]] > PANEL_X[new[1]]:
        c += SPIN_COST if (new[0] == R and new[1] == L) else crossed
    return c


_TABLES = None


def tables() -> dict:
    """Lazily built lookup tables shared by the annotator and the generator.

      next[s, tok]            next state
      p_alt[s, tok]           panel pressed by the alternating foot (-1 = none)
      p_same[s, tok]          panel pressed by the repeating foot (-1 = none)
      moves_l / moves_r[s,tok] whether that physical foot presses
      cost[dtb, lock, s, tok] transition_cost, lock = lock_l | lock_r << 1
      single_tok[panel, s, foot]   token for `foot` stepping on `panel`
      jump_tok[p, q, s, c]    token for a {p, q} jump; c=0 puts L on p, c=1 on q
      lockbits[held, s]       feet pinned when the columns in bitmask `held`
                              carry an active freeze (annotator's positional rule)
    """
    global _TABLES
    if _TABLES is not None:
        return _TABLES
    nxt = np.zeros((N_STATES, N_TOKENS), np.int16)
    p_alt = np.full((N_STATES, N_TOKENS), -1, np.int8)
    p_same = np.full((N_STATES, N_TOKENS), -1, np.int8)
    moves = np.zeros((2, N_STATES, N_TOKENS), bool)
    cost = np.zeros((N_DT, 4, N_STATES, N_TOKENS), np.int32)
    for s in range(N_STATES):
        pl, pr, fl = state_tuple(s)
        for tok in range(N_TOKENS):
            npl, npr, nf, pa, ps = apply_token(pl, pr, fl, tok)
            nxt[s, tok] = state_index(npl, npr, nf)
            p_alt[s, tok], p_same[s, tok] = pa, ps
            if pa >= 0:
                moves[1 - fl, s, tok] = True
            if ps >= 0:
                moves[fl, s, tok] = True
            for dtb in range(N_DT):
                for lock in range(4):
                    cost[dtb, lock, s, tok] = transition_cost(
                        pl, pr, fl, tok, dtb, bool(lock & 1), bool(lock & 2))

    single_tok = np.zeros((4, N_STATES, 2), np.int8)
    jump_tok = np.zeros((4, 4, N_STATES, 2), np.int8)
    for s in range(N_STATES):
        _, _, fl = state_tuple(s)
        alt, same = 1 - fl, fl
        for p in range(4):
            for foot in (LEFT, RIGHT):
                z = Z_OF[foot][p]
                single_tok[p, s, foot] = z if foot == alt else 4 + z
            for q in range(4):
                if p == q:
                    continue
                for c, (left_p, right_p) in enumerate(((p, q), (q, p))):
                    panel = (left_p, right_p)
                    jump_tok[p, q, s, c] = 8 + _JUMP_INDEX[
                        (Z_OF[alt][panel[alt]], Z_OF[same][panel[same]])]

    lockbits = np.zeros((16, N_STATES), np.int8)
    for held in range(16):
        for s in range(N_STATES):
            pl, pr, fl = state_tuple(s)
            # A foot standing on a held column is the one holding it — except in
            # a same-panel state, where only the foot that arrived last can be.
            ll = bool(held >> pl & 1) and not (pl == pr and fl != LEFT)
            lr = bool(held >> pr & 1) and not (pl == pr and fl != RIGHT)
            lockbits[held, s] = int(ll) | int(lr) << 1

    _TABLES = {"next": nxt, "p_alt": p_alt, "p_same": p_same,
               "moves_l": moves[0], "moves_r": moves[1], "cost": cost,
               "single_tok": single_tok, "jump_tok": jump_tok,
               "lockbits": lockbits}
    return _TABLES


def mirror_state(s: int) -> int:
    pl, pr, fl = state_tuple(s)
    return state_index(MIRROR_PANEL[pr], MIRROR_PANEL[pl], 1 - fl)


def ergo_matrix() -> np.ndarray:
    """C_ergo for the Stage B regularizer: shape (N_DT, 20, 20), values in
    [0, 1]. Entry [dtb, a, b] is the discomfort of token `b` directly after
    token `a`, averaged over every state `a` could plausibly have been taken
    from, capped at the double-step level and normalized by it."""
    t = tables()
    cap = float(DSTEP_COST[0])
    base = t["cost"][:, 0]                              # (N_DT, S, 20), no locks
    out = np.zeros((N_DT, N_TOKENS, N_TOKENS), np.float32)
    for dtb in range(N_DT):
        for a in range(N_TOKENS):
            ok = base[dtb, :, a] < HARD_COST            # states where `a` is sane
            if not ok.any():
                ok[:] = True
            after = t["next"][ok, a]
            out[dtb, a] = np.minimum(base[dtb, after, :], cap).mean(0) / cap
    return out


# --------------------------------------------------------------------------- #
# Chart containers
# --------------------------------------------------------------------------- #
@dataclass
class ChartEvents:
    """A dance-single chart as 1- and 2-panel events on the 48-tick grid.

    Arrays are parallel, sorted by `tick`. `p1 == -1` marks a single step; for
    a jump `p0 < p1`. `k*` is TAP/HOLD/ROLL for that panel and `d*` its freeze
    length in ticks (0 for a tap). `mines` is (M, 2) rows of (tick, column).
    """
    tick: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    k0: np.ndarray
    k1: np.ndarray
    d0: np.ndarray
    d1: np.ndarray
    mines: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))

    def __len__(self) -> int:
        return len(self.tick)

    @classmethod
    def from_rows(cls, rows, mines=()) -> "ChartEvents":
        """rows: iterable of (tick, [(panel, kind, dur), ...]) with 1 or 2
        panels per row. Convenience constructor for tests and decoding."""
        rows = sorted(rows, key=lambda r: r[0])
        n = len(rows)
        tick = np.zeros(n, np.int32)
        p = np.full((2, n), -1, np.int8)
        k = np.zeros((2, n), np.int8)
        d = np.zeros((2, n), np.int32)
        for i, (t, cells) in enumerate(rows):
            cells = sorted((c if isinstance(c, (tuple, list)) else (c, TAP, 0))
                           for c in cells)
            if not 1 <= len(cells) <= 2:
                raise ValueError(f"row at tick {t} has {len(cells)} panels")
            tick[i] = t
            for j, cell in enumerate(cells):
                panel, kind, dur = (tuple(cell) + (TAP, 0))[:3]
                p[j, i], k[j, i], d[j, i] = panel, kind, dur if kind else 0
        m = np.asarray(sorted(mines), np.int32).reshape(-1, 2)
        return cls(tick, p[0], p[1], k[0], k[1], d[0], d[1], m)

    def rows(self) -> list:
        out = []
        for i in range(len(self)):
            cells = [(int(self.p0[i]), int(self.k0[i]), int(self.d0[i]))]
            if self.p1[i] >= 0:
                cells.append((int(self.p1[i]), int(self.k1[i]), int(self.d1[i])))
            out.append((int(self.tick[i]), cells))
        return out

    def mirrored(self) -> "ChartEvents":
        m = np.asarray(MIRROR_PANEL, np.int8)
        a = m[self.p0]
        b = np.where(self.p1 >= 0, m[np.maximum(self.p1, 0)], -1).astype(np.int8)
        swap = (b >= 0) & (b < a)
        mines = self.mines.copy()
        if len(mines):
            mines[:, 1] = m[mines[:, 1]]
            mines = mines[np.lexsort((mines[:, 1], mines[:, 0]))]
        return ChartEvents(
            self.tick.copy(), np.where(swap, b, a).astype(np.int8),
            np.where(swap, a, b).astype(np.int8),
            np.where(swap, self.k1, self.k0), np.where(swap, self.k0, self.k1),
            np.where(swap, self.d1, self.d0), np.where(swap, self.d0, self.d1), mines)

    def equals(self, other: "ChartEvents") -> bool:
        return all(np.array_equal(getattr(self, f), getattr(other, f))
                   for f in ("tick", "p0", "p1", "k0", "k1", "d0", "d1", "mines"))


@dataclass
class EncodedChart:
    f_init: int
    tick: np.ndarray            # (N,) int32
    token: np.ndarray           # (N,) uint8 in 0..19
    flavor: np.ndarray          # (N,) uint8 in 0..8
    dur_alt: np.ndarray         # (N,) int32 ticks, freeze on the alt foot's panel
    dur_same: np.ndarray        # (N,) int32 ticks, freeze on the same foot's panel
    mines: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))

    def __len__(self) -> int:
        return len(self.tick)


@dataclass
class EncodeInfo:
    cost: np.ndarray            # (N,) per-transition discomfort along the path
    total_cost: int
    lock_violations: int        # transitions that had to move a holding foot
    spins: int

    @property
    def mean_cost(self) -> float:
        return float(self.cost.mean()) if len(self.cost) else 0.0


# --------------------------------------------------------------------------- #
# Encode (inverse footflow) / decode
# --------------------------------------------------------------------------- #
def event_times_ms(tick, bpm: float = 120.0):
    """Tick -> ms on a constant-tempo grid (tests and constant-BPM inference)."""
    return np.asarray(tick, np.float64) * (60000.0 / bpm / TICKS_PER_BEAT)


def _held_masks(ev: ChartEvents, inclusive: bool = False) -> np.ndarray:
    """Per event, the bitmask of columns whose freeze is in progress.

    By default strictly so (head < tick < tail): when reading human charts the
    tail tick is free, because releasing a hold and stepping on the same row is
    ordinary. `inclusive=True` also pins the tail tick (head < tick <= tail),
    which is the stricter rule the generator holds itself to (spec §4.3)."""
    n = len(ev)
    held = np.zeros(n, np.int64)
    tick = ev.tick
    for p, k, d in ((ev.p0, ev.k0, ev.d0), (ev.p1, ev.k1, ev.d1)):
        for i in np.nonzero((k > 0) & (d > 0))[0]:
            j0 = i + 1
            j1 = int(np.searchsorted(tick, tick[i] + d[i],
                                     side="right" if inclusive else "left"))
            if j1 > j0:
                held[j0:j1] |= 1 << int(p[i])
    return held


def encode(ev: ChartEvents, times_ms=None, bpm: float = 120.0):
    """Infer the minimum-discomfort foot assignment and return
    (EncodedChart, EncodeInfo).

    `times_ms` are the events' wall-clock times (ms); they only select the gap
    bucket of the cost model. Ties between equally comfortable assignments are
    broken toward the lexicographically smallest token sequence, then toward
    `f_init = LEFT`, which makes the result exactly mirror-equivariant.
    """
    n = len(ev)
    if n == 0:
        z = np.zeros(0, np.int32)
        return (EncodedChart(LEFT, z, z.astype(np.uint8), z.astype(np.uint8), z, z,
                             ev.mines.copy()),
                EncodeInfo(z, 0, 0, 0))
    t = tables()
    if times_ms is None:
        times_ms = event_times_ms(ev.tick, bpm)
    times_ms = np.asarray(times_ms, np.float64)
    dt = np.empty(n)
    dt[0] = 1e9
    dt[1:] = np.diff(times_ms)
    dtb = dt_bucket(dt)
    held = _held_masks(ev)

    # choice tables: tok[i, s, c] for the two candidate foot assignments
    is_jump = ev.p1 >= 0
    p0 = ev.p0.astype(np.int64)
    p1 = np.where(is_jump, ev.p1, 0).astype(np.int64)
    tok = np.where(is_jump[:, None, None],
                   t["jump_tok"][p0, np.where(is_jump, p1, (p0 + 1) % 4)],
                   t["single_tok"][p0]).astype(np.int64)             # (N, S, 2)
    s_idx = np.arange(N_STATES)[None, :, None]
    lock = t["lockbits"][held][:, :, None].astype(np.int64)           # (N, S, 1)
    cost = t["cost"][dtb[:, None, None], lock, s_idx, tok]            # (N, S, 2)
    nxt = t["next"][s_idx, tok].astype(np.int64)                      # (N, S, 2)

    # backward cost-to-go
    v = np.zeros((n + 1, N_STATES), np.int64)
    q = np.empty((n, N_STATES, 2), np.int64)
    for i in range(n - 1, -1, -1):
        q[i] = cost[i] + v[i + 1][nxt[i]]
        v[i] = q[i].min(1)

    def walk(f_init: int):
        s = start_state(f_init)
        toks = np.empty(n, np.int64)
        states = np.empty(n, np.int64)
        for i in range(n):
            states[i] = s
            qa, qb = q[i, s]
            ta, tb = tok[i, s]
            c = 0 if (qa < qb or (qa == qb and ta <= tb)) else 1
            toks[i] = tok[i, s, c]
            s = nxt[i, s, c]
        return toks, states

    best = None
    for f_init in (LEFT, RIGHT):
        total = int(v[0, start_state(f_init)])
        toks, states = walk(f_init)
        key = (total, toks.tolist(), f_init)
        if best is None or key < best[0]:
            best = (key, toks, states, f_init, total)
    _, toks, states, f_init, total = best

    step_cost = t["cost"][dtb, t["lockbits"][held, states], states, toks]
    pa = t["p_alt"][states, toks]
    ps = t["p_same"][states, toks]
    # which event column (0/1) does the alt / same foot press?
    alt_is_0 = pa == ev.p0
    same_is_0 = ps == ev.p0
    k_alt = np.where(pa < 0, 0, np.where(alt_is_0, ev.k0, ev.k1))
    k_same = np.where(ps < 0, 0, np.where(same_is_0, ev.k0, ev.k1))
    d_alt = np.where(pa < 0, 0, np.where(alt_is_0, ev.d0, ev.d1))
    d_same = np.where(ps < 0, 0, np.where(same_is_0, ev.d0, ev.d1))
    flavor = np.array([_FLAVOR_OF[(int(a), int(b))] for a, b in zip(k_alt, k_same)],
                      np.uint8)
    enc = EncodedChart(int(f_init), ev.tick.astype(np.int32), toks.astype(np.uint8),
                       flavor, d_alt.astype(np.int32), d_same.astype(np.int32),
                       ev.mines.copy())
    info = EncodeInfo(step_cost.astype(np.int32), total,
                      int((step_cost >= LOCK_VIOLATION).sum()),
                      int(((step_cost % LOCK_VIOLATION) >= SPIN_COST).sum()))
    return enc, info


def decode(enc: EncodedChart) -> ChartEvents:
    """Exact inverse of `encode`: tokens + f_init -> concrete panels."""
    n = len(enc)
    p = np.full((2, n), -1, np.int8)
    k = np.zeros((2, n), np.int8)
    d = np.zeros((2, n), np.int32)
    pl, pr, fl = L, R, int(enc.f_init)
    for i in range(n):
        pl, pr, fl, pa, ps = apply_token(pl, pr, fl, int(enc.token[i]))
        ka, ks = FLAVOR_KINDS[int(enc.flavor[i])]
        cells = []
        if pa >= 0:
            cells.append((pa, ka, int(enc.dur_alt[i]) if ka else 0))
        if ps >= 0:
            cells.append((ps, ks, int(enc.dur_same[i]) if ks else 0))
        cells.sort()
        for j, (panel, kind, dur) in enumerate(cells):
            p[j, i], k[j, i], d[j, i] = panel, kind, dur
    return ChartEvents(np.asarray(enc.tick, np.int32).copy(), p[0], p[1], k[0], k[1],
                       d[0], d[1], np.asarray(enc.mines, np.int32).reshape(-1, 2).copy())


def foot_path(enc: EncodedChart):
    """Replay an encoding: per event (state_before, p_alt, p_same, f_last_after).
    Used by conditioning statistics and QA."""
    n = len(enc)
    before = np.empty(n, np.int64)
    pa = np.empty(n, np.int8)
    ps = np.empty(n, np.int8)
    pl, pr, fl = L, R, int(enc.f_init)
    for i in range(n):
        before[i] = state_index(pl, pr, fl)
        pl, pr, fl, pa[i], ps[i] = apply_token(pl, pr, fl, int(enc.token[i]))
    return before, pa, ps


# --------------------------------------------------------------------------- #
# Ingestion: raw note rows -> ChartEvents
# --------------------------------------------------------------------------- #
@dataclass
class IngestStats:
    notes: int = 0              # tap/hold/roll heads seen
    rows: int = 0               # rows kept
    hands: int = 0              # rows that had 3+ panels (thinned to 2)
    dropped: int = 0            # notes discarded (hands surplus, on a held column)
    orphan_tails: int = 0
    open_holds: int = 0         # heads with no tail (kept as taps)
    mines: int = 0

    @property
    def hand_share(self) -> float:
        return self.hands / self.rows if self.rows else 0.0


_HEADS = {"1": TAP, "2": HOLD, "4": ROLL, "L": TAP}


def events_from_notes(notes) -> tuple:
    """Build ChartEvents from (tick, column, type_char) triples, where type_char
    is the simfile note character ('1' tap, '2' hold head, '3' tail, '4' roll
    head, 'M' mine, 'L' lift; anything else is ignored).

    Repairs what the codec cannot represent, and counts each repair:
    3+ panel rows are thinned to two (freeze heads kept first), a note on a
    column that is still being held is dropped, an unterminated head becomes a
    tap. Returns (ChartEvents, IngestStats).
    """
    st = IngestStats()
    notes = sorted(notes, key=lambda x: (x[0], x[1]))
    open_head: dict = {}                # column -> [tick, kind]
    cells: dict = {}                    # tick -> {col: [kind, dur]}
    mines = []
    for tick, col, ch in notes:
        if not 0 <= col < 4:
            continue
        if ch == "3":
            head = open_head.pop(col, None)
            if head is None:
                st.orphan_tails += 1
                continue
            cells[head[0]][col][1] = tick - head[0]
            continue
        if ch == "M":
            if col not in open_head:
                mines.append((tick, col))
                st.mines += 1
            continue
        kind = _HEADS.get(ch)
        if kind is None:
            continue
        st.notes += 1
        if col in open_head:
            if not kind:                # a tap on a column that is being held
                st.dropped += 1
                continue
            # a second head before any tail: the first one never closed
            cells[open_head[col][0]][col] = [TAP, 0]
            del open_head[col]
            st.open_holds += 1
        cells.setdefault(tick, {})[col] = [kind, 0]
        if kind:
            open_head[col] = [tick, kind]
    for col, head in open_head.items():          # never closed -> plain tap
        cells[head[0]][col] = [TAP, 0]
        st.open_holds += 1

    rows = []
    for tick in sorted(cells):
        row = [(c, k, d) for c, (k, d) in cells[tick].items()]
        for i, (c, k, d) in enumerate(row):      # zero-length freeze -> tap
            if k and d <= 0:
                row[i] = (c, TAP, 0)
        if len(row) > 2:
            st.hands += 1
            st.dropped += len(row) - 2
            row = sorted(row, key=lambda x: (x[1] == TAP, x[0]))[:2]
        rows.append((tick, row))
    st.rows = len(rows)
    return ChartEvents.from_rows(rows, mines), st


def simfile_chart_to_ticks(chart, bpm_at, snap_tol_ms: float = 15.0) -> tuple:
    """Read a `simfile` chart's notes onto the 48-tick grid.

    Returns (triples, offgrid_dropped, snapped) with triples ready for
    `events_from_notes`. A note between ticks (a 32nd, a 64th) is snapped to
    the nearest tick when that moves it <= `snap_tol_ms`, and dropped
    otherwise; `bpm_at(beat)` supplies the tempo that conversion needs.
    """
    from simfile.notes import NoteData
    triples, dropped, snapped = [], 0, 0
    for n in NoteData(chart):
        if n.column >= 4:
            continue
        pos = n.beat * TICKS_PER_BEAT
        if pos < 0:
            dropped += 1
            continue
        if pos.denominator == 1:
            tick = int(pos)
        else:
            tick = int(round(float(pos)))
            err_ms = (abs(float(pos) - tick) / TICKS_PER_BEAT * 60000.0
                      / float(bpm_at(float(n.beat))))
            if err_ms > snap_tol_ms:
                dropped += 1
                continue
            snapped += 1
        triples.append((tick, int(n.column), n.note_type.value))
    return triples, dropped, snapped
