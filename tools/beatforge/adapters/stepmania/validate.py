"""validate.py — legality + ergonomic gates with repair (STEPFORGE §5,
REQ-SM-04/11). Runs AFTER realization: the DP already minimized foot-flow cost,
so repair is mostly NPS thinning, hold-overlap and jump-cap fixes, plus a
foot-flow-ceiling check. >15% of notes repaired -> fail -> re-prompt."""
from __future__ import annotations

import math

from dataclasses import dataclass

from ... import config
from . import footflow as ff
from .grammar import BUDGETS
from .quantize import Placement, measure_resolution_ok, snap_beat


@dataclass
class RepairReport:
    placements: list
    repairs: list
    original: int
    ok: bool                    # False if >15% repaired (re-prompt)


def validate_repair(placements: list[Placement], analysis: dict, difficulty: str) -> RepairReport:
    b = BUDGETS[difficulty]
    bpm, offset = analysis["bpm"], analysis["offset"]
    last_beat = (analysis["duration_s"] - config.PLAYABLE_TAIL_S - offset) * bpm / 60.0
    original = len(placements)
    repairs = []

    # 1. hard legality: window, ≤2 panels, quantization
    kept = []
    for p in sorted(placements, key=lambda p: p.beat):
        if p.beat < config.FIRST_PLAYABLE_BEAT - 1e-6 or p.beat > last_beat + 1e-6:
            repairs.append(("window", round(p.beat, 3))); continue
        panels = tuple(dict.fromkeys(p.panels))[:2]
        if p.hold_beats and len(panels) != 1:
            panels = panels[:1]
        kept.append(Placement(p.beat, panels, p.kind, p.hold_beats, p.meta))

    # 2. iterative ergonomic repair
    kept = _repair(kept, b, bpm, offset, repairs)

    # 3. re-cap the jump share AFTER repair (pad ergonomics).
    #
    # `decide_jumps` caps jumps as a fraction of the notes it was handed, but
    # `_nps` thinning then drops *single taps* preferentially — jumps survive. The
    # surviving share therefore drifts UP as density rises, and it drifted badly:
    # Round 2's `hard` charts measured 21.5% jumps against a 12% budget, more than
    # its own `challenge` tier. On a dance pad a jump is a two-foot commitment, so
    # that is precisely the wrong thing to let inflate.
    kept = cap_jump_share(kept, b, meter=4, repairs=repairs)

    # 4. two feet. Last, because the ergonomic repair above can shift and thin
    # notes but never introduces a hold — so capping here is capping what ships.
    kept = cap_holds(kept, b, repairs=repairs)
    kept = cap_concurrent_holds(kept, repairs=repairs)

    # 4b. ...and bound how LONG both feet are pinned, not just the peak. A chart
    # can never need a third foot and still be miserable to play.
    kept = cap_two_foot_time(kept, b, repairs=repairs)

    # 5. hold tails, AFTER the cap — the cap turns holds into taps, which can
    # free a column and change who a hold's next same-column neighbour is.
    kept = trim_hold_tails(kept, b, repairs=repairs)

    repaired_frac = (original - len(kept)) / original if original else 0.0
    return RepairReport(kept, repairs, original, repaired_frac <= 0.15 + 1e-9)


def cap_jump_share(notes, b, meter: int = 4, repairs: list | None = None):
    """Demote surplus jumps to single taps so the realized share honours the tier.

    Demotes rather than deletes: the note stays on the music (timing untouched,
    SACRED-02 safe), it just stops costing a second foot. Off-downbeat jumps go
    first — a jump on beat 1 is the accent the chart is built around, one on an
    off-beat is the one that makes a pad chart feel scrappy.
    """
    from .realize import _JUMP_MAX_FRAC

    jumps = [i for i, p in enumerate(notes) if len(p.panels) > 1 and not p.hold_beats]
    if not jumps:
        return notes
    cap = int(_JUMP_MAX_FRAC.get(b.jumps, 0.10) * len(notes))
    surplus = len(jumps) - cap
    if surplus <= 0:
        return notes

    def priority(i):
        beat = notes[i].beat
        on_downbeat = abs(beat % meter) < 1e-3
        on_beat = abs(beat - round(beat)) < 1e-3
        return (on_downbeat, on_beat, beat)      # False sorts first -> demoted first

    for i in sorted(jumps, key=priority)[:surplus]:
        p = notes[i]
        notes[i] = Placement(p.beat, p.panels[:1], p.kind, p.hold_beats, p.meta)
        if repairs is not None:
            repairs.append(("jump_demoted", round(p.beat, 3)))
    return notes


#: A dancer has two feet. Anything above this is not "hard", it is impossible.
#:
#: Measured rather than assumed: across the 130 reference charts in the
#: FoFo-Compare-Mix pack, DDC and AutoStepper NEVER exceed two concurrent holds —
#: at any difficulty, Beginner included (15 of 26 Beginner charts use exactly 2).
#: So 2 is both the physical limit and observed practice, and it does not need to
#: be scaled per tier.
MAX_CONCURRENT_HOLDS = 2


def cap_concurrent_holds(notes, repairs: list | None = None,
                         limit: int = MAX_CONCURRENT_HOLDS):
    """Demote holds that would need a third foot, keeping the note on the music.

    `_hold_overlap` already rejects a hold colliding with a note in the SAME
    column, which is why this looked covered and was not: nothing tracked holds
    overlapping each other across DIFFERENT columns. A CHARTFORGE-LoRA run made
    that visible — 10 of 45 shipped charts needed three or four simultaneous
    holds, 6 of them pinning all four panels at once, and no gate said a word.

    Demotes rather than deletes, the same trade `cap_jump_share` makes: the
    timing is the music's (SACRED-02), so the note stays exactly where it is and
    only loses its hold-ness. The newest breaching hold is the one demoted —
    earlier holds have already been committed to by the surrounding steps, and
    dropping the later one is what a dancer with two feet would do anyway.
    """
    active: list[tuple[float, int]] = []          # (end_beat, index)
    for i, p in enumerate(sorted(range(len(notes)), key=lambda j: notes[j].beat)):
        n = notes[p]
        active = [(e, j) for e, j in active if e > n.beat + 1e-6]
        if not n.hold_beats:
            continue
        if len(active) >= limit:
            notes[p] = Placement(n.beat, n.panels, "tap", None, n.meta)
            if repairs is not None:
                repairs.append(("hold_demoted_concurrent", round(n.beat, 3)))
            continue
        active.append((n.beat + n.hold_beats, p))
    return notes


def cap_holds(notes, b, repairs: list | None = None):
    """Enforce the hold budget — both the share AND the absolute count.

    `hold_share` was only ever a GATE. Nothing reduced holds; the gate reported a
    failure and the chart shipped as-is, which is why tightening the ceiling
    changed nothing a player could feel. This is the repair that was missing.

    The absolute cap matters more than the ratio, because a ratio does not bound
    a count on a dense chart: the worst chart measured had 38 holds at a share of
    0.085 — inside its 0.12 ceiling — simply because it had 449 notes.

    Shortest holds go first. A long sustain is the one the music actually asks
    for; a one-beat hold is a tap wearing a costume, and dropping it costs the
    chart nothing.
    """
    holds = [i for i, p in enumerate(notes) if p.hold_beats]
    if not holds:
        return notes
    allowed = min(int(b.hold_share[1] * len(notes)), b.max_holds)
    surplus = len(holds) - allowed
    if surplus <= 0:
        return notes
    for i in sorted(holds, key=lambda j: (notes[j].hold_beats, notes[j].beat))[:surplus]:
        p = notes[i]
        notes[i] = Placement(p.beat, p.panels, "tap", None, p.meta)
        if repairs is not None:
            repairs.append(("hold_demoted_budget", round(p.beat, 3)))
    return notes


def two_foot_fraction(notes) -> float:
    """Fraction of charted time with BOTH feet pinned by holds.

    This is the number that tracks whether a chart is playable, and nothing in
    the pipeline measured it. `hold_share` counts holds; it cannot tell a chart
    that spreads eight holds evenly from one that stacks them in pairs, and those
    play nothing alike. Real charts sit at a MEDIAN of 0.00% here (n=171 across
    DDC, AutoStepper and Founders Mix #2) with p75 under 1%; a CHARTFORGE run
    passing every hold gate measured 4.08% median, i.e. the reference p88.
    """
    events = []
    for p in notes:
        if p.hold_beats:
            events.append((p.beat, 1))
            events.append((p.beat + p.hold_beats, -1))
    if not events:
        return 0.0
    span = max(p.beat for p in notes) - min(p.beat for p in notes)
    if span <= 0:
        return 0.0
    events.sort()
    depth = 0
    last = events[0][0]
    two = 0.0
    for beat, delta in events:
        if depth >= 2:
            two += beat - last
        depth += delta
        last = beat
    return two / span


def cap_two_foot_time(notes, b, repairs: list | None = None):
    """Demote holds until both-feet-pinned time is inside the tier's budget.

    Greedy and deliberately simple: repeatedly demote whichever hold contributes
    the most two-foot time. That converges fast because the offenders are
    clustered — the pattern this exists to break up is a burst of holds stacked
    in pairs, so removing the worst one usually clears several overlaps at once.

    `cap_concurrent_holds` bounds the PEAK (never a third foot); this bounds the
    DURATION. Both are needed: two concurrent holds is legal and real charts do
    it, just rarely and briefly. Licensing it as a routine budget is what made
    the charts unplayable while every existing gate stayed green.
    """
    budget = getattr(b, "max_two_foot_frac", 0.02)
    for _ in range(len(notes)):
        frac = two_foot_fraction(notes)
        if frac <= budget + 1e-9:
            break
        best_i, best_gain = None, 0.0
        for i, p in enumerate(notes):
            if not p.hold_beats:
                continue
            trial = notes[:i] + [Placement(p.beat, p.panels, "tap", None, p.meta)] + notes[i + 1:]
            gain = frac - two_foot_fraction(trial)
            if gain > best_gain:
                best_i, best_gain = i, gain
        if best_i is None:                  # no single demotion helps; stop
            break
        p = notes[best_i]
        notes[best_i] = Placement(p.beat, p.panels, "tap", None, p.meta)
        if repairs is not None:
            repairs.append(("hold_demoted_two_foot", round(p.beat, 3)))
    return notes


def trim_hold_tails(notes, b, repairs: list | None = None):
    """Keep each hold's SNAPPED tail strictly before the next note in its column.

    `_hold_overlap` does this check in float beats, and that is not where the
    collision happens. `quantize.py:62` writes the tail at
    `snap_beat(head + hold_beats, finest_subdiv)` — it ROUNDS, so a tail can move
    forward onto a later row. A hold ending at 117.8 with a tap at 118.0 in the
    same column passes the float check by 0.2 beats and then snaps straight onto
    it, and the tail is dropped: the head is left orphaned and the hold runs to
    the end of the chart. Measured on a shipped chart (Banana Banana / Medium,
    head at beat 115.5 with no tail before 192.5), and REQ-SM-11 forbids exactly
    that.

    Trimming, not deleting: the hold shortens to the last grid row before its
    neighbour. Only if that leaves it under the tier's minimum length does it
    demote to a tap — a hold too short to read as one is a tap anyway.
    """
    subdiv = b.finest_subdiv
    min_len = b.hold_len_beats[0]
    for i, p in enumerate(notes):
        if not p.hold_beats:
            continue
        col = p.panels[0]
        nxt = min((q.beat for j, q in enumerate(notes)
                   if j != i and q.beat > p.beat + 1e-9 and col in q.panels),
                  default=None)
        if nxt is None:
            continue
        if float(snap_beat(p.beat + p.hold_beats, subdiv)) < nxt - 1e-9:
            continue
        # Last grid row strictly before the neighbour.
        steps = math.ceil((nxt - 1e-9) / subdiv) - 1
        new_len = steps * subdiv - p.beat
        if new_len >= min_len - 1e-9:
            notes[i] = Placement(p.beat, p.panels, p.kind, new_len, p.meta)
            tag = "hold_trimmed"
        else:
            notes[i] = Placement(p.beat, p.panels, "tap", None, p.meta)
            tag = "hold_demoted_tail_collision"
        if repairs is not None:
            repairs.append((tag, round(p.beat, 3)))
    return notes


def _repair(notes, b, bpm, offset, repairs, guard_max=5000):
    guard = 0
    while guard < guard_max:
        guard += 1
        drop = (_hold_overlap(notes)
                or _jack(notes, max(1, b.jack_limit))   # jack_limit N = max same-panel run
                or _nps(notes, b.max_nps_4s, bpm, offset))
        if drop is None:
            break
        idx, reason = drop
        repairs.append((reason, round(notes[idx].beat, 3)))
        notes.pop(idx)
    return notes


def _hold_overlap(notes):
    for i, h in enumerate(notes):
        if not h.hold_beats:
            continue
        end = h.beat + h.hold_beats
        hp = set(h.panels)
        for j, n in enumerate(notes):
            if j == i:
                continue
            if h.beat - 1e-6 < n.beat < end + 1e-6 and hp & set(n.panels):
                return (j, "hold_overlap")
    return None


def _jack(notes, limit):
    run = 1
    for i in range(1, len(notes)):
        same = (notes[i].panels == notes[i - 1].panels and not notes[i].hold_beats)
        run = run + 1 if same else 1
        if run > limit:
            return (i, "jack")
    return None


def _nps(notes, max_nps, bpm, offset):
    if not notes:
        return None
    times = [offset + n.beat / bpm * 60 for n in notes]
    for i in range(len(notes)):
        j = i
        while j < len(notes) and times[j] - times[i] < 4.0:
            j += 1
        if (j - i) / 4.0 > max_nps + 1e-6:
            mid = (i + j) // 2
            taps = [k for k in range(i, j) if not notes[k].hold_beats and len(notes[k].panels) == 1]
            pool = taps or list(range(i, j))
            return (min(pool, key=lambda k: abs(k - mid)), "nps")
    return None
