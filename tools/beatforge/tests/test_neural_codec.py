"""REQ-SFN-03 / REQ-SFN-04 — the foot-relative codec and the style descriptor.

The two properties the whole model rests on are checked exhaustively rather
than by example: every chart decodes back to itself bit-for-bit, and a chart
and its left-right mirror produce the same tokens.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from beatforge.adapters.stepmania.grammar import BUDGETS
from beatforge.adapters.stepmania.neural import codec as C
from beatforge.adapters.stepmania.neural import conditioning as cond

FIXTURE = Path(__file__).parent / "fixtures" / "neural" / "fixture_song.ssc"
NAMES = "LDUR"


def chart(seq: str, step: int = 3, start: int = 0) -> C.ChartEvents:
    """'L D LR U' -> events, one row every `step` ticks ('.' is a rest)."""
    rows = []
    for i, cell in enumerate(seq.split()):
        if cell != ".":
            rows.append((start + i * step, [(NAMES.index(c), C.TAP, 0) for c in cell]))
    return C.ChartEvents.from_rows(rows)


def random_chart(rng, n: int) -> C.ChartEvents:
    rows, t, busy = [], 0, {}
    for _ in range(n):
        t += int(rng.choice([3, 6, 12, 12, 24, 4]))
        free = [c for c in range(4) if busy.get(c, -1) < t]
        if not free:
            continue
        k = 2 if (rng.random() < 0.15 and len(free) >= 2) else 1
        cells = []
        for c in rng.choice(free, k, replace=False):
            if rng.random() < 0.15:
                d = int(rng.choice([6, 12, 24, 36]))
                busy[int(c)] = t + d
                cells.append((int(c), int(rng.choice([C.HOLD, C.ROLL])), d))
            else:
                cells.append((int(c), C.TAP, 0))
        rows.append((t, cells))
    return C.ChartEvents.from_rows(rows)


PATTERNS = {
    "stream": "L D U R L D U R L D U R L D U R",
    "crossover": "L D R D L D R D L U R U L U R U",
    "candle": "L D R U L U R D L D R U",
    "jacks": "L L L L D D D D R R U U",
    "du_footswitch": "L D D R U U L D D R",
    "jumps": "LR D U LR DU L R LD UR LU DR L",
    "staircase": "L D U R R U D L L D U R",
}


def assert_round_trip_and_mirror(ev: C.ChartEvents, bpm: float = 150.0):
    enc, info = C.encode(ev, bpm=bpm)
    assert C.decode(enc).equals(ev), "decode(encode(x)) != x"
    assert enc.token.max(initial=0) < C.N_TOKENS
    mirrored = ev.mirrored()
    enc_m, _ = C.encode(mirrored, bpm=bpm)
    assert np.array_equal(enc.token, enc_m.token), "tokens differ under L<->R"
    assert np.array_equal(enc.flavor, enc_m.flavor)
    assert np.array_equal(enc.dur_alt, enc_m.dur_alt)
    assert np.array_equal(enc.dur_same, enc_m.dur_same)
    assert np.array_equal(enc.tick, enc_m.tick)
    if not mirrored.equals(ev):
        assert enc_m.f_init != enc.f_init, "mirror must flip only the parity bit"
    assert C.decode(enc_m).equals(mirrored)
    return enc, info


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #
def test_vocabulary_is_twenty_tokens_with_twelve_legal_jumps():
    assert C.N_TOKENS == 20 and C.MASK == 20
    assert len(C.JUMP_PAIRS) == 12
    collide = {(C.DOWN, C.DOWN), (C.UP, C.UP), (C.HOME, C.CROSS), (C.CROSS, C.HOME)}
    assert collide.isdisjoint(C.JUMP_PAIRS)
    assert list(C.JUMP_PAIRS) == sorted(C.JUMP_PAIRS)       # lexicographic


def test_panel_decoder_commutes_with_mirror():
    for f in (C.LEFT, C.RIGHT):
        for z in range(4):
            assert (C.PANEL_OF[1 - f][z]
                    == C.MIRROR_PANEL[C.PANEL_OF[f][z]])


def test_every_jump_token_lands_on_two_distinct_panels():
    for s in range(C.N_STATES):
        pl, pr, fl = C.state_tuple(s)
        for tok in C.JUMP_TOKENS:
            *_, pa, ps = C.apply_token(pl, pr, fl, tok)
            assert pa >= 0 and ps >= 0 and pa != ps


def test_post_jump_parity_follows_the_foot_that_moved():
    # feet at rest (L, R), last foot RIGHT -> alt = LEFT.
    # jump where only the alt (left) foot moves: L stays? choose (DOWN, HOME):
    # left -> D (moved), right -> R (did not) => last = LEFT
    tok = 8 + C.JUMP_PAIRS.index((C.DOWN, C.HOME))
    *_, fl, _, _ = C.apply_token(C.L, C.R, C.RIGHT, tok)
    assert fl == C.LEFT
    # only the same (right) foot moves: (HOME, UP) => last = RIGHT
    tok = 8 + C.JUMP_PAIRS.index((C.HOME, C.UP))
    *_, fl, _, _ = C.apply_token(C.L, C.R, C.RIGHT, tok)
    assert fl == C.RIGHT
    # neither moves (jump-jack), and both move: the alternating foot is last
    for pair in ((C.HOME, C.HOME), (C.DOWN, C.UP)):
        tok = 8 + C.JUMP_PAIRS.index(pair)
        *_, fl, _, _ = C.apply_token(C.L, C.R, C.RIGHT, tok)
        assert fl == C.LEFT


# --------------------------------------------------------------------------- #
# REQ-SFN-03: lossless + mirror invariant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(PATTERNS))
@pytest.mark.parametrize("bpm", [90.0, 150.0, 200.0])
def test_patterns_round_trip_and_mirror(name, bpm):
    assert_round_trip_and_mirror(chart(PATTERNS[name]), bpm)


def test_random_charts_round_trip_and_mirror():
    rng = np.random.default_rng(7)
    for _ in range(100):
        ev = random_chart(rng, int(rng.integers(4, 220)))
        assert_round_trip_and_mirror(ev, float(rng.uniform(80, 210)))


def test_freeze_arrow_tap_transitions_round_trip():
    # a hold on L while the other foot taps; a roll into a jump; a jump whose
    # two panels carry different freezes (the case six flavors cannot express)
    ev = C.ChartEvents.from_rows([
        (0, [(C.L, C.HOLD, 24)]), (6, [(C.D,)]), (12, [(C.U,)]), (18, [(C.R,)]),
        (24, [(C.R, C.ROLL, 12)]), (30, [(C.D,)]), (36, [(C.L,), (C.R,)]),
        (48, [(C.L, C.HOLD, 12), (C.R, C.ROLL, 24)]),
        (96, [(C.D, C.HOLD, 6), (C.U,)]), (108, [(C.L, C.ROLL, 12), (C.R, C.ROLL, 12)]),
    ])
    enc, _ = assert_round_trip_and_mirror(ev, 120.0)
    assert enc.flavor.max() > C.ROLL_SAME, "mixed-kind jump needs an extended flavor"
    assert set(C.flavor6(enc.flavor).tolist()) <= set(range(C.N_FLAVORS))


def test_real_ssc_fixture_round_trips():
    import simfile
    from beatforge.adapters.stepmania.neural import grid_features as GF
    from beatforge.adapters.stepmania.neural.train.corpus import chart_notes_to_ticks
    sf = simfile.open(str(FIXTURE))
    timing = GF.TimingMap.constant(120.0, 0.25)
    seen = 0
    for ch in sf.charts:
        triples, off, snapped = chart_notes_to_ticks(ch, timing)
        ev, st = C.events_from_notes(triples)
        assert off == 0 and snapped == 0 and st.hands == 0
        enc, info = assert_round_trip_and_mirror(ev, 120.0)
        assert info.lock_violations == 0
        seen += 1
    assert seen == 2


def test_self_mirrored_chart_encodes_identically():
    ev = chart("D U D U LR D U DU")
    assert ev.mirrored().equals(ev)
    a, _ = C.encode(ev)
    b, _ = C.encode(ev.mirrored())
    assert a.f_init == b.f_init and np.array_equal(a.token, b.token)


# --------------------------------------------------------------------------- #
# the foot model reads charts the way a player does
# --------------------------------------------------------------------------- #
def test_stream_alternates_with_zero_cost():
    enc, info = C.encode(chart(PATTERNS["stream"]), bpm=150)
    assert set(enc.token.tolist()) <= set(C.ALT_TOKENS)
    assert info.total_cost == 0


def test_repeated_arrow_is_a_jack_not_a_footswitch():
    enc, _ = C.encode(chart("L L L L", step=12), bpm=120)
    assert enc.token[0] in C.ALT_TOKENS
    assert all(t in C.SAME_TOKENS for t in enc.token[1:])


def test_first_step_is_never_crossed():
    for seq in ("L D R D L D R D", "R U L U R U L U", "L U R"):
        ev = chart(seq)
        enc, _ = C.encode(ev, bpm=150)
        assert int(enc.token[0]) == C.HOME, seq          # ALT + HOME


def test_hold_lock_pins_the_holding_foot():
    # left foot holds L for two beats; everything else must come from the right
    ev = C.ChartEvents.from_rows(
        [(0, [(C.L, C.HOLD, 24)])] + [(3 * i, [(p,)]) for i, p in
                                      enumerate((C.D, C.U, C.R, C.U, C.D), start=1)])
    enc, info = C.encode(ev, bpm=150)
    assert info.lock_violations == 0
    before, pa, ps = C.foot_path(enc)
    t = C.tables()
    for i in range(1, len(enc)):
        s, tok = int(before[i]), int(enc.token[i])
        left_moves = bool(t["moves_l"][s, tok])
        assert not left_moves, f"event {i} moved the foot that is holding"
    # after the first alternation onto the free foot, the rest are forced SAMEs
    assert all(int(x) in C.SAME_TOKENS for x in enc.token[2:])
    # ...and a forced double-step under a hold is not charged as one
    assert info.cost[2:].max() < C.DSTEP_COST[-2]


def test_cost_table_is_mirror_symmetric():
    t = C.tables()
    ms = np.array([C.mirror_state(s) for s in range(C.N_STATES)])
    swap_locks = np.array([0, 2, 1, 3])
    assert np.array_equal(t["cost"], t["cost"][:, swap_locks][:, :, ms])
    assert np.array_equal(ms[t["next"]], t["next"][ms])


def test_spin_is_a_hard_cost():
    # left foot on R, right foot steps onto L: a 180-degree spin
    c = C.transition_cost(C.R, C.D, C.LEFT, C.CROSS, dtb=1)   # alt = RIGHT -> L
    assert c >= C.HARD_COST


# --------------------------------------------------------------------------- #
# ingestion repairs
# --------------------------------------------------------------------------- #
def test_ingest_thins_hands_and_reports_it():
    notes = [(0, 0, "1"), (0, 1, "1"), (0, 3, "1"),           # a 3-panel hand
             (12, 0, "2"), (18, 0, "1"), (24, 0, "3"),        # tap on a held column
             (36, 2, "2"),                                      # head with no tail
             (48, 1, "3"),                                      # orphan tail
             (60, 3, "M"), (72, 2, "4"), (84, 2, "3")]
    ev, st = C.events_from_notes(notes)
    assert st.hands == 1 and st.dropped == 2
    assert st.open_holds == 1 and st.orphan_tails == 1 and st.mines == 1
    assert all(ev.p1[i] == -1 or ev.p0[i] < ev.p1[i] for i in range(len(ev)))
    rows = dict(ev.rows())
    assert rows[12] == [(C.L, C.HOLD, 12)]
    assert rows[36] == [(C.U, C.TAP, 0)]
    assert rows[72] == [(C.U, C.ROLL, 12)]
    assert ev.mines.tolist() == [[60, 3]]
    assert_round_trip_and_mirror(ev)


def test_dur_bucket_and_flavor_projection():
    assert [int(C.dur_bucket(b)) for b in C.HOLD_DUR_BUCKETS] == list(range(10))
    assert int(C.dur_bucket(1)) == 0 and int(C.dur_bucket(500)) == 9
    assert C.flavor6([0, 1, 2, 3, 4, 5, 6, 7, 8]).tolist() == [0, 1, 2, 3, 4, 5, 3, 3, 3]


def test_ergo_matrix_is_normalized_and_penalizes_double_steps():
    m = C.ergo_matrix()
    assert m.shape == (C.N_DT, 20, 20)
    assert m.min() >= 0.0 and m.max() <= 1.0
    fast = m[0]
    # after an ALT+DOWN, a clean ALT+HOME is free, a SAME+HOME (double-step) is not
    assert fast[C.DOWN, C.HOME] < 0.05 < fast[C.DOWN, 4 + C.HOME]
    # the long-gap bucket forgives what the fast one punishes
    assert m[-1][C.DOWN, 4 + C.HOME] < fast[C.DOWN, 4 + C.HOME]


# --------------------------------------------------------------------------- #
# REQ-SFN-04: style descriptor + presets
# --------------------------------------------------------------------------- #
def style_of(ev: C.ChartEvents, bpm: float = 150.0, meter: int = 9) -> np.ndarray:
    enc, _ = C.encode(ev, bpm=bpm)
    return cond.compute_style_vector(enc, C.event_times_ms(ev.tick, bpm) / 1000.0, meter)


def test_style_vector_is_deterministic_and_bounded():
    rng = np.random.default_rng(3)
    for _ in range(40):
        ev = random_chart(rng, int(rng.integers(10, 300)))
        a, b = style_of(ev), style_of(ev)
        assert np.array_equal(a, b)
        assert a.shape == (cond.STYLE_DIM,) and a.dtype == np.float32
        assert (a >= 0).all() and (a <= 1.0).all()
        assert (a <= cond.MAXIMA + 1e-6).all()


def test_pure_stream_scores_high_on_stream_share():
    c = style_of(chart(" ".join(["L D U R"] * 16)))
    i = cond.INDEX
    assert c[i["stream_share"]] > 0.95
    assert c[i["jump_share"]] == 0 and c[i["crossover_share"]] == 0
    assert c[i["jack_share"]] == 0


def test_jumpstream_scores_high_on_jump_share():
    c = style_of(chart(" ".join(["LR D U LD R L UR D"] * 8)))
    assert c[cond.INDEX["jump_share"]] > 0.30
    plain = style_of(chart(" ".join(["L D U R"] * 16)))
    assert c[cond.INDEX["jump_share"]] > plain[cond.INDEX["jump_share"]]


def test_crossover_heavy_scores_high_on_crossover_share():
    c = style_of(chart(" ".join(["L D R D L U R U"] * 8)))
    plain = style_of(chart(" ".join(["L D U R"] * 16)))
    assert c[cond.INDEX["crossover_share"]] > 0.15 > plain[cond.INDEX["crossover_share"]]


def test_jack_and_hold_and_mine_shares():
    c = style_of(chart(" ".join(["L L D D U U R R"] * 6), step=6))
    assert c[cond.INDEX["jack_share"]] > 0.15
    held = C.ChartEvents.from_rows(
        [(i * 24, [(i % 4, C.HOLD, 12)]) for i in range(20)],
        mines=[(i * 24 + 18, (i + 2) % 4) for i in range(4)])
    c = style_of(held)
    assert c[cond.INDEX["hold_share"]] == pytest.approx(0.35)       # clamped
    assert 0 < c[cond.INDEX["mine_share"]] <= 0.15


def test_nps_and_meter_normalization():
    ev = chart(" ".join(["L D U R"] * 32))               # 16ths at 150 BPM = 10 nps
    c = style_of(ev, bpm=150.0, meter=11)
    assert c[0] == pytest.approx(11 / 15)
    assert c[1] == pytest.approx(10 / 16, abs=0.03)
    assert style_of(ev, meter=40)[0] == 1.0 and style_of(ev, meter=0)[0] == pytest.approx(1 / 15)


def test_presets_are_inside_the_tier_budgets():
    assert set(cond.PRESET_NAMES) == {"casual_flow", "itg_stamina", "tech_footswitch",
                                      "jumpstream"}
    for (diff, name), c in cond.PRESETS.items():
        b = BUDGETS[diff]
        ceil = cond.tier_ceilings(diff)
        assert c.shape == (10,) and (c >= 0).all() and (c <= 1).all(), (diff, name)
        assert b.meter_range[0] / 15 - 1e-6 <= c[0] <= b.meter_range[1] / 15 + 1e-6
        assert c[1] * 16 <= b.max_nps_4s + 1e-6
        assert c[4] <= b.hold_share[1] + 1e-6
        assert c[3] <= ceil["jump"] + 1e-6 and c[5] <= ceil["cross"] + 1e-6
        if not b.footswitch:
            assert c[7] == 0
        if b.crossover == "none":
            assert c[5] == 0
    # presets differ where they should
    hard = {n: cond.PRESETS[("hard", n)] for n in cond.PRESET_NAMES}
    assert hard["jumpstream"][3] > hard["itg_stamina"][3]
    assert hard["itg_stamina"][2] > hard["casual_flow"][2]
    assert hard["tech_footswitch"][7] > hard["casual_flow"][7]


def test_cond19_masks_dropped_fields():
    c = cond.PRESETS[("hard", "jumpstream")]
    v = cond.cond19(c)
    assert v.shape == (cond.COND_DIM,) and np.array_equal(v[10:], np.ones(9))
    v = cond.cond19(c, drop=[1, 3])
    assert v[1] == cond.MASKED and v[3] == cond.MASKED
    assert v[10 + 1] == 0 and v[10 + 3] == 0 and v[10 + 2] == 1
    t = cond.cond19(cond.tier_only_vector("medium"))
    assert t[0] > 0 and (t[1:9] == cond.MASKED).all() and t[11:19].sum() == 0


def test_resolve_style_applies_slider_overrides_and_clamps():
    c = cond.resolve_style("hard", overrides={"jump_share": 0.9, "nps": 5.0})
    assert c[3] == pytest.approx(cond.tier_ceilings("hard")["jump"])
    assert c[1] == pytest.approx(5 / 16)
    assert c[2] == cond.MASKED                             # untouched slider stays free
    c = cond.resolve_style("easy", preset="tech_footswitch")
    assert c[5] == 0 and c[7] == 0                         # easy forbids both
    with pytest.raises(ValueError):
        cond.resolve_style("hard", overrides={"jmup_share": 0.1})
    with pytest.raises(ValueError):
        cond.resolve_style("hard", preset="nope")
