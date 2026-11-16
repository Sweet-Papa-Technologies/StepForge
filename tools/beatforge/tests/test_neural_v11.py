"""StepForge Flow v1.1: decoder, foot-model and grid refinements
(docs/v1.1/improvements.v1.1.md). No retraining: every test here is about what
happens between the networks' logits and the chart.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from beatforge.adapters.stepmania import footflow as ff  # noqa: E402
from beatforge.adapters.stepmania import qa as sm_qa  # noqa: E402
from beatforge.adapters.stepmania.grammar import BUDGETS  # noqa: E402
from beatforge.adapters.stepmania.neural import codec as C  # noqa: E402
from beatforge.adapters.stepmania.neural import conditioning as cond  # noqa: E402
from beatforge.adapters.stepmania.neural import grid_features as GF  # noqa: E402
from beatforge.adapters.stepmania.neural import infer, pipeline  # noqa: E402
from beatforge.adapters.stepmania.quantize import Placement  # noqa: E402

from test_neural_infer import BPM, OFFSET, fake_logits, tick_times  # noqa: E402


def singles(ticks, difficulty: str, unary, tt, style=None):
    """Solve the foot path for a run of single notes the way generate() does."""
    ticks = np.asarray(ticks, np.int64)
    n = len(ticks)
    z = np.zeros(n, np.int64)
    plan = infer.RhythmPlan(ticks, z.astype(bool), z, z, z)
    rules = infer.TierRules.for_difficulty(difficulty)
    times_ms = tt[ticks] * 1000.0
    dt = np.concatenate([[1e9], np.diff(times_ms)])
    style = cond.resolve_style(difficulty) if style is None else style
    extra, no_repeat = infer.timing_penalties(rules, style, dt / 1000.0)
    slots, kind = infer.lock_schedule(ticks, plan.is_jump, plan.flavor, plan.dur)
    res = infer.enforce_run_limit(
        dict(unary=unary, dtb=C.dt_bucket(dt), is_jump=plan.is_jump, slots=slots,
             hold_kind=kind, rules=rules, **extra), ticks, rules, no_repeat=no_repeat)
    return res, infer.token_rows(res.token, res.f_init)


def longest_run(rows) -> int:
    run = best = 1
    for a, b in zip(rows, rows[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


# --------------------------------------------------------------------------- #
# 1. default density
# --------------------------------------------------------------------------- #
def test_default_density_is_scaled_per_tier_and_the_slider_is_not(monkeypatch):
    t = 48 * 60
    tt = tick_times(t)
    logits = fake_logits(t, 1)
    style = cond.tier_only_vector("beginner")
    scaled = infer.decode_rhythm(logits, tt, "beginner", style, float(tt[-1]) + 2)
    monkeypatch.setitem(infer.DENSITY_SCALE, "beginner", 1.0)
    plain = infer.decode_rhythm(logits, tt, "beginner", style, float(tt[-1]) + 2)
    assert infer.DENSITY_SCALE["easy"] <= 1.0
    assert scaled.stats["expected_count"] < plain.stats["expected_count"]
    assert len(scaled.tick) <= len(plain.tick)
    # an explicit NPS slider is a request, not a default: untouched by the scale
    asked = cond.resolve_style("beginner", overrides={"nps": 1.5})
    a = infer.decode_rhythm(logits, tt, "beginner", asked, float(tt[-1]) + 2)
    monkeypatch.setitem(infer.DENSITY_SCALE, "beginner", 0.5)
    b = infer.decode_rhythm(logits, tt, "beginner", asked, float(tt[-1]) + 2)
    assert np.array_equal(a.tick, b.tick)


# --------------------------------------------------------------------------- #
# 2. slow repeats at the tiers with no jack budget
# --------------------------------------------------------------------------- #
def test_footflow_scores_a_repeat_by_how_fast_it_is():
    assert ff.repeat_scale(None) == 1.0
    assert ff.repeat_scale(ff.JACK_FULL_S - 0.01) == 1.0
    assert ff.repeat_scale(ff.JACK_FREE_S) == 0.0
    assert 0.0 < ff.repeat_scale(0.5) < 1.0
    b = BUDGETS["beginner"]
    state = (0, 3, ff.LEFT)                         # the left foot just stepped on L
    full, _ = ff.step(state, ff.LEFT, 0, b, "static", 0.0)
    fast, _ = ff.step(state, ff.LEFT, 0, b, "static", 0.0, dt_s=0.2)
    slow, _ = ff.step(state, ff.LEFT, 0, b, "static", 0.0, dt_s=1.0)
    assert full == fast == ff.DOUBLE_STEP + ff.JACK
    assert slow == 0.0
    # a same-foot move to ANOTHER panel is a double-step however slow it is
    dstep, _ = ff.step(state, ff.LEFT, 1, b, "static", 0.0, dt_s=1.0)
    assert dstep >= ff.DOUBLE_STEP - 1.0


def test_qa_flow_gate_passes_slow_repeats_only_when_told_the_tempo():
    b = BUDGETS["beginner"]
    chart = [Placement(float(i * 2), (0,), "tap", None, {"foot": ff.LEFT})
             for i in range(6)]                      # L L L L L L, two beats apart
    worst_blind, _ = sm_qa._flow_costs(chart, b)
    worst_slow, _ = sm_qa._flow_costs(chart, b, bpm=120.0)      # 1.0 s apart
    worst_fast, _ = sm_qa._flow_costs(chart, b, bpm=480.0)      # 0.25 s apart
    ceiling = ff.DOUBLE_STEP + ff.JACK
    assert worst_blind >= ceiling and worst_fast >= ceiling
    assert worst_slow < ceiling


@pytest.mark.parametrize("difficulty", ["beginner", "easy"])
def test_low_tiers_repeat_an_arrow_only_when_it_is_slow(difficulty):
    tt = tick_times(48 * 40)
    n = 24
    # every "same foot again" token is strongly preferred; with double-steps
    # forbidden the only way to take one is to repeat the arrow
    unary = np.zeros((n, C.N_TOKENS))
    unary[:, 4:8] = 6.0
    rules = infer.TierRules.for_difficulty(difficulty)
    assert rules.slow_repeats_only and rules.panel_run_limit == infer.SLOW_REPEAT_RUN
    slow = 48 + 36 * np.arange(n)                   # 36 ticks at 150 BPM = 1.2 s
    fast = 48 + 9 * np.arange(n)                    # 9 ticks = 0.3 s
    _, rows = singles(slow, difficulty, unary, tt)
    assert 1 < longest_run(rows) <= infer.SLOW_REPEAT_RUN
    _, rows = singles(fast, difficulty, unary, tt)
    assert longest_run(rows) == 1


def test_tiers_with_a_jack_budget_keep_their_own_rule():
    rules = infer.TierRules.for_difficulty("medium")
    assert not rules.slow_repeats_only
    assert rules.panel_run_limit == BUDGETS["medium"].jack_limit
    extra, no_repeat = infer.timing_penalties(rules, cond.resolve_style("medium"),
                                              np.full(8, 0.2))
    assert no_repeat is None and "repeat_pen" not in extra


# --------------------------------------------------------------------------- #
# 3. footswitches at Hard and Challenge
# --------------------------------------------------------------------------- #
def footswitches(res) -> int:
    g = infer._gen_tables()
    return int(g["fsw"][res.states, res.token].sum())


@pytest.mark.parametrize("difficulty", ["hard", "challenge"])
def test_footswitch_penalty_removes_the_casual_ones(difficulty, monkeypatch):
    tt = tick_times(48 * 60)
    ticks = 48 + 6 * np.arange(200)                 # 8ths at 150 BPM: 0.2 s apart
    with_pen = without = 0
    for seed in range(6):
        unary = np.random.default_rng(seed).normal(0, 1.0, (len(ticks), C.N_TOKENS))
        res, _ = singles(ticks, difficulty, unary, tt)
        with_pen += footswitches(res)
        monkeypatch.setattr(infer, "FOOTSWITCH_PENALTY", 0.0)
        res, _ = singles(ticks, difficulty, unary, tt)
        without += footswitches(res)
        monkeypatch.undo()
    assert without > 0 and with_pen < 0.5 * without


def test_footswitch_slider_lifts_the_penalty_and_slow_ones_are_free():
    rules = infer.TierRules.for_difficulty("challenge")
    dt = np.array([0.2, 0.2, 1.0])
    default, _ = infer.timing_penalties(rules, cond.resolve_style("challenge"), dt)
    assert default["fsw_pen"].tolist() == [infer.FOOTSWITCH_PENALTY] * 2 + [0.0]
    assert infer.FOOTSWITCH_SLOW_S == 0.60
    ceil = cond.tier_ceilings("challenge")["fsw"]
    asked = cond.resolve_style("challenge", overrides={"footswitch_share": ceil})
    wanted, _ = infer.timing_penalties(rules, asked, dt)
    assert not wanted["fsw_pen"].any()
    half = cond.resolve_style("challenge", overrides={"footswitch_share": ceil / 2})
    some, _ = infer.timing_penalties(rules, half, dt)
    assert some["fsw_pen"][0] == pytest.approx(infer.FOOTSWITCH_PENALTY / 2)
    # a tier that may not footswitch at all needs no penalty
    low, _ = infer.timing_penalties(infer.TierRules.for_difficulty("medium"),
                                    cond.resolve_style("medium"), dt)
    assert "fsw_pen" not in low


# --------------------------------------------------------------------------- #
# 4. energy following
# --------------------------------------------------------------------------- #
def test_every_tier_has_an_energy_tilt_and_low_tiers_lean_harder():
    assert set(infer.ENERGY_TILT) == set(BUDGETS)
    assert infer.ENERGY_TILT["beginner"][0][0] > infer.ENERGY_TILT["challenge"][0][0]
    assert infer.ENERGY_TILT["medium"][1] > infer.ENERGY_TILT["challenge"][1]
    t = 48 * 64
    tt = tick_times(t)
    energy = np.concatenate([np.full(32, 0.2), np.full(32, 0.9)])
    for d in BUDGETS:
        plan = infer.decode_rhythm(fake_logits(t, 4), tt, d, cond.resolve_style(d),
                                   float(tt[-1]) + 2, energy)
        assert plan.stats["energy_weight"] in infer.ENERGY_TILT[d][0]
        assert (plan.tick >= 32 * 48).sum() > (plan.tick < 32 * 48).sum(), d


def test_the_tilt_cannot_saturate_and_ties_are_thinned_evenly():
    # multiplying the odds keeps every score inside (0, 1) and in order
    p = np.array([0.2, 0.5, 0.9, 0.99])
    for m in (0.1, 1.0, 4.5, 50.0):
        out = infer._scale_odds(p, m)
        assert (out > 0).all() and (out < 1).all() and (np.diff(out) > 0).all()
    assert infer._scale_odds(p, 1.0) == pytest.approx(p)
    # twelve equal-scored notes, 0.4 s apart, in a tier that allows 8 per 4 s:
    # the survivors must not all be at one end
    tt = tick_times(48 * 4)
    ticks = 12 * np.arange(12)
    keep = infer._thin_to_peak_nps(ticks, np.ones(12), tt, 2.0)
    kept = ticks[keep]
    assert 6 <= keep.sum() < 12
    assert kept.min() <= 12 and kept.max() >= 120


# --------------------------------------------------------------------------- #
# 5. jumps, freezes, streams, the 16th floor, mines
# --------------------------------------------------------------------------- #
def test_jump_and_freeze_shares_are_boosted_but_stay_inside_the_tier():
    t = 48 * 160
    tt = tick_times(t)
    logits = fake_logits(t, 3)
    logits["hold_type"][:, 0] -= 2.0                # a model that likes freezes
    b = BUDGETS["challenge"]
    plan = infer.decode_rhythm(logits, tt, "challenge", cond.resolve_style("challenge"),
                               float(tt[-1]) + 2)
    n = len(plan.tick)
    holds = int((plan.flavor > 0).sum())
    assert plan.stats["hold_share_model"] * infer.HOLD_SHARE_BOOST > b.hold_share[1]
    assert holds <= b.hold_share[1] * n
    assert holds > b.max_holds                      # the share is the limit, not a count
    want = min(infer.JUMP_SHARE_BOOST * plan.stats["jump_share_model"],
               infer._JUMP_MAX_FRAC[b.jumps])
    assert plan.is_jump.sum() <= want * n + 1
    assert infer._JUMP_MAX_FRAC["downbeats"] >= 0.05


def test_a_beat_is_counted_in_16ths_or_triplets_never_both():
    #            beat 0: 16ths + one triplet   beat 1: triplets + one 16th
    ticks = np.array([0, 3, 4, 6, 9, 12, 15, 16, 20, 24])
    scores = np.array([.9, .8, .2, .8, .8, .9, .1, .8, .8, .9])
    keep = infer._one_grid_per_beat(ticks, scores, np.ones(len(ticks), bool))
    assert ticks[keep].tolist() == [0, 3, 6, 9, 12, 16, 20, 24]
    assert np.diff(ticks[keep]).min() >= 3
    # nothing to do when a beat already uses one family
    clean = np.array([0, 3, 6, 9, 12, 16, 20])
    assert infer._one_grid_per_beat(clean, np.ones(7), np.ones(7, bool)).all()


def test_challenge_has_no_24ths_unless_asked(monkeypatch):
    t = 48 * 60
    tt = tick_times(t)
    logits = fake_logits(t, 7)
    logits["onset"][np.arange(t) % 2 == 0] += 4.0   # 24th-note positions light up
    style = cond.resolve_style("challenge", overrides={"nps": 9.0})
    plan = infer.decode_rhythm(logits, tt, "challenge", style, float(tt[-1]) + 2)
    assert set((plan.tick % 12).tolist()) <= {0, 3, 4, 6, 8, 9}
    for beat in np.unique(plan.tick // 12):         # and never both grids in a beat
        res = set((plan.tick[plan.tick // 12 == beat] % 12).tolist())
        assert not (res & {3, 6, 9} and res & {4, 8})
    monkeypatch.setattr(infer, "ALLOW_24THS", True)
    plan = infer.decode_rhythm(logits, tt, "challenge", style, float(tt[-1]) + 2)
    assert set((plan.tick % 12).tolist()) & {2, 10}


def test_near_complete_16th_bars_in_the_loudest_part_are_filled():
    n_bars = 10
    t = n_bars * C.G
    tt = tick_times(t)
    p = np.full(t, 0.2)
    allowed = np.ones(t, bool)
    rank = np.linspace(0.0, 1.0, n_bars)            # bar 9 loudest, bar 0 quietest
    sixteenths = lambda bar, missing: [bar * C.G + 3 * i for i in range(16)
                                       if i not in missing]
    ticks = np.array(sorted(
        sixteenths(9, {5, 11})                      # loud, 14 of 16: filled
        + sixteenths(8, {1, 2, 3, 4, 5, 6})         # loud, 10 of 16: too sparse
        + sixteenths(1, {5, 11})))                  # 14 of 16 but quiet: left alone
    out, filled = infer._commit_streams(ticks, p, allowed, rank, tt, 16.0, 0.05)
    assert filled == 1
    in_bar = lambda bar: int(((out >= bar * C.G) & (out < (bar + 1) * C.G)).sum())
    assert (in_bar(9), in_bar(8), in_bar(1)) == (16, 10, 14)
    # not where the model hears nothing, and not past the tier's peak rate
    quiet = p.copy()
    quiet[9 * C.G + 15] = 0.0
    assert infer._commit_streams(ticks, quiet, allowed, rank, tt, 16.0, 0.05)[1] == 0
    assert infer._commit_streams(ticks, p, allowed, rank, tt, 3.0, 0.05)[1] == 0
    assert infer.STREAM_COMMIT_MIN >= 11


def crafted_logits(t: int, notes, head: int | None = None, mines: bool = False) -> dict:
    onset = np.full(t, -9.0, np.float32)
    onset[list(notes)] = 6.0
    hold_type = np.zeros((t, 6), np.float32)
    hold_type[:, 0] = 6.0
    hold_dur = np.zeros((t, 10), np.float32)
    if head is not None:
        hold_type[head] = (-6, 6, 0, 0, 0, 0)
        hold_dur[head, C.HOLD_DUR_BUCKETS.index(48)] = 6.0
    return {"onset": onset, "jump": np.full(t, -9.0, np.float32), "hold_type": hold_type,
            "hold_dur": hold_dur,
            "mine": np.full(t, 3.0 if mines else -9.0, np.float32)}


def test_a_freeze_is_released_before_the_free_foot_has_to_jack():
    t = 48 * 40
    tt = tick_times(t)                              # 150 BPM: one beat is 0.4 s
    quarters = [48 + 12 * i for i in range(24)] + [480 + 48 * i for i in range(20)]
    head = 48
    style = cond.resolve_style("hard", overrides={"hold_share": 0.06, "nps": 2.5})

    def hold_end(extra):
        plan = infer.decode_rhythm(crafted_logits(t, quarters + extra, head), tt, "hard",
                                   style, float(tt[-1]) + 2)
        i = int(np.nonzero(plan.tick == head)[0][0])
        assert plan.flavor[i] > 0
        return int(plan.tick[i] + plan.dur[i]), plan

    # notes a beat apart (0.4 s < 0.45 s) inside the freeze: the second one ends it
    end, plan = hold_end([])
    inside = plan.tick[(plan.tick > head) & (plan.tick <= end)]
    assert len(inside) == 1 and end <= 72 - infer.HOLD_RELEASE_TICKS
    for a, b in zip(inside, inside[1:]):
        assert tt[b] - tt[a] >= infer.PINNED_MIN_GAP_S
    # with the notes two beats apart (0.8 s) the freeze may run over more of them
    sparse = [48 + 24 * i for i in range(24)]
    plan = infer.decode_rhythm(crafted_logits(t, sparse, head), tt, "hard", style,
                               float(tt[-1]) + 2)
    i = int(np.nonzero(plan.tick == head)[0][0])
    inside = plan.tick[(plan.tick > head) & (plan.tick <= plan.tick[i] + plan.dur[i])]
    assert len(inside) >= 2


def test_no_mine_within_a_beat_of_a_16th_run():
    t = 48 * 60
    tt = tick_times(t)
    run = [480 + 3 * i for i in range(8)]           # two beats of 16ths
    quarters = [48 + 24 * i for i in range(100) if not 440 <= 48 + 24 * i <= 540]
    style = cond.resolve_style("challenge", overrides={"mine_share": 0.05, "nps": 3.0})
    plan = infer.decode_rhythm(crafted_logits(t, quarters + run, mines=True), tt,
                               "challenge", style, float(tt[-1]) + 2)
    assert len(plan.mine_tick) > 0
    fast = plan.tick[np.isin(plan.tick, run)]
    assert len(fast) >= 4
    for m in plan.mine_tick:
        assert np.abs(fast - m).min() > infer.MINE_CLEARANCE_TICKS


# --------------------------------------------------------------------------- #
# 6. tempo that is not a round number
# --------------------------------------------------------------------------- #
def click_track(bpm: float, dur: float, offset: float, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(int(dur * GF.SR), np.float32)
    step, i = 15.0 / bpm, 0
    while offset + i * step < dur - 0.1:
        k = int((offset + i * step) * GF.SR)
        amp = (1.0, 0.35, 0.6, 0.35)[i % 4]
        y[k:k + 60] += amp * np.hanning(120)[60:].astype(np.float32) * rng.standard_normal(60)
        i += 1
    return y / np.abs(y).max()


def test_coherence_refines_a_non_round_tempo_and_leaves_a_right_one_alone():
    true_bpm, offset, dur = 127.37, 0.40, 90.0
    env = GF.frame_features(click_track(true_bpm, dur, offset)).onset_envelope
    # the regression-based refinement declines: nothing round to snap to
    assert not pipeline.refine_tempo(env, 127.45, offset, dur)["snapped"]
    r = pipeline.refine_tempo_coherence(env, 127.45, offset, dur)
    assert r["refined"] and abs(r["bpm"] - true_bpm) < 0.02
    assert r["gain"] >= pipeline.COHERENCE_MIN_GAIN
    assert r["jitter_ms"][1] < r["jitter_ms"][0]
    beat = 60.0 / r["bpm"] / 4                      # the clicks are 16ths
    assert abs((r["offset"] - offset + beat / 2) % beat - beat / 2) < 0.006
    # given the right tempo there is nothing to gain, and nothing changes
    same = pipeline.refine_tempo_coherence(env, true_bpm, offset, dur)
    assert not same["refined"] and same["bpm"] == true_bpm and same["offset"] == offset
    # too little to go on
    none = pipeline.refine_tempo_coherence(np.zeros(4000), 120.0, 0.1, 40.0)
    assert not none["refined"]


# --------------------------------------------------------------------------- #
# beat phase and bar line (the cause of the Beginner / Easy placement gap)
# --------------------------------------------------------------------------- #
class LoudnessModel:
    """Stands in for RhythmNet: an onset is likely where the audio is loud."""

    def rhythm(self, x, cv):
        mel = x[:, GF.MEL].sum(1).astype(np.float64)
        top = np.percentile(mel, 99.5)
        return {"onset": ((mel - 0.8 * top) / (0.05 * top)).astype(np.float32)}


class LoudnessEngine:
    model = LoudnessModel()

    def cond_for(self, style, has_stems=False):
        return np.zeros(19, np.float32)


def accented(bpm: float, dur: float, first_beat: float, strong_every: int = 4,
             seed: int = 2) -> np.ndarray:
    """A click on every beat, a loud one every `strong_every` beats."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(dur * GF.SR), np.float32)
    i = 0
    while first_beat + i * 60.0 / bpm < dur - 0.1:
        k = int((first_beat + i * 60.0 / bpm) * GF.SR)
        amp = 1.0 if i % strong_every == 0 else 0.3
        y[k:k + 80] += amp * np.hanning(160)[80:].astype(np.float32) * rng.standard_normal(80)
        i += 1
    return y / np.abs(y).max()


def grid_for(audio, bpm, offset):
    g = GF.extract_grid_features(audio, GF.TimingMap.constant(bpm, offset))
    return g.core, g.dt_ms


def test_a_grid_on_the_off_beats_is_moved_half_a_beat():
    bpm, dur = 120.0, 60.0
    audio = accented(bpm, dur, first_beat=0.50, strong_every=1)
    right = pipeline.verify_phase(LoudnessEngine(), *grid_for(audio, bpm, 0.50))
    assert not right["half_beat"] and right["gain"] < -pipeline.PHASE_MARGIN
    wrong = pipeline.verify_phase(LoudnessEngine(), *grid_for(audio, bpm, 0.25))
    assert wrong["half_beat"] and wrong["gain"] > pipeline.PHASE_MARGIN


def test_a_model_can_carry_its_own_phase_margin():
    class Model:
        meta = {"decoder": {"phase_margin": 0.3}}

    class Engine:
        model = Model()

    assert pipeline.phase_margin(Engine()) == 0.3
    assert pipeline.phase_margin(LoudnessEngine()) == pipeline.PHASE_MARGIN
    Model.meta = {"decoder": {"phase_margin": "high"}}          # unusable: fall back
    assert pipeline.phase_margin(Engine()) == pipeline.PHASE_MARGIN


@pytest.mark.parametrize("late_by", [0, 1, 2, 3])
def test_the_bar_line_is_put_on_the_downbeat(late_by):
    bpm, dur = 120.0, 80.0
    audio = accented(bpm, dur, first_beat=0.50)     # the loud beat is every fourth
    # a grid whose beat 0 is `late_by` beats before a loud one
    start = 0.50 + (4 - late_by) % 4 * 0.5
    out = pipeline.verify_phase(LoudnessEngine(), *grid_for(audio, bpm, start))
    assert not out["half_beat"]
    assert out["bar_shift_beats"] == late_by
    assert int(np.argmax(out["bar_share"])) == late_by


def test_too_short_a_song_is_left_alone():
    core = np.zeros((4 * C.G, 31), np.float32)
    out = pipeline.verify_phase(LoudnessEngine(), core, np.full(len(core), 40.0))
    assert out == {"half_beat": False, "gain": 0.0, "bar_shift_beats": 0,
                   "bar_share": None}


def test_rebar_follows_a_moved_bar_line():
    an = {"bpm": 120.0, "offset": 0.0, "duration_s": 60.0, "density_plan": {"x": 1},
          "energy_curve": [float(i) for i in range(30)],
          "sections": [{"start_bar": 0, "end_bar": 8}, {"start_bar": 8, "end_bar": 30}]}
    out = pipeline._rebar(an, 120.0, 2.0)           # the grid starts one bar later
    assert out["energy_curve"][:3] == [1.0, 2.0, 3.0]
    assert [s["start_bar"] for s in out["sections"]] == [0, 7]
    assert out["sections"][1]["end_bar"] == 29 and out["density_plan"] == {}
