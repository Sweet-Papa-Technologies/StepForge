"""Decoder options: a cap on the subdivision, Stream mode, re-generating a bar
range with its own settings, and the per-model decoder settings.

Like the tier rules, every one of these is enforced by construction: the tests
run on made-up logits and on the tiny random model, so nothing here depends on
what a trained network would have said.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from beatforge.adapters.stepmania.grammar import ALL_DIFFICULTIES, BUDGETS  # noqa: E402
from beatforge.adapters.stepmania.neural import codec as C  # noqa: E402
from beatforge.adapters.stepmania.neural import conditioning as cond  # noqa: E402
from beatforge.adapters.stepmania.neural import grid_features as GF  # noqa: E402
from beatforge.adapters.stepmania.neural import infer, pipeline  # noqa: E402
from beatforge.adapters.stepmania.neural.infer import DecodeOptions  # noqa: E402

TINY = Path(__file__).parent / "fixtures" / "neural" / "tiny_model"
BPM, OFFSET, DUR = 150.0, 0.2, 40.0


def click_audio(bpm=BPM, dur=DUR, offset=OFFSET, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(int(dur * GF.SR), np.float32)
    t, i = offset, 0
    while t < dur - 0.1:
        k = int(t * GF.SR)
        amp = 1.0 if i % 2 == 0 else 0.5
        y[k:k + 60] += amp * np.hanning(120)[60:].astype(np.float32) * rng.standard_normal(60)
        t += 60.0 / bpm / 2
        i += 1
    return y / np.abs(y).max()


@pytest.fixture(scope="module")
def engine():
    return infer.StepManiaNeuralEngine(TINY)


@pytest.fixture(scope="module")
def grid():
    return GF.extract_grid_features(click_audio(), GF.TimingMap.constant(BPM, OFFSET))


def four_on_the_floor(t: int, seed: int = 0, quiet_bars=()) -> dict:
    """RhythmNet output for a steady kick on every beat with off-beat hats and
    some 16th-note noise the decoder would happily use — the hardstyle track of
    the stamina report. `quiet_bars` are bars where the music drops out."""
    rng = np.random.default_rng(seed)
    k = np.arange(t)
    onset = np.full(t, -6.0)
    onset[k % 3 == 0] = -0.5          # 16ths: tempting
    onset[k % 6 == 0] = 1.0           # off-beat hats
    onset[k % 12 == 0] = 3.0          # the kick
    onset += rng.normal(0, 0.6, t)
    for b in quiet_bars:
        onset[b * 48:(b + 1) * 48] -= 4.0
    return {"onset": onset.astype(np.float32),
            "jump": rng.normal(-1.0, 1.5, t).astype(np.float32),
            "hold_type": rng.normal(0, 1.0, (t, 6)).astype(np.float32),
            "hold_dur": rng.normal(0, 1.0, (t, 10)).astype(np.float32),
            "mine": rng.normal(-2.0, 1.0, t).astype(np.float32)}


def tick_times(t: int, bpm: float = BPM) -> np.ndarray:
    return OFFSET + np.arange(t) * 60.0 / bpm / 12


def longest_run(ticks, max_gap: int) -> int:
    run = longest = 1
    for g in np.diff(ticks):
        run = run + 1 if g <= max_gap else 1
        longest = max(longest, run)
    return longest


# --------------------------------------------------------------------------- #
# The options object
# --------------------------------------------------------------------------- #
def test_options_round_trip_and_reject_what_they_do_not_know():
    assert DecodeOptions().to_dict() == {}
    o = DecodeOptions.from_dict({"max_subdivision": 8, "stream": True, "min_break_bars": 4})
    assert o.max_subdivision == 8 and o.stream and o.min_break_bars == 4
    assert DecodeOptions.from_dict(o.to_dict()) == o
    # the Stream settings are recorded only when Stream mode is on
    assert "stream_coverage" in o.to_dict()
    assert DecodeOptions(stream_coverage=0.5).to_dict() == {}
    for bad in ({"max_subdivison": 8}, {"max_subdivision": 12}, {"density": 5.0},
                {"stream_coverage": 0.0}, {"min_break_bars": 0}):
        with pytest.raises(ValueError):
            DecodeOptions.from_dict(bad)


@pytest.mark.parametrize("difficulty", ALL_DIFFICULTIES)
@pytest.mark.parametrize("cap", [4, 8, 16])
def test_a_subdivision_cap_only_ever_coarsens_a_tier(difficulty, cap):
    tier = set(infer.subdivision_residues(difficulty))
    capped = set(infer.subdivision_residues(difficulty, cap))
    assert capped <= tier and capped <= set(infer.SUBDIVISION_RESIDUES[cap])
    assert 0 in capped                                   # quarter notes always remain


# --------------------------------------------------------------------------- #
# Max subdivision
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("difficulty", ["hard", "challenge"])
def test_max_subdivision_8ths_leaves_no_16th_and_no_triplet(difficulty):
    t = 48 * 60
    tt = tick_times(t)
    style = cond.resolve_style(difficulty)
    free = infer.decode_rhythm(four_on_the_floor(t), tt, difficulty, style, tt[-1] + 2)
    assert (free.tick % 6 != 0).any(), "the test needs a chart that WOULD use 16ths"
    plan = infer.decode_rhythm(four_on_the_floor(t), tt, difficulty, style, tt[-1] + 2,
                               options=DecodeOptions(max_subdivision=8))
    assert len(plan.tick) > 50
    assert set((plan.tick % 12).tolist()) <= {0, 6}
    assert set((plan.mine_tick % 12).tolist()) <= {0, 6}
    assert plan.stats["options"] == {"max_subdivision": 8}


def test_max_subdivision_4ths_and_16ths():
    t = 48 * 60
    tt = tick_times(t)
    style = cond.resolve_style("challenge")
    q = infer.decode_rhythm(four_on_the_floor(t), tt, "challenge", style, tt[-1] + 2,
                            options=DecodeOptions(max_subdivision=4))
    assert len(q.tick) > 20 and set((q.tick % 12).tolist()) == {0}
    s = infer.decode_rhythm(four_on_the_floor(t), tt, "challenge", style, tt[-1] + 2,
                            options=DecodeOptions(max_subdivision=16))
    assert set((s.tick % 12).tolist()) <= {0, 3, 6, 9}          # no triplets


# --------------------------------------------------------------------------- #
# Stream mode
# --------------------------------------------------------------------------- #
def check_stream_plan(plan, opts: DecodeOptions, step: int):
    """The acceptance test of the stamina report, as assertions: nothing finer
    than the stream's subdivision anywhere, every stream unbroken, every break
    at least `min_break_bars` long, nothing in a break as fast as the stream."""
    info = plan.stats["stream"]
    have = set(plan.tick.tolist())
    assert all(tk % step == 0 for tk in have), "a note finer than the stream's subdivision"
    in_stream = np.zeros(int(plan.tick.max()) + 49, bool)
    for a, b in info["segments"]:
        assert b - a >= infer.STREAM_MIN_BARS
        slots = range(a * 48, b * 48, step)
        missing = [s for s in slots if s not in have]
        assert not missing, f"stream {a}:{b} is broken at ticks {missing[:5]}"
        in_stream[a * 48:b * 48 + 1] = True              # ...and its closing downbeat
    # breaks between two streams are real breaks
    for (_, e0), (s1, _) in zip(info["segments"], info["segments"][1:]):
        assert s1 - e0 >= opts.min_break_bars, f"a break of {s1 - e0} bars"
    # nothing in a break is as fast as the stream
    brk = plan.tick[~in_stream[plan.tick]]
    if len(brk) > 1:
        gaps = np.diff(brk)
        same_break = gaps < 48 * opts.min_break_bars
        assert (gaps[same_break] > step).all()
    # streams carry plain steps: no jump and no freeze starts inside one
    inside = in_stream[plan.tick] & ~np.isin(plan.tick, [b * 48 for _, b in info["segments"]])
    assert not plan.is_jump[inside].any() and not (plan.flavor[inside] > 0).any()
    # ...and the breaks are not made to carry the whole chart's share of them
    n_break = int((~in_stream[plan.tick]).sum())
    assert plan.is_jump.sum() <= 0.12 * n_break + 1
    assert (plan.flavor > 0).sum() <= 0.10 * n_break + 1
    return info


def test_stream_of_8ths_on_a_four_on_the_floor_track():
    """max subdivision = 8ths, Stream mode: zero 16th notes, and streams are
    unbroken outside the configured breaks."""
    t = 48 * 64
    tt = tick_times(t)
    quiet = (16, 17, 18, 19, 40, 41, 42, 43)
    opts = DecodeOptions(max_subdivision=8, stream=True, stream_coverage=0.8,
                         min_break_bars=4)
    plan = infer.decode_rhythm(four_on_the_floor(t, quiet_bars=quiet), tt, "hard",
                               cond.resolve_style("hard"), tt[-1] + 2,
                               energy_per_bar=np.where(np.isin(np.arange(64), quiet), 0.1, 0.9),
                               options=opts)
    info = check_stream_plan(plan, opts, step=6)
    assert info["subdivision"] == 8
    assert not (plan.tick % 6 != 0).any()                    # zero 16ths, zero triplets
    # the breaks went where the music dropped out
    stream_bars = {b for a, e in info["segments"] for b in range(a, e)}
    assert not stream_bars & set(quiet)
    assert info["stream_bars"] >= 0.7 * info["playable_bars"]
    # far beyond what the tier's ordinary run limit would allow
    assert longest_run(plan.tick, 6) >= 64 > infer.RUN_LIMIT["hard"]


@pytest.mark.parametrize("difficulty,cap,step", [
    ("challenge", 16, 3), ("hard", 16, 3), ("hard", 8, 6), ("medium", 8, 6),
    ("medium", 16, 6),          # Medium writes nothing finer than 8ths: the cap is clipped
    ("easy", 4, 12), ("beginner", 16, 12)])
def test_stream_runs_at_the_chosen_subdivision_up_to_the_tiers_finest(difficulty, cap, step):
    t = 48 * 48
    tt = tick_times(t, bpm=120.0)
    opts = DecodeOptions(max_subdivision=cap, stream=True, stream_coverage=0.7,
                         min_break_bars=2)
    plan = infer.decode_rhythm(four_on_the_floor(t, seed=2), tt, difficulty,
                               cond.resolve_style(difficulty), tt[-1] + 2, options=opts)
    info = check_stream_plan(plan, opts, step)
    assert infer.SUBDIVISION_TICKS[info["subdivision"]] == step
    assert info["segments"], "no stream at all"


def test_full_coverage_is_one_unbroken_stream():
    t = 48 * 40
    tt = tick_times(t)
    opts = DecodeOptions(max_subdivision=16, stream=True, stream_coverage=1.0)
    plan = infer.decode_rhythm(four_on_the_floor(t), tt, "challenge",
                               cond.resolve_style("challenge"), tt[-1] + 2, options=opts)
    info = check_stream_plan(plan, opts, step=3)
    assert len(info["segments"]) == 1
    assert info["stream_bars"] == info["playable_bars"]
    assert longest_run(plan.tick, 3) == len(plan.tick)


def test_stream_subdivision_left_open_fits_the_tiers_rate():
    # 16ths at 120 BPM are 8 notes/s: inside Challenge's 10, and Hard's 8
    assert infer.stream_subdivision("challenge", None, 0.5) == 16
    assert infer.stream_subdivision("hard", None, 0.5) == 16
    # Stream mode has its own speed limit (16ths to 170 BPM at Hard, 200 at
    # Challenge); past it the stream runs in 8ths
    assert infer.stream_subdivision("challenge", None, 60 / 175) == 16
    assert infer.stream_subdivision("hard", None, 60 / 175) == 8
    assert infer.stream_subdivision("challenge", None, 60 / 210) == 8
    # ...unless the user asked for 16ths by name
    assert infer.stream_subdivision("challenge", 16, 60 / 210) == 16
    # and never finer than the tier writes at all
    assert infer.stream_subdivision("medium", 16, 0.5) == 8
    assert infer.stream_subdivision("beginner", None, 0.5) == 4
    t = 48 * 32
    plan = infer.decode_rhythm(four_on_the_floor(t), tick_times(t, 210.0), "challenge",
                               cond.resolve_style("challenge"), tick_times(t, 210.0)[-1] + 2,
                               options=DecodeOptions(max_subdivision=16, stream=True))
    assert plan.stats["stream"]["over_tier_rate"] is True     # said so in the report


@pytest.mark.parametrize("difficulty", ["medium", "hard", "challenge"])
def test_a_stream_chart_is_legal_to_stand_on(engine, grid, difficulty):
    opts = DecodeOptions(max_subdivision=8, stream=True, stream_coverage=0.8)
    chart = engine.generate(grid.X, grid.tick_times_sec, difficulty,
                            cond.resolve_style(difficulty), DUR, seed=4, options=opts,
                            veto=pipeline.flow_veto(difficulty, BPM))
    assert chart.stats["options"]["stream"] is True
    ticks = chart.events.tick.astype(np.int64)
    assert (ticks % 6 == 0).all()
    assert longest_run(ticks, 6) >= 32
    st = infer.chart_stats(chart, grid.tick_times_sec)
    assert st["spins"] == 0 and st["lock_violations"] == 0 and st["double_steps"] == 0
    # it survives the trip through the serializer's note model
    assert len(infer.to_placements(chart)) >= len(chart.events)


# --------------------------------------------------------------------------- #
# No crossovers
# --------------------------------------------------------------------------- #
def test_crossovers_can_be_switched_off(engine, grid):
    style = cond.resolve_style("challenge", overrides={"crossover_share": 0.15})
    crossed = 0
    for seed in range(4):
        on = engine.generate(grid.X, grid.tick_times_sec, "challenge", style, DUR, seed=seed)
        crossed += infer.chart_stats(on, grid.tick_times_sec)["crossed"]
        off = engine.generate(grid.X, grid.tick_times_sec, "challenge", style, DUR, seed=seed,
                              options=DecodeOptions(crossovers=False))
        assert infer.chart_stats(off, grid.tick_times_sec)["crossed"] == 0
    assert crossed > 0, "the test needs charts that WOULD cross"
    assert infer.TierRules.for_difficulty("challenge", crossovers=False).no_cross
    assert not infer.TierRules.for_difficulty("challenge").no_cross


# --------------------------------------------------------------------------- #
# Decoding a window / re-generating a bar range
# --------------------------------------------------------------------------- #
def test_a_window_decodes_only_its_own_ticks_and_respects_blocked_ones():
    t = 48 * 40
    tt = tick_times(t)
    blocked = np.zeros(t, bool)
    blocked[48 * 10:48 * 11] = True
    plan = infer.decode_rhythm(four_on_the_floor(t), tt, "hard", cond.resolve_style("hard"),
                               tt[-1] + 2, window=(48 * 8, 48 * 16), blocked=blocked)
    assert len(plan.tick) > 10
    assert plan.tick.min() >= 48 * 8 and plan.tick.max() < 48 * 16
    assert not blocked[plan.tick].any()
    # density is relative to what the model expects there
    few = infer.decode_rhythm(four_on_the_floor(t), tt, "hard", cond.resolve_style("hard"),
                              tt[-1] + 2, window=(48 * 8, 48 * 16),
                              options=DecodeOptions(density=0.5))
    many = infer.decode_rhythm(four_on_the_floor(t), tt, "hard", cond.resolve_style("hard"),
                               tt[-1] + 2, window=(48 * 8, 48 * 16),
                               options=DecodeOptions(density=1.5))
    assert len(few.tick) < len(plan.tick) < len(many.tick)


def outside_rows(events, lo, hi):
    return [r for r in events.rows() if not lo <= r[0] < hi]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "challenge"])
def test_regenerating_a_range_leaves_the_rest_alone_and_stays_legal(engine, grid, difficulty):
    style = cond.resolve_style(difficulty)
    veto = pipeline.flow_veto(difficulty, BPM)
    base = engine.generate(grid.X, grid.tick_times_sec, difficulty, style, DUR, seed=3,
                           veto=veto)
    start, end = 6, 12
    lo, hi = start * 48, end * 48
    cap = 4 if difficulty in ("hard", "challenge") else None
    seen = set()
    for seed in (1, 2, 3):
        opts = DecodeOptions(max_subdivision=cap, crossovers=False, density=0.8)
        out = engine.regenerate_section(base.events, grid.X, grid.tick_times_sec, difficulty,
                                        style, start, end, DUR, options=opts, seed=seed,
                                        veto=veto)
        # notes outside the range: unchanged, bit for bit
        assert outside_rows(out.events, lo, hi) == outside_rows(base.events, lo, hi)
        assert [tuple(m) for m in out.events.mines.tolist() if not lo <= m[0] < hi] == \
               [tuple(m) for m in base.events.mines.tolist() if not lo <= m[0] < hi]
        inside = out.events.tick[(out.events.tick >= lo) & (out.events.tick < hi)]
        assert len(inside) > 0
        if cap:
            assert (inside % 12 == 0).all(), "the range's own subdivision cap"
        # footing is legal across both boundaries, as everywhere else
        st = infer.chart_stats(out, grid.tick_times_sec)
        assert st["spins"] == 0 and st["lock_violations"] == 0
        assert st["double_steps"] <= infer.chart_stats(base, grid.tick_times_sec)["double_steps"]
        # the tier's shortest gap holds at the seams too
        gaps = np.diff(grid.tick_times_sec[out.events.tick])
        assert gaps.min() >= infer.min_gap_s(difficulty) - 1e-9
        # and the chart still encodes/decodes to itself
        assert C.decode(out.encoded).equals(out.events)
        assert out.stats["notes_in_section"] == len(inside)
        seen.add(tuple(map(str, out.events.rows())))
    assert len(seen) > 1 or difficulty == "easy"


def test_regenerating_a_range_changes_its_rhythm_not_just_its_arrows(engine, grid):
    style = cond.resolve_style("hard")
    base = engine.generate(grid.X, grid.tick_times_sec, "hard", style, DUR, seed=3)
    lo, hi = 6 * 48, 12 * 48
    before = base.events.tick[(base.events.tick >= lo) & (base.events.tick < hi)]
    sparse = engine.regenerate_section(base.events, grid.X, grid.tick_times_sec, "hard", style,
                                       6, 12, DUR, options=DecodeOptions(density=0.4), seed=1)
    stream = engine.regenerate_section(
        base.events, grid.X, grid.tick_times_sec, "hard", style, 6, 12, DUR, seed=1,
        options=DecodeOptions(max_subdivision=8, stream=True, stream_coverage=1.0))
    n = lambda c: int(((c.events.tick >= lo) & (c.events.tick < hi)).sum())
    assert n(sparse) < len(before)
    # unbroken 8ths over the range — all of it but a bar still under a freeze
    # of the chart before it, which stays clear
    info = stream.stats["rhythm"]["stream"]
    assert info["subdivision"] == 8 and info["stream_bars"] >= 5
    have = set(stream.events.tick.tolist())
    for a, b in info["segments"]:
        assert lo <= a * 48 and b * 48 <= hi
        assert all(tk in have for tk in range(a * 48, b * 48, 6))
    assert n(stream) >= 8 * info["stream_bars"] and n(stream) > len(before)
    assert outside_rows(stream.events, lo, hi) == outside_rows(base.events, lo, hi)
    st = infer.chart_stats(stream, grid.tick_times_sec)
    assert st["spins"] == 0 and st["lock_violations"] == 0


def test_a_range_under_an_outside_freeze_keeps_clear_of_it(engine, grid):
    """A freeze that starts before the range and is still held inside it pins a
    foot: no new note may land under it."""
    rows = [(48 * 4 + 12 * i, [(i % 2 * 3, C.TAP, 0)]) for i in range(16)]      # bars 4-7
    rows.append((48 * 8 - 12, [(C.D, C.HOLD, 60)]))                             # held into bar 9
    rows += [(48 * 12 + 12 * i, [(i % 2 * 3, C.TAP, 0)]) for i in range(8)]     # bars 12-13
    base = C.ChartEvents.from_rows(rows)
    out = engine.regenerate_section(base, grid.X, grid.tick_times_sec, "hard",
                                    cond.resolve_style("hard"), 8, 12, DUR, seed=2,
                                    options=DecodeOptions(density=1.5))
    held_until = 48 * 8 - 12 + 60
    new = out.events.tick[(out.events.tick >= 48 * 8) & (out.events.tick < 48 * 12)]
    assert len(new) > 0 and new.min() > held_until
    assert outside_rows(out.events, 48 * 8, 48 * 12) == outside_rows(base, 48 * 8, 48 * 12)
    assert infer.chart_stats(out, grid.tick_times_sec)["lock_violations"] == 0


def test_regenerating_an_empty_chart_range_and_a_range_past_the_notes(engine, grid):
    style = cond.resolve_style("medium")
    base = engine.generate(grid.X, grid.tick_times_sec, "medium", style, DUR, seed=3)
    # a range past the end of the song: nothing to write, nothing changes
    out = engine.regenerate_section(base.events, grid.X, grid.tick_times_sec, "medium", style,
                                    500, 510, DUR, seed=1)
    assert out.events.equals(base.events) and out.stats["notes_in_section"] == 0


# --------------------------------------------------------------------------- #
# Per-model decoder settings
# --------------------------------------------------------------------------- #
def test_a_model_can_carry_its_own_density_scale(tmp_path):
    import shutil
    d = tmp_path / "model"
    shutil.copytree(TINY, d)
    plain = infer.FlowModel(d)
    assert plain.density_scale("easy") == infer.DENSITY_SCALE["easy"]
    meta = json.loads((d / "model_meta.json").read_text())
    meta["decoder"] = {"density_scale": {"easy": 0.5, "hard": "lots"}}
    (d / "model_meta.json").write_text(json.dumps(meta))
    own = infer.FlowModel(d)
    assert own.density_scale("easy") == 0.5
    assert own.density_scale("hard") == infer.DENSITY_SCALE["hard"]     # unusable: fall back
    assert own.density_scale("medium") == infer.DENSITY_SCALE["medium"]
    t = 48 * 40
    tt = tick_times(t)
    style = cond.resolve_style("easy")
    a = infer.decode_rhythm(four_on_the_floor(t), tt, "easy", style, tt[-1] + 2)
    b = infer.decode_rhythm(four_on_the_floor(t), tt, "easy", style, tt[-1] + 2,
                            density_scale=0.5)
    assert len(b.tick) < len(a.tick)


def test_the_shipped_model_carries_its_decoder_settings():
    meta = json.loads((infer.default_model_dir() / "model_meta.json").read_text())
    dec = meta.get("decoder") or {}
    # v1.5's rhythm network decides the half beat by itself; v1.4 needed the vote.
    assert dec.get("phase_rule") == "margin" and dec.get("phase_margin", 0) > 0
    assert set(dec.get("density_scale") or {}) == set(BUDGETS)
    assert all(0.3 <= v <= 1.2 for v in dec["density_scale"].values())


# --------------------------------------------------------------------------- #
# The half-beat vote
# --------------------------------------------------------------------------- #
class VoteModel:
    """A rhythm model whose two heads can be made to disagree: the onset head
    follows loudness (as the real one does); the jump head either follows it
    too or prefers wherever the music is quiet."""
    meta = {"decoder": {"phase_rule": "vote"}}
    jump_agrees = True

    def rhythm(self, x, cv):
        mel = x[:, GF.MEL].sum(1).astype(np.float64)
        top = np.percentile(mel, 99.5)
        onset = (mel - 0.8 * top) / (0.05 * top)
        jump = onset if self.jump_agrees else -onset
        return {"onset": onset.astype(np.float32), "jump": jump.astype(np.float32)}


class VoteEngine:
    def __init__(self, jump_agrees=True):
        self.model = VoteModel()
        self.model.jump_agrees = jump_agrees

    def cond_for(self, style, has_stems=False):
        return np.zeros(19, np.float32)


def clicks(bpm: float, dur: float, first_beat: float, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(int(dur * GF.SR), np.float32)
    i = 0
    while first_beat + i * 60.0 / bpm < dur - 0.1:
        k = int((first_beat + i * 60.0 / bpm) * GF.SR)
        y[k:k + 80] += np.hanning(160)[80:].astype(np.float32) * rng.standard_normal(80)
        i += 1
    return y / np.abs(y).max()


def test_the_vote_moves_a_grid_on_the_off_beats_and_leaves_a_right_one():
    bpm, dur = 120.0, 60.0
    audio = clicks(bpm, dur, first_beat=0.50)
    g = GF.extract_grid_features(audio, GF.TimingMap.constant(bpm, 0.50))
    right = pipeline.verify_phase(VoteEngine(), g.core, g.dt_ms)
    assert right["rule"] == "vote" and not right["half_beat"]
    assert all(v < 0 for v in right["votes"].values())
    g = GF.extract_grid_features(audio, GF.TimingMap.constant(bpm, 0.25))
    wrong = pipeline.verify_phase(VoteEngine(), g.core, g.dt_ms)
    assert wrong["half_beat"]
    assert all(v > 0 for v in wrong["votes"].values())


def test_two_votes_carry_it_and_one_does_not():
    bpm, dur = 120.0, 60.0
    audio = clicks(bpm, dur, first_beat=0.50)
    g = GF.extract_grid_features(audio, GF.TimingMap.constant(bpm, 0.25))   # off the beat
    # the jump head disagrees (it likes the grid as it is): still two to one
    out = pipeline.verify_phase(VoteEngine(jump_agrees=False), g.core, g.dt_ms)
    assert out["votes"]["jump"] < 0 and out["half_beat"]
    # on a right grid, one dissenting vote moves nothing
    g = GF.extract_grid_features(audio, GF.TimingMap.constant(bpm, 0.50))
    out = pipeline.verify_phase(VoteEngine(jump_agrees=False), g.core, g.dt_ms)
    assert out["votes"]["jump"] > 0 and not out["half_beat"]


def test_models_without_the_setting_keep_the_margin_rule():
    class Model:
        meta = {}

    class Engine:
        model = Model()

    assert pipeline.phase_rule(Engine()) == "margin"
    Model.meta = {"decoder": {"phase_rule": "coin"}}
    assert pipeline.phase_rule(Engine()) == "margin"
    assert pipeline.phase_rule(VoteEngine()) == "vote"


# --------------------------------------------------------------------------- #
# The analysis a re-generated range is decoded against
# --------------------------------------------------------------------------- #
def test_analysis_is_put_back_on_the_charts_own_grid():
    cached = {"bpm": 120.0, "offset": 0.10, "duration_s": 64.0, "meter": 4,
              "energy_curve": [float(i) for i in range(32)],
              "onsets": [{"time": 0.10 + 0.5 * i} for i in range(100)],
              "sections": [{"start_bar": 4, "end_bar": 8}]}
    row = {"grid_shift_ms": 70.0, "sync_lead_ms": 8.0}
    # the simfile's beat 0 = DSP offset + clock shift - sync lead
    out = pipeline._analysis_on_grid(cached, row, 120.0, 0.10 + 0.070 - 0.008, 64.0)
    assert out["offset"] == pytest.approx(0.10) and out["bpm"] == 120.0
    assert out["energy_curve"][:8] == cached["energy_curve"][:8]
    assert out["sections"] == cached["sections"]
    # every tick with an onset under it is found again
    on = pipeline.transient_ticks(out, 48 * 8)
    assert on[::12].all() and not on[6::12].any()
    # no analysis on disk: a neutral one that blocks nothing
    bare = pipeline._analysis_on_grid(None, {}, 150.0, 0.2, 40.0)
    assert bare["onsets"] == [] and pipeline.transient_ticks(bare, 96).all()


# --------------------------------------------------------------------------- #
# The CLI end to end
# --------------------------------------------------------------------------- #
@pytest.fixture
def cli_track(tmp_path, monkeypatch):
    """A catalogue track backed by synthetic audio, with every output directory
    redirected into tmp_path."""
    import soundfile as sf
    from beatforge import config
    from beatforge.adapters.stepmania import qa as sm_qa
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    sf.write(tracks / "nrl_click.ogg", click_audio(), GF.SR, format="OGG", subtype="VORBIS")
    monkeypatch.setattr(config, "TRACKS_PUB", tracks)
    monkeypatch.setattr(config, "TRACKS_SRC", tracks)
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path / "analysis")
    monkeypatch.setattr(config, "STEPMANIA_DIR", tmp_path / "stepmania")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(config.TRACK_CATALOGUE, "nrl_click", "nrl_click")
    monkeypatch.setitem(config.TRACK_META, "nrl_click", ("Neural Click", "Tests"))
    monkeypatch.setattr(sm_qa, "render_previews", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "render_previews", lambda *a, **k: {})
    monkeypatch.delenv("BEATFORGE_ENGINE", raising=False)
    monkeypatch.delenv("BEATFORGE_NEURAL_MODEL_DIR", raising=False)
    return tmp_path / "stepmania" / "nrl_click"


def chart_beats(chart) -> list:
    """Beats of the note rows of a simfile chart (4 beats per measure)."""
    out = []
    for m, measure in enumerate(str(chart.notes).split(",")):
        rows = [r.strip() for r in measure.strip().splitlines() if r.strip()]
        for i, row in enumerate(rows):
            if any(c in "124" for c in row):
                out.append(round(4 * (m + i / len(rows)), 6))
    return out


def test_cli_stream_chart_and_its_report(cli_track, capsys):
    import simfile
    from beatforge.__main__ import main
    main(["stepforge", "--track", "nrl_click", "--engine", "neural",
          "--difficulties", "medium,hard", "--model-dir", str(TINY), "--options-json",
          '{"max_subdivision": 8, "stream": true, "stream_coverage": 0.8, "min_break_bars": 2}'])
    report = json.loads((cli_track / "report.json").read_text())
    assert report["engine"]["options"]["stream"] is True
    for diff in ("medium", "hard"):
        eng = report["charts"][diff]["engine"]
        assert eng["options"] == {"max_subdivision": 8, "stream": True,
                                  "stream_coverage": 0.8, "min_break_bars": 2}
        assert eng["rhythm"]["stream"]["subdivision"] == 8
    sf = simfile.loads((cli_track / "nrl_click.ssc").read_text())
    for chart in sf.charts:
        beats = np.array(chart_beats(chart))
        assert len(beats) > 60
        assert np.allclose(beats * 2, np.round(beats * 2)), "something finer than an 8th"
    with pytest.raises(SystemExit, match="unknown decoder option"):
        main(["stepforge", "--track", "nrl_click", "--engine", "neural", "--model-dir",
              str(TINY), "--options-json", '{"streem": true}'])


def test_cli_regenerates_a_range_with_its_own_settings(cli_track, capsys):
    import simfile
    from beatforge.__main__ import main
    main(["stepforge", "--track", "nrl_click", "--engine", "neural",
          "--difficulties", "easy,hard", "--model-dir", str(TINY)])
    ssc = cli_track / "nrl_click.ssc"
    before = simfile.loads(ssc.read_text())
    before_report = json.loads((cli_track / "report.json").read_text())
    capsys.readouterr()
    main(["stepforge-inpaint", "--track", "nrl_click", "--difficulty", "hard",
          "--bars", "4:10", "--seed", "7", "--model-dir", str(TINY), "--regenerate",
          "--options-json", '{"max_subdivision": 4, "crossovers": false, "density": 1.2}',
          "--style-json", '{"jump_share": 0.0}'])
    out = capsys.readouterr().out
    assert "bars 4:10 re-generated" in out
    after = simfile.loads(ssc.read_text())
    assert str(after) == ssc.read_text()                       # byte-stable
    easy_b, hard_b = before.charts
    easy_a, hard_a = after.charts
    assert str(easy_a.notes) == str(easy_b.notes), "the other chart must not change"
    mb = [m.strip() for m in str(hard_b.notes).split(",")]
    ma = [m.strip() for m in str(hard_a.notes).split(",")]
    assert len(mb) == len(ma)
    changed = [i for i, (x, y) in enumerate(zip(mb, ma)) if x != y]
    assert changed and all(4 <= i < 10 for i in changed), changed
    # inside the range: quarter notes only, and no jumps
    inside = [b for b in chart_beats(hard_a) if 16 <= b < 40]
    assert inside and all(abs(b - round(b)) < 1e-6 for b in inside)
    for m in ma[4:10]:
        assert all(sum(c in "124" for c in row.strip()) <= 1 for row in m.splitlines())
    # the .sm twin got the same edit
    sm_after = simfile.loads((cli_track / "nrl_click.sm").read_text())
    assert str(sm_after.charts[1].notes).strip() == str(hard_a.notes).strip()
    # the report logs the edit and keeps what the chart as a whole was made with
    report = json.loads((cli_track / "report.json").read_text())
    eng = report["charts"]["hard"]["engine"]
    edit = eng["inpainted"][0]
    assert edit["bars"] == [4, 10] and edit["seed"] == 7 and edit["regenerated"] is True
    assert edit["options"] == {"max_subdivision": 4, "crossovers": False, "density": 1.2}
    assert edit["style"] == {"jump_share": 0.0}
    assert eng["style"] == before_report["charts"]["hard"]["engine"]["style"]
    assert "options" not in eng                     # the chart itself had none
    assert report["charts"]["easy"] == before_report["charts"]["easy"]
    # options that need new notes are refused without --regenerate
    with pytest.raises(SystemExit, match="re-generated"):
        main(["stepforge-inpaint", "--track", "nrl_click", "--difficulty", "hard",
              "--bars", "4:10", "--model-dir", str(TINY),
              "--options-json", '{"max_subdivision": 4}'])
    # ...but "no crossovers" alone is a plain re-roll
    main(["stepforge-inpaint", "--track", "nrl_click", "--difficulty", "hard",
          "--bars", "4:10", "--seed", "3", "--model-dir", str(TINY),
          "--options-json", '{"crossovers": false}'])
    report = json.loads((cli_track / "report.json").read_text())
    assert len(report["charts"]["hard"]["engine"]["inpainted"]) == 2


# --------------------------------------------------------------------------- #
# Crossovers in a stream stay within what human stream charts have
# --------------------------------------------------------------------------- #
def test_a_stream_is_held_to_the_crossover_cap(engine, grid, monkeypatch):
    opts = DecodeOptions(max_subdivision=8, stream=True, stream_coverage=1.0)
    style = cond.resolve_style("challenge")
    free = []
    monkeypatch.setattr(infer, "STREAM_MAX_CROSS", 1.0)             # no cap at all
    for seed in range(6):
        c = engine.generate(grid.X, grid.tick_times_sec, "challenge", style, DUR, seed=seed,
                            options=opts)
        free.append(infer._cross_share(c.encoded.token))
    assert max(free) > 0.02, "the test needs streams that WOULD cross"
    monkeypatch.setattr(infer, "STREAM_MAX_CROSS", 0.02)
    capped = []
    for seed in range(6):
        c = engine.generate(grid.X, grid.tick_times_sec, "challenge", style, DUR, seed=seed,
                            options=opts)
        capped.append(infer._cross_share(c.encoded.token))
        st = infer.chart_stats(c, grid.tick_times_sec)
        assert st["spins"] == 0 and st["lock_violations"] == 0 and st["double_steps"] == 0
    assert all(b <= a + 1e-9 for a, b in zip(free, capped))
    assert max(capped) < max(free)
    # the crossover slider can ask for more than the cap
    asked = cond.resolve_style("challenge", overrides={"crossover_share": 0.15})
    c = engine.generate(grid.X, grid.tick_times_sec, "challenge", asked, DUR,
                        seed=int(np.argmax(free)), options=opts)
    share = infer._cross_share(c.encoded.token)
    assert 0.02 < share <= max(0.15, max(free))
