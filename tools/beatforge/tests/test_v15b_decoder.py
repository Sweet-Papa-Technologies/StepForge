"""v1.5b decoder pieces: a chart that is asked to be a stream chart is made in
Stream mode (and hits the share of notes it was asked for), ordinary charts
are untouched by that, and the optional per-model decoder settings are off
unless a model declares them."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from beatforge.adapters.stepmania.neural import codec as C  # noqa: E402
from beatforge.adapters.stepmania.neural import conditioning as cond  # noqa: E402
from beatforge.adapters.stepmania.neural import grid_features as GF  # noqa: E402
from beatforge.adapters.stepmania.neural import infer, pipeline  # noqa: E402
from beatforge.adapters.stepmania.neural.infer import DecodeOptions  # noqa: E402

import synth_helpers as H  # noqa: E402

TINY = Path(__file__).parent / "fixtures" / "neural" / "tiny_model"
OFFSET = 0.2
STREAM = cond.INDEX["stream_share"]


# --------------------------------------------------------------------------- #
# which charts are routed
# --------------------------------------------------------------------------- #
def _route(difficulty, preset=None, overrides=None, options=None):
    style = cond.resolve_style(difficulty, preset, overrides)
    return infer.route_stream(difficulty, style, preset, overrides, options)


def test_ordinary_charts_are_not_routed_and_keep_their_options_object():
    for diff in ("beginner", "easy", "medium", "hard", "challenge"):
        assert _route(diff) == (None, None)
        given = DecodeOptions(max_subdivision=8)
        got, why = _route(diff, options=given)
        assert got is given and why is None
    # the other presets carry a stream value too; they stay note by note
    for preset in ("casual_flow", "jumpstream", "tech_footswitch"):
        for diff in ("hard", "challenge"):
            assert _route(diff, preset) == (None, None), (preset, diff)
    assert cond.preset_vector("challenge", "jumpstream")[STREAM] > infer.STREAM_ROUTE_MIN_SHARE


def test_the_stamina_preset_is_routed_at_hard_and_challenge_only():
    for diff, share in (("hard", 0.45), ("challenge", 0.80)):
        opts, why = _route(diff, "itg_stamina")
        assert why == "preset itg_stamina"
        assert opts.stream and opts.stream_fill == "neural"
        assert opts.stream_share == pytest.approx(share)
    for diff in ("beginner", "easy", "medium"):
        assert _route(diff, "itg_stamina") == (None, None)
    # the caller's other options survive the routing
    opts, _ = _route("challenge", "itg_stamina", options=DecodeOptions(crossovers=False,
                                                                        stream_fill="rule"))
    assert opts.stream and not opts.crossovers and opts.stream_fill == "rule"


def test_a_stream_slider_set_by_hand_routes_from_45_percent():
    opts, why = _route("challenge", overrides={"stream_share": 0.6})
    assert why == "stream_share slider" and opts.stream_share == pytest.approx(0.6)
    opts, why = _route("hard", overrides={"stream_share": 0.9})      # clamped to the tier's 0.45
    assert why == "stream_share slider" and opts.stream_share == pytest.approx(0.45)
    assert _route("challenge", overrides={"stream_share": 0.44}) == (None, None)
    assert _route("medium", overrides={"stream_share": 0.9}) == (None, None)
    # lowering the slider under a stamina preset takes the chart out again
    assert _route("challenge", "itg_stamina", {"stream_share": 0.2}) == (None, None)
    # a value that was not the caller's (style retrieval fills sliders in) does not count
    style = cond.resolve_style("challenge", overrides={"stream_share": 0.7})
    assert infer.route_stream("challenge", style, None, None, None) == (None, None)
    assert infer.route_stream("challenge", style, None, {"jump_share": 0.1}, None) == (None, None)


def test_stream_mode_the_user_switched_on_is_left_as_they_set_it():
    given = DecodeOptions(stream=True, stream_coverage=0.5)
    got, why = _route("challenge", "itg_stamina", options=given)
    assert got is given and why is None


def test_stream_share_option_is_validated_and_reported_only_in_stream_mode():
    assert DecodeOptions().stream_share is None and DecodeOptions().to_dict() == {}
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="stream_share"):
            DecodeOptions(stream=True, stream_share=bad)
    d = DecodeOptions(stream=True, stream_share=0.8).to_dict()
    assert d["stream_share"] == 0.8 and DecodeOptions.from_dict(d).stream_share == 0.8
    assert "stream_share" not in DecodeOptions(stream_share=0.8).to_dict()
    # asking the model for a stream chart names the share of notes, not of bars
    s = infer.stream_style(cond.resolve_style("challenge"), "challenge",
                           DecodeOptions(stream=True, stream_coverage=0.3, stream_share=0.8,
                                         max_subdivision=16), 0.5)
    assert s[STREAM] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# the share of notes asked for is the share of notes written
# --------------------------------------------------------------------------- #
def _fake_rhythm(x, cond19):
    """8ths everywhere, quieter quarters and 16ths: enough for streams at any
    subdivision and for breaks between them."""
    t = len(x)
    k = np.arange(t)
    onset = np.full(t, -6.0, np.float32)
    onset[k % 3 == 0] = 0.5
    onset[k % 6 == 0] = 1.5
    onset[k % 12 == 0] = 2.5
    return {"onset": onset, "jump": np.full(t, -5.0, np.float32),
            "hold_type": np.tile(np.array([9, 0, 0, 0, 0, 0], np.float32), (t, 1)),
            "hold_dur": np.zeros((t, 10), np.float32),
            "mine": np.full(t, -9.0, np.float32)}


def _song(bpm: float, bars: int = 48):
    y = H.clicks(bars=bars, bpm=bpm)
    g = GF.extract_grid_features(y, GF.TimingMap.constant(bpm, OFFSET))
    return g, len(y) / GF.SR


def _engine(monkeypatch, neural: bool = True):
    engine = infer.StepManiaNeuralEngine(TINY)
    monkeypatch.setattr(engine.model, "rhythm", _fake_rhythm)
    monkeypatch.setitem(engine.model.meta, "decoder", {"neural_stream": neural})
    return engine


def _measured(chart) -> float:
    ev = chart.events
    return cond.stream_share(ev.tick, ev.p1 >= 0)


@pytest.mark.parametrize("share", [0.45, 0.6, 0.8])
def test_coverage_is_solved_so_the_asked_share_of_notes_is_stream(monkeypatch, share):
    engine = _engine(monkeypatch)
    g, dur = _song(120.0)                          # 16ths are 8 notes/s: inside Challenge
    chart = engine.generate(g.X, g.tick_times_sec, "challenge", cond.resolve_style("challenge"),
                            dur, options=DecodeOptions(stream=True, stream_share=share))
    info = chart.stats["rhythm"]["stream"]
    assert info["subdivision"] == 16
    assert info["share_asked"] == pytest.approx(share)
    assert _measured(chart) == pytest.approx(info["share"], abs=1e-3)
    assert abs(_measured(chart) - share) <= 0.08
    # a bar of 16ths is many more notes than a bar of break: fewer bars than notes
    assert info["coverage"] < share
    assert info["stream_bars"] / info["playable_bars"] < share
    # and the streams are whole
    ticks = set(chart.events.tick.tolist())
    for a, b in info["segments"]:
        assert all(t in ticks for t in range(a * C.G, b * C.G, 3))
    st = infer.chart_stats(chart, g.tick_times_sec)
    assert st.get("spins", 0) == 0 and st.get("lock_violations", 0) == 0


def test_where_the_tier_only_allows_8ths_the_share_is_taken_in_bars(monkeypatch):
    engine = _engine(monkeypatch)
    g, dur = _song(180.0)                          # 16ths would be 12 notes/s
    chart = engine.generate(g.X, g.tick_times_sec, "challenge", cond.resolve_style("challenge"),
                            dur, options=DecodeOptions(stream=True, stream_share=0.8))
    info = chart.stats["rhythm"]["stream"]
    assert info["subdivision"] == 8 and not info["over_tier_rate"]
    assert info["coverage"] == pytest.approx(0.8)
    assert info["stream_bars"] / info["playable_bars"] == pytest.approx(0.8, abs=0.08)
    assert info["share"] == 0.0                    # the slider's statistic counts 16ths only


def test_without_stream_share_stream_mode_is_what_it_was(monkeypatch):
    engine = _engine(monkeypatch)
    g, dur = _song(120.0, bars=24)
    opts = DecodeOptions(stream=True, stream_coverage=0.5)
    chart = engine.generate(g.X, g.tick_times_sec, "challenge", cond.resolve_style("challenge"),
                            dur, options=opts)
    info = chart.stats["rhythm"]["stream"]
    assert "coverage" not in info and "share_asked" not in info
    assert info["stream_bars"] / info["playable_bars"] == pytest.approx(0.5, abs=0.1)


# --------------------------------------------------------------------------- #
# through the pipeline
# --------------------------------------------------------------------------- #
def _context(bpm: float, bars: int = 32) -> pipeline.SongContext:
    g, dur = _song(bpm, bars)
    analysis = {"duration_s": dur, "bpm": bpm, "offset": OFFSET, "energy_curve": None}
    return pipeline.SongContext(g=g, align={"bpm": bpm, "offset": OFFSET}, analysis=analysis,
                                synced=analysis, on_transient=None, dsp_bpm=bpm,
                                dsp_offset=OFFSET)


def _same(a, b) -> bool:
    ea, eb = a.events, b.events
    return all(np.array_equal(getattr(ea, f), getattr(eb, f))
               for f in ("tick", "p0", "p1", "k0", "k1", "d0", "d1", "mines"))


def test_default_charts_are_identical_with_and_without_the_routing(monkeypatch):
    """The routing may not move a single note of a chart nobody asked to be a
    stream chart: every tier by default, and the presets that are not routed."""
    engine = _engine(monkeypatch)
    ctx = _context(150.0)
    tiers = ("beginner", "easy", "medium", "hard", "challenge")
    cases = [(None, None), ("casual_flow", None), ("jumpstream", None),
             ("tech_footswitch", None), (None, {"stream_share": 0.3}),
             (None, {"jump_share": 0.1})]
    with_routing = [pipeline.chart_song(ctx, tiers, engine, track_id="t", style_preset=p,
                                        style_overrides=o)[0] for p, o in cases]
    monkeypatch.setattr(infer, "route_stream", lambda d, s, p, o, options: (options, None))
    without = [pipeline.chart_song(ctx, tiers, engine, track_id="t", style_preset=p,
                                   style_overrides=o)[0] for p, o in cases]
    for (preset, over), a, b in zip(cases, with_routing, without):
        for diff in tiers:
            assert _same(a[diff], b[diff]), (preset, over, diff)
            assert "stream" not in a[diff].stats["rhythm"], (preset, over, diff)
            assert a[diff].stats == b[diff].stats, (preset, over, diff)


def test_the_stamina_preset_comes_out_as_a_stream_chart(monkeypatch):
    engine = _engine(monkeypatch)
    ctx = _context(120.0, bars=48)
    tiers = ("medium", "hard", "challenge")
    charts, styles, _ = pipeline.chart_song(ctx, tiers, engine, track_id="t",
                                            style_preset="itg_stamina")
    assert "stream" not in charts["medium"].stats["rhythm"]
    for diff in ("hard", "challenge"):
        info = charts[diff].stats["rhythm"]["stream"]
        assert info["routed"] == "preset itg_stamina"
        assert charts[diff].stats["options"]["stream"] is True
        asked = float(styles[diff][STREAM])
        assert info["subdivision"] == 16
        assert abs(_measured(charts[diff]) - asked) <= 0.08, diff
        st = infer.chart_stats(charts[diff], ctx.g.tick_times_sec)
        assert st.get("spins", 0) == 0 and st.get("lock_violations", 0) == 0
        note = pipeline._design_note(diff, styles[diff], "itg_stamina", charts[diff].stats)
        assert "switched on by the preset itg_stamina" in note["design_notes"]
    # the unrouted chart of the same request would have had almost no stream
    monkeypatch.setattr(infer, "route_stream", lambda d, s, p, o, options: (options, None))
    plain, _, _ = pipeline.chart_song(ctx, ("challenge",), engine, track_id="t",
                                      style_preset="itg_stamina")
    assert _measured(plain["challenge"]) < 0.5 * _measured(charts["challenge"])


def test_a_hand_set_stream_slider_is_routed_and_says_so(monkeypatch):
    engine = _engine(monkeypatch)
    ctx = _context(120.0, bars=48)
    charts, _, _ = pipeline.chart_song(ctx, ("challenge",), engine, track_id="t",
                                       style_overrides={"stream_share": 0.6})
    info = charts["challenge"].stats["rhythm"]["stream"]
    assert info["routed"] == "stream_share slider"
    assert abs(_measured(charts["challenge"]) - 0.6) <= 0.08


# --------------------------------------------------------------------------- #
# 16th-note figures at Hard (decoder.hard_sixteenth_boost)
# --------------------------------------------------------------------------- #
def _promote(tk, score, theta=0.5, boost=8.0, on_transient=None, max_nps=8.0,
             fine_frac=0.999, run_limit=64, bpm=120.0):
    t = len(score)
    times = np.arange(t) * (60.0 / bpm / C.TICKS_PER_BEAT)
    return infer._promote_sixteenths(np.asarray(tk, np.int64), np.asarray(score, float), theta,
                                     boost, np.ones(t, bool), on_transient, times, max_nps,
                                     fine_frac, run_limit, infer.min_gap_s("hard"))


def test_a_16th_is_promoted_only_between_two_notes_the_chart_has():
    score = np.zeros(96)
    eighths = [0, 6, 12, 18, 30, 36, 48]
    score[eighths] = 0.9
    score[[3, 15, 21, 33, 39, 45]] = [0.2, 0.05, 0.2, 0.2, 0.2, 0.2]
    out, n = _promote(eighths, score)
    # 3: both neighbours there, 0.2 boosted 8x is 0.67 >= 0.5 -> in
    # 15: neighbours there but 0.05 boosted is 0.30 -> out
    # 21: 24 carries no note -> out;  33: in;  39 and 45: 42 carries no note -> out
    assert n == 2 and sorted(set(out) - set(eighths)) == [3, 33]
    # every promoted note is inside a figure that begins and ends on an 8th
    assert all((k - 3) in out and (k + 3) in out for k in (3, 33))
    # nothing without a transient under it, when the caller knows where they are
    heard = np.ones(96, bool)
    heard[33] = False
    out, n = _promote(eighths, score, on_transient=heard)
    assert n == 1 and 33 not in out
    # and a boost of 1 or less is no boost
    assert _promote(eighths, score, boost=1.0)[1] == 0


def test_figures_are_three_five_or_nine_notes_never_seven_or_longer():
    def figure_lengths(ticks):
        ticks, out, run = sorted(ticks), [], 1
        for a, b in zip(ticks, ticks[1:]):
            if b - a == 3:
                run += 1
            else:
                out.append(run)
                run = 1
        return [r for r in out + [run] if r > 1]

    eighths = list(range(0, 96, 6))
    score = np.zeros(96)
    score[eighths] = 0.9
    score[[3, 9, 15]] = [0.4, 0.3, 0.35]                   # three in a row would be seven
    out, n = _promote(eighths, score)
    assert n == 2 and 9 not in out and figure_lengths(out) == [3, 3]
    score[:] = 0.0
    score[eighths] = 0.9
    score[list(range(3, 96, 6))] = np.linspace(0.45, 0.2, 16)   # every 16th wants in
    out, _ = _promote(eighths, score, max_nps=64.0)
    assert set(figure_lengths(out)) <= {3, 5, 9}
    assert max(figure_lengths(out)) <= infer.SIXTEENTH_BURST_MAX_RUN


def test_promotion_respects_the_tiers_rate_run_length_and_off_beat_share():
    eighths = list(range(0, 192, 6))
    score = np.zeros(192)
    score[eighths] = 0.9
    score[list(range(3, 192, 6))] = 0.3
    # run length: 8ths-or-faster runs may not outgrow the limit
    assert _promote(eighths, score, run_limit=len(eighths), max_nps=64.0)[1] == 0
    # peak rate: 8ths at 120 BPM are 4 notes/s; a 4.5/s ceiling leaves room for 2 per 4 s
    out, n = _promote(eighths, score, max_nps=4.5)
    times = np.sort(out) * (60.0 / 120.0 / C.TICKS_PER_BEAT)
    worst = max(int(((times >= a) & (times < a + 4.0)).sum()) for a in times)
    assert 0 < n and worst <= 18
    # off-beat share: half the notes are off the quarter grid already
    assert _promote(eighths, score, fine_frac=0.5, max_nps=64.0)[1] == 0


def _hard_logits(t: int) -> dict:
    """8ths with weaker 16ths between them on every other bar. (A little
    noise: perfectly tied scores are not something a model produces, and the
    run limiter breaks ties from the left.)"""
    k = np.arange(t)
    onset = np.full(t, -6.0, np.float32)
    onset[k % 6 == 0] = 0.8
    onset[k % 12 == 0] = 3.0
    onset[(k % 6 == 3) & ((k // C.G) % 2 == 0)] = -0.8
    onset += np.random.default_rng(0).normal(0, 0.5, t).astype(np.float32)
    return {"onset": onset, "jump": np.full(t, -5.0, np.float32),
            "hold_type": np.tile(np.array([9, 0, 0, 0, 0, 0], np.float32), (t, 1)),
            "hold_dur": np.zeros((t, 10), np.float32),
            "mine": np.full(t, -9.0, np.float32)}


def test_the_boost_is_off_unless_given_and_only_acts_at_hard():
    t = 24 * C.G
    times = np.arange(t) * (60.0 / 110.0 / C.TICKS_PER_BEAT)
    dur = float(times[-1]) + 2.0
    lg = _hard_logits(t)

    def plan(diff, **kw):
        return infer.decode_rhythm(lg, times, diff, cond.resolve_style(diff), dur, **kw)

    base = plan("hard")
    assert "sixteenth_boost" not in base.stats
    for same in (plan("hard", sixteenth_boost=None), plan("hard", sixteenth_boost=1.0)):
        assert np.array_equal(same.tick, base.tick)
    for diff in ("beginner", "easy", "medium", "challenge"):
        assert np.array_equal(plan(diff, sixteenth_boost=16.0).tick, plan(diff).tick), diff
    boosted = plan("hard", sixteenth_boost=16.0)
    six = lambda p: int((p.tick % 6 == 3).sum())            # noqa: E731
    assert six(boosted) > six(base)
    assert boosted.stats["sixteenth_boost"] == {"boost": 16.0,
                                                "promoted": boosted.stats["sixteenth_boost"]["promoted"]}
    assert boosted.stats["sixteenth_boost"]["promoted"] > 0
    # the figures replace the chart's weakest notes: the count is still calibrated
    assert abs(len(boosted.tick) - len(base.tick)) <= max(2, 0.03 * len(base.tick))
    # a cap on the subdivision still wins
    capped = plan("hard", sixteenth_boost=16.0, options=DecodeOptions(max_subdivision=8))
    assert six(capped) == 0


def test_the_model_declares_its_boost_for_hard_only(monkeypatch):
    engine = infer.StepManiaNeuralEngine(TINY)
    monkeypatch.setitem(engine.model.meta, "decoder", {})
    assert engine.model.sixteenth_boost("hard") is None
    monkeypatch.setitem(engine.model.meta, "decoder", {"hard_sixteenth_boost": 4})
    assert engine.model.sixteenth_boost("hard") == 4.0
    for diff in ("beginner", "easy", "medium", "challenge"):
        assert engine.model.sixteenth_boost(diff) is None
    for off in (1.0, 0.5, "4", None):
        monkeypatch.setitem(engine.model.meta, "decoder", {"hard_sixteenth_boost": off})
        assert engine.model.sixteenth_boost("hard") is None


# --------------------------------------------------------------------------- #
# the cheapest reading of a chart has no forbidden step either
# --------------------------------------------------------------------------- #
_L, _D, _U, _R = 0, 1, 2, 3
#: Two shapes from benchmark charts whose own footing was fine: an arrow
#: repeated a third of a second later, which the least-effort reading plays
#: with the feet crossed (A) or after a crossed jump (B).
FAST_REPEAT_CROSSED = [(_L,), (_R,), (_L, _R), (_D,), (_L,), (_L,), (_U,), (_R,), (_D,)]
REPEAT_AFTER_JUMP = [(_D,), (_L,), (_U, _D), (_R,), (_L,), (_L, _D), (_D,), (_R,), (_U,)]
PLAIN = [(_L,), (_R,), (_L,), (_R,), (_D,), (_U,), (_L,), (_R,), (_D,)]
REPEAT_BPM = 198.0


def _chart_of(rows, bpm=REPEAT_BPM):
    triples = [(i * C.TICKS_PER_BEAT, p, "1") for i, r in enumerate(rows) for p in r]
    ev, _ = C.events_from_notes(triples)
    enc, _ = C.encode(ev, ev.tick / C.TICKS_PER_BEAT * 60000.0 / bpm)
    return infer.NeuralChart(C.decode(enc), enc, {})


def _brute_force_total(placements, difficulty, bpm):
    """Cheapest total footflow cost over every way of assigning feet."""
    import itertools
    from beatforge.adapters.stepmania import footflow as ff
    from beatforge.adapters.stepmania.grammar import BUDGETS
    budget, best = BUDGETS[difficulty], None
    for choice in itertools.product((0, 1), repeat=len(placements)):
        state, total, prev = ff.REST_STATE, 0.0, None
        for p, c in zip(placements, choice):
            dt = None if prev is None else (p.beat - prev) * 60.0 / bpm
            prev = p.beat
            if len(p.panels) == 2:
                a, b = p.panels if c == 0 else p.panels[::-1]
                dc, state = ff.jump_cost(state, a, b, budget)
            else:
                dc, state = ff.step(state, c, p.panels[0], budget, "static", 0.0, dt_s=dt)
            total += dc
        best = total if best is None else min(best, total)
    return best


def test_cheapest_reading_is_the_least_total_cost_assignment():
    from beatforge.adapters.stepmania import footflow as ff
    from beatforge.adapters.stepmania.grammar import BUDGETS
    for rows in (FAST_REPEAT_CROSSED, REPEAT_AFTER_JUMP, PLAIN):
        placements = infer.to_placements(_chart_of(rows))
        got = pipeline.cheapest_reading(placements, "medium", REPEAT_BPM)
        assert [i for i, _ in got] == list(range(len(rows)))
        # replay the reading it implies: its total is the brute-force minimum.
        # (costs are reported floored at 0; the minimum is over the raw costs,
        # so compare through the DP's own total)
        budget = BUDGETS["medium"]
        dp = {ff.REST_STATE: 0.0}
        prev = None
        for p in placements:
            dt = None if prev is None else (p.beat - prev) * 60.0 / REPEAT_BPM
            prev, ndp = p.beat, {}
            for st, c0 in dp.items():
                moves = ([ff.jump_cost(st, a, b, budget) for a, b in (p.panels, p.panels[::-1])]
                         if len(p.panels) == 2 else
                         [ff.step(st, f, p.panels[0], budget, "static", 0.0, dt_s=dt)
                          for f in (0, 1)])
                for dc, ns in moves:
                    ndp[ns] = min(ndp.get(ns, 1e18), c0 + dc)
            dp = ndp
        assert min(dp.values()) == pytest.approx(_brute_force_total(placements, "medium",
                                                                    REPEAT_BPM))
    assert pipeline.cheapest_reading(infer.to_placements(_chart_of(PLAIN[:1])), "medium") == []


def test_the_referee_objects_to_a_step_that_is_forbidden_in_the_cheapest_reading(monkeypatch):
    from beatforge.adapters.stepmania import footflow as ff
    ceiling = ff.DOUBLE_STEP + ff.JACK + 1e-6
    veto = pipeline.flow_veto("medium", REPEAT_BPM)
    for rows in (FAST_REPEAT_CROSSED, REPEAT_AFTER_JUMP):
        chart = _chart_of(rows)
        costs = dict(pipeline.cheapest_reading(infer.to_placements(chart), "medium", REPEAT_BPM))
        assert [i for i, c in costs.items() if c >= ceiling] == [5]
        assert veto(chart) == [5]
    assert veto(_chart_of(PLAIN)) == []
    # the charts' own footing is not what is objected to: without the second
    # reading the referee has nothing to say
    monkeypatch.setattr(pipeline, "cheapest_reading", lambda *a, **k: [])
    for rows in (FAST_REPEAT_CROSSED, REPEAT_AFTER_JUMP):
        assert pipeline.flow_veto("medium", REPEAT_BPM)(_chart_of(rows)) == []


def test_slow_repeats_are_not_objected_to():
    # the same shapes at a third of the tempo: a repeat every 0.9 s is no jack
    slow = REPEAT_BPM / 3
    for rows in (FAST_REPEAT_CROSSED, REPEAT_AFTER_JUMP):
        assert pipeline.flow_veto("medium", slow)(_chart_of(rows, slow)) == []


@pytest.mark.parametrize("diff", ["medium", "hard"])
def test_generated_charts_pass_the_cheapest_reading(monkeypatch, diff):
    """End to end on the fixture model: with the referee in the loop no chart
    is left with a forbidden-tier step in its cheapest reading."""
    from beatforge.adapters.stepmania import footflow as ff
    ceiling = ff.DOUBLE_STEP + ff.JACK + 1e-6
    engine = _engine(monkeypatch)
    for bpm in (150.0, 198.0):
        g, dur = _song(bpm, bars=32)
        for seed in range(3):
            chart = engine.generate(g.X, g.tick_times_sec, diff, cond.resolve_style(diff), dur,
                                    seed=seed, veto=pipeline.flow_veto(diff, bpm))
            worst = max((c for _, c in pipeline.cheapest_reading(
                infer.to_placements(chart), diff, bpm)), default=0.0)
            assert worst < ceiling, (diff, bpm, seed)


# --------------------------------------------------------------------------- #
# fitting the 16th boost: a target, and two things it may not cost
# --------------------------------------------------------------------------- #
def test_placement_f1_ignores_a_constant_offset_and_counts_each_note_once():
    from beatforge.adapters.stepmania.neural import calibrate_decoder as CAL
    truth = np.arange(0.0, 60.0, 0.5)
    assert CAL.placement_f1(truth + 0.013, truth) == pytest.approx(1.0)
    # every other note a third of a second away: half of them are hits
    moved = truth.copy()
    moved[::2] += 0.2
    assert CAL.placement_f1(moved, truth) == pytest.approx(0.5)
    # two predictions on one human note are one hit, not two
    doubled = np.sort(np.concatenate([truth, truth + 0.005]))
    assert CAL.placement_f1(doubled, truth) == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert CAL.placement_f1(truth[:3], truth) is None


def test_the_boost_is_the_smallest_that_reaches_the_target_without_costing_placement():
    from beatforge.adapters.stepmania.neural import calibrate_decoder as CAL
    off = {"median": 0.0, "human_median": 18.0, "f1": 0.720, "gates": 0.98}

    def row(median, f1, gates=0.98):
        return {"median": median, "f1": f1, "gates": gates}

    choose = CAL.choose_sixteenth_boost
    # reaches 55 % of 18 (= 9.9) at boost 4, inside 0.005 of placement
    tried = [(2.0, row(3, 0.719)), (4.0, row(10, 0.716)), (8.0, row(15, 0.712))]
    assert choose(off, tried, 0.55, 0.005) == 4.0
    # not there yet: nothing chosen so far
    assert choose(off, tried[:1], 0.55, 0.005) is None
    # the target is only reached after placement has paid more than allowed: none
    tried = [(2.0, row(3, 0.7175)), (4.0, row(8, 0.7139)), (8.0, row(15, 0.708))]
    assert choose(off, tried, 0.55, 0.005) is None
    # nor at the price of charts failing the QA gates
    tried = [(2.0, row(3, 0.720)), (4.0, row(10, 0.720, gates=0.95))]
    assert choose(off, tried, 0.55, 0.005) is None
    assert CAL.SIXTEENTH_BOOSTS == tuple(sorted(CAL.SIXTEENTH_BOOSTS))


# --------------------------------------------------------------------------- #
# A model's own energy-tilt ladder (decoder.energy_tilt)
# --------------------------------------------------------------------------- #
def test_a_model_without_an_energy_tilt_uses_the_engines():
    from beatforge.adapters.stepmania.neural import calibrate_decoder as CD
    model = infer.FlowModel.__new__(infer.FlowModel)
    model.meta = {"decoder": {"density_scale": {"beginner": 0.7}}}
    assert model.energy_tilt("beginner") is None
    model.meta = {"decoder": {"energy_tilt": {"beginner": [1.5, 3.0, 6.0, 12.0]}}}
    assert model.energy_tilt("beginner") == (1.5, 3.0, 6.0, 12.0)
    assert model.energy_tilt("hard") is None
    model.meta = {"decoder": {"energy_tilt": {"beginner": "steep"}}}
    assert model.energy_tilt("beginner") is None           # nonsense is ignored
    # every ladder the calibrator may store starts with the engine's own steps
    own = infer.ENERGY_TILT["beginner"][0]
    assert all(lad[:len(own)] == own for lad in CD.LOW_TIER_TILTS)


def test_the_calibrator_only_lengthens_the_tilt_when_charts_fail_the_gates():
    from beatforge.adapters.stepmania.neural import calibrate_decoder as CD
    a, b = CD.LOW_TIER_TILTS
    off = {"charts": 120, "gates": 0.86, "f1": 0.670}
    assert CD.choose_energy_tilt({**off, "gates": 0.985}, []) is None
    # the mildest ladder that reaches the target
    tried = [(a, {"gates": 0.97, "f1": 0.671}), (b, {"gates": 0.983, "f1": 0.671})]
    assert CD.choose_energy_tilt(off, tried) == b
    assert CD.choose_energy_tilt(off, [(a, {"gates": 0.99, "f1": 0.670})] + tried[1:]) == a
    # none reaches it: the one that passes most, if placement holds
    assert CD.choose_energy_tilt(off, [(a, {"gates": 0.93, "f1": 0.669}),
                                       (b, {"gates": 0.95, "f1": 0.668})]) == b
    assert CD.choose_energy_tilt(off, [(a, {"gates": 0.99, "f1": 0.660})]) is None
    assert CD.choose_energy_tilt(off, [(a, {"gates": 0.86, "f1": 0.670})]) is None


def test_qa_gates_do_not_count_mines_as_steps():
    """The tech preset writes mines; walked through the foot model as if they
    were arrows they failed the flow-ceiling gate on charts whose steps are fine."""
    from beatforge.adapters.stepmania import qa as sm_qa
    from beatforge.adapters.stepmania.quantize import Placement
    analysis = {"bpm": 120.0, "offset": 0.0,
                "onsets": [{"time": 0.5 * i, "nearest_beat": float(i), "strength": 1.0}
                           for i in range(64)],
                "energy_curve": [0.5] * 16, "duration_s": 32.0, "sections": []}
    steps = [Placement(float(i), (i % 4,), "tap", None, {"foot": i % 2}) for i in range(32)]
    base = sm_qa.chart_metrics(steps, analysis, "challenge")
    # a mine half a beat after a step, on the panel the same foot would have to jump to
    mines = [Placement(i + 0.5, ((i + 2) % 4,), "mine", None, {}) for i in range(0, 32, 3)]
    mixed = sm_qa.chart_metrics(sorted(steps + mines, key=lambda p: p.beat), analysis, "challenge")
    for key in ("notes", "flow_cost_max", "flow_cost_mean", "panel_balance", "onset_alignment",
                "gates"):
        assert mixed[key] == base[key], key


def test_the_referee_objects_to_a_double_step_in_the_annotators_reading():
    """Feet kept crossed for a stretch can be read more cheaply with one
    double-step; the benchmark counts that reading, so the referee sees it."""
    from beatforge.adapters.stepmania.neural import pipeline as P
    L, D, U, R = C.L, C.D, C.U, C.R
    # plain alternation: nothing to object to
    walk = C.ChartEvents.from_rows([(i * 6, [(p,)]) for i, p in enumerate([L, D, U, R] * 6)])
    assert P.read_double_steps(walk, walk.tick * 50.0) == []
    # L then D then U at 8th-note speed after a jump pinned both feet: the
    # cheapest reading sends one foot D -> U, a double-step
    rows = [(0, [(L,), (R,)]), (6, [(D,)]), (12, [(U,)]), (18, [(D,)]), (24, [(U,)])]
    ev = C.ChartEvents.from_rows(rows)
    bad = P.read_double_steps(ev, ev.tick * 50.0)
    enc, _ = C.encode(ev, ev.tick * 50.0)
    same_new = [(4 <= int(t) < 8) for t in enc.token]
    assert bad == [] or all(same_new[i] for i in bad)
    # the same steps a second apart are not double-steps: there is time to reset
    assert P.read_double_steps(ev, ev.tick * 400.0) == []
    # and what it reports is exactly what the benchmark's definition counts
    found = 0
    for seed in range(120):
        rng = np.random.default_rng(seed)
        panels = rng.integers(0, 4, 60)
        gaps = rng.choice([3, 6, 6, 12], 60)
        ev = C.ChartEvents.from_rows([(int(t), [(int(p),)])
                                      for t, p in zip(np.cumsum(gaps), panels)])
        times = ev.tick * 35.0
        enc, _ = C.encode(ev, times)
        before, _, p_same = C.foot_path(enc)
        stood = np.where(before & 1 == C.LEFT, (before >> 1) // 4, (before >> 1) % 4)
        tok = enc.token.astype(int)
        dt = np.concatenate([[1e9], np.diff(times)])
        want = [i for i in range(len(tok))
                if 4 <= tok[i] < 8 and p_same[i] != stood[i] and dt[i] < 480.0]
        assert P.read_double_steps(ev, times) == want
        found += bool(want)
    assert found > 0
