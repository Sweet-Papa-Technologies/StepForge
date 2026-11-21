"""v1.5c decoder pieces: footswitches that are asked for are placed so that
they are READ as footswitches (and nothing runs when they are not asked for),
and Stream mode has its own speed limit while an ordinary chart keeps the
tier's."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from beatforge.adapters.stepmania.grammar import BUDGETS  # noqa: E402
from beatforge.adapters.stepmania.neural import codec as C  # noqa: E402
from beatforge.adapters.stepmania.neural import conditioning as cond  # noqa: E402
from beatforge.adapters.stepmania.neural import grid_features as GF  # noqa: E402
from beatforge.adapters.stepmania.neural import infer, pipeline  # noqa: E402
from beatforge.adapters.stepmania.neural.infer import DecodeOptions  # noqa: E402

import synth_helpers as H  # noqa: E402

TINY = Path(__file__).parent / "fixtures" / "neural" / "tiny_model"
OFFSET = 0.2
FSW = cond.INDEX["footswitch_share"]


def _fake_rhythm(x, cond19):
    """8ths everywhere, quieter quarters and 16ths."""
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


def _fake_choreo(ctx, scalars, tokens, cond19):
    """A step network that alternates feet over three panels and never asks
    for a crossover, a jack or a footswitch of its own: what the placement
    adds is then all there is."""
    n = len(tokens)
    out = np.zeros((n, C.N_TOKENS), np.float32)
    out[:, :4] = (2.0, 1.5, 1.5, -6.0)                # ALT: own side, D, U, far side
    out[:, 4:8] = -8.0                                # SAME foot
    out[np.arange(n), np.arange(n) % 3] += 6.0        # ...and it is sure of itself
    return out


def _song(bpm: float, bars: int = 48):
    y = H.clicks(bars=bars, bpm=bpm)
    g = GF.extract_grid_features(y, GF.TimingMap.constant(bpm, OFFSET))
    return g, len(y) / GF.SR


def _engine(monkeypatch):
    engine = infer.StepManiaNeuralEngine(TINY)
    monkeypatch.setattr(engine.model, "rhythm", _fake_rhythm)
    monkeypatch.setattr(engine.model, "choreo", _fake_choreo)
    monkeypatch.setitem(engine.model.meta, "decoder", {"neural_stream": True})
    return engine


def _style(difficulty: str, fsw: float | None) -> np.ndarray:
    s = cond.resolve_style(difficulty)
    s[cond.INDEX["crossover_share"]] = 0.0
    s[cond.INDEX["jack_share"]] = 0.0
    if fsw is not None:
        s[FSW] = fsw
    return s


def _read_share(chart, g) -> float:
    ev = chart.events
    flags = infer.read_footswitches(ev, g.tick_times_sec[ev.tick] * 1000.0)
    return float(flags.sum()) / max(1, int((ev.p1 < 0).sum()))


# --------------------------------------------------------------------------- #
# footswitches
# --------------------------------------------------------------------------- #
def test_the_du_footswitch_table_is_the_style_vectors_footswitch():
    g = infer._gen_tables()
    assert not g["fsw_du"][:, 4:].any()                     # an ALT step only
    assert (g["fsw"] | ~g["fsw_du"]).all()                  # ...that is a footswitch
    for s in range(C.N_STATES):
        pl, pr, fl = C.state_tuple(s)
        for a in range(4):
            foot = 1 - fl
            panel = C.PANEL_OF[foot][a]
            other = pr if foot == C.LEFT else pl
            own = pl if foot == C.LEFT else pr
            want = panel == other and panel != own and panel in (C.D, C.U)
            assert bool(g["fsw_du"][s, a]) == want, (s, a)


def test_a_footswitch_bonus_puts_a_du_footswitch_at_that_note_only():
    n = 24
    rules = infer.TierRules.for_difficulty("challenge")
    unary = np.full((n, C.N_TOKENS), -3.0)
    unary[:, 8:] = infer.NEG
    args = dict(unary=unary, dtb=np.full(n, 2), is_jump=np.zeros(n, bool),
                slots=np.zeros(n, np.int64), hold_kind=np.zeros(n, np.int64), rules=rules)
    g = infer._gen_tables()
    plain = infer.constrained_viterbi(**args)
    assert not g["fsw"][plain.states, plain.token.astype(int)].any()
    bonus = np.zeros(n)
    bonus[10] = infer.FSW_BONUS
    res = infer.constrained_viterbi(**args, fsw_bonus=bonus)
    hit = g["fsw_du"][res.states, res.token.astype(int)]
    assert hit[10] and hit.sum() == 1
    # the penalty (and the placement's block) still wins where it is applied
    block = np.zeros(n)
    block[10] = infer.FSW_BLOCK
    res = infer.constrained_viterbi(**args, fsw_bonus=bonus, fsw_pen=block)
    assert not g["fsw"][res.states, res.token.astype(int)].any()


@pytest.mark.parametrize("asked", [None, 0.0, 0.03])
def test_placement_never_runs_unless_footswitches_are_asked_for(monkeypatch, asked):
    engine = _engine(monkeypatch)
    g, dur = _song(140.0)

    def boom(*a, **k):
        raise AssertionError("footswitch placement ran")

    for difficulty in ("medium", "hard", "challenge"):
        style = _style(difficulty, asked)
        before = engine.generate(g.X, g.tick_times_sec, difficulty, style, dur,
                                 veto=pipeline.flow_veto(difficulty, 140.0))
        monkeypatch.setattr(infer, "place_footswitches", boom)
        after = engine.generate(g.X, g.tick_times_sec, difficulty, style, dur,
                                veto=pipeline.flow_veto(difficulty, 140.0))
        monkeypatch.undo()
        engine = _engine(monkeypatch)
        assert "footswitches" not in before.stats["choreo"]
        assert np.array_equal(before.encoded.token, after.encoded.token)
        assert np.array_equal(before.events.tick, after.events.tick)


def test_a_tier_without_footswitches_never_gets_one_whatever_is_asked(monkeypatch):
    engine = _engine(monkeypatch)
    g, dur = _song(140.0)
    chart = engine.generate(g.X, g.tick_times_sec, "medium", _style("medium", 0.2), dur,
                            veto=pipeline.flow_veto("medium", 140.0))
    assert "footswitches" not in chart.stats["choreo"]
    assert _read_share(chart, g) == 0.0


@pytest.mark.parametrize("difficulty,bpm", [("hard", 150.0), ("challenge", 140.0),
                                            ("challenge", 170.0)])
def test_footswitches_asked_for_are_read_as_footswitches(monkeypatch, difficulty, bpm):
    engine = _engine(monkeypatch)
    g, dur = _song(bpm)
    veto = pipeline.flow_veto(difficulty, bpm)
    none = engine.generate(g.X, g.tick_times_sec, difficulty, _style(difficulty, 0.0), dur,
                           veto=veto)
    some = engine.generate(g.X, g.tick_times_sec, difficulty, _style(difficulty, 0.06), dur,
                           veto=veto)
    # the same notes, other arrows
    assert np.array_equal(none.events.tick, some.events.tick)
    info = some.stats["choreo"]["footswitches"]
    share = _read_share(some, g)
    assert _read_share(none, g) < 0.01
    assert 0.045 <= share <= 0.08, share
    assert info["read_share"] == pytest.approx(share, abs=1e-3)
    # ...and still a chart anyone can stand on: no step the gate forbids, no
    # unforced double-step in the annotator's reading, no spin, no held foot moved
    assert veto(some) == []
    times_ms = g.tick_times_sec[some.events.tick] * 1000.0
    assert pipeline.read_double_steps(some.events, times_ms) == []
    st = infer.chart_stats(some, g.tick_times_sec)
    assert st.get("spins", 0) == 0 and st.get("lock_violations", 0) == 0
    # a footswitch the reader would play as a jack is not left in the chart
    own = infer._gen_tables()["fsw"][C.foot_path(some.encoded)[0],
                                     some.encoded.token.astype(int)]
    read = infer.read_footswitches(some.events, times_ms)
    assert int((own & ~read).sum()) <= max(1, int(0.2 * read.sum()))


def test_more_asked_is_more_read(monkeypatch):
    engine = _engine(monkeypatch)
    g, dur = _song(140.0)
    shares = []
    for asked in (0.0, 0.04, 0.08):
        chart = engine.generate(g.X, g.tick_times_sec, "challenge", _style("challenge", asked),
                                dur, veto=pipeline.flow_veto("challenge", 140.0))
        shares.append(_read_share(chart, g))
    assert shares[0] < shares[1] < shares[2]


# --------------------------------------------------------------------------- #
# Stream mode's speed limit
# --------------------------------------------------------------------------- #
def test_stream_mode_keeps_16ths_to_200_bpm_at_challenge_and_170_at_hard():
    sub = infer.stream_subdivision
    for bpm in (150.0, 175.0, 190.0, 200.0):
        assert sub("challenge", None, 60 / bpm) == 16, bpm
    assert sub("challenge", None, 60 / 204.0) == 8
    for bpm in (120.0, 150.0, 170.0):
        assert sub("hard", None, 60 / bpm) == 16, bpm
    assert sub("hard", None, 60 / 174.0) == 8
    assert infer.stream_rate_limit("challenge") == pytest.approx(1 / 0.075)
    assert infer.stream_rate_limit("hard") == pytest.approx(1 / 0.088)
    # the tiers that do not write 16ths are where they were
    assert infer.stream_rate_limit("medium") == BUDGETS["medium"].max_nps_4s
    assert sub("medium", None, 0.5) == 8 and sub("beginner", None, 0.5) == 4


def test_an_ordinary_charts_speed_limit_is_the_tiers_own():
    assert BUDGETS["challenge"].max_nps_4s == 10.0 and BUDGETS["hard"].max_nps_4s == 8.0
    assert infer.min_gap_s("challenge") == pytest.approx(0.05)
    assert infer.min_gap_s("hard") == pytest.approx(0.0625)


@pytest.mark.parametrize("difficulty,bpm", [("challenge", 180.0), ("hard", 165.0)])
def test_a_fast_song_streams_in_16ths_only_when_asked_to_stream(monkeypatch, difficulty, bpm):
    engine = _engine(monkeypatch)
    g, dur = _song(bpm)
    style = cond.resolve_style(difficulty)
    share = 0.8 if difficulty == "challenge" else 0.45
    stream = engine.generate(g.X, g.tick_times_sec, difficulty, style, dur,
                             veto=pipeline.flow_veto(difficulty, bpm),
                             options=DecodeOptions(stream=True, stream_share=share))
    info = stream.stats["rhythm"]["stream"]
    assert info["subdivision"] == 16 and info["over_tier_rate"]
    assert info["nps"] == pytest.approx(4 * bpm / 60.0, abs=0.05)
    ev = stream.events
    assert abs(cond.stream_share(ev.tick, ev.p1 >= 0) - share) <= 0.08
    ticks = set(ev.tick.tolist())
    for a, b in info["segments"]:
        assert all(t in ticks for t in range(a * C.G, b * C.G, 3))
    # the stream is still legal to stand on at that speed
    st = infer.chart_stats(stream, g.tick_times_sec)
    assert st.get("spins", 0) == 0 and st.get("lock_violations", 0) == 0
    assert pipeline.flow_veto(difficulty, bpm)(stream) == []
    times_ms = g.tick_times_sec[ev.tick] * 1000.0
    assert pipeline.read_double_steps(ev, times_ms) == []

    # the ordinary chart of the same song never exceeds the tier's peak rate
    plain = engine.generate(g.X, g.tick_times_sec, difficulty, style, dur,
                            veto=pipeline.flow_veto(difficulty, bpm))
    t = g.tick_times_sec[plain.events.tick]
    in_4s = np.searchsorted(t, t + 4.0, side="left") - np.arange(len(t))
    assert in_4s.max() <= int(BUDGETS[difficulty].max_nps_4s * 4.0)
    assert "stream" not in plain.stats["rhythm"]


def test_the_stream_limit_does_not_touch_an_ordinary_chart(monkeypatch):
    engine = _engine(monkeypatch)
    g, dur = _song(180.0)
    style = cond.resolve_style("challenge")
    a = engine.generate(g.X, g.tick_times_sec, "challenge", style, dur)
    monkeypatch.setattr(infer, "STREAM_MIN_GAP_S", {})          # as before v1.5c
    b = engine.generate(g.X, g.tick_times_sec, "challenge", style, dur)
    assert np.array_equal(a.events.tick, b.events.tick)
    assert np.array_equal(a.encoded.token, b.encoded.token)
    # ...and with the limit gone, Stream mode is back to 8ths at this tempo
    assert infer.stream_subdivision("challenge", None, 60 / 180.0) == 8
