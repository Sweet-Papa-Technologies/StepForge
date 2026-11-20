"""v1.5 decoder pieces: the three half-beat rules and how one is chosen on
tuning songs, the per-model density fit, and Stream mode's learned fill with
its rule fallback."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from beatforge.adapters.stepmania.neural import calibrate_decoder as CAL  # noqa: E402
from beatforge.adapters.stepmania.neural import codec as C  # noqa: E402
from beatforge.adapters.stepmania.neural import conditioning as cond  # noqa: E402
from beatforge.adapters.stepmania.neural import grid_features as GF  # noqa: E402
from beatforge.adapters.stepmania.neural import infer, pipeline  # noqa: E402
from beatforge.adapters.stepmania.neural.infer import DecodeOptions  # noqa: E402

import synth_helpers as H  # noqa: E402

TINY = Path(__file__).parent / "fixtures" / "neural" / "tiny_model"
BPM, OFFSET = 160.0, 0.2


# --------------------------------------------------------------------------- #
# the half-beat rules
# --------------------------------------------------------------------------- #
def test_phase_decision_rules():
    d = pipeline.phase_decision
    assert d("margin", 1.5, 2.0) and not d("margin", 1.5, 1.0) and not d("margin", 1.5, -9.0)
    # vote: two of three by sign, the margin plays no part
    assert d("vote", 9.0, 0.1, 0.1, -5.0) and not d("vote", 0.0, 5.0, -0.1, -0.1)
    # margin_vote: the onset statistic alone when it is sure, either way...
    assert d("margin_vote", 1.0, 1.5, -3.0, -3.0)
    assert not d("margin_vote", 1.0, -1.5, 3.0, 3.0)
    # ...and the vote when it is not
    assert d("margin_vote", 1.0, -0.5, 3.0, 3.0)
    assert not d("margin_vote", 1.0, 0.5, -3.0, -3.0)
    assert pipeline.PHASE_RULES == ("margin", "vote", "margin_vote")


class _Engine:
    def __init__(self, decoder):
        self.model = type("M", (), {"meta": {"decoder": decoder}})()


def test_phase_rule_is_read_from_the_model_and_unknown_values_fall_back():
    assert pipeline.phase_rule(_Engine({"phase_rule": "margin_vote"})) == "margin_vote"
    assert pipeline.phase_rule(_Engine({"phase_rule": "vote"})) == "vote"
    assert pipeline.phase_rule(_Engine({"phase_rule": "coin"})) == "margin"
    assert pipeline.phase_rule(_Engine(None)) == "margin"


def _rows(spec):
    return [{"name": str(i), "truth": t, "votes": {"onset": o, "jump": j, "audio": a}}
            for i, (t, o, j, a) in enumerate(spec)]


def test_fit_keeps_the_current_rule_unless_another_fixes_enough_grids():
    # onset is right on every song; so is everything else: nothing to gain
    rows = _rows([(True, 5.0, 2.0, 1.0)] * 6 + [(False, -5.0, -2.0, -1.0)] * 14)
    got = CAL.fit_phase_rule(rows, {"phase_rule": "vote"})
    assert got["chosen"] == {"rule": "vote", "margin": pipeline.PHASE_MARGIN, "errors": 0}
    assert got["songs"] == 20 and got["off_before"] == 6
    assert got["errors"]["vote"] == 0 and set(got["errors"]["margin"]) == {
        str(m) for m in CAL.MARGINS}
    # the vote gets one wrong and the margin rule none: one grid is not enough
    rows[0]["votes"].update(jump=-2.0, audio=-1.0)
    assert CAL.fit_phase_rule(rows, {"phase_rule": "vote"})["chosen"]["rule"] == "vote"
    assert CAL.fit_phase_rule(rows, {"phase_rule": "vote"}, min_gain=1)["chosen"]["rule"] \
        == "margin"


def test_fit_picks_margin_vote_when_the_statistic_is_right_only_when_sure():
    sure = [(True, 4.0, -1.0, -1.0)] * 5 + [(False, -4.0, 1.0, 1.0)] * 5      # vote wrong
    unsure = [(True, -0.3, 1.0, 1.0)] * 4 + [(False, 0.3, -1.0, -1.0)] * 4    # onset wrong
    got = CAL.fit_phase_rule(_rows(sure + unsure), {"phase_rule": "vote"})
    assert got["current"]["errors"] == 10
    assert got["errors"]["margin_vote"]["1.0"] == 0
    assert min(got["errors"]["margin"].values()) == 4          # misses the unsure moves
    assert got["chosen"]["rule"] == "margin_vote" and got["chosen"]["errors"] == 0
    # the margin is the middle of the flat stretch that works (0.3 .. 3.9 here,
    # of which 0.3 .. 2.0 was swept), not its edge
    assert 0.8 <= got["chosen"]["margin"] <= 1.5


def test_songs_without_a_usable_truth_are_left_out_of_the_fit():
    rows = _rows([(None, 5.0, 5.0, 5.0), (True, 5.0, 5.0, 5.0), (False, -5.0, -5.0, -5.0)])
    rows.append({"name": "x", "truth": True, "votes": None})
    assert CAL.fit_phase_rule(rows, {})["songs"] == 2


def test_half_beat_off_reads_the_phase_between_two_grids():
    human = {"bpm": 120.0, "offset_s": 0.50, "constant": True}
    assert CAL.half_beat_off(120.0, 0.50, human) is False
    assert CAL.half_beat_off(120.0, 0.52, human) is False           # 0.04 beat
    assert CAL.half_beat_off(120.0, 0.75, human) is True            # half a beat later
    assert CAL.half_beat_off(120.0, 0.25, human) is True            # ...or earlier
    assert CAL.half_beat_off(120.0, 1.00, human) is False           # a whole beat
    assert CAL.half_beat_off(120.0, 0.625, human) is None           # a quarter: neither
    assert CAL.half_beat_off(121.0, 0.75, human) is None            # another tempo
    assert CAL.half_beat_off(120.0, 0.75, dict(human, constant=False)) is None


def test_sixteenths_counts_only_16th_ticks():
    assert CAL.sixteenths(np.array([0, 3, 6, 9, 12, 15, 4, 8, 48 + 3])) == 4


def test_median_ratio_per_difficulty():
    res = [{"charts": {"easy": {"n": 90, "n_human": 100}, "hard": {"n": 300, "n_human": 200}}},
           {"charts": {"easy": {"n": 110, "n_human": 100}}},
           {"charts": {"easy": {"n": 130, "n_human": 100}, "hard": {"n": 100, "n_human": 200}}}]
    got = CAL.median_ratios(res)
    assert got == {"easy": pytest.approx(1.1), "hard": pytest.approx(1.0)}


def test_tuning_songs_are_read_from_folders_with_audio_and_a_simfile(tmp_path):
    H.make_song_folder(tmp_path, "one")
    (tmp_path / "empty").mkdir()
    songs = CAL.find_songs(tmp_path)
    assert [s[0] for s in songs] == ["one"]
    human = CAL.human_charts(songs[0][2])
    assert human["bpm"] == H.BPM and human["offset_s"] == pytest.approx(H.OFFSET_S)
    assert human["constant"] and set(human["ticks"]) == {
        "beginner", "easy", "medium", "hard", "challenge"}
    assert len(human["ticks"]["challenge"]) == len(H.chart_ticks("Challenge"))
    assert CAL.sixteenths(human["ticks"]["challenge"]) == 4 * len(H.RUN_BARS)
    assert CAL.sixteenths(human["ticks"]["hard"]) == 0


# --------------------------------------------------------------------------- #
# Stream mode: the learned fill and its fallback
# --------------------------------------------------------------------------- #
def test_stream_fill_is_validated_and_only_reported_in_stream_mode():
    assert DecodeOptions().stream_fill == "neural"
    with pytest.raises(ValueError, match="stream_fill"):
        DecodeOptions(stream_fill="magic")
    assert DecodeOptions(stream_fill="rule").to_dict() == {}
    assert DecodeOptions(stream=True, stream_fill="rule").to_dict()["stream_fill"] == "rule"
    assert "stream_fill" not in DecodeOptions(stream=True).to_dict()
    assert DecodeOptions.from_dict({"stream": True, "stream_fill": "rule"}).stream_fill == "rule"


def test_stream_style_describes_the_stream_the_user_asked_for():
    base = cond.resolve_style("challenge", overrides={"jump_share": 0.02})
    beat_s = 60.0 / 160.0
    s = infer.stream_style(base, "challenge", DecodeOptions(
        stream=True, stream_coverage=0.8, max_subdivision=16), beat_s)
    rate = 4.0 / beat_s                                            # 16ths at 160 BPM
    assert s[cond.INDEX["nps_norm"]] == pytest.approx(rate / 16.0)
    assert s[cond.INDEX["stream_share"]] == pytest.approx(0.8)
    assert s[cond.INDEX["meter_norm"]] == pytest.approx(cond.effective_meter(rate) / 15.0)
    assert s[cond.INDEX["jump_share"]] == pytest.approx(0.02)      # the user's slider stays
    assert s[cond.INDEX["crossover_share"]] == cond.MASKED         # the rest stay open
    assert base[cond.INDEX["stream_share"]] == cond.MASKED         # the input is not edited
    # a stream of 8ths is not what the stream slider counts: density only
    e = infer.stream_style(base, "challenge",
                           DecodeOptions(stream=True, max_subdivision=8), beat_s)
    assert e[cond.INDEX["stream_share"]] == 0.0
    assert e[cond.INDEX["nps_norm"]] == pytest.approx(2.0 / beat_s / 16.0)


@pytest.fixture
def stream_setup(monkeypatch):
    engine = infer.StepManiaNeuralEngine(TINY)
    y = H.clicks(bars=24, bpm=BPM)
    g = GF.extract_grid_features(y, GF.TimingMap.constant(BPM, OFFSET))
    calls = []

    def fake_rhythm(answers: bool):
        def rhythm(x, cond19):
            """A model that writes 16ths everywhere when asked for a stream
            (if `answers`) and a plain 8th-note chart otherwise."""
            asked = float(cond19[cond.INDEX["stream_share"]]) >= 0.5
            calls.append(asked)
            t = len(x)
            k = np.arange(t)
            onset = np.full(t, -6.0, np.float32)
            onset[k % 6 == 0] = 2.0
            if asked and answers:
                onset[k % 3 == 0] = 3.0
            return {"onset": onset, "jump": np.full(t, -5.0, np.float32),
                    "hold_type": np.tile(np.array([9, 0, 0, 0, 0, 0], np.float32), (t, 1)),
                    "hold_dur": np.zeros((t, 10), np.float32),
                    "mine": np.full(t, -9.0, np.float32)}
        return rhythm

    def chart(answers: bool, declared: bool, **opt):
        calls.clear()
        monkeypatch.setattr(engine.model, "rhythm", fake_rhythm(answers))
        monkeypatch.setitem(engine.model.meta, "decoder",
                            {"neural_stream": True} if declared else {})
        opts = DecodeOptions(stream=True, max_subdivision=16, stream_coverage=0.75, **opt)
        out = engine.generate(g.X, g.tick_times_sec, "challenge",
                              cond.resolve_style("challenge"), len(y) / GF.SR, options=opts)
        return out, list(calls)

    return chart


def test_a_model_that_writes_the_stream_keeps_its_own_chart(stream_setup):
    chart, calls = stream_setup(answers=True, declared=True)
    info = chart.stats["rhythm"]["stream"]
    assert calls == [True]                                # asked once, as a stream chart
    assert info["fill"] == "neural" and info["model_share"] == 1.0
    assert info["subdivision"] == 16 and info["stream_bars"] >= 8
    # the streams are still whole: every slot of every stream bar
    ticks = set(chart.events.tick.tolist())
    for a, b in info["segments"]:
        assert all(t in ticks for t in range(a * C.G, b * C.G, 3))
    st = infer.chart_stats(chart, np.arange(len(chart.events) + 1))
    assert st.get("spins", 0) == 0 and st.get("lock_violations", 0) == 0


def test_a_model_that_does_not_write_it_falls_back_to_the_rule_fill(stream_setup):
    chart, calls = stream_setup(answers=False, declared=True)
    info = chart.stats["rhythm"]["stream"]
    assert calls == [True, False]                         # asked, then charted plainly
    assert info["fill"].startswith("rule (the model wrote 50%")
    assert info["stream_bars"] >= 8                       # the user still gets streams


def test_rule_fill_on_request_and_for_models_that_do_not_declare_it(stream_setup):
    chart, calls = stream_setup(answers=True, declared=True, stream_fill="rule")
    assert calls == [False] and chart.stats["rhythm"]["stream"]["fill"] == "rule"
    # weights from before v1.5 are never asked for a stream chart
    chart, calls = stream_setup(answers=True, declared=False)
    assert calls == [False] and chart.stats["rhythm"]["stream"]["fill"] == "rule"


def test_the_shipped_model_reports_whether_it_writes_streams():
    engine = infer.StepManiaNeuralEngine()
    declared = bool((engine.model.meta.get("decoder") or {}).get("neural_stream"))
    assert engine.model.neural_stream is declared
