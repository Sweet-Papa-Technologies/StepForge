"""Tempo maps: one tempo where one tempo holds, #BPMS + #STOPS where it does not.

The songs here are synthetic drum loops spliced the way real recordings break
a single-tempo grid — a band stops and comes back in, a bar is cut short, the
tempo jumps — so the true beat times are known before and after the splice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from beatforge.adapters.stepmania import serialize, timing as T
from beatforge.adapters.stepmania.neural import grid_features as GF
from beatforge.adapters.stepmania.neural import infer, pipeline
from beatforge.adapters.stepmania.neural import tempo_map as TM

TINY = Path(__file__).parent / "fixtures" / "neural" / "tiny_model"
SR = GF.SR
OFFSET = 0.40                       # time of beat 0 in every loop below


# --------------------------------------------------------------------------- #
# Synthetic songs
# --------------------------------------------------------------------------- #
def _hit(y: np.ndarray, t: float, amp: float, length: int, rng) -> None:
    i = int(round(t * SR))
    if 0 <= i < len(y) - length:
        y[i:i + length] += amp * np.hanning(2 * length)[length:].astype(np.float32) \
            * rng.standard_normal(length).astype(np.float32)


def drum_loop(bpm: float, beats: int, seed: int = 7) -> np.ndarray:
    """Kick on every beat, a louder snare on 2 and 4, hats on the 8ths and a
    few 16ths, over a quiet held chord (music has a level between its hits;
    a pause is where that goes away): enough of a rhythm for a grid to be read
    off it. Beat 0 is at OFFSET and the loop ends on a beat line."""
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    n = int(round((OFFSET + beats * period) * SR))
    k = np.arange(n) / SR
    y = (0.03 * (np.sin(2 * np.pi * 110.0 * k) + np.sin(2 * np.pi * 165.0 * k))).astype(np.float32)
    for b in range(beats):
        t = OFFSET + b * period
        _hit(y, t, 1.0 if b % 2 == 0 else 1.4, 400, rng)
        _hit(y, t + period / 2, 0.45, 120, rng)
        if b % 4 in (1, 3):
            _hit(y, t + 3 * period / 4, 0.3, 120, rng)
    return y


def beat_times(bpm: float, beats: int) -> np.ndarray:
    return OFFSET + np.arange(beats) * 60.0 / bpm


def measure(y: np.ndarray, bpm: float):
    """What the pipeline hands the tempo-map check for audio `y`, with the
    single-tempo grid of the song's opening."""
    ff = GF.frame_features(y)
    dur = len(y) / SR
    return TM.analyze(ff.onset_envelope, bpm, OFFSET, dur, TM.frame_rms(y, ff.n_frames))


def on_grid(tm: TM.TempoMap, truth: np.ndarray, duration: float, step: float = 0.25
            ) -> float:
    """Share of the true beats within 20 ms of a line of the mapped grid, read
    every `step` beats (0.25: the 16th-note lines a chart's notes sit on)."""
    timing = tm.timing()
    n = int(timing.beat_at(duration) / step) + 4
    lines = timing.time_at(np.arange(-4, n) * step)
    j = np.clip(np.searchsorted(lines, truth), 1, len(lines) - 1)
    err = np.minimum(np.abs(lines[j] - truth), np.abs(lines[j - 1] - truth))
    return float((err <= 0.020).mean())


# --------------------------------------------------------------------------- #
# The shared beat <-> seconds arithmetic
# --------------------------------------------------------------------------- #
def test_beat_to_time_walks_bpm_segments_and_stops():
    bpms, stops = [(0, 120.0), (8, 240.0)], [(4, 0.5)]
    # beat 4 is played, THEN the stop: it sits at its plain time
    assert T.beat_to_time(4.0, bpms, stops, 0.5) == pytest.approx(0.5 + 2.0)
    assert T.beat_to_time(6.0, bpms, stops, 0.5) == pytest.approx(0.5 + 3.0 + 0.5)
    # past the tempo change: 8 beats at 120, the stop, then 4 beats at 240
    assert T.beat_to_time(12.0, bpms, stops, 0.5) == pytest.approx(0.5 + 4.0 + 0.5 + 1.0)
    b = np.linspace(-2.0, 20.0, 441)
    back = T.time_to_beat(T.beat_to_time(b, bpms, stops, 0.5), bpms, stops, 0.5)
    np.testing.assert_allclose(back, b, atol=1e-9)
    # during the stop the music waits on the beat it stopped at
    t4 = T.beat_to_time(4.0, bpms, stops, 0.5)
    assert T.time_to_beat(t4 + 0.25, bpms, stops, 0.5) == pytest.approx(4.0)
    assert T.time_to_beat(t4 + 0.5 + 0.25, bpms, stops, 0.5) == pytest.approx(4.5)


def test_grid_features_and_timing_share_one_timing_map():
    assert GF.TimingMap is T.TimingMap
    tm = T.TimingMap(0.1, [(0, 150.0)], [(16, 0.3)])
    assert T.TimingMap(0.1, **{k: tm.to_dict()[k] for k in ("bpms", "stops")}).time_at(40.0) \
        == pytest.approx(float(tm.time_at(40.0)))
    assert tm.dominant_bpm(60.0) == 150.0
    assert T.TimingMap(0.0, [(0, 120.0), (8, 180.0)]).dominant_bpm(60.0) == 180.0


def test_single_tempo_analysis_is_timed_exactly_as_before():
    """A song with no tempo map goes through the expression every caller used
    before one could exist: bit for bit, not merely close."""
    an = {"bpm": 173.913, "offset": 0.3187}
    beats = np.arange(0, 400) / 12.0
    old = np.array([an["offset"] + b / an["bpm"] * 60 for b in beats])
    assert (T.beats_to_times(an, beats) == old).all()
    assert not T.has_tempo_map(an) and T.timing_of(an).is_constant
    mapped = dict(an, tempo_map={"bpms": [[0.0, 120.0], [8.0, 240.0]], "stops": [[4.0, 0.5]]})
    assert T.beats_to_times(mapped, [12.0])[0] == pytest.approx(
        T.beat_to_time(12.0, [(0, 120.0), (8, 240.0)], [(4, 0.5)], an["offset"]))


# --------------------------------------------------------------------------- #
# The check, and the map when the check fails
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bpm", [96.0, 128.0, 174.0])
def test_a_steady_song_is_left_alone(bpm):
    beats = int(110 * bpm / 60)
    y = drum_loop(bpm, beats)
    tm = measure(y, bpm)
    assert tm.constant, tm.diag
    assert tm.diag["aligned_share"] >= TM.FAST_PATH_SHARE
    timing = tm.timing()
    assert timing.is_constant and timing.bpms == [(0.0, bpm)] and timing.offset_s == OFFSET


def test_a_pause_becomes_a_stop():
    """The band stops for two thirds of a second and comes back in between
    the lines of the old grid: the map carries a #STOPS entry and the beats
    after the pause are back on it."""
    bpm, before, after = 126.0, 112, 112
    a, b = drum_loop(bpm, before), drum_loop(bpm, after, seed=8)
    gap = 5.5 * 15.0 / bpm              # five and a half 16th notes: 0.655 s
    y = np.concatenate([a, np.zeros(int(round(gap * SR)), np.float32), b[int(OFFSET * SR):]])
    cut = OFFSET + before * 60.0 / bpm
    t_before = beat_times(bpm, before)
    t_after = cut + gap + np.arange(after) * 60.0 / bpm
    single = TM.TempoMap(True, bpm, OFFSET)
    assert on_grid(single, t_after, len(y) / SR) < 0.2           # the bug
    tm = measure(y, bpm)
    assert not tm.constant, tm.diag
    assert [e["kind"] for e in tm.events] == ["pause"]
    assert len(tm.stops) == 1 and 0.0 < tm.stops[0][1] < 60.0 / bpm
    assert abs(tm.events[0]["time_s"] - cut) < 60.0 / bpm       # at the pause
    assert len(tm.bpms) == 1 and tm.bpms[0][1] == pytest.approx(bpm, rel=2e-3)
    assert on_grid(tm, t_before[4:-2], len(y) / SR) >= 0.9
    assert on_grid(tm, t_after[2:-4], len(y) / SR) >= 0.9


def test_a_cut_bar_is_reseated_without_a_stop():
    """1.37 beats go missing with no silence (an odd-meter turnaround): the
    grid after it is re-seated through a short BPM segment, not a stop."""
    bpm, before, after = 132.0, 120, 120
    period = 60.0 / bpm
    a, b = drum_loop(bpm, before), drum_loop(bpm, after, seed=8)
    skip = int(round((OFFSET + 1.37 * period) * SR))
    y = np.concatenate([a, b[skip:]])
    cut = OFFSET + before * period
    t_after = cut + (np.arange(2, after) - 1.37) * period
    tm = measure(y, bpm)
    assert not tm.constant, tm.diag
    assert tm.stops == [] and len(tm.bpms) >= 2
    assert [e["kind"] for e in tm.events] == ["phase"]
    assert abs(tm.events[0]["time_s"] - cut) < 2 * period
    assert on_grid(tm, beat_times(bpm, before)[4:-2], len(y) / SR) >= 0.9
    assert on_grid(tm, t_after[2:-4], len(y) / SR) >= 0.9
    # the join is short, and the grid after it starts on a whole beat
    assert tm.events[0]["beats"] <= 2.0 and float(tm.bpms[-1][0]).is_integer()
    assert all(float(2 * beat).is_integer() for beat, _ in tm.bpms)


def test_a_tempo_jump_gets_its_own_segment():
    bpm, new, before, after = 120.0, 135.0, 128, 144
    a, b = drum_loop(bpm, before), drum_loop(new, after, seed=8)
    y = np.concatenate([a, b[int(OFFSET * SR):]])
    cut = OFFSET + before * 60.0 / bpm
    t_after = cut + np.arange(after) * 60.0 / new
    tm = measure(y, bpm)
    assert not tm.constant, tm.diag
    assert tm.events[-1]["kind"] == "tempo"
    assert tm.bpms[0][1] == pytest.approx(bpm, rel=1e-3)
    assert tm.bpms[-1][1] == pytest.approx(new, rel=1.5e-3)
    assert abs(tm.events[-1]["time_s"] - cut) < 4 * 60.0 / bpm
    assert on_grid(tm, beat_times(bpm, before)[4:-4], len(y) / SR) >= 0.9
    assert on_grid(tm, t_after[4:-4], len(y) / SR) >= 0.9
    assert tm.timing().dominant_bpm(len(y) / SR) == pytest.approx(new, rel=1.5e-3)


def test_two_whole_beats_cut_changes_nothing():
    """A 2/4 turnaround bar drops whole beats: the beat grid is still locked
    (only the bar line moves), so the single tempo stands."""
    bpm, before, after = 140.0, 120, 120
    period = 60.0 / bpm
    a, b = drum_loop(bpm, before), drum_loop(bpm, after, seed=8)
    y = np.concatenate([a, b[int(round((OFFSET + 2 * period) * SR)):]])
    tm = measure(y, bpm)
    assert tm.constant, tm.diag
    truth = np.concatenate([beat_times(bpm, before),
                            OFFSET + (before + np.arange(after - 2)) * period])
    assert on_grid(tm, truth[4:-4], len(y) / SR, step=1.0) >= 0.9


def test_a_small_wobble_is_not_an_event():
    """12 ms of push in the second half is feel, not a break: no event."""
    bpm, before, after = 128.0, 112, 112
    a, b = drum_loop(bpm, before), drum_loop(bpm, after, seed=8)
    y = np.concatenate([a, np.zeros(int(0.012 * SR), np.float32), b[int(OFFSET * SR):]])
    assert measure(y, bpm).constant


def test_the_check_is_cheap():
    import time
    bpm = 150.0
    y = drum_loop(bpm, int(180 * bpm / 60))
    ff = GF.frame_features(y)
    rms = TM.frame_rms(y, ff.n_frames)
    t0 = time.perf_counter()
    for _ in range(5):
        TM.analyze(ff.onset_envelope, bpm, OFFSET, len(y) / SR, rms)
    assert (time.perf_counter() - t0) / 5 < 0.15


# --------------------------------------------------------------------------- #
# What is written, and what the rest of the engine reads
# --------------------------------------------------------------------------- #
def _placement(beat: float):
    from beatforge.adapters.stepmania.quantize import Placement
    return Placement(beat=beat, panels=(0,), kind="tap")


def test_single_tempo_simfile_is_written_as_it_always_was():
    an = {"bpm": 128.0, "offset": 0.25, "sections": []}
    assert serialize.timing_tags(an) == ("0.000=128.000", None)
    sf = serialize.build_simfile({"title": "t", "artist": "a", "music": "t.ogg"}, an,
                                 {"easy": (3, [_placement(float(b)) for b in range(8)])})
    text = str(sf)
    assert "#BPMS:0.000=128.000;" in text and "#STOPS:;" in text


def test_a_tempo_map_is_written_as_bpms_and_stops():
    an = {"bpm": 135.0, "offset": 0.25, "sections": [{"start_bar": 2, "energy_pct": 0.9}],
          "tempo_map": {"bpms": [[0.0, 120.0], [16.0, 135.0]], "stops": [[8.0, 0.312]]}}
    assert serialize.timing_tags(an) == ("0.000=120.000,16.000=135.000", "8.000=0.312")
    sf = serialize.build_simfile({"title": "t", "artist": "a", "music": "t.ogg"}, an,
                                 {"easy": (3, [_placement(float(b)) for b in range(24)])})
    text = str(sf)
    assert "#BPMS:0.000=120.000,16.000=135.000;" in text
    assert "#STOPS:8.000=0.312;" in text
    # the preview sample starts where bar 2 really is: 8 beats at 120
    assert f"#SAMPLESTART:{0.25 + 4.0:.3f};" in text
    # and garcia/simfile reads the same times back out of the file
    import simfile
    from simfile.timing import TimingData
    from simfile.timing.engine import TimingEngine
    eng = TimingEngine(TimingData(simfile.loads(text)))
    mine = T.timing_of(an)
    for beat in (0, 7.5, 8, 8.25, 15, 16, 23.75):
        assert float(eng.time_at(beat)) == pytest.approx(float(mine.time_at(beat)), abs=1e-6)


def test_qa_and_meter_time_notes_through_the_map():
    from beatforge.adapters.stepmania import qa as sm_qa
    from beatforge.adapters.stepmania.difficulty import compute_meter
    stops = [[8.0, 0.27]]
    an = {"bpm": 120.0, "offset": 0.0, "duration_s": 30.0, "sections": [], "energy_curve": [],
          "tempo_map": {"bpms": [[0.0, 120.0]], "stops": stops},
          # onsets exactly where the mapped notes are: after the stop, 270 ms late
          "onsets": [{"time": float(t), "nearest_beat": float(b)}
                     for b, t in enumerate(T.beat_to_time(np.arange(0, 32), [(0, 120.0)],
                                                          stops, 0.0))]}
    notes = [_placement(float(b)) for b in range(32)]
    assert sm_qa.chart_metrics(notes, an, "easy")["onset_alignment"] == 1.0
    plain = {k: v for k, v in an.items() if k != "tempo_map"}
    assert sm_qa.chart_metrics(notes, plain, "easy")["onset_alignment"] < 0.5
    # a double-time stretch doubles the notes per second the meter sees
    fast = T.TimingMap(0.0, [(0, 120.0), (16, 240.0)])
    eighths = [_placement(b / 2.0) for b in range(96)]
    assert compute_meter(eighths, 120.0, "hard", fast) >= compute_meter(eighths, 120.0, "hard")


def test_rebar_follows_the_map():
    an = {"bpm": 120.0, "offset": 0.0, "duration_s": 40.0,
          "energy_curve": [float(i) for i in range(20)],
          "sections": [{"start_bar": 0, "end_bar": 10}, {"start_bar": 10, "end_bar": 20}]}
    tm = T.TimingMap(0.0, [(0, 120.0)], [(20.0, 2.0)])       # 2 s stop after bar 5
    out = pipeline._rebar(an, 120.0, 0.0, tm)
    # old bar 10 starts at 20 s, which the map reaches one bar earlier (bar 9)
    assert out["sections"][1]["start_bar"] == 9
    assert out["energy_curve"][4] == 4.0 and out["energy_curve"][6] == 7.0


# --------------------------------------------------------------------------- #
# After a pause: the same grid, another beat
# --------------------------------------------------------------------------- #
class _LoudModel:
    """Stands in for RhythmNet: onset probability where the music is loud. On
    the drum loops above that is the beats, whichever tick they fall on."""
    meta = None

    def rhythm(self, x, cv):
        mel = x[:, GF.MEL].sum(1).astype(np.float64)
        top = np.percentile(mel, 99.0)
        return {"onset": ((mel - 0.7 * top) / (0.05 * abs(top) + 1e-9)).astype(np.float32)}


class _LoudEngine:
    model = _LoudModel()

    def cond_for(self, style, has_stems=False):
        return np.zeros(19, np.float32)


def _song_with_gap(gap_beats: float, bpm: float = 126.0, before: int = 112, after: int = 112):
    """Two drum loops with `gap_beats` of silence between them -> (audio, the
    pause in seconds, the true beat times after it)."""
    period = 60.0 / bpm
    a, b = drum_loop(bpm, before), drum_loop(bpm, after, seed=8)
    gap = gap_beats * period
    y = np.concatenate([a, np.zeros(int(round(gap * SR)), np.float32), b[int(OFFSET * SR):]])
    cut = OFFSET + before * period
    return y, (cut, cut + gap), cut + gap + np.arange(after) * period


@pytest.mark.parametrize("gap_beats, seat", [(2.5, 0.5), (2.25, 0.25), (2.75, 0.75)])
def test_a_pause_that_ends_on_another_16th_is_found_by_the_model(gap_beats, seat):
    """Every onset is still on a grid line after such a pause, so the tempo-map
    check sees one tempo; the rhythm model, asked on each side, sees the beat
    has moved."""
    bpm = 126.0
    y, pause, _ = _song_with_gap(gap_beats, bpm)
    assert measure(y, bpm).constant
    g = GF.extract_grid_features(y, GF.TimingMap.constant(bpm, OFFSET))
    out = pipeline.pause_seats(_LoudEngine(), g.core, g.dt_ms, g.tick_times_sec, [pause])
    assert out is not None and out["margin"] >= pipeline.PAUSE_SEAT_MARGIN
    assert out["before_beats"] == 0.0 and out["after_beats"] == seat
    assert out["pause"] == pytest.approx(list(pause))


def test_a_pause_of_whole_beats_changes_no_seat():
    bpm = 126.0
    y, pause, _ = _song_with_gap(2.0, bpm)
    g = GF.extract_grid_features(y, GF.TimingMap.constant(bpm, OFFSET))
    assert pipeline.pause_seats(_LoudEngine(), g.core, g.dt_ms, g.tick_times_sec,
                                [pause]) is None
    # nor is a pause asked about when it has too little song on one side
    assert pipeline.pause_seats(_LoudEngine(), g.core, g.dt_ms, g.tick_times_sec,
                                [(2.0, 2.6)]) is None


def test_the_seat_check_reuses_the_phase_checks_passes():
    """verify_phase hands back the passes it ran, and is not changed by doing so."""
    bpm = 126.0
    y, pause, _ = _song_with_gap(2.5, bpm)
    g = GF.extract_grid_features(y, GF.TimingMap.constant(bpm, OFFSET))
    plain = pipeline.verify_phase(_LoudEngine(), g.core, g.dt_ms)
    passes: dict = {}
    assert pipeline.verify_phase(_LoudEngine(), g.core, g.dt_ms, passes) == plain
    assert {0, 6} <= set(passes)

    class Counting(_LoudModel):
        calls = 0

        def rhythm(self, x, cv):
            Counting.calls += 1
            return super().rhythm(x, cv)

    eng = _LoudEngine()
    eng.model = Counting()
    out = pipeline.pause_seats(eng, g.core, g.dt_ms, g.tick_times_sec, [pause], 0, passes)
    assert out["after_beats"] == 0.5
    assert Counting.calls == 2                   # the two 16th-off grids only
    # and a pause with nothing odd on either side of it costs no pass at all:
    # the screen reads the pass the phase check already ran
    y2, pause2, _ = _song_with_gap(2.0, bpm)
    g2 = GF.extract_grid_features(y2, GF.TimingMap.constant(bpm, OFFSET))
    passes2: dict = {}
    pipeline.verify_phase(_LoudEngine(), g2.core, g2.dt_ms, passes2)
    Counting.calls = 0
    assert pipeline.pause_seats(eng, g2.core, g2.dt_ms, g2.tick_times_sec, [pause2], 0,
                                passes2) is None
    assert Counting.calls == 0


def test_reseat_at_pause_writes_one_stop():
    bpm = 126.0
    y, pause, truth_after = _song_with_gap(2.5, bpm)
    dur = len(y) / SR
    tm = TM.reseat_at_pause(bpm, OFFSET, dur, pause, 0.0, 0.5)
    assert not tm.constant and tm.bpms == [(0.0, bpm)]
    assert len(tm.stops) == 1 and tm.stops[0][1] == pytest.approx(0.5 * 60.0 / bpm, abs=2e-3)
    assert [e["kind"] for e in tm.events] == ["pause"]
    # the beats before the pause and after it are on BEAT lines now
    assert on_grid(tm, beat_times(bpm, 112)[2:-2], dur, step=1.0) >= 0.95
    assert on_grid(tm, truth_after[2:-2], dur, step=1.0) >= 0.95
    assert on_grid(TM.TempoMap(True, bpm, OFFSET), truth_after[2:-2], dur, step=1.0) < 0.05


def test_features_reseat_the_beat_after_such_a_pause(tmp_path):
    import soundfile as sf
    bpm = 126.0
    y, pause, truth_after = _song_with_gap(2.5, bpm)
    path = tmp_path / "half.wav"
    sf.write(path, y / np.abs(y).max(), SR)
    early = OFFSET - pipeline.DSP_EARLY_S["local_dsp"]
    # without the model there is one tempo and nothing to see
    g0, a0 = pipeline._features(str(path), bpm, early, len(y) / SR)
    assert a0["tempo_map"] is None and a0["tempo_check"]["constant"]
    assert a0["tempo_check"]["pauses"] == 1
    g, align = pipeline._features(str(path), bpm, early, len(y) / SR, engine=_LoudEngine())
    assert align["tempo_check"]["pause_seat"]["after_beats"] == 0.5
    assert align["tempo_map"]["bpms"] == [[0.0, bpm]] and len(align["tempo_map"]["stops"]) == 1
    assert align["bpm"] == bpm
    beats = g.tick_times_sec[::12]
    j = np.clip(np.searchsorted(beats, truth_after[2:-4]), 1, len(beats) - 1)
    err = np.minimum(np.abs(beats[j] - truth_after[2:-4]),
                     np.abs(beats[j - 1] - truth_after[2:-4]))
    assert (err <= 0.020).mean() >= 0.9
    # a pause of whole beats: asked about, and left exactly as it was
    y2, _, _ = _song_with_gap(2.0, bpm)
    path2 = tmp_path / "whole.wav"
    sf.write(path2, y2 / np.abs(y2).max(), SR)
    g2, a2 = pipeline._features(str(path2), bpm, early, len(y2) / SR, engine=_LoudEngine())
    assert a2["tempo_map"] is None and "pause_seat" not in a2["tempo_check"]


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def engine():
    return infer.StepManiaNeuralEngine(TINY)


def _paused_song(tmp_path):
    import soundfile as sf
    bpm, before, after = 126.0, 112, 112
    gap = 5.5 * 15.0 / bpm
    a, b = drum_loop(bpm, before), drum_loop(bpm, after, seed=8)
    y = np.concatenate([a, np.zeros(int(round(gap * SR)), np.float32), b[int(OFFSET * SR):]])
    path = tmp_path / "paused.wav"
    sf.write(path, y / np.abs(y).max(), SR)
    cut = OFFSET + before * 60.0 / bpm
    truth_after = cut + gap + np.arange(after) * 60.0 / bpm
    # the analysis the DSP would hand over: one tempo, on its early clock
    an = {"bpm": bpm, "offset": OFFSET - pipeline.DSP_EARLY_S["local_dsp"],
          "duration_s": len(y) / SR, "beat_backend": "local_dsp", "onsets": [],
          "sections": [], "energy_curve": [], "meter": 4}
    return path, an, bpm, truth_after


def test_features_are_warped_on_the_map(tmp_path):
    path, an, bpm, truth_after = _paused_song(tmp_path)
    g, align = pipeline._features(str(path), an["bpm"], an["offset"], an["duration_s"])
    assert align["tempo_map"] is not None and align["tempo_map"]["stops"]
    assert align["tempo_check"]["events"][0]["kind"] == "pause"
    assert align["bpm"] == pytest.approx(bpm, rel=2e-3)
    # every true beat after the pause has a 16th-note tick of the grid on it
    ticks = g.tick_times_sec[::3]
    j = np.clip(np.searchsorted(ticks, truth_after[2:-4]), 1, len(ticks) - 1)
    err = np.minimum(np.abs(ticks[j] - truth_after[2:-4]),
                     np.abs(ticks[j - 1] - truth_after[2:-4]))
    assert (err <= 0.020).mean() >= 0.9
    # the stop shows in the tick lengths the networks are told
    assert g.dt_ms.max() > 1.5 * np.median(g.dt_ms)
    assert g.X.shape == (len(g.tick_times_sec), 64) and np.isfinite(g.X).all()


def test_a_paused_song_is_charted_and_written_with_its_stop(engine, tmp_path):
    import simfile
    from simfile.timing import TimingData
    from simfile.timing.engine import TimingEngine
    path, an, bpm, truth_after = _paused_song(tmp_path)
    out = tmp_path / "song"
    report = pipeline.chart_audio(str(path), an, out, {"title": "Paused", "artist": "t",
                                                       "music": "paused.wav"},
                                  ("easy", "hard"), engine, previews=False,
                                  log=lambda *a, **k: None)
    assert report["tempo_map"]["stops"] and report["engine"]["tempo_check"]["events"]
    text = Path(report["files"]["ssc"]).read_text()
    assert "#STOPS:" in text and "#STOPS:;" not in text
    assert Path(report["files"]["sm"]).read_text().count("=") >= 2
    # the file's own timing puts its notes where the engine put them
    sf = simfile.loads(text)
    eng = TimingEngine(TimingData(sf))
    mapped = T.TimingMap(report["offset"], report["tempo_map"]["bpms"],
                         report["tempo_map"]["stops"])
    for beat in np.arange(0.0, 400.0, 7.25):
        assert float(eng.time_at(float(beat))) == pytest.approx(float(mapped.time_at(beat)),
                                                               abs=2e-3)
    # and a section of it can be re-rolled: the stop is read back from the file
    res = pipeline.inpaint_song("easy", (8, 12), ssc=report["files"]["ssc"], engine=engine,
                                seed=5)
    assert res["files"] and "#STOPS:" in Path(report["files"]["ssc"]).read_text()
