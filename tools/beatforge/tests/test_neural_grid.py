"""REQ-SFN-02 — the beat-warped 48-tick feature tensor."""
from __future__ import annotations

import numpy as np
import pytest

from beatforge.adapters.stepmania.neural import grid_features as GF


def click_track(bpm: float, dur_s: float = 9.0, offset_s: float = 0.25,
                seed: int = 1) -> np.ndarray:
    """Short broadband clicks on every beat."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(dur_s * GF.SR), np.float32)
    beat = 0
    while True:
        i = int(round((offset_s + beat * 60.0 / bpm) * GF.SR))
        if i + 400 >= len(y):
            break
        y[i:i + 40] += np.hanning(80)[40:].astype(np.float32) * rng.standard_normal(40)
        beat += 1
    return y


@pytest.mark.parametrize("bpm", [120.0, 175.0])
def test_click_track_flux_lands_on_beat_ticks(bpm):
    timing = GF.TimingMap.constant(bpm, 0.25)
    g = GF.extract_grid_features(click_track(bpm), timing)
    t = len(g.X)
    assert g.X.shape == (t, 64) and g.X.dtype == np.float32
    assert g.tick_times_sec.shape == (t,) and g.dt_ms.shape == (t,)
    assert t % 48 == 0
    assert np.isfinite(g.X).all()
    flux = g.X[:, GF.FLUX].sum(1)
    # every interior beat: the flux maximum of the surrounding beat-wide window
    # sits exactly on the tick with k % 12 == 0
    n_beats = 0
    for k in range(12, t - 24, 12):
        if g.tick_times_sec[k] > 8.0:                 # past the last click
            break
        window = flux[k - 6:k + 6]
        assert int(window.argmax()) == 6, f"beat tick {k}: peak off by {window.argmax() - 6}"
        assert flux[k] > 4 * np.delete(window, 6).max()
        n_beats += 1
    assert n_beats >= 12
    np.testing.assert_allclose(g.dt_ms, 60000.0 / bpm / 12, rtol=1e-5)


def test_shapes_identical_with_and_without_stems():
    bpm = 128.0
    y = click_track(bpm, dur_s=6.0)
    timing = GF.TimingMap.constant(bpm, 0.25)
    plain = GF.extract_grid_features(y, timing)
    stems = {"drums": y, "bass": np.zeros_like(y), "vocals": np.zeros_like(y),
             "other": 0.1 * y}
    with_stems = GF.extract_grid_features(y, timing, stems=stems)
    assert plain.X.shape == with_stems.X.shape
    assert not plain.has_stems and with_stems.has_stems
    assert np.all(plain.X[:, GF.STEM] == 0)
    assert np.abs(with_stems.X[:, GF.STEM]).sum() > 0
    assert np.all(with_stems.X[:, 36:40] == 0)        # the silent bass stem
    # every non-stem channel is unaffected by supplying stems
    keep = np.ones(64, bool)
    keep[GF.STEM] = False
    np.testing.assert_array_equal(plain.X[:, keep], with_stems.X[:, keep])
    assert np.isfinite(with_stems.X).all()


def test_channel_layout_covers_sixty_four_channels():
    blocks = (GF.MEL, GF.FLUX, GF.DFLUX, GF.STEM, GF.CHROMA, GF.METRIC)
    idx = np.concatenate([np.arange(64)[b] for b in blocks])
    assert sorted(idx.tolist()) == list(range(64))
    assert [len(np.arange(64)[b]) for b in blocks] == [20, 6, 6, 16, 6, 10]


def test_metric_channels_encode_subdivision_and_phase():
    dt = np.full(96, 40.0, np.float32)
    m = GF.metric_features(96, dt)
    onehot = m[:, :6]
    assert np.all(onehot.sum(1) == 1)
    cls = onehot.argmax(1)
    assert cls[0] == 0 and cls[12] == 0                # quarter
    assert cls[6] == 1                                  # eighth
    assert cls[4] == 2 and cls[8] == 2                  # triplet
    assert cls[3] == 3 and cls[9] == 3                  # sixteenth
    assert cls[2] == 4 and cls[10] == 4                 # twenty-fourth
    assert cls[1] == 5 and cls[11] == 5                 # forty-eighth
    np.testing.assert_allclose(m[:, 8], 0.0, atol=1e-6)            # log(40/40)
    np.testing.assert_allclose(m[0, 6:8], [0.0, 1.0], atol=1e-6)   # phase 0
    np.testing.assert_allclose(m[48, 6:8], [1.0, 0.0], atol=1e-6)  # one bar in = 1/4
    assert m[0, 9] == 0 and 0.98 < m[-1, 9] < 1.0


def test_timing_map_handles_bpm_changes_and_stops():
    tm = GF.TimingMap(0.5, [(0, 120.0), (8, 240.0)], stops=[(4, 0.5)])
    np.testing.assert_allclose(tm.time_at([0, 4]), [0.5, 2.5])
    assert tm.time_at(4.0001) == pytest.approx(3.00005)     # the stop follows beat 4
    np.testing.assert_allclose(tm.time_at([8, 12]), [5.0, 6.0])
    assert not tm.is_constant and GF.TimingMap.constant(120, 0).is_constant
    np.testing.assert_allclose(tm.bpm_at([0, 7.9, 8, 20]), [120, 120, 240, 240])
    # a delay pauses BEFORE the note on its beat
    td = GF.TimingMap(0.0, [(0, 120.0)], delays=[(4, 0.25)])
    np.testing.assert_allclose(td.time_at([3.999, 4.0]), [1.9995, 2.25], atol=1e-6)
    shifted = tm.shifted(0.02)
    assert shifted.time_at(0) == pytest.approx(0.52) and tm.time_at(0) == pytest.approx(0.5)


def test_dt_ms_tracks_tempo_and_clamps():
    tm = GF.TimingMap(0.0, [(0, 120.0), (4, 180.0)], stops=[(6, 2.0)])
    y = np.zeros(int(8 * GF.SR), np.float32)
    g = GF.extract_grid_features(y, tm, n_ticks=96)
    np.testing.assert_allclose(g.dt_ms[:47], 60000 / 120 / 12, rtol=1e-5)
    np.testing.assert_allclose(g.dt_ms[49:71], 60000 / 180 / 12, rtol=1e-5)
    assert g.dt_ms[72] == 500.0                         # the stop, clamped
    assert g.dt_ms.min() >= 10.0 and g.dt_ms.max() <= 500.0
    # silence: finite features, flux exactly zero
    assert np.isfinite(g.X).all() and np.all(g.X[:, GF.FLUX] == 0)


def test_ticks_cover_the_audio_in_whole_measures():
    tm = GF.TimingMap.constant(150.0, 0.1)
    n = tm.ticks_for_duration(30.0)
    assert n % 48 == 0
    assert tm.time_at((n - 1) / 12) >= 30.0 - 60 / 150 / 12
    assert tm.time_at((n - 48) / 12) < 30.0


def test_assemble_rebuilds_x_from_the_storable_core():
    bpm = 140.0
    timing = GF.TimingMap.constant(bpm, 0.25)
    g = GF.extract_grid_features(click_track(bpm, dur_s=5.0), timing)
    assert g.core.shape == (len(g.X), GF.D_CORE)
    rebuilt = GF.assemble(g.core.astype(np.float16).astype(np.float32), g.dt_ms)
    np.testing.assert_allclose(rebuilt, g.X, atol=2e-2)   # float16 storage round-off


def test_estimate_sync_recovers_a_known_offset():
    bpm = 150.0
    y = click_track(bpm, dur_s=12.0, offset_s=0.25)
    ff = GF.frame_features(y)
    true_times = 0.25 + np.arange(24) * 60.0 / bpm
    good = GF.estimate_sync(ff.onset_envelope, true_times)
    assert abs(good["shift_s"]) <= 0.006
    assert good["contrast"] > 2.0
    late = GF.estimate_sync(ff.onset_envelope, true_times - 0.024)   # chart 24 ms early
    assert late["shift_s"] == pytest.approx(0.024, abs=0.006)
    # notes that ignore the audio score no better than their surroundings
    rng = np.random.default_rng(0)
    junk = GF.estimate_sync(ff.onset_envelope, np.sort(rng.uniform(0.3, 9.0, 60)))
    assert junk["contrast"] < good["contrast"] / 2
