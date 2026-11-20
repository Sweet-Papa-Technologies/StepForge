"""REQ-SFV15-01 / REQ-SFV15-02 — the 4-stem cache, stem channels in the corpus,
and leak-free, timing-exact acoustic augmentation."""
from __future__ import annotations

import json

import numpy as np
import pytest

from beatforge.adapters.stepmania.neural import codec as C
from beatforge.adapters.stepmania.neural import grid_features as GF
from beatforge.adapters.stepmania.neural.synth import audio_augment as AUG
from beatforge.adapters.stepmania.neural.synth import shards, stems_cache
from beatforge.adapters.stepmania.neural.train import corpus

import synth_helpers as H

SR = GF.SR
quiet = dict(log=lambda *_: None)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("aug")
    r = H.make_corpus_root(tmp, train=2, val=1)
    AUG.augment_corpus(r / "real", tmp / "stems", r / "aug_stem", workers=1, **quiet)
    return r


# --------------------------------------------------------------------------- #
# REQ-SFV15-01: stems
# --------------------------------------------------------------------------- #
def test_stem_channels_are_populated_with_stems_and_exactly_zero_without():
    y = H.clicks(bars=8)
    timing = GF.TimingMap.constant(H.BPM, H.OFFSET_S)
    stems = {n: (y * g).astype(np.float32) for n, g in H.STEM_SPLIT.items()}
    with_stems = GF.extract_grid_features(y, timing, stems=stems)
    without = GF.extract_grid_features(y, timing)
    assert with_stems.has_stems and not without.has_stems
    block = with_stems.X[:, GF.STEM]
    assert block.shape[1] == 16 and np.isfinite(block).all()
    for i in range(4):                               # every stem has level and flux
        assert np.abs(block[:, 4 * i:4 * i + 4]).max() > 0
    assert not without.X[:, GF.STEM].any()
    # nothing outside the stem block depends on the stems
    other = np.r_[0:GF.STEM.start, GF.STEM.stop:GF.D_IN]
    assert np.array_equal(with_stems.X[:, other], without.X[:, other])
    assert np.isfinite(with_stems.X).all() and np.isfinite(without.X).all()


def test_stem_cache_round_trip_keeps_length_and_level(tmp_path):
    mix = H.clicks(bars=8)
    stems = {n: (mix * g).astype(np.float32) for n, g in H.STEM_SPLIT.items()}
    row = stems_cache.save_stems(tmp_path, "k1", mix, stems, {"folder": "f"})
    assert row["n_samples"] == len(mix) and row["snr_db"] > 100       # they sum to the mix
    got = stems_cache.load_stems(tmp_path, "k1", n_samples=len(mix), match_mix=mix)
    assert set(got) == set(GF.STEM_NAMES)
    for name in GF.STEM_NAMES:
        assert len(got[name]) == len(mix)                             # +-0 samples
        assert np.array_equal(got[name], stems[name])
    assert stems_cache.load_stems(tmp_path, "absent") is None
    assert stems_cache.validate(tmp_path) == []
    assert stems_cache.write_manifest(tmp_path)["n_songs"] == 1


def test_stem_cache_refuses_drift_and_foreign_audio(tmp_path):
    mix = H.clicks(bars=8)
    stems = {n: (mix * g).astype(np.float32) for n, g in H.STEM_SPLIT.items()}
    stems_cache.save_stems(tmp_path, "k1", mix, stems, {})
    one_ms = stems_cache.MAX_LENGTH_DRIFT
    # within a millisecond: padded / trimmed to the exact length asked for
    got = stems_cache.load_stems(tmp_path, "k1", n_samples=len(mix) - one_ms)
    assert all(len(v) == len(mix) - one_ms for v in got.values())
    with pytest.raises(ValueError, match="samples off"):
        stems_cache.load_stems(tmp_path, "k1", n_samples=len(mix) + one_ms + 1)
    other = H.clicks(bars=8, bpm=97.0, seed=5)[:len(mix)]
    with pytest.raises(ValueError, match="do not match"):
        stems_cache.load_stems(tmp_path, "k1", n_samples=len(other), match_mix=other)
    with pytest.raises(ValueError, match="samples"):
        stems_cache.save_stems(tmp_path, "k2", mix, {**stems, "bass": stems["bass"][:-5]}, {})


def test_corpus_compiled_with_a_stem_cache_stores_stem_channels(root):
    for key in H.real_keys(root, "train") + H.real_keys(root, "val"):
        meta, arrays = corpus.load_song(root / "real" / "songs" / f"{key}.npz")
        assert meta["has_stems"] and meta["source_type"] == "real"
        st = arrays["stems"].astype(np.float32)
        assert st.shape == (meta["n_ticks"], 16) and np.isfinite(st).all() and st.any()
        # the timing block reproduces the grid the shard was compiled on
        timing = corpus.timing_from_facts(meta["timing"])
        assert timing.time_at(0.0) == pytest.approx(meta["tick0_time_s"], abs=1e-9)


def test_a_song_missing_from_the_stem_cache_compiles_without_stems(tmp_path):
    H.make_song_folder(tmp_path / "packs", "solo")
    (tmp_path / "stems").mkdir()
    corpus.compile_corpus(tmp_path / "packs", tmp_path / "c", workers=1,
                          stems_dir=str(tmp_path / "stems"), **quiet)
    meta, arrays = corpus.load_song(next((tmp_path / "c" / "songs").glob("*.npz")))
    assert not meta["has_stems"] and "stems" not in arrays


def test_locked_split_copies_ids_splits_and_verdicts(root, tmp_path):
    import shutil
    real = tmp_path / "real"                         # the module fixture is shared
    shutil.copytree(root / "real", real)
    locked = json.loads((real / "manifest.json").read_text())
    locked["songs"][0]["charts"][0].update(keep=False, reject="by_hand")
    for s in locked["songs"]:
        s["song_id"] = "locked_" + s["key"]
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(locked))
    fresh = corpus.build_manifest(real, split_manifest=lock_path, **quiet)
    by_key = {s["key"]: s for s in fresh["songs"]}
    for s in locked["songs"]:
        got = by_key[s["key"]]
        assert (got["song_id"], got["split"]) == (s["song_id"], s["split"])
        assert [c["keep"] for c in got["charts"]] == [c["keep"] for c in s["charts"]]
    assert fresh["locked_split"] and fresh["rejections"] == {"by_hand": 1}
    assert sum(v["charts"] for v in fresh["splits"].values()) == 14
    assert fresh["n_charts_kept"] == 14
    # a chart that changed since the lock is an error, never a silent guess
    locked["songs"][1]["charts"][2]["n_events"] += 1
    with pytest.raises(ValueError, match="no longer matches"):
        corpus.apply_locked_split(json.loads(json.dumps(fresh)), locked)
    locked["songs"][1]["charts"][2]["n_events"] -= 1
    # ...and so is a locked song that was not compiled, unless allowed
    locked["songs"].append(dict(locked["songs"][0], key="ghost",
                                charts=[dict(locked["songs"][0]["charts"][1])]))
    with pytest.raises(ValueError, match="were not compiled"):
        corpus.apply_locked_split(json.loads(json.dumps(fresh)), locked)
    corpus.apply_locked_split(json.loads(json.dumps(fresh)), locked, allow_missing=True)


# --------------------------------------------------------------------------- #
# REQ-SFV15-02 (1): no validation song is ever augmented
# --------------------------------------------------------------------------- #
def test_only_training_songs_are_augmented_and_ids_are_inherited(root):
    real = json.loads((root / "real" / "manifest.json").read_text())
    aug = json.loads((root / "aug_stem" / "manifest.json").read_text())
    train_ids = {s["key"]: s["song_id"] for s in real["songs"] if s["split"] == "train"}
    val_keys = set(H.real_keys(root, "val"))
    assert len(aug["songs"]) == AUG.VARIANTS * len(train_ids)
    for s in aug["songs"]:
        assert s["split"] == "train"
        assert s["parent"]["source"] == "real" and s["parent"]["key"] in train_ids
        assert s["parent"]["key"] not in val_keys
        assert s["song_id"] == train_ids[s["parent"]["key"]]
    assert shards.check_no_leak(root, **quiet)["aug_stem"] == len(aug["songs"])


def test_leak_check_catches_a_validation_song_in_a_derived_source(root, tmp_path):
    import shutil
    bad = tmp_path / "corpus"
    shutil.copytree(root, bad)
    m = json.loads((bad / "aug_stem" / "manifest.json").read_text())
    m["songs"][0]["parent"]["key"] = H.real_keys(root, "val")[0]
    (bad / "aug_stem" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="not a training song"):
        shards.check_no_leak(bad, **quiet)
    m["songs"][0]["split"] = "val"
    (bad / "aug_stem" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="outside the training split"):
        shards.check_no_leak(bad, **quiet)


# --------------------------------------------------------------------------- #
# REQ-SFV15-02 (2): a tempo change is exact in time
# --------------------------------------------------------------------------- #
def _click_track(bpm: float, beats: int, offset: float = 0.3):
    y = np.zeros(int((offset + beats * 60.0 / bpm + 0.5) * SR), np.float32)
    rng = np.random.default_rng(1)
    env = np.exp(-np.arange(int(0.03 * SR)) / (0.004 * SR)).astype(np.float32)
    times = offset + np.arange(beats) * 60.0 / bpm
    for t in times:
        i = int(round(t * SR))
        y[i:i + len(env)] += env * rng.standard_normal(len(env)).astype(np.float32)
    return (y / np.abs(y).max()).astype(np.float32), times


def _onsets(y: np.ndarray) -> np.ndarray:
    import scipy.signal
    k = int(0.002 * SR)
    env = np.convolve(np.abs(y), np.ones(k) / k, "same")
    peaks, _ = scipy.signal.find_peaks(env, height=0.25 * env.max(), distance=int(0.1 * SR))
    out = []
    for p in peaks:
        a = max(0, p - int(0.03 * SR))
        out.append((a + int(np.argmax(env[a:p + 1] >= 0.3 * env[p]))) / SR)
    return np.array(out)


BACKENDS = ["resample"] + (["rubberband"] if AUG.have_rubberband() else [])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rate", [1.06, 0.93])
def test_stretched_click_track_stays_on_the_grid(backend, rate):
    bpm, offset = 128.0, 0.3
    y, times = _click_track(bpm, 48, offset)
    z, used = AUG.time_stretch(y, rate, backend)
    assert used == backend and len(z) == int(round(len(y) / rate))
    # every click is where the rescaled timing says, within 3 ms
    got = _onsets(z)
    assert len(got) == len(times)
    assert np.abs(got - _onsets(y) / rate).max() <= 0.003
    # on the rescaled grid (BPM' = r BPM, OFFSET' = OFFSET / r) the flux peaks
    # sit on the quarter-note ticks, and a tick lasts 1/r as long
    timing = GF.TimingMap.constant(bpm * rate, offset / rate)
    base = GF.TimingMap.constant(bpm, offset)
    n = 48 * C.TICKS_PER_BEAT
    core, tick_times, dt = GF.warp_core(GF.frame_features(z), timing, n)
    _, _, dt0 = GF.warp_core(GF.frame_features(y), base, n)
    assert np.allclose(dt / dt0, 1.0 / rate, rtol=1e-3)
    # ...where "sits on" is the tick the flux rises on: a stretched click can
    # ring into the next tick with an all but equal peak, which is not a move
    flux = core[:, 20:26].sum(1)
    for beat in range(1, 47):
        k = beat * C.TICKS_PER_BEAT
        near = flux[k - 5:k + 6]
        assert int(np.argmax(near >= 0.5 * near.max())) - 5 == 0, (backend, rate, beat)
        assert flux[k] >= 0.9 * near.max()
    assert np.abs(tick_times[::C.TICKS_PER_BEAT][:len(got)] - got).max() <= 0.003


def test_tempo_variants_rescale_time_but_not_ticks(root):
    key = H.real_keys(root, "train")[0]
    pm, pa = corpus.load_song(root / "real" / "songs" / f"{key}.npz")
    for k in (4, 5):
        m, a = corpus.load_song(root / "aug_stem" / "songs" / f"{AUG.aug_key(key, k)}.npz")
        r = m["recipe"]["rate"]
        assert (0.92 <= r <= 0.96) if k == 4 else (1.04 <= r <= 1.08)
        assert m["kind"] == "tempo" and m["n_ticks"] == pm["n_ticks"]
        assert m["duration_s"] == pytest.approx(pm["duration_s"] / r, abs=0.02)
        assert m["timing"]["bpms"][0][1] == pytest.approx(pm["timing"]["bpms"][0][1] * r)
        assert m["timing"]["offset_s"] == pytest.approx(pm["timing"]["offset_s"] / r)
        assert np.allclose(a["dt_ms"] / pa["dt_ms"], 1.0 / r, rtol=1e-3)
        for c in m["charts"]:
            j, pj = c["j"], c["parent_j"]
            assert np.array_equal(a[f"c{j}_tick"], pa[f"c{pj}_tick"])
            assert np.allclose(a[f"c{j}_tms"], pa[f"c{pj}_tms"] / r, atol=0.05)
            src = next(x for x in pm["charts"] if x["j"] == pj)
            assert c["difficulty"] == src["difficulty"]
            # density is measured again on the new timing, not copied
            span, span0 = np.ptp(a[f"c{j}_tms"]), np.ptp(pa[f"c{pj}_tms"])
            assert span0 / span == pytest.approx(r, rel=1e-3)
            if src["difficulty"] in ("hard", "challenge"):
                assert (a[f"c{j}_style"][1] > pa[f"c{pj}_style"][1]) == (r > 1)


def test_every_variant_is_a_finite_shard_with_all_channels(root):
    key = H.real_keys(root, "train")[1]
    pm, pa = corpus.load_song(root / "real" / "songs" / f"{key}.npz")
    kinds = []
    for k in range(AUG.VARIANTS):
        m, a = corpus.load_song(root / "aug_stem" / "songs" / f"{AUG.aug_key(key, k)}.npz")
        kinds.append(m["kind"])
        assert m["source_type"] == "aug_stem" and m["variant"] == k and m["has_stems"]
        assert a["core"].shape == pa["core"].shape and a["stems"].shape == pa["stems"].shape
        assert np.isfinite(a["core"].astype(np.float32)).all()
        assert np.isfinite(a["stems"].astype(np.float32)).all() and a["stems"].any()
        assert len(m["charts"]) == len(pm["charts"])
        # the notes still sit on the transients of the re-rendered audio
        flux = a["core"].astype(np.float32)[:, 20:26].sum(1)
        tick = a["c0_tick"]
        assert flux[tick].mean() > 2.0 * flux.mean(), (k, m["kind"])
    assert tuple(kinds) == AUG.KINDS


# --------------------------------------------------------------------------- #
# recipes
# --------------------------------------------------------------------------- #
def test_recipes_are_deterministic_bounded_and_distinct():
    seen = set()
    for key in ("a", "b", "c", "d", "e", "f"):
        for k in range(AUG.VARIANTS):
            r = AUG.recipe(key, k, seed=3)
            assert r == AUG.recipe(key, k, seed=3)
            seen.add(json.dumps(r, sort_keys=True))
            for name, g in r["gains"].items():
                lo, hi = AUG.GAIN_RANGE[name]
                assert min(lo, 0.8) - 1e-9 <= g <= max(hi, 1.2) + 1e-9
            if r["kind"] == "tempo":
                assert abs(r["rate"] - 1.0) <= 0.08 + 1e-9 and r["rate"] != 1.0
            else:
                assert r["rate"] == 1.0
            if r["kind"] == "pitch":
                assert 1 <= abs(r["pitch"]) <= 3 and max(map(abs, r["tilt_db"])) <= 4.0
        assert AUG.recipe(key, 0)["gains"]["vocals"] == 0.0         # the instrumental
    assert len(seen) == 6 * AUG.VARIANTS
    with pytest.raises(ValueError):
        AUG.recipe("a", AUG.VARIANTS)


def test_drums_are_never_pitch_shifted(monkeypatch):
    shifted = []
    monkeypatch.setattr(AUG, "pitch_shift",
                        lambda y, st, backend="auto": (shifted.append(float(st)) or y, "fake"))
    y = H.clicks(bars=4)
    stems = {n: (y * g).astype(np.float32) for n, g in H.STEM_SPLIT.items()}
    rec = dict(AUG.recipe("song", 2), tilt_db=[0.0, 0.0], drive=0.0)
    mix, out, info = AUG.render(stems, rec)
    assert len(shifted) == len(AUG.HARMONIC) == 3 and info["pitch"] == "fake"
    assert "drums" not in AUG.HARMONIC
    assert np.abs(mix).max() == pytest.approx(1.0)
    # one gain for the mix and its stems: they still sum to it
    assert np.allclose(np.sum([out[n] for n in GF.STEM_NAMES], axis=0), mix, atol=1e-5)


def test_tonal_tilt_and_soft_clip_do_not_move_a_transient():
    y, _ = _click_track(120.0, 16)
    before = _onsets(y)
    for z in (AUG.tonal_tilt(y, 4.0, -4.0), AUG.tonal_tilt(y, -4.0, 4.0),
              AUG.soft_clip(y, 2.5), AUG.lowpass(y, 7000.0)):
        after = _onsets(z)
        assert len(after) == len(before) and np.abs(after - before).max() <= 0.001


def test_lossy_roundtrip_is_realigned_to_its_input():
    y, _ = _click_track(120.0, 16)
    z, codec = AUG.lossy_roundtrip(y, 64)
    if z is None:
        pytest.skip("ffmpeg with an MP3 encoder is not available")
    assert codec == "mp3" and len(z) == len(y)
    assert np.abs(_onsets(z) - _onsets(y)).max() <= 0.002
