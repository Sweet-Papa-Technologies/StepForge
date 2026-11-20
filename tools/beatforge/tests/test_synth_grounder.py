"""Seating programmatic step patterns on training audio (audio_grounder.py)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from beatforge.adapters.stepmania.neural import codec as C
from beatforge.adapters.stepmania.neural import conditioning as cond
from beatforge.adapters.stepmania.neural.synth import audio_grounder as AG
from beatforge.adapters.stepmania.neural.synth import shards, style_patterns
from beatforge.adapters.stepmania.neural.train import corpus

import synth_helpers as H

quiet = dict(log=lambda *_: None)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = H.make_corpus_root(tmp_path_factory.mktemp("ground"), train=3, val=1)
    style_patterns.generate(r / "synth_style", total_charts=24, seed=7, workers=1, **quiet)
    AG.ground_corpus(r, share=0.5, seed=3, **quiet)
    return r


def test_the_asked_share_of_every_archetype_is_grounded_and_the_rest_stay_symbolic(root):
    style = json.loads((root / "synth_style" / "manifest.json").read_text())
    grounded = json.loads((root / "synth_style_grounded" / "manifest.json").read_text())
    flags = {s["key"]: s["grounded"] for s in style["songs"]}
    assert sum(flags.values()) == len(grounded["songs"]) == style["n_grounded"]
    by_arch: dict = {}
    for s in style["songs"]:
        by_arch.setdefault(s["archetype"], []).append(s["grounded"])
    assert len(by_arch) == len(style_patterns.ARCHETYPES)
    for arch, got in by_arch.items():
        assert sum(got) == round(0.5 * len(got)), arch
    assert {s["key"] for s in grounded["songs"]} == {k + "__g" for k, v in flags.items() if v}


def test_grounded_patterns_sit_on_training_audio_and_carry_its_song_id(root):
    real = json.loads((root / "real" / "manifest.json").read_text())
    train = {s["key"]: s["song_id"] for s in real["songs"] if s["split"] == "train"}
    val = set(H.real_keys(root, "val"))
    for f in shards.shard_files(root / "synth_style_grounded"):
        meta, arrays = corpus.load_song(f)
        par = meta["parent"]
        assert par["source"] == "real" and par["key"] in train and par["key"] not in val
        assert meta["song_id"] == train[par["key"]]
        assert "core" not in arrays and not meta["has_audio"]          # attached
        pm, pa = corpus.load_song(root / "real" / "songs" / f"{par['key']}.npz")
        tick = arrays["c0_tick"]
        assert tick[0] >= meta["start_tick"] and meta["start_tick"] % C.G == 0
        assert tick[-1] < pm["n_ticks"]
        # timed on the song's own grid
        dt = pa["dt_ms"].astype(np.float64)
        times = pm["tick0_time_s"] + np.concatenate([[0.0], np.cumsum(dt[:-1])]) / 1000.0
        assert np.allclose(arrays["c0_tms"], times[tick] * 1000.0, atol=0.05)
    shards.check_no_leak(root, **quiet)


def test_a_grounded_pattern_keeps_its_steps_and_is_relabelled_on_its_new_timing(root):
    seen = 0
    for f in shards.shard_files(root / "synth_style_grounded"):
        meta, arrays = corpus.load_song(f)
        sm, sa = corpus.load_song(root / "synth_style" / "songs" / f"{meta['pattern']}.npz")
        # the same panels, bar-shifted; a pattern longer than the song is cut
        # at the song's end and nowhere else
        n = len(arrays["c0_tick"])
        assert n == len(sa["c0_tick"]) - meta["cut"] and n >= min(AG.MIN_EVENTS, n)
        assert np.array_equal(arrays["c0_raw"], sa["c0_raw"][:, :n])
        assert np.array_equal(arrays["c0_tick"] - meta["start_tick"], sa["c0_tick"][:n])
        assert arrays["c0_tick"][-1] < (meta["n_ticks"] // C.G) * C.G
        row = meta["charts"][0]
        assert row["lock_violations"] == 0 and row["spins"] == 0
        # label = measured, on the audio's tempo
        ev = AG.chart_events(arrays, 0)
        times_ms = arrays["c0_tms"].astype(np.float64)
        enc, _ = C.encode(ev, times_ms)
        assert np.array_equal(enc.token, arrays["c0_tok"])
        want = cond.compute_style_vector(enc, times_ms / 1000.0, row["meter"],
                                         has_stems=meta["has_stems"])
        assert np.allclose(arrays["c0_style"], want, atol=1e-6)
        ratio = meta["bpm_audio"] / meta["bpm_pattern"]
        assert arrays["c0_style"][1] == pytest.approx(sa["c0_style"][1] * ratio, rel=0.15)
        seen += 1
    assert seen > 0


def test_loud_archetypes_are_seated_on_louder_bars_than_casual_ones():
    rng = np.random.default_rng(0)
    rank = np.linspace(0.0, 1.0, 64).astype(np.float32)             # ever louder
    pool = [{"source": "real", "key": "k", "song_id": "s", "bpm": 150.0, "n_bars": 64,
             "rank": rank, "archetype": None}]
    by_bpm = np.array([150.0])

    def level(arch):
        got = []
        for _ in range(40):
            _, i, start = AG._window({"n_ticks": 16 * C.G, "bpm": 150.0, "archetype": arch},
                                     pool, by_bpm, rng)[0]
            got.append(float(rank[start:start + 16].mean()))
        return float(np.mean(got))

    assert level("itg_stamina") > level("crossover_flow") > level("casual_ddr_flow")
    # a pattern longer than every song starts at the song's first bar
    long = AG._window({"n_ticks": 80 * C.G, "bpm": 150.0, "archetype": "itg_stamina"},
                      pool, by_bpm, rng)
    assert long and all(start == 0 for _, _, start in long)
    short = [dict(pool[0], n_bars=AG.MIN_BARS - 1, rank=rank[:AG.MIN_BARS - 1])]
    assert AG._window({"n_ticks": 80 * C.G, "bpm": 150.0, "archetype": "itg_stamina"},
                      short, by_bpm, rng) == []
    # no song in tempo range: the nearest tempo is used rather than none
    far = AG._window({"n_ticks": 16 * C.G, "bpm": 90.0, "archetype": "casual_ddr_flow"},
                     pool, by_bpm, rng)
    assert far and far[0][1] == 0


def test_grounding_twice_replaces_the_first_run(root, tmp_path):
    import shutil
    again = tmp_path / "corpus"
    shutil.copytree(root, again)
    first = {f.name for f in shards.shard_files(again / "synth_style_grounded")}
    res = AG.ground_corpus(again, share=0.25, seed=3, **quiet)
    second = {f.name for f in shards.shard_files(again / "synth_style_grounded")}
    assert len(second) == res["grounded"] < len(first)
    m = json.loads((again / "synth_style" / "manifest.json").read_text())
    assert sum(s["grounded"] for s in m["songs"]) == len(second)
