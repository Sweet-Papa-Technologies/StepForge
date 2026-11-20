"""REQ-SFV15-04 — Beginner / Easy charts thinned from the densest human chart:
a strict subset of its notes, on the tier's ticks, with its rests intact."""
from __future__ import annotations

import json

import numpy as np
import pytest

from beatforge.adapters.stepmania.grammar import BUDGETS
from beatforge.adapters.stepmania.neural import codec as C
from beatforge.adapters.stepmania.neural import conditioning as cond
from beatforge.adapters.stepmania.neural.synth import chart_thinning as TH
from beatforge.adapters.stepmania.neural.synth import shards
from beatforge.adapters.stepmania.neural.train import corpus

import synth_helpers as H

quiet = dict(log=lambda *_: None)
GRID = {"beginner": 12, "easy": 6}


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = H.make_corpus_root(tmp_path_factory.mktemp("thin"), train=3, val=1)
    TH.thin_corpus(r / "real", r / "synth_thin", **quiet)
    return r


def thinned(root):
    for f in shards.shard_files(root / "synth_thin"):
        meta, arrays = corpus.load_song(f)
        parent = corpus.load_song(root / "real" / "songs" / f"{meta['parent']['key']}.npz")
        for c in meta["charts"]:
            yield meta, arrays, parent, c


def test_every_training_song_gets_both_tiers_and_no_validation_song_does(root):
    manifest = json.loads((root / "synth_thin" / "manifest.json").read_text())
    parents = {s["parent"]["key"] for s in manifest["songs"]}
    assert parents == set(H.real_keys(root, "train"))
    assert not parents & set(H.real_keys(root, "val"))
    assert all(s["split"] == "train" for s in manifest["songs"])
    assert all([c["difficulty"] for c in s["charts"]] == ["beginner", "easy"]
               for s in manifest["songs"])
    shards.check_no_leak(root, **quiet)


def test_thinned_notes_are_a_subset_of_the_parent_on_legal_ticks(root):
    seen = 0
    for meta, arrays, (pm, pa), c in thinned(root):
        j, tier = c["j"], c["difficulty"]
        tick = arrays[f"c{j}_tick"]
        src_tick = pa[f"c{c['source_j']}_tick"]
        assert np.isin(tick, src_tick).all()                    # nothing invented
        assert (tick % GRID[tier] == 0).all()                   # the tier's grid
        assert len(np.unique(tick)) == len(tick) < len(src_tick)
        # each note keeps one of the panels the human chart had on that tick
        src = TH.chart_events(pa, c["source_j"])
        ev = TH.chart_events(arrays, j)
        at = {int(t): {int(src.p0[i]), int(src.p1[i])} for i, t in enumerate(src.tick)}
        for i, t in enumerate(ev.tick):
            assert int(ev.p0[i]) in at[int(t)]
            assert int(ev.p1[i]) == -1 or int(ev.p1[i]) in at[int(t)]
        assert not len(ev.mines)
        assert c["source_difficulty"] in ("hard", "challenge")
        assert c["lock_violations"] == 0 and c["spins"] == 0
        seen += 1
    assert seen == 6


def test_parent_rests_of_a_bar_or_more_survive(root):
    for meta, arrays, (pm, pa), c in thinned(root):
        tick = arrays[f"c{c['j']}_tick"]
        src_tick = pa[f"c{c['source_j']}_tick"]
        gaps = np.nonzero(np.diff(src_tick) >= 4 * C.TICKS_PER_BEAT)[0]
        assert len(gaps) >= 1                                   # the fixture has a rest
        for g in gaps:
            a, b = src_tick[g], src_tick[g + 1]
            assert not ((tick > a) & (tick < b)).any()          # 100 % preserved


def test_density_and_tier_rules(root):
    by_parent: dict = {}
    for meta, arrays, (pm, pa), c in thinned(root):
        j, tier = c["j"], c["difficulty"]
        budget = BUDGETS[tier]
        times_s = arrays[f"c{j}_tms"] / 1000.0
        assert cond.peak_nps(times_s) <= budget.max_nps_4s + 1e-9
        tick = arrays[f"c{j}_tick"]
        cand = pa[f"c{c['source_j']}_tick"]
        cand = cand[cand % GRID[tier] == 0]
        lo, hi = TH.KEEP_SHARE[tier][2:]
        assert lo - 0.02 <= len(tick) / len(cand) <= hi + 0.02
        ev = TH.chart_events(arrays, j)
        jumps = int((ev.p1 >= 0).sum())
        assert jumps <= np.floor(TH.JUMP_SHARE[tier] * len(tick))
        assert (ev.tick[ev.p1 >= 0] % 24 == 0).all()            # only beats 1 and 3
        assert not (ev.k0 == C.ROLL).any()
        assert c["hold_share"] <= budget.hold_share[1] + 1e-9
        if tier == "easy":
            off = tick % 12 != 0
            assert off.sum() <= np.floor(budget.fine_frac * len(tick))
        assert c["companion"] is True                           # the fixture has both tiers
        lo_m, hi_m = budget.meter_range
        assert lo_m <= c["meter"] <= hi_m
        by_parent.setdefault(meta["parent"]["key"], {})[tier] = len(tick)
    for counts in by_parent.values():
        assert counts["beginner"] < counts["easy"]


def test_salience_prefers_strong_beats_and_real_transients():
    n_bars = 8
    core = np.zeros((n_bars * 48, 31), np.float32)
    core[::12, 23:26] = 1.0                                     # percussion on the beats
    ticks = np.array([0, 6, 12, 24, 36, 48])
    s = TH.salience(ticks, core, None)
    assert s[0] == s[5] > s[3] > s[2] == s[4] > s[1]            # 1 > 3 > 2,4 > off-beat
    # the same metric position scores higher where a drum actually hits
    core2 = core.copy()
    core2[24, 23:26] = 0.0
    assert TH.salience(ticks, core2, None)[3] < s[3]
    # with stems, the drum stem's flux is what counts
    stems = np.zeros((n_bars * 48, 16), np.float32)
    stems[6, 1:4] = 5.0
    flat = np.zeros_like(core)
    assert (TH.salience(ticks, flat, stems)[1]
            > TH.salience(ticks, flat, np.zeros_like(stems))[1])


def test_select_respects_the_window_cap_and_run_limit():
    # 16 bars of 8th notes at 180 BPM: 6 notes/s, far above either tier
    ticks = np.arange(0, 16 * 48, 6)
    times = ticks * (60.0 / 180.0 / 12.0)
    score = np.where(ticks % 12 == 0, 0.9, 0.8).astype(np.float64)
    for tier in TH.TIERS:
        keep = TH.select(ticks, times, score, tier, 0.7)
        kt, ks = ticks[keep], times[keep]
        assert (kt % GRID[tier] == 0).all()
        assert cond.peak_nps(ks) <= BUDGETS[tier].max_nps_4s + 1e-9
        if tier == "easy":
            run = longest = 1
            for d in np.diff(kt):
                run = run + 1 if d <= 6 else 1
                longest = max(longest, run)
            assert longest <= BUDGETS[tier].max_run + 1
    assert len(TH.select(ticks[ticks % 12 == 6], times[ticks % 12 == 6],
                         score[ticks % 12 == 6], "beginner", 0.7)) == 0


def test_thinning_is_deterministic_and_seeded(root):
    key = H.real_keys(root, "train")[0]
    manifest = shards.training_parents(root / "real")
    meta, arrays = corpus.load_song(root / "real" / "songs" / f"{key}.npz")
    a = TH.thin_song(manifest[key], meta, arrays, seed=0)
    b = TH.thin_song(manifest[key], meta, arrays, seed=0)
    assert a[1]["charts"] == b[1]["charts"]
    assert all(np.array_equal(a[0][k], b[0][k]) for k in a[0])
    assert a[1]["song_id"] == manifest[key]["song_id"]
    assert a[1]["parent"] == {"source": "real", "key": key} and not a[1]["has_audio"]
    assert "core" not in a[0]                                   # attached: charts only
    shares = {TH.keep_share(key, "beginner", seed) for seed in range(6)}
    assert len(shares) > 1 and all(0.25 <= s <= 0.70 for s in shares)


def test_a_song_with_nothing_dense_enough_is_skipped(root):
    key = H.real_keys(root, "train")[0]
    row = json.loads(json.dumps(shards.training_parents(root / "real")[key]))
    meta, arrays = corpus.load_song(root / "real" / "songs" / f"{key}.npz")
    for c in row["charts"]:
        c["keep"] = c["difficulty"] in ("beginner", "easy")
    assert TH.thin_song(row, meta, arrays) is None
    # a Medium chart can still be thinned for the tiers below it
    for c in row["charts"]:
        c["keep"] = c["difficulty"] in ("beginner", "medium")
    got = TH.thin_song(row, meta, arrays)[1]["charts"]
    assert [c["source_difficulty"] for c in got] == ["medium", "medium"]
    assert [c["companion"] for c in got] == [True, False]
