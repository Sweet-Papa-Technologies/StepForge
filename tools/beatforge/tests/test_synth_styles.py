"""REQ-SFV15-06 (1) — programmatic style patterns: every sequence is physically
clean on the codec's own reading of it, never one loop, labelled with the style
it measures, and the six archetypes come out as six different styles."""
from __future__ import annotations

import json

import numpy as np
import pytest

from beatforge.adapters.stepmania.grammar import BUDGETS
from beatforge.adapters.stepmania.neural import codec as C
from beatforge.adapters.stepmania.neural import conditioning as cond
from beatforge.adapters.stepmania.neural.synth import shards
from beatforge.adapters.stepmania.neural.synth import style_patterns as SP
from beatforge.adapters.stepmania.neural.train import corpus

SEED = 20261119
PER = 10
quiet = dict(log=lambda *_: None)
L, D, U, R = C.L, C.D, C.U, C.R


@pytest.fixture(scope="module")
def batch():
    """PER sequences of each archetype, at indices of their own."""
    return {a: [SP.generate_sequence(1000 * k + i, SEED, a) for i in range(PER)]
            for k, a in enumerate(SP.ARCHETYPES)}


def every(batch):
    return [s for a in SP.ARCHETYPES for s in batch[a]]


def replay(ev: C.ChartEvents, times_ms) -> dict:
    """Encode a chart and walk the result one transition at a time with
    `codec.transition_cost`, under the generator's hold rule (a freeze pins its
    foot up to and including the tail tick). Shares no code with `verify`."""
    enc, info = C.encode(ev, times_ms)
    pinned = C._held_masks(ev, inclusive=True)
    strict = C._held_masks(ev)
    lockbits = C.tables()["lockbits"]
    gaps = np.diff(times_ms, prepend=-1e9)
    pl, pr, fl = L, R, enc.f_init
    out = dict(info=info, enc=enc, spins=0, locks=0, hard=0, fast_dsteps=0, on_held=0,
               total=0)
    for i, tok in enumerate(int(t) for t in enc.token):
        state = C.state_index(pl, pr, fl)
        dtb = int(C.dt_bucket(gaps[i]))
        bits = int(lockbits[pinned[i], state])
        cost = C.transition_cost(pl, pr, fl, tok, dtb, bool(bits & 1), bool(bits & 2))
        out["locks"] += cost >= C.LOCK_VIOLATION
        out["spins"] += cost % C.LOCK_VIOLATION >= C.SPIN_COST
        free = int(lockbits[strict[i], state])
        plain = C.transition_cost(pl, pr, fl, tok, dtb, bool(free & 1), bool(free & 2))
        out["hard"] += plain >= C.HARD_COST
        out["total"] += plain
        was = (pl, pr)[fl]
        other_free = not free >> (1 - fl) & 1
        pl, pr, fl, pa, ps = C.apply_token(pl, pr, fl, tok)
        if 4 <= tok < 8 and ps != was and other_free and C.DSTEP_COST[dtb] >= 60:
            out["fast_dsteps"] += 1
        out["on_held"] += any(p >= 0 and pinned[i] >> p & 1 for p in (pa, ps))
    return out


def style_of(seq) -> np.ndarray:
    enc, _ = C.encode(seq.events, seq.times_ms)
    return cond.compute_style_vector(enc, seq.times_ms / 1000.0, seq.meter)


def median(batch, archetype: str, field: str) -> float:
    return float(np.median([style_of(s)[cond.INDEX[field]] for s in batch[archetype]]))


# --------------------------------------------------------------------------- #
# the module's contract
# --------------------------------------------------------------------------- #
def test_six_archetypes_and_their_default_counts():
    assert SP.ARCHETYPES == ("itg_stamina", "jumpstream", "crossover_flow",
                             "tech_footswitch", "modern_hard_burst", "casual_ddr_flow")
    assert SP.DEFAULT_COUNTS == {"itg_stamina": 500, "jumpstream": 450,
                                 "crossover_flow": 400, "tech_footswitch": 400,
                                 "modern_hard_burst": 350, "casual_ddr_flow": 300}
    assert set(SP.PROFILES) == set(SP.BPM_RANGES) == set(SP.SLOTS) == set(SP.ARCHETYPES)
    for profile in SP.PROFILES.values():
        for field, (lo, hi) in profile.items():
            assert 0.0 <= lo <= hi <= cond.MAXIMA[cond.INDEX[field]]


def test_other_totals_scale_the_counts():
    count = lambda total: {a: SP.allocation(total).count(a) for a in SP.ARCHETYPES}
    assert count(2400) == SP.DEFAULT_COUNTS
    assert count(240) == {a: n // 10 for a, n in SP.DEFAULT_COUNTS.items()}
    for total in (1, 7, 60, 999):
        assert len(SP.allocation(total)) == total
    assert set(SP.allocation(12)) == set(SP.ARCHETYPES)


# --------------------------------------------------------------------------- #
# verifier: every emitted sequence passes it, and an independent replay agrees
# --------------------------------------------------------------------------- #
def test_every_sequence_passes_the_verifier(batch):
    for s in every(batch):
        assert SP.verify(s.events, s.times_ms, s.difficulty) == [], s.seq_id


def test_no_forbidden_transition_hold_lock_or_spin(batch):
    for s in every(batch):
        r = replay(s.events, s.times_ms)
        info = r["info"]
        assert (info.spins, info.lock_violations) == (0, 0), s.seq_id
        assert int(info.cost.max()) < C.HARD_COST, s.seq_id
        assert r["total"] == info.total_cost, s.seq_id          # the replay is the encoding
        assert (r["spins"], r["locks"], r["hard"]) == (0, 0, 0), s.seq_id
        assert r["fast_dsteps"] == 0, s.seq_id
        assert r["on_held"] == 0, s.seq_id
        assert info.mean_cost <= corpus.DISCOMFORT_CEILING[s.difficulty], s.seq_id


def test_holds_release_before_their_foot_or_panel_is_used_again(batch):
    held_any = 0
    for s in every(batch):
        ev = s.events
        for p, k, d in ((ev.p0, ev.k0, ev.d0), (ev.p1, ev.k1, ev.d1)):
            for i in np.nonzero(k > 0)[0]:
                held_any += 1
                assert d[i] > 0
                tail = ev.tick[i] + d[i]
                later = ev.tick > ev.tick[i]
                same_panel = later & ((ev.p0 == p[i]) | (ev.p1 == p[i]))
                assert not (same_panel & (ev.tick <= tail)).any(), s.seq_id
                assert tail < s.n_ticks
    assert held_any > 50                                     # the tech and burst holds


def test_no_sequence_is_one_loop(batch):
    for s in every(batch):
        enc, _ = C.encode(s.events, s.times_ms)
        assert SP.top_ngram_share(enc.token) <= 0.20, s.seq_id


def test_top_ngram_share_counts_overlapping_windows():
    loop = np.tile([0, 1, 2, 0], 50)                         # 200 tokens, 193 windows
    assert SP.top_ngram_share(loop) == pytest.approx(49 / 193)
    assert SP.top_ngram_share(loop, n=4) == pytest.approx(50 / 197)
    assert SP.top_ngram_share(np.arange(20) % 20) == pytest.approx(1 / 13)
    assert SP.top_ngram_share([0, 1, 2]) == 0.0


def test_round_trip(batch):
    for s in every(batch):
        enc, _ = C.encode(s.events, s.times_ms)
        assert C.decode(enc).equals(s.events), s.seq_id


# --------------------------------------------------------------------------- #
# label = measured
# --------------------------------------------------------------------------- #
def test_stored_style_is_the_measured_style_and_inside_the_profile(batch):
    for a in SP.ARCHETYPES:
        for s in batch[a]:
            arrays, meta = SP.sequence_shard(s)
            row = meta["charts"][0]
            enc = C.EncodedChart(row["f_init"], arrays["c0_tick"], arrays["c0_tok"],
                                 arrays["c0_flav"], arrays["c0_da"], arrays["c0_ds"],
                                 arrays["c0_mines"])
            times_ms = arrays["c0_tick"] * (5000.0 / meta["bpm"])
            assert np.allclose(times_ms, arrays["c0_tms"], rtol=1e-6)
            measured = cond.compute_style_vector(enc, times_ms / 1000.0, row["meter"])
            assert np.array_equal(arrays["c0_style"], measured), s.seq_id
            assert np.array_equal(measured, style_of(s))
            for field, (lo, hi) in SP.PROFILES[a].items():
                assert lo <= measured[cond.INDEX[field]] <= hi, (s.seq_id, field)
            assert SP.profile_misses(a, measured) == []
            assert measured[cond.INDEX["has_stems"]] == 0.0


def test_profile_misses_names_the_fields_outside():
    style = np.zeros(cond.STYLE_DIM, np.float32)
    assert SP.profile_misses("casual_ddr_flow", style) == []
    style[cond.INDEX["jack_share"]] = 0.01
    assert SP.profile_misses("casual_ddr_flow", style) == ["jack_share"]
    assert set(SP.profile_misses("jumpstream", style)) == {"stream_share", "jump_share"}


def test_archetypes_are_separated_on_the_fields_that_define_them(batch):
    stream = {a: median(batch, a, "stream_share") for a in SP.ARCHETYPES}
    assert stream["itg_stamina"] > stream["modern_hard_burst"] > stream["casual_ddr_flow"]
    for field, top in (("jump_share", "jumpstream"), ("crossover_share", "crossover_flow"),
                       ("footswitch_share", "tech_footswitch")):
        med = {a: median(batch, a, field) for a in SP.ARCHETYPES}
        assert max(med, key=med.get) == top, (field, med)
        assert all(med[top] > v for a, v in med.items() if a != top), (field, med)


# --------------------------------------------------------------------------- #
# determinism, lengths, tempo, tier
# --------------------------------------------------------------------------- #
def test_a_sequence_is_a_function_of_seed_and_index(batch):
    for k, a in enumerate(SP.ARCHETYPES):
        first = batch[a][0]
        again = SP.generate_sequence(1000 * k, SEED, a)
        assert again.events.equals(first.events)
        assert (again.seq_id, again.bpm, again.difficulty, again.meter, again.n_ticks,
                again.attempts) == (first.seq_id, first.bpm, first.difficulty,
                                    first.meter, first.n_ticks, first.attempts)
        assert np.array_equal(again.times_ms, first.times_ms)
        assert not batch[a][1].events.equals(first.events)            # another index
        assert not SP.generate_sequence(1000 * k, SEED + 1, a).events.equals(first.events)
    assert len({s.events.tick.tobytes() for s in every(batch)}) == len(every(batch))


def test_default_archetype_follows_the_2400_layout():
    assert SP.generate_sequence(0, SEED).archetype == "itg_stamina"
    assert SP.generate_sequence(500, SEED).archetype == "jumpstream"
    assert SP.generate_sequence(2399, SEED).archetype == "casual_ddr_flow"
    assert SP.generate_sequence(2399, SEED).seq_id == "synth_style_02399"


def test_lengths_bars_tempo_and_tier(batch):
    for a in SP.ARCHETYPES:
        lo, hi = SP.BPM_RANGES[a]
        for s in batch[a]:
            ev = s.events
            assert s.archetype == a and s.seed == SEED and s.attempts >= 1
            assert 128 <= len(ev) <= 768, s.seq_id
            assert s.n_ticks % C.G == 0 and 16 <= s.n_ticks // C.G <= 64, s.seq_id
            assert 0 <= ev.tick[0] and ev.tick[-1] < s.n_ticks
            assert (np.diff(ev.tick) > 0).all()
            assert lo <= s.bpm <= hi, s.seq_id
            assert np.allclose(s.times_ms, ev.tick * 5000.0 / s.bpm)
            nps = cond.sustained_nps(s.times_ms / 1000.0)
            assert s.meter == int(np.clip(round(cond.effective_meter(nps)), 1, 15))
            slots = SP.SLOTS[a]
            assert s.difficulty in slots
            easier = slots[:slots.index(s.difficulty)]
            assert all(nps > BUDGETS[t].max_nps_4s for t in easier), s.seq_id
            if s.difficulty != slots[-1]:
                assert nps <= BUDGETS[s.difficulty].max_nps_4s
    assert {s.difficulty for s in batch["casual_ddr_flow"]} <= {"easy", "medium"}
    assert {s.difficulty for s in batch["itg_stamina"]} <= {"hard", "challenge"}


def test_rhythm_is_on_the_grid_each_archetype_calls_for(batch):
    for s in batch["casual_ddr_flow"]:                       # 4ths and 8ths only
        assert (s.events.tick % 6 == 0).all()
        assert (s.events.p1 >= 0).any()                      # with jump accents
    for a in SP.ARCHETYPES:                                  # nothing finer than a 16th
        for s in batch[a]:
            assert (s.events.tick % 3 == 0).all()
    for s in batch["tech_footswitch"]:
        assert (s.events.k0 == C.HOLD).any()


# --------------------------------------------------------------------------- #
# shards
# --------------------------------------------------------------------------- #
def test_generate_writes_symbolic_training_shards(tmp_path):
    out = tmp_path / "synth_style"
    summary = SP.generate(out, total_charts=12, seed=SEED, workers=1, **quiet)
    files = shards.shard_files(out)
    assert [f.stem for f in files] == [f"synth_style_{i:05d}" for i in range(12)]
    for f, archetype in zip(files, SP.allocation(12)):
        meta, arrays = corpus.load_song(f)
        assert not {"core", "dt_ms", "stems"} & set(arrays)
        assert all(k.startswith("c0_") for k in arrays)       # one chart, nothing else
        assert meta["source_type"] == "synth_style" and meta["has_audio"] is False
        assert meta["key"] == meta["song_id"] == f.stem
        assert meta["archetype"] == archetype and meta["seed"] == SEED
        assert meta["n_ticks"] % C.G == 0
        assert meta["duration_s"] == pytest.approx(meta["n_ticks"] * 5.0 / meta["bpm"], abs=0.01)
        (row,) = meta["charts"]
        assert row["difficulty"] in SP.SLOTS[archetype]
        assert corpus.chart_verdict(meta, row) is None       # trainable as it stands
        seq = SP.generate_sequence(int(f.stem[-5:]), SEED, archetype)
        assert np.array_equal(arrays["c0_tick"], seq.events.tick)
        assert np.array_equal(arrays["c0_raw"][0], seq.events.p0)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["source_type"] == "synth_style" and manifest["n_songs"] == 12
    assert manifest["splits"]["val"]["songs"] == manifest["splits"]["test"]["songs"] == 0
    for song in manifest["songs"]:
        assert song["split"] == "train"
        assert song["song_id"].startswith("synth_style_")
        assert song["archetype"] in SP.ARCHETYPES
        assert all(c["keep"] for c in song["charts"])

    profile = json.loads((out / "profile.json").read_text())
    assert profile["archetypes"] == summary["archetypes"]
    assert sum(p["count"] for p in profile["archetypes"].values()) == 12
    for a, p in profile["archetypes"].items():
        assert p["mean_attempts"] >= 1.0
        assert set(p["fields"]) == set(cond.FIELDS)
        for field, (lo, hi) in SP.PROFILES[a].items():
            spread = p["fields"][field]
            assert lo <= spread["min"] <= spread["median"] <= spread["max"] <= hi


def test_generate_is_the_same_with_a_worker_pool_and_drops_stale_shards(tmp_path):
    serial, pooled = tmp_path / "serial", tmp_path / "pooled"
    SP.generate(serial, total_charts=12, seed=3, workers=1, **quiet)
    SP.generate(pooled, total_charts=14, seed=3, workers=2, **quiet)
    SP.generate(pooled, total_charts=12, seed=3, workers=2, **quiet)
    a, b = shards.shard_files(serial), shards.shard_files(pooled)
    assert [f.name for f in a] == [f.name for f in b] and len(a) == 12
    for fa, fb in zip(a, b):
        (meta_a, arr_a), (meta_b, arr_b) = corpus.load_song(fa), corpus.load_song(fb)
        assert meta_a == meta_b
        assert all(np.array_equal(arr_a[k], arr_b[k]) for k in arr_a)


# --------------------------------------------------------------------------- #
# the verifier rejects what it should
# --------------------------------------------------------------------------- #
def at_100ms(ev: C.ChartEvents) -> np.ndarray:
    """16ths 100 ms apart (150 BPM): the fastest gap bucket."""
    return ev.tick * (100.0 / 3.0)


def taps(panels, step: int = 3) -> C.ChartEvents:
    return C.ChartEvents.from_rows([(i * step, [(p, C.TAP, 0)]) for i, p in enumerate(panels)])


def test_verify_rejects_a_spin():
    # The left foot freezes L while the right taps sixteen 16ths, then the
    # right freezes D; the left, the only free foot, must cross to freeze R;
    # D is released, and the note on L can only be the right foot's: a spin.
    k = 16
    rows = [(0, [(L, C.HOLD, 3 * (k + 3))])]
    rows += [(3 * (i + 1), [((D, U)[i % 2], C.TAP, 0)]) for i in range(k)]
    rows += [(3 * (k + 1), [(D, C.HOLD, 12)]), (3 * (k + 4), [(R, C.HOLD, 12)]),
             (3 * (k + 6), [(L, C.TAP, 0)])]
    ev = C.ChartEvents.from_rows(rows)
    _, info = C.encode(ev, at_100ms(ev))
    assert info.spins == 1 and info.lock_violations == 0
    reasons = SP.verify(ev, at_100ms(ev), "challenge")
    assert "spin" in reasons and "forbidden_transition" in reasons
    assert "hold_lock" not in reasons and "double_step" not in reasons


def test_verify_rejects_a_step_on_a_held_panel_and_a_moved_holding_foot():
    on_held = C.ChartEvents.from_rows([(0, [(L, C.TAP, 0)]), (3, [(D, C.HOLD, 12)]),
                                       (6, [(R, C.TAP, 0)]), (9, [(D, C.TAP, 0)]),
                                       (18, [(U, C.TAP, 0)])])
    assert "note_on_held_panel" in SP.verify(on_held, at_100ms(on_held), "challenge")

    # both feet are freezing when the third arrow comes
    both = C.ChartEvents.from_rows([(0, [(L, C.HOLD, 24)]), (3, [(R, C.HOLD, 24)]),
                                    (9, [(D, C.TAP, 0)]), (30, [(U, C.TAP, 0)])])
    _, info = C.encode(both, at_100ms(both))
    assert info.lock_violations == 1
    reasons = SP.verify(both, at_100ms(both), "challenge")
    assert "hold_lock" in reasons and "spin" not in reasons
    assert "note_on_held_panel" not in reasons

    # the annotator lets a foot leave on the tail tick; the generator's rule does not
    on_tail = C.ChartEvents.from_rows([(0, [(L, C.HOLD, 12)]), (6, [(D, C.TAP, 0)]),
                                       (12, [(U, C.TAP, 0)]), (18, [(R, C.TAP, 0)])])
    _, info = C.encode(on_tail, at_100ms(on_tail))
    assert info.lock_violations == 0 and info.total_cost == 0
    assert SP.verify(on_tail, at_100ms(on_tail), "challenge") == ["hold_lock"]
    cleared = C.ChartEvents.from_rows([(0, [(L, C.HOLD, 9)]), (6, [(D, C.TAP, 0)]),
                                       (12, [(U, C.TAP, 0)]), (18, [(R, C.TAP, 0)])])
    assert SP.verify(cleared, at_100ms(cleared), "challenge") == []


def test_verify_rejects_a_fast_double_step_stream():
    ev = taps([L, D, R] * 20)                                # L and R two steps apart, for ever
    reasons = SP.verify(ev, at_100ms(ev), "challenge")
    assert "double_step" in reasons
    assert "spin" not in reasons and "hold_lock" not in reasons
    # the same panels with a second between them are no longer a FAST double-step
    slow = taps([L, D, R] * 20, step=48)
    assert "double_step" not in SP.verify(slow, slow.tick * (1000.0 / 48.0), "challenge")
    # and with the other foot pinned by a freeze the same foot may walk freely
    forced = C.ChartEvents.from_rows([(0, [(L, C.HOLD, 18)]), (3, [(D, C.TAP, 0)]),
                                      (6, [(U, C.TAP, 0)]), (9, [(R, C.TAP, 0)]),
                                      (12, [(D, C.TAP, 0)]), (24, [(L, C.TAP, 0)])])
    assert SP.verify(forced, at_100ms(forced), "challenge") == []


def test_verify_rejects_one_repeated_loop():
    ev = taps([L, D, U, R] * 50)                             # 200 comfortable notes
    _, info = C.encode(ev, at_100ms(ev))
    assert info.total_cost == 0
    assert SP.verify(ev, at_100ms(ev), "challenge") == ["loop"]


def test_verify_holds_each_tier_to_its_own_ceiling():
    # a jump on every other 8th: comfortable for Hard, too much for Easy
    rows = []
    for i in range(40):
        pair = ((L, R), (D, U), (L, D), (U, R), (L, U), (D, R))[(i * 5 + i // 6) % 6]
        rows.append((i * 12, [(p, C.TAP, 0) for p in pair]))
    ev = C.ChartEvents.from_rows(rows)
    times = ev.tick * (5000.0 / 120.0)
    _, info = C.encode(ev, times)
    assert corpus.DISCOMFORT_CEILING["easy"] < info.mean_cost <= corpus.DISCOMFORT_CEILING["hard"]
    assert SP.verify(ev, times, "hard") == []
    assert SP.verify(ev, times, "easy") == ["discomfort"]
    assert SP.verify(C.ChartEvents.from_rows([]), [], "hard") == ["empty"]
