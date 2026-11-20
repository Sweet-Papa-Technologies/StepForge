"""REQ-SFV15-03 — synthesized grooves: stems that sum to the mix, charts that
sit on the rendered transients, tier rules, and corpus-format shards.

Density is checked against `grammar.BUDGETS[tier].max_nps_4s` as
`conditioning.sustained_nps` for every tier (the figure the network is
conditioned on), and additionally as `conditioning.peak_nps` for Beginner
through Hard. Challenge is exempt from the peak form only: a continuous 16th
stream at 150+ BPM is 10 notes/s or more by itself.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from beatforge.adapters.stepmania.grammar import ALL_DIFFICULTIES, BUDGETS
from beatforge.adapters.stepmania.neural import codec as C
from beatforge.adapters.stepmania.neural import conditioning as cond
from beatforge.adapters.stepmania.neural import grid_features as GF
from beatforge.adapters.stepmania.neural.synth import groove_synth as GS
from beatforge.adapters.stepmania.neural.synth import shards
from beatforge.adapters.stepmania.neural.train import corpus

SEED = 20261119
DRUM_OR_ARP = {"kick", "snare", "clap", "hat_closed", "hat_open", "crash", "lead"}
TRIPLET = "triplet_groove"


@pytest.fixture(scope="module")
def tracks():
    """One default-length track per archetype."""
    return {a: GS.render_track(i, SEED, archetype=a) for i, a in enumerate(GS.ARCHETYPES)}


@pytest.fixture(scope="module")
def packed(tracks):
    return {a: GS.track_shard(t) for a, t in tracks.items()}


def timing_of(track) -> GF.TimingMap:
    return GF.TimingMap.constant(track.bpm, track.offset_s)


def note_times(track, ticks) -> np.ndarray:
    return timing_of(track).time_at(np.asarray(ticks) / C.TICKS_PER_BEAT)


def sixteenths(ev) -> np.ndarray:
    """Ticks of the notes on a 16th that is not also an 8th."""
    return ev.tick[(ev.tick % 3 == 0) & (ev.tick % 6 != 0)]


def chains(ticks, step: int) -> list:
    """Maximal groups of two or more notes spaced exactly `step` ticks."""
    out, cur = [], [int(ticks[0])]
    for a, b in zip(ticks[:-1], ticks[1:]):
        if b - a == step:
            cur.append(int(b))
            continue
        if len(cur) > 1:
            out.append(cur)
        cur = [int(b)]
    return out + ([cur] if len(cur) > 1 else [])


def hits_by_tick(track, voices=None) -> dict:
    by: dict = {}
    for e in track.events:
        if e.voice != "pad" and (voices is None or e.voice in voices):
            by.setdefault(e.tick, []).append(e.time_s)
    return by


def off_transient(track, ticks, voices=None, tol_s: float = 0.003) -> int:
    """How many of `ticks` have no logged hit within `tol_s` of their time."""
    by = hits_by_tick(track, voices)
    return sum(not any(abs(h - t) <= tol_s for h in by.get(int(k), ()))
               for k, t in zip(ticks, note_times(track, ticks)))


def test_default_set_is_300_tracks_in_the_stated_mix():
    assert set(GS.DEFAULT_COUNTS) == set(GS.ARCHETYPES) and len(GS.ARCHETYPES) == 5
    assert sum(GS.DEFAULT_COUNTS.values()) == 300
    dealt = [GS.archetype_for(i) for i in range(300)]
    assert {a: dealt.count(a) for a in GS.ARCHETYPES} == GS.DEFAULT_COUNTS
    # a smaller set keeps the proportions
    for a in GS.ARCHETYPES:
        assert abs(dealt[:60].count(a) - GS.DEFAULT_COUNTS[a] / 5) <= 1
    assert GS.render_track(7, SEED, bars=24).archetype == GS.archetype_for(7)


def test_tracks_follow_their_archetype(tracks):
    for i, (a, tr) in enumerate(tracks.items()):
        assert tr.song_id == f"synth_groove_{i:04d}" and tr.archetype == a and tr.seed == SEED
        lo, hi = GS.BPM_RANGE[a]
        assert lo <= tr.bpm <= hi and 0.05 <= tr.offset_s <= 1.5
        bars = sum(s.bars for s in tr.sections)
        assert 48 <= bars <= 64 and all(s.bars % 4 == 0 for s in tr.sections)
        assert {s.name for s in tr.sections} == {"intro", "verse", "breakdown", "chorus",
                                                 "outro"}
        assert 45.0 < len(tr.audio) / GF.SR < 200.0
        assert tr.n_ticks == timing_of(tr).ticks_for_duration(len(tr.audio) / GF.SR)
        assert set(tr.charts) == set(ALL_DIFFICULTIES)
        # the log: tick order, humanized, never more than 1.5 ms off the grid
        ticks = np.array([e.tick for e in tr.events])
        drift = np.array([e.time_s for e in tr.events]) - note_times(tr, ticks)
        assert np.all(np.diff(ticks) >= 0)
        assert np.abs(drift).max() <= 0.0015 and drift.std() > 0.0005
        assert np.std([e.velocity for e in tr.events if e.voice == "kick"]) > 0.03


def test_stems_sum_to_the_mix(tracks):
    for tr in tracks.values():
        assert tuple(tr.stems) == GF.STEM_NAMES
        assert tr.audio.dtype == np.float32 and tr.audio.ndim == 1
        total = np.zeros_like(tr.audio)
        for name in GF.STEM_NAMES:
            stem = tr.stems[name]
            assert stem.dtype == np.float32 and stem.shape == tr.audio.shape
            assert np.isfinite(stem).all() and np.abs(stem).max() > 0.01
            total += stem
        np.testing.assert_allclose(total, tr.audio, rtol=0.0, atol=1e-6)
        assert np.isfinite(tr.audio).all()
        assert 0.9 < np.abs(tr.audio).max() <= 0.99


def test_mix_balance_resembles_a_real_song(tracks, packed):
    """Measured as the network reads it: the stem level channels of the shard
    (drums, bass, lead, pads + chords) and the low end of the mix."""
    bed = []
    for a, tr in tracks.items():
        arrays, _ = packed[a]
        drums, bass, lead, other = arrays["stems"].astype(np.float32)[:, [0, 4, 8, 12]].mean(0)
        # a harmonic bed to hear the drums through, and nothing missing
        assert other > 0.10, (a, other)
        assert drums > 0.10 and bass > 0.02 and lead > 0.02, (a, drums, bass, lead)
        bed.append(float(other))
        power = np.abs(np.fft.rfft(tr.audio.astype(np.float64))) ** 2
        freqs = np.fft.rfftfreq(len(tr.audio), 1.0 / GF.SR)
        assert power[freqs < 150.0].sum() / power.sum() < 0.5, a
        # the harmonic content sits in the mid-range, not on the sub
        core = arrays["core"].astype(np.float32)
        assert core[:, 27].mean() > 0.2 and core[:, 30].mean() > 0.45, a
    # not one fixed balance
    assert max(bed) - min(bed) > 0.03, bed


def test_a_track_is_a_function_of_seed_and_index(tracks):
    a = GS.render_track(3, SEED, bars=24)
    b = GS.render_track(3, SEED, bars=24)
    assert a.bpm == b.bpm and a.events == b.events and a.sections == b.sections
    np.testing.assert_array_equal(a.audio, b.audio)
    for name in GF.STEM_NAMES:
        np.testing.assert_array_equal(a.stems[name], b.stems[name])
    assert all(a.charts[d].equals(b.charts[d]) for d in ALL_DIFFICULTIES)
    assert sum(s.bars for s in a.sections) == 24

    for other in (GS.render_track(3, SEED + 1, bars=24), GS.render_track(4, SEED, bars=24)):
        assert other.bpm != a.bpm or len(other.audio) != len(a.audio)
        assert not all(a.charts[d].equals(other.charts[d]) for d in ALL_DIFFICULTIES)

    # the default (full-length) path repeats too
    ref = tracks["hard_16th_fills"]
    again = GS.render_track(2, SEED, archetype="hard_16th_fills")
    np.testing.assert_array_equal(ref.audio, again.audio)
    assert all(ref.charts[d].equals(again.charts[d]) for d in ALL_DIFFICULTIES)


def test_every_chart_note_sits_on_a_rendered_transient(tracks):
    for a, tr in tracks.items():
        for d in ALL_DIFFICULTIES:
            assert off_transient(tr, tr.charts[d].tick) == 0, (a, d)


def test_hard_sixteenths_are_on_drum_or_arp_hits_and_audible(tracks):
    for a, tr in tracks.items():
        hard = tr.charts["hard"]
        fine = hard.tick[hard.tick % 12 == 4] if a == TRIPLET else sixteenths(hard)
        assert len(fine) >= 8, a
        # the log says a drum or arpeggio hit is there, within 3 ms
        assert off_transient(tr, fine, DRUM_OR_ARP) == 0, a
        # ...and so does the audio: the percussive flux peaks within one frame
        # (10 ms) of the note's time, above the track's mean
        ff = GF.frame_features(tr.audio / np.abs(tr.audio).max())
        flux = ff.flux[:, 3:6].sum(1)
        peak = np.zeros(len(flux), bool)
        peak[1:-1] = ((flux[1:-1] >= flux[:-2]) & (flux[1:-1] > flux[2:])
                      & (flux[1:-1] > flux.mean()))
        f = np.round(note_times(tr, fine) / GF.HOP_S).astype(int)
        heard = peak[f - 1] | peak[f] | peak[f + 1]
        assert heard.mean() >= 0.9, (a, float(heard.mean()))


def test_hard_charts_carry_sixteenth_runs(tracks):
    count = {a: len(sixteenths(tr.charts["hard"])) for a, tr in tracks.items()}
    assert np.median(list(count.values())) >= 18, count
    assert 16 <= count["hard_16th_fills"] <= 36 and 16 <= count["stamina_drive"] <= 36, count
    assert max(count.values()) <= 36, count
    for a, tr in tracks.items():
        if a == TRIPLET:
            continue
        runs = chains(tr.charts["hard"].tick, 3)
        # 3-, 5- and 9-note runs, each ending on a beat
        assert {len(r) for r in runs} <= {3, 5, 9}, a
        assert all(r[-1] % 12 == 0 for r in runs), a
    # the triplet groove is charted on its 12ths instead, and never on 16ths
    triplet = tracks[TRIPLET].charts["hard"].tick
    assert count[TRIPLET] == 0 and (triplet % 12 == 4).sum() >= 8
    assert all(r[-1] % 12 == 0 for r in chains(triplet, 4))


def test_tier_rules(tracks):
    for a, tr in tracks.items():
        ch = tr.charts
        seconds = len(tr.audio) / GF.SR
        assert np.all(ch["beginner"].tick % 12 == 0), a
        assert np.all(ch["beginner"].p1 < 0) and np.all(ch["beginner"].k0 == C.TAP), a
        assert 0.7 <= len(ch["beginner"]) / seconds <= 1.9, a
        assert np.all(ch["easy"].tick % 6 == 0), a
        if a != TRIPLET:
            assert np.all(ch["medium"].tick % 6 == 0), a
        assert 0.03 <= (ch["medium"].p1 >= 0).mean() <= 0.15, a
        assert (ch["challenge"].p1 >= 0).any() and (ch["medium"].k0 == C.HOLD).any(), a

        n = [len(ch[d]) for d in ALL_DIFFICULTIES]
        assert n[0] < n[1] < n[2] < n[3] <= n[4], (a, n)

        for d in ALL_DIFFICULTIES:
            t = note_times(tr, ch[d].tick)
            cap = BUDGETS[d].max_nps_4s
            assert cond.sustained_nps(t) <= cap, (a, d)
            if d != "challenge":
                assert cond.peak_nps(t) <= cap, (a, d)
            # a jump is two different panels; nothing lands on a held panel
            assert np.all((ch[d].p1 < 0) | (ch[d].p0 < ch[d].p1)), (a, d)
            for i in np.nonzero(ch[d].k0 == C.HOLD)[0]:
                head, tail = ch[d].tick[i], ch[d].tick[i] + ch[d].d0[i]
                under = (ch[d].tick > head) & (ch[d].tick < tail)
                assert not np.any((ch[d].p0[under] == ch[d].p0[i])
                                  | (ch[d].p1[under] == ch[d].p0[i])), (a, d)


def test_stamina_challenge_streams_through_a_drop(tracks):
    tr = tracks["stamina_drive"]
    ticks = tr.charts["challenge"].tick
    longest = max(chains(ticks, 3), key=len)
    assert len(longest) >= 8 * 16 + 1                      # eight bars of 16ths
    drops = [range(s.bar0 * 48, (s.bar0 + s.bars) * 48 + 1)
             for s in tr.sections if s.name == "chorus"]
    assert any(sum(t in d for t in longest) >= 8 * 16 + 1 for d in drops)
    assert len(sixteenths(tr.charts["challenge"])) > 2 * len(sixteenths(tr.charts["hard"]))


def test_shards_round_trip_in_the_corpus_format(tracks, packed, tmp_path):
    for a, tr in tracks.items():
        arrays, meta = packed[a]
        path = tmp_path / "songs" / f"{tr.song_id}.npz"
        corpus.save_shard(path, arrays, meta)
        got_meta, got = corpus.load_song(path)
        assert got_meta == json.loads(json.dumps(meta))
        assert got["core"].shape == (tr.n_ticks, GF.D_CORE) == (tr.n_ticks, 31)
        assert got["stems"].shape == (tr.n_ticks, 16) and got["dt_ms"].shape == (tr.n_ticks,)
        for key in ("core", "stems", "dt_ms", "fingerprint"):
            assert np.isfinite(got[key].astype(np.float32)).all(), (a, key)
        # every stem has both level and flux
        stems = got["stems"].astype(np.float32)
        assert all(np.abs(stems[:, 4 * i:4 * i + 4]).sum(0).min() > 0 for i in range(4)), a
        np.testing.assert_allclose(got["dt_ms"], 60000.0 / tr.bpm / 12, rtol=1e-5)

        assert got_meta["key"] == got_meta["song_id"] == tr.song_id
        assert got_meta["source_type"] == "synth_groove" and got_meta["has_stems"]
        assert got_meta["archetype"] == a and got_meta["bpm"] == tr.bpm
        assert got_meta["n_ticks"] == tr.n_ticks and got_meta["seed"] == SEED
        assert got_meta["tick0_time_s"] == pytest.approx(tr.offset_s)
        restored = corpus.timing_from_facts(got_meta["timing"])
        assert restored.time_at(8.0) == pytest.approx(timing_of(tr).time_at(8.0))

        assert [c["difficulty"] for c in got_meta["charts"]] == list(ALL_DIFFICULTIES)
        for c in got_meta["charts"]:
            j, d = c["j"], c["difficulty"]
            ev = tr.charts[d]
            assert c["lock_violations"] == 0 and c["spins"] == 0, (a, d)
            assert c["n_events"] == len(ev) and c["n_jumps"] == int((ev.p1 >= 0).sum())
            lo, hi = BUDGETS[d].meter_range
            assert lo <= c["meter"] <= hi
            assert c["sustained_nps"] <= BUDGETS[d].max_nps_4s
            assert c["hold_share"] <= BUDGETS[d].hold_share[1], (a, d)
            assert corpus.chart_verdict(got_meta, c) is None, (a, d)
            # the stored tokens decode back to the chart that was emitted
            enc = C.EncodedChart(c["f_init"], got[f"c{j}_tick"], got[f"c{j}_tok"],
                                 got[f"c{j}_flav"], got[f"c{j}_da"], got[f"c{j}_ds"])
            assert C.decode(enc).equals(ev), (a, d)
            assert got[f"c{j}_style"][cond.INDEX["has_stems"]] == 1.0

    manifest = shards.write_source_manifest(tmp_path, "synth_groove", log=lambda *_: None)
    assert manifest["n_songs"] == 5 and manifest["splits"]["val"]["songs"] == 0
    assert manifest == json.loads((tmp_path / "manifest.json").read_text())
    for song in manifest["songs"]:
        assert song["split"] == "train" and song["song_id"].startswith("synth_groove_")
        assert song["archetype"] in GS.ARCHETYPES
        assert len(song["charts"]) == 5 and all(c["keep"] for c in song["charts"])


def test_generate_is_independent_of_the_worker_count(tmp_path):
    lines: list = []
    one = GS.generate(tmp_path / "one", num_tracks=2, seed=SEED, workers=1, log=lines.append)
    two = GS.generate(tmp_path / "two", num_tracks=2, seed=SEED, workers=2, log=lines.append)
    assert one == two and one["n_songs"] == 2 and one["source_type"] == "synth_groove"
    assert [s["key"] for s in one["songs"]] == ["synth_groove_0000", "synth_groove_0001"]
    assert all(s["split"] == "train" for s in one["songs"])
    assert any("synth_groove_0001" in line for line in lines)
    for key in ("synth_groove_0000", "synth_groove_0001"):
        meta_a, a = corpus.load_song(tmp_path / "one" / "songs" / f"{key}.npz")
        meta_b, b = corpus.load_song(tmp_path / "two" / "songs" / f"{key}.npz")
        assert meta_a == meta_b and meta_a["archetype"] == GS.archetype_for(int(key[-4:]))
        assert a.keys() == b.keys()
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
    # and each shard is the track that (seed, index) alone names
    arrays, _ = GS.track_shard(GS.render_track(1, SEED))
    np.testing.assert_array_equal(arrays["core"], a["core"])


def test_edm_offbeat_hat_is_the_loudest_onset_but_beginner_stays_on_the_beat(tracks, packed):
    tr = tracks["edm_offbeat_trap"]
    arrays, _ = packed["edm_offbeat_trap"]
    high = arrays["core"][:, 25].astype(np.float32)     # percussive flux, 2 kHz and up
    k = np.arange(len(high))
    drums_on = np.zeros(len(high), bool)
    for s in tr.sections:
        if s.energy >= 1:
            drums_on[s.bar0 * 48:(s.bar0 + s.bars) * 48] = True
    at = [float(high[drums_on & (k % 12 == r)].mean()) for r in range(12)]
    # the "&" carries more high-band attack than any other position off the
    # beat — and more than the beat itself, which is what makes the grid ambiguous
    assert at[6] > 1.5 * max(at[r] for r in range(1, 12) if r != 6), at
    assert at[6] > at[0], at
    hats = [e for e in tr.events if e.voice in ("hat_open", "hat_closed") and e.velocity > 0.6]
    assert len(hats) > 100 and all(e.tick % 12 == 6 for e in hats)
    # the kick is on every beat there, and that is where the Beginner chart steps
    kicks = {e.tick for e in tr.events if e.voice == "kick"}
    assert all(t in kicks for t in np.nonzero(drums_on & (k % 12 == 0))[0])
    assert np.all(tr.charts["beginner"].tick % 12 == 0)
    assert len(tr.charts["beginner"]) >= 40
