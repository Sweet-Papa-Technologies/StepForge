"""v1.5b: counterfactual footing variants (synth/counterfactual.py) and the
Stage B data path that trains on them."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beatforge.adapters.stepmania.neural import codec as C  # noqa: E402
from beatforge.adapters.stepmania.neural import conditioning as cond  # noqa: E402
from beatforge.adapters.stepmania.neural.synth import counterfactual as CF  # noqa: E402
from beatforge.adapters.stepmania.neural.synth import shards  # noqa: E402
from beatforge.adapters.stepmania.neural.train import corpus  # noqa: E402
from beatforge.adapters.stepmania.neural.train import dataset as D  # noqa: E402
from beatforge.adapters.stepmania.neural.train import train_stage_b as TB  # noqa: E402

import synth_helpers as H  # noqa: E402

S16, S8, BEAT, BAR = 3, 6, 12, 48


def _skeleton(bpm: float = 150.0, bars: int = 24, jumps: bool = False, seed: int = 0):
    """A rhythm with 16th runs, 8ths and a few freezes; no arrows."""
    rng = np.random.default_rng(seed)
    ticks, is_jump, hold = [], [], []
    t = BAR
    while t < bars * BAR:
        if rng.random() < 0.6:                       # a run of 16ths
            for _ in range(int(rng.integers(8, 33))):
                ticks.append(t)
                is_jump.append(False)
                hold.append(0)
                t += S16
        else:                                        # 8ths, some with a freeze
            for _ in range(int(rng.integers(4, 13))):
                ticks.append(t)
                is_jump.append(bool(jumps and rng.random() < 0.2))
                hold.append(S8 if rng.random() < 0.15 and not is_jump[-1] else 0)
                t += S8 if not hold[-1] else 2 * S8
        t += int(rng.choice((S8, BEAT)))
    ticks = np.asarray(ticks, np.int64)
    hold = np.asarray(hold, np.int64)
    z = np.zeros(len(ticks), np.int64)
    return CF.Skeleton(ticks, ticks * (5000.0 / bpm), np.asarray(is_jump, bool),
                       (hold > 0).astype(np.int64), z.copy(), hold, z.copy())


@pytest.fixture(scope="module")
def group():
    sk = _skeleton()
    kept, lost = CF.realize_group(sk, CF.uniform_prior(), "challenge", 11, seed=3)
    return sk, kept, lost


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #
def test_a_fast_rhythm_gets_every_variant(group):
    _, kept, lost = group
    assert set(kept) == set(CF.VARIANTS), lost


def test_all_variants_of_a_group_are_the_same_rhythm_with_different_arrows(group):
    sk, kept, _ = group
    base = kept["clean_alt"][0]
    assert np.array_equal(base.tick, sk.tick)
    seen = set()
    for variant, (ev, _) in kept.items():
        assert CF.same_rhythm(base, ev), variant
        assert np.array_equal(ev.p1 >= 0, sk.is_jump)
        seen.add((ev.p0.tobytes(), ev.p1.tobytes()))
    assert len(seen) == len(kept)                    # no two variants share their arrows


def test_same_rhythm_notices_a_moved_note_a_jump_and_a_freeze(group):
    _, kept, _ = group
    ev = kept["clean_alt"][0]
    assert CF.same_rhythm(ev, ev)
    moved = C.ChartEvents(ev.tick + (np.arange(len(ev)) == 5) * 1, ev.p0, ev.p1, ev.k0, ev.k1,
                          ev.d0, ev.d1)
    assert not CF.same_rhythm(ev, moved)
    i = int(np.nonzero(ev.p1 < 0)[0][3])
    p1 = ev.p1.copy()
    p1[i] = (ev.p0[i] + 1) % 4
    assert not CF.same_rhythm(ev, C.ChartEvents(ev.tick, ev.p0, p1, ev.k0, ev.k1, ev.d0, ev.d1))
    k0, d0 = ev.k0.copy(), ev.d0.copy()
    j = int(np.nonzero(ev.k0 == 0)[0][0])
    k0[j], d0[j] = 1, 3
    assert not CF.same_rhythm(ev, C.ChartEvents(ev.tick, ev.p0, ev.p1, k0, ev.k1, d0, ev.d1))


def test_the_label_is_what_the_annotator_measures_not_what_was_asked(group):
    sk, kept, _ = group
    for variant, (ev, m) in kept.items():
        enc, info = C.encode(ev, sk.times_ms)
        style = cond.compute_style_vector(enc, sk.times_ms / 1000.0, 11)
        assert m["cross"] == pytest.approx(float(style[cond.INDEX["crossover_share"]]))
        assert m["jack"] == pytest.approx(float(style[cond.INDEX["jack_share"]]))
        assert m["fsw"] == pytest.approx(float(style[cond.INDEX["footswitch_share"]]))
        flags = CF.footing_flags(enc)
        n_single = max(1, int((enc.token < 8).sum()))
        for axis in CF.AXES:                         # the per-note flags add up to the label
            assert flags[axis].sum() / n_single == pytest.approx(m[axis], abs=1e-6)


def test_every_variant_is_legal_under_the_32_state_foot_model(group):
    sk, kept, _ = group
    for variant, (ev, m) in kept.items():
        enc, info = C.encode(ev, sk.times_ms)
        assert info.spins == 0 and info.lock_violations == 0, variant
        assert (info.cost < C.HARD_COST).all(), variant
        assert m["reasons"] == [] and m["loop"] <= CF.MAX_NGRAM_SHARE


def test_variants_are_apart_on_their_own_axis_and_quiet_on_the_others(group):
    _, kept, _ = group
    clean = kept["clean_alt"][1]
    assert max(clean[a] for a in CF.AXES) <= CF.CLEAN_MAX
    assert kept["crossover_heavy"][1]["cross"] >= clean["cross"] + 0.06
    assert kept["footswitch_heavy"][1]["fsw"] >= clean["fsw"] + 0.05
    assert kept["jack_heavy"][1]["jack"] >= clean["jack"] + 0.03
    b = kept["blended_tech"][1]
    assert b["cross"] >= clean["cross"] + 0.04 and b["fsw"] >= clean["fsw"] + 0.03
    for variant in ("crossover_heavy", "footswitch_heavy", "jack_heavy"):
        m = kept[variant][1]
        for axis in CF.AXES:
            if axis not in CF.TARGETS[variant]:
                assert m[axis] <= clean[axis] + CF.OFF_AXIS_SLACK, (variant, axis)


def test_accept_rejects_what_the_spec_says_to_reject():
    clean = {"cross": 0.0, "jack": 0.0, "fsw": 0.0, "reasons": [], "loop": 0.05}
    ok = dict(clean, cross=0.12)
    assert CF.accept("crossover_heavy", ok, clean) is None
    assert CF.accept("crossover_heavy", dict(ok, cross=0.05), clean) == "not_apart_cross"
    assert CF.accept("crossover_heavy", dict(ok, reasons=["forbidden_transition"]), clean) \
        == "forbidden_transition"
    assert CF.accept("crossover_heavy", dict(ok, reasons=["hold_lock"]), clean) == "hold_lock"
    assert CF.accept("crossover_heavy", dict(ok, loop=0.3), clean) == "loop"
    assert CF.accept("crossover_heavy", dict(ok, jack=0.09), clean) == "drift_jack"
    assert CF.accept("footswitch_heavy", dict(clean, fsw=0.04), clean) == "not_apart_fsw"
    assert CF.accept("jack_heavy", dict(clean, jack=0.02), clean) == "not_apart_jack"
    assert CF.accept("clean_alt", dict(clean, jack=0.08), None) == "not_clean"
    assert CF.accept("jack_heavy", dict(clean, jack=0.08), None) == "no_clean_variant"


def test_a_slow_rhythm_gets_no_footswitch_variant():
    """A footswitch only reads as one on fast steps; on plain 8ths at 120 BPM
    the same arrows are a jack, so the variant must be refused, not mislabelled."""
    ticks = np.arange(BAR, 30 * BAR, S8, dtype=np.int64)
    z = np.zeros(len(ticks), np.int64)
    sk = CF.Skeleton(ticks, ticks * (5000.0 / 120.0), z.astype(bool), z, z.copy(), z.copy(),
                     z.copy())
    kept, lost = CF.realize_group(sk, CF.uniform_prior(), "medium", 6, seed=1)
    assert "footswitch_heavy" not in kept and lost["footswitch_heavy"].startswith("not_apart")
    assert {"clean_alt", "crossover_heavy"} <= set(kept)


def test_skeleton_cuts_freezes_the_generator_could_not_stand_on():
    # a freeze that ends exactly on the next note, and a jump during a freeze
    tick = np.array([0, 12, 24, 36, 48, 60], np.int64)
    tok = np.array([0, 0, 0, 8, 0, 0], np.uint8)           # the 4th note is a jump
    flav = np.array([C._FLAVOR_OF[(1, 0)], 0, C._FLAVOR_OF[(1, 0)], 0, 0, 0], np.uint8)
    da = np.array([12, 0, 24, 0, 0, 0], np.int32)          # ends ON note 1; spans the jump
    sk = CF.skeleton_of(tick, tok, flav, da, np.zeros(6, np.int32), tick * 50.0)
    length = sk.hold_length()
    assert length[0] == 12 - 3                              # let go a 16th before note 1
    assert tick[2] + length[2] < tick[3]                    # over before the jump needs both feet
    slots, kind = CF._lock_schedule(sk)
    assert slots[3] == 0 and kind[0] == 4


def test_stream_rhythms_are_long_fast_and_reproducible():
    a = CF.stream_rhythm(np.random.default_rng(5), jumpstream=True)
    b = CF.stream_rhythm(np.random.default_rng(5), jumpstream=True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    ticks, is_jump, bpm, n_bars = a
    assert 120.0 <= bpm <= 195.0 and 16 <= n_bars <= 48
    assert (np.diff(ticks) > 0).all() and (np.diff(ticks) == S16).mean() > 0.5
    assert 0 < is_jump.mean() < 0.3
    assert not CF.stream_rhythm(np.random.default_rng(5), jumpstream=False)[1].any()


# --------------------------------------------------------------------------- #
# The corpus sources
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def root(tmp_path_factory):
    src = H.make_corpus_root(tmp_path_factory.mktemp("cf"), train=3, val=1, stems=False)
    out = CF.generate(src, streams=6, seed=7, workers=1, log=lambda *a, **k: None)
    return src, out


def test_generate_writes_two_train_only_sources_that_pass_the_leak_check(root):
    src, out = root
    counts = shards.check_no_leak(src, log=lambda *a, **k: None)
    assert counts[CF.SOURCE_STREAM] == 6 and out[CF.SOURCE_STREAM]["groups"] == 6
    real = json.loads((src / "real" / "manifest.json").read_text())
    held = {s["song_id"] for s in real["songs"] if s["split"] != "train"}
    train = {s["key"]: s["song_id"] for s in real["songs"] if s["split"] == "train"}
    assert held, "the fixture needs a validation song for this test to mean anything"
    for source in (CF.SOURCE_REAL, CF.SOURCE_STREAM):
        m = json.loads((src / source / "manifest.json").read_text())
        assert all(s["split"] == "train" for s in m["songs"])
        assert not {s["song_id"] for s in m["songs"]} & held
        if source == CF.SOURCE_REAL:
            for s in m["songs"]:
                assert s["parent"]["source"] == "real"
                assert train[s["parent"]["key"]] == s["song_id"]


def test_stored_charts_carry_their_group_and_the_annotators_reading(root):
    src, _ = root
    n = 0
    for f in shards.shard_files(src / CF.SOURCE_STREAM):
        meta, arrays = corpus.load_song(f)
        groups = {}
        for c in meta["charts"]:
            j = c["j"]
            raw, rawd = arrays[f"c{j}_raw"], arrays[f"c{j}_rawd"]
            ev = C.ChartEvents(arrays[f"c{j}_tick"], raw[0], raw[1], raw[2], raw[3],
                               rawd[0], rawd[1])
            tms = arrays[f"c{j}_tms"].astype(np.float64)
            enc, _ = C.encode(ev, tms)
            assert np.array_equal(enc.token, arrays[f"c{j}_tok"])
            style = cond.compute_style_vector(enc, tms / 1000.0, c["meter"])
            assert np.allclose(style, arrays[f"c{j}_style"], atol=1e-6)
            assert c["variant"] in CF.VARIANTS and c["rhythm_group"] == meta["key"]
            groups.setdefault(c["rhythm_group"], []).append(ev)
            n += 1
        for members in groups.values():
            assert all(CF.same_rhythm(members[0], ev) for ev in members[1:])
    assert n >= 12


def test_a_validation_song_is_never_a_base_rhythm(root, tmp_path):
    src, _ = root
    real = json.loads((src / "real" / "manifest.json").read_text())
    val_keys = {s["key"] for s in real["songs"] if s["split"] != "train"}
    made = {f.name.split("__")[0] for f in shards.shard_files(src / CF.SOURCE_REAL)}
    assert not made & val_keys
    # and a source that claims one is refused by the loader
    bad = tmp_path / "root"
    shutil.copytree(src, bad, symlinks=True)
    m = json.loads((bad / CF.SOURCE_STREAM / "manifest.json").read_text())
    m["songs"][0]["song_id"] = next(s["song_id"] for s in real["songs"] if s["split"] != "train")
    (bad / CF.SOURCE_STREAM / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="held-out"):
        D.load_sources(bad, (CF.SOURCE_STREAM,), log=lambda *a, **k: None)


# --------------------------------------------------------------------------- #
# Training on them
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def loaded(root):
    src, _ = root
    return D.load_sources(src, (CF.SOURCE_REAL, CF.SOURCE_STREAM), log=lambda *a, **k: None)


def test_footing_sliders_are_never_hidden_on_a_counterfactual_chart(loaded):
    ds = D.GroupedStageBDataset(loaded[CF.SOURCE_STREAM], train=True, seed=0)
    avail = np.stack([ds.item(i % len(ds), np.random.default_rng(i))["cond"].numpy()
                      [cond.STYLE_DIM:] for i in range(200)])
    footing = [i - 0 for i in D.FOOTING_FIELDS]
    assert (avail[:, footing] == 1.0).all()
    others = [i for i in range(1, 9) if i not in D.FOOTING_FIELDS]
    hidden = 1.0 - avail[:, others].mean()
    assert 0.25 < hidden < 0.65                     # the other sliders still drop out
    # ...and on a human chart every slider, the footing ones too, can be hidden
    real = D.StageBDataset(loaded["real"], train=True, seed=0)
    r = np.stack([real.item(i % len(real), np.random.default_rng(i))["cond"].numpy()
                  [cond.STYLE_DIM:] for i in range(200)])
    assert (r[:, footing] == 0.0).any()


def test_pairs_are_two_variants_of_one_rhythm_in_the_same_window(loaded):
    g = D.GroupedStageBDataset(loaded[CF.SOURCE_STREAM], train=True, seed=0)
    r = D.StageBDataset(loaded["real"], train=True, seed=0)
    mix = D.MixedDataset({"real": r, "synth_cf_stream": g},
                         [(0, {"real": 0.5, "synth_cf_stream": 0.5})], 200, seed=0)
    seen = {"real": 0, "synth_cf_stream": 0}
    for k in range(0, 200, 2):
        n0, j0, _ = mix.pick(k)
        n1, j1, _ = mix.pick(k + 1)
        seen[n0] += 1
        assert n0 == n1
        if n0 != "synth_cf_stream":
            continue
        c0, c1 = g.charts[j0], g.charts[j1]
        assert c0.group == c1.group and c0.variant != c1.variant
        a, b = mix[k], mix[k + 1]
        assert torch.equal(a["ctx"], b["ctx"]) and torch.equal(a["dtb"], b["dtb"])
        assert not torch.equal(a["tokens"], b["tokens"])
        diff = (a["cond"] - b["cond"]).abs().numpy()
        other = [i for i in range(cond.COND_DIM) if i not in D.FOOTING_FIELDS]
        assert diff[other].max() == 0.0             # same sliders hidden, same values
        assert diff[list(D.FOOTING_FIELDS)].max() > 0.0
    assert seen["real"] > 20 and seen["synth_cf_stream"] > 20


def test_stage_b_trains_the_counterfactual_mix_and_selects_on_the_probe(root, tmp_path):
    src, _ = root
    out, mirror = tmp_path / "b", tmp_path / "mirror"
    TB.main(["--corpus-root", str(src), "--out", str(out), "--mirror-dir", str(mirror),
             "--phase1-epochs", "1", "--phase2-epochs", "1", "--epoch-items", "8",
             "--batch", "4", "--workers", "0", "--amp", "off", "--warmup", "2",
             "--crop", str(8 * C.G), "--d-model", "32", "--layers", "1", "--heads", "2",
             "--d-audio", "16", "--p-drop-audio", "0.15", "--p-drop-audio2", "0.15",
             "--device", "cpu"])
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["select"] == "ce_probe" and cfg["data"]["mix"] == "v15b"
    assert cfg["data"]["phase1"]["real"] == pytest.approx(0.45)
    assert cfg["data"]["phase2"]["real_polish"] == pytest.approx(0.65)
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    for r in rows:
        assert 0.0 <= r["val_probe_sens"] <= 1.0
        for name in ("cross", "jack", "fsw"):
            assert r[f"val_probe_{name}_delta"] == pytest.approx(
                r[f"val_probe_{name}_hi"] - r[f"val_probe_{name}_lo"])
    for name in ("best.pt", "best_ce.pt", "last.pt", "log.jsonl", "config.json"):
        assert (mirror / name).exists(), name
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert "no_audio_embed" in state["model"]
