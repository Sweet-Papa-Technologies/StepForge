"""REQ-SFV15-05 / REQ-SFV15-06 (2) — the v1.5 training terms, the multi-source
loader and its dropouts, and the exported graph's unchanged signature.
PyTorch-only; skipped cleanly where torch is not installed."""
from __future__ import annotations

import json
import math
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beatforge.adapters.stepmania.neural import codec as C  # noqa: E402
from beatforge.adapters.stepmania.neural import conditioning as cond  # noqa: E402
from beatforge.adapters.stepmania.neural import event_features as EF  # noqa: E402
from beatforge.adapters.stepmania.neural import grid_features as GF  # noqa: E402
from beatforge.adapters.stepmania.neural import infer  # noqa: E402
from beatforge.adapters.stepmania.neural.synth import audio_augment as AUG  # noqa: E402
from beatforge.adapters.stepmania.neural.synth import chart_thinning as TH  # noqa: E402
from beatforge.adapters.stepmania.neural.train import dataset as D  # noqa: E402
from beatforge.adapters.stepmania.neural.train import export_onnx as X  # noqa: E402
from beatforge.adapters.stepmania.neural.train import losses as L  # noqa: E402
from beatforge.adapters.stepmania.neural.train import train_stage_a as TA  # noqa: E402
from beatforge.adapters.stepmania.neural.train import train_stage_b as TB  # noqa: E402
from beatforge.adapters.stepmania.neural.train.choreo_net import (  # noqa: E402
    INPUT_NAMES, TRAINING_ONLY, ChoreoConfig, ChoreoNet)
from beatforge.adapters.stepmania.neural.train.rhythm_net import (  # noqa: E402
    OUTPUT_NAMES, RhythmConfig, RhythmNet)

import synth_helpers as H  # noqa: E402

quiet = dict(log=lambda *_: None)
T = 4 * C.G
SMALL_A = RhythmConfig(d_model=32, n_layers=1, n_heads=2)
SMALL_B = ChoreoConfig(d_model=32, n_layers=1, n_heads=2, d_audio=16)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v15")
    r = H.make_corpus_root(tmp, train=2, val=1)
    AUG.augment_corpus(r / "real", tmp / "stems", r / "aug_stem", variants_per_song=3,
                       workers=1, **quiet)
    TH.thin_corpus(r / "real", r / "synth_thin", **quiet)
    return r


# --------------------------------------------------------------------------- #
# REQ-SFV15-05 (1): the beat-phase margin
# --------------------------------------------------------------------------- #
def _phase_logits(score: float) -> "torch.Tensor":
    """Onset logits whose phase score is exactly `score`."""
    z = torch.zeros(1, T)
    z[0, ::12] = score
    return z.requires_grad_(True)


@pytest.mark.parametrize("score", [-3.0, 0.0, 1.0, 1.99])
def test_phase_margin_has_a_gradient_below_the_margin(score):
    z = _phase_logits(score)
    valid = torch.ones(1, T, dtype=torch.bool)
    loss = L.phase_margin_loss(z, valid, torch.tensor([True]))
    assert float(loss) == pytest.approx(2.0 - score, abs=1e-5)
    loss.backward()
    k = torch.arange(T) % 12
    g = z.grad[0]
    assert (g[k == 0] < 0).all()            # raise the quarter notes
    assert (g[k == 6] > 0).all()            # lower the off-beats
    assert (g[(k != 0) & (k != 6)] == 0).all()


@pytest.mark.parametrize("score", [2.0, 2.5, 9.0])
def test_phase_margin_gradient_is_exactly_zero_at_or_above_the_margin(score):
    z = _phase_logits(score)
    loss = L.phase_margin_loss(z, torch.ones(1, T, dtype=torch.bool), torch.tensor([True]))
    assert float(loss) == 0.0
    loss.backward()
    assert torch.count_nonzero(z.grad) == 0


def test_phase_margin_ignores_ineligible_items_and_invalid_ticks():
    z = torch.zeros(2, T, requires_grad=True)
    valid = torch.ones(2, T, dtype=torch.bool)
    valid[0, T // 2:] = False
    loss = L.phase_margin_loss(z, valid, torch.tensor([True, False]))
    loss.backward()
    assert torch.count_nonzero(z.grad[1]) == 0                  # not eligible
    assert torch.count_nonzero(z.grad[0, T // 2:]) == 0         # padding
    assert torch.count_nonzero(z.grad[0, :T // 2]) > 0
    none = L.phase_margin_loss(z, valid, torch.tensor([False, False]))
    assert float(none) == 0.0


def _batch(meter_norm: float, on: str) -> dict:
    onset = torch.zeros(1, T)
    onset[0, (0 if on == "beat" else 6)::12] = 1.0
    style = np.full(10, -1.0, np.float32)
    style[0], style[9] = meter_norm, 0.0
    return {"onset": onset, "valid": torch.ones(1, T, dtype=torch.bool),
            "cond": torch.from_numpy(cond.cond19(style))[None],
            "jump": torch.zeros(1, T), "hold_type": torch.zeros(1, T, dtype=torch.long),
            "hold_dur": torch.zeros(1, T, dtype=torch.long), "mine": torch.zeros(1, T),
            "x": torch.randn(1, T, GF.D_IN)}


def test_only_low_tier_charts_on_the_beat_are_eligible():
    assert bool(L.phase_eligible(_batch(0.10, "beat")))
    assert bool(L.phase_eligible(_batch(0.25, "beat")))
    assert not bool(L.phase_eligible(_batch(0.30, "beat")))     # not a low tier
    assert not bool(L.phase_eligible(_batch(0.10, "off")))      # notes on the off-beats


def test_stage_a_loss_adds_the_phase_term_only_when_asked():
    net = RhythmNet(SMALL_A)
    b = _batch(0.10, "beat")
    out = net(b["x"], b["cond"], b["valid"])
    plain, parts = L.stage_a_loss(out, b)
    with_phase, parts2 = L.stage_a_loss(out, b, lambda_phase=L.LAMBDA_PHASE)
    assert "phase" not in parts and parts2["phase"] > 0
    assert float(with_phase) == pytest.approx(
        float(plain) + L.LAMBDA_PHASE * parts2["phase"], rel=1e-5)


def test_half_beat_shift_moves_audio_and_targets_but_not_the_metric_block():
    b = _batch(0.10, "beat")
    b["x"] = torch.arange(T, dtype=torch.float32)[None, :, None].repeat(1, 1, GF.D_IN)
    for direction in (1, -1):
        m = L.shift_half_beat(b, direction)
        k = direction * 6
        assert torch.equal(m["x"][0, :, :54], torch.roll(b["x"][0, :, :54], k, dims=0))
        assert torch.equal(m["x"][0, :, 54:], b["x"][0, :, 54:])        # metric_timing
        assert torch.equal(m["onset"], torch.roll(b["onset"], k, dims=1))
        on = torch.nonzero(m["onset"][0]).flatten() % 12
        assert (on == 6).all()                                           # off the grid's beats
        wrapped = m["valid"][0, :6] if direction == 1 else m["valid"][0, -6:]
        assert not wrapped.any() and int(m["valid"].sum()) == T - 6
    assert GF.METRIC.start == 54


def test_phase_contrast_wants_the_true_grid_to_win_by_the_margin():
    true = torch.tensor([3.0, 0.0, -1.0], requires_grad=True)
    moved = torch.tensor([0.0, 0.0, 2.0])
    loss = L.phase_contrast_loss(true, moved)
    assert float(loss) == pytest.approx((0.0 + 1.5 + 4.5) / 3)
    loss.backward()
    assert true.grad[0] == 0 and true.grad[1] < 0 and true.grad[2] < 0
    assert float(L.phase_contrast_loss(true[:0], moved[:0])) == 0.0


# --------------------------------------------------------------------------- #
# the 16th-note weight
# --------------------------------------------------------------------------- #
def test_subdivision_weight_marks_16ths_and_triplets_of_dense_charts_only():
    w = L.subdivision_weight(48, torch.tensor([0.60, 0.45, 0.44, -1.0]))
    k = torch.arange(48)
    is16 = (k % 3 == 0) & (k % 6 != 0)
    is12 = (k % 4 == 0) & (k % 12 != 0)
    for row in (0, 1):
        assert (w[row][is16] == L.SUB16_WEIGHT).all()
        assert (w[row][is12] == L.SUB12_WEIGHT).all()
        assert (w[row][~is16 & ~is12] == 1.0).all()
    assert (w[2] == 1.0).all() and (w[3] == 1.0).all()          # Medium and below; masked
    assert int(is16.sum()) == 8 and int(is12.sum()) == 8 and not (is16 & is12).any()


def test_the_weight_raises_the_cost_of_a_missed_16th_and_nothing_else():
    def loss_at(tick: int, weighted: bool) -> float:
        b = _batch(0.60, "beat")
        b["onset"] = torch.zeros(1, T)
        b["onset"][0, tick] = 1.0
        out = (torch.full((1, T), -4.0), torch.zeros(1, T), torch.zeros(1, T, 6),
               torch.zeros(1, T, 10), torch.full((1, T), -9.0))
        kw = dict(sub16_weight=L.SUB16_WEIGHT, sub12_weight=L.SUB12_WEIGHT) if weighted else {}
        return L.stage_a_loss(out, b, **kw)[1]["focal"]

    assert loss_at(3, True) == pytest.approx(L.SUB16_WEIGHT * loss_at(3, False), rel=1e-4)
    assert loss_at(4, True) == pytest.approx(L.SUB12_WEIGHT * loss_at(4, False), rel=1e-4)
    assert loss_at(0, True) == pytest.approx(loss_at(0, False), rel=1e-6)
    assert loss_at(6, True) == pytest.approx(loss_at(6, False), rel=1e-6)
    # a false alarm on a 16th costs the same as before: the weight is on positives
    b = _batch(0.60, "beat")
    b["onset"] = torch.zeros(1, T)
    hot = torch.full((1, T), -4.0)
    hot[0, 3] = 4.0
    out = (hot, torch.zeros(1, T), torch.zeros(1, T, 6), torch.zeros(1, T, 10),
           torch.full((1, T), -9.0))
    assert L.stage_a_loss(out, b, sub16_weight=1.45)[1]["focal"] == pytest.approx(
        L.stage_a_loss(out, b)[1]["focal"], rel=1e-6)


# --------------------------------------------------------------------------- #
# REQ-SFV15-05 (2): validation holds no synthetic or augmented item
# --------------------------------------------------------------------------- #
def test_validation_is_real_songs_only(root):
    src = D.load_sources(root, TA.STAGE_A_SOURCES, **quiet)
    val_keys = set(H.real_keys(root, "val"))
    assert {s.key for s in src["val"]} == val_keys
    assert all(s.source == "real" for s in src["val"])
    val_ids = {s.song_id for s in src["val"]}
    for name in ("real", "aug_stem", "synth_thin"):
        assert src[name], name
        assert not {s.song_id for s in src[name]} & val_ids, name
        assert not {s.key.split("__")[0] for s in src[name]} & val_keys, name
    assert {s.source for s in src["aug_stem"]} == {"aug_stem"}
    val_ds = D.StageADataset(src["val"], train=False, stems="off")
    assert len(val_ds) == 5 and all(c.song.source == "real" for c in val_ds.charts)


def test_a_training_only_source_listing_a_validation_song_is_refused(root, tmp_path):
    bad = tmp_path / "corpus"
    shutil.copytree(root, bad)
    path = bad / "aug_stem" / "manifest.json"
    m = json.loads(path.read_text())
    m["songs"][0]["split"] = "val"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="outside the training split"):
        D.load_sources(bad, TA.STAGE_A_SOURCES, **quiet)
    m["songs"][0]["split"] = "train"
    real = json.loads((bad / "real" / "manifest.json").read_text())
    m["songs"][0]["song_id"] = next(s["song_id"] for s in real["songs"] if s["split"] == "val")
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="held-out song id"):
        D.load_sources(bad, TA.STAGE_A_SOURCES, **quiet)
    with pytest.raises(ValueError, match="unknown corpus source"):
        D.load_sources(root, ("made_up",), **quiet)


def test_attached_charts_borrow_the_parents_audio_and_companions_weigh_half(root):
    src = D.load_sources(root, TA.STAGE_A_SOURCES, **quiet)
    real = {s.key: s for s in src["real"]}
    for s in src["synth_thin"]:
        parent = real[s.key.split("__")[0]]
        assert s.core is parent.core and s.stems is parent.stems     # shared, not copied
        assert s.song_id == parent.song_id
        assert [c.difficulty for c in s.charts] == ["beginner", "easy"]
        assert all(c.weight == D.COMPANION_WEIGHT for c in s.charts)
    assert all(c.weight == 1.0 for s in src["real"] for c in s.charts)


# --------------------------------------------------------------------------- #
# the mixer
# --------------------------------------------------------------------------- #
def _mixed(root, seed=0):
    src = D.load_sources(root, TA.STAGE_A_SOURCES, **quiet)
    parts = {k: D.StageADataset(v, crop=4 * C.G) for k, v in src.items() if k != "val"}
    parts["aug_mild"] = D.StageADataset(
        [s for s in src["aug_stem"] if s.kind in TA.MILD_KINDS], crop=4 * C.G)
    return D.MixedDataset(parts, [(0, TA.PHASE1_SHARES), (3, TA.PHASE2_SHARES)], 400, seed)


def test_mixer_follows_the_schedule_and_redistributes_missing_sources(root):
    mix = _mixed(root)
    # no synthesized grooves in this corpus: their share goes to the others
    p1, p2 = mix.shares(0), mix.shares(3)
    assert set(p1) == {"real", "aug_stem", "synth_thin"} and sum(p1.values()) == pytest.approx(1)
    assert p1["real"] == pytest.approx(0.35 / 0.80) and p1["aug_stem"] == pytest.approx(0.30 / 0.80)
    assert set(p2) == {"real", "aug_mild", "synth_thin"}
    assert p2["real"] == pytest.approx(0.55 / 0.95)
    assert mix.shares(2) == p1 and mix.shares(99) == p2
    for epoch, want in ((0, p1), (5, p2)):
        mix.epoch = epoch
        names = [mix.pick(i)[0] for i in range(len(mix))]
        for name, share in want.items():
            assert names.count(name) / len(names) == pytest.approx(share, abs=0.08)
        assert set(names) == set(want)


def test_mixer_items_are_a_function_of_seed_epoch_and_index(root):
    a, b = _mixed(root), _mixed(root)
    a.epoch = b.epoch = 2
    for i in (0, 7, 123):
        assert a.pick(i)[:2] == b.pick(i)[:2]
        assert torch.equal(a[i]["x"], b[i]["x"]) and torch.equal(a[i]["onset"], b[i]["onset"])
    b.epoch = 3
    assert [a.pick(i)[:2] for i in range(40)] != [b.pick(i)[:2] for i in range(40)]
    c = _mixed(root, seed=1)
    c.epoch = 2
    assert [a.pick(i)[:2] for i in range(40)] != [c.pick(i)[:2] for i in range(40)]
    with pytest.raises(ValueError, match="no training data"):
        D.MixedDataset({"real": D.StageADataset([])}, [(0, {"real": 1.0})], 10)


# --------------------------------------------------------------------------- #
# stem dropout (REQ-SFV15-01)
# --------------------------------------------------------------------------- #
def test_stem_dropout_zeroes_the_stem_channels_and_the_flag(root):
    songs = D.load_corpus(root / "real", "train", **quiet)
    ds = D.StageADataset(songs, crop=4 * C.G, train=True, p_drop_stems=0.5)
    dropped = kept = 0
    for epoch in range(8):
        ds.epoch = epoch
        for i in range(len(ds)):
            it = ds[i]
            stem = it["x"][:, GF.STEM]
            if float(it["cond"][9]) == 0.0:
                assert not stem.any()                            # exactly zero
                dropped += 1
            else:
                assert float(it["cond"][9]) == 1.0 and stem.any()
                kept += 1
            assert torch.isfinite(it["x"]).all()
    assert 0.3 < dropped / (dropped + kept) < 0.7
    always = D.StageADataset(songs, crop=4 * C.G, train=True, p_drop_stems=1.0)
    assert all(not always[i]["x"][:, GF.STEM].any() for i in range(len(always)))
    # evaluation: 'off' is the app (no separator), 'on' the stem-fed score
    off = D.StageADataset(songs, crop=4 * C.G, train=False, stems="off")
    on = D.StageADataset(songs, crop=4 * C.G, train=False, stems="on")
    for i in range(len(off)):
        assert not off[i]["x"][:, GF.STEM].any() and float(off[i]["cond"][9]) == 0.0
        assert on[i]["x"][:, GF.STEM].any() and float(on[i]["cond"][9]) == 1.0
        other = np.r_[0:GF.STEM.start, GF.STEM.stop:GF.D_IN]
        assert torch.equal(off[i]["x"][:, other], on[i]["x"][:, other])


# --------------------------------------------------------------------------- #
# audio dropout in Stage B and the training-only vector
# --------------------------------------------------------------------------- #
def _symbolic_song() -> D.Song:
    s = D.Song()
    s.key, s.song_id, s.source, s.kind = "synth_style_00000", "synth_style_00000", "synth_style", None
    s.core = s.dt_ms = s.stems = None
    s.n_ticks = 8 * C.G
    c = D.Chart()
    c.song, c.j, c.difficulty, c.meter = s, 0, "hard", 9
    c.tick = np.arange(0, 8 * C.G, 6, dtype=np.int64)
    n = len(c.tick)
    c.tms = C.event_times_ms(c.tick, 150.0)
    c.tok = (np.arange(n) % 4).astype(np.uint8)
    c.flav = np.zeros(n, np.uint8)
    c.da = c.ds = np.zeros(n, np.int32)
    c.mines = np.zeros((0, 2), np.int32)
    c.style = np.array([0.6, 0.3, 0.9, 0, 0, 0, 0, 0, 0, 0], np.float32)
    c.uid, c.weight = 1, 1.0
    s.charts = [c]
    return s


def test_symbolic_sequences_always_drop_audio_and_real_ones_at_the_asked_rate(root):
    sym = D.StageBDataset([_symbolic_song()], train=True)
    it = sym[0]
    assert bool(it["drop_audio"]) and not it["ctx"].any()
    assert it["ctx"].shape[1:] == (EF.CTX_K, GF.D_IN) and float(it["cond"][9]) == 0.0
    songs = D.load_corpus(root / "real", "train", **quiet)
    never = D.StageBDataset(songs, train=True, p_drop_audio=0.0)
    assert not any(bool(never[i]["drop_audio"]) for i in range(len(never)))
    ds = D.StageBDataset(songs, train=True, p_drop_audio=0.5)
    flags = []
    for epoch in range(8):
        ds.epoch = epoch
        flags += [bool(ds[i]["drop_audio"]) for i in range(len(ds))]
    assert 0.3 < np.mean(flags) < 0.7
    val = D.StageBDataset(songs, train=False, p_drop_audio=1.0)
    assert not any(bool(val[i]["drop_audio"]) for i in range(len(val)))   # never in eval
    batch = D.collate_b([sym[0], ds[0]])
    assert batch["drop_audio"].dtype == torch.bool and batch["drop_audio"].shape == (2,)
    assert bool(batch["drop_audio"][0])


def _b_inputs(n=12, batch=2, seed=0):
    return X.stage_b_inputs(batch, n, SMALL_B, seed=seed)


def test_dropped_audio_is_replaced_by_the_learned_vector():
    net = ChoreoNet(SMALL_B).eval()
    ctx, scal, tok, cv = _b_inputs()
    other_ctx = torch.randn_like(ctx)
    drop = torch.tensor([True, False])
    a = net(ctx, scal, tok, cv, None, drop)
    b = net(other_ctx, scal, tok, cv, None, drop)
    assert torch.allclose(a[0], b[0], atol=1e-6)        # dropped: the audio is not read
    assert not torch.allclose(a[1], b[1], atol=1e-4)    # kept: it is
    plain = net(ctx, scal, tok, cv)
    assert torch.allclose(plain[1], a[1], atol=1e-6)    # an undropped row is untouched
    assert torch.allclose(net(ctx, scal, tok, cv, None, torch.tensor([False, False])),
                          plain, atol=1e-6)


def test_no_audio_vector_learns_only_from_dropped_sequences():
    net = ChoreoNet(SMALL_B)
    assert net.no_audio_embed.shape == (SMALL_B.d_audio,)
    assert 0 < float(net.no_audio_embed.abs().max()) < 0.2          # N(0, 0.02) init
    ctx, scal, tok, cv = _b_inputs()
    net(ctx, scal, tok, cv, None, torch.tensor([False, False])).sum().backward()
    assert net.no_audio_embed.grad is None or not net.no_audio_embed.grad.any()
    net.zero_grad()
    net(ctx, scal, tok, cv, None, torch.tensor([True, False])).sum().backward()
    assert net.no_audio_embed.grad.any()


# --------------------------------------------------------------------------- #
# REQ-SFV15-06 (2): the exported graphs
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    out = tmp_path_factory.mktemp("export")
    a, b = RhythmNet(SMALL_A), ChoreoNet(SMALL_B)
    with torch.no_grad():
        b.no_audio_embed.fill_(0.37)        # a value that would be unmistakable
    meta = X.export_models(a, b, out, version="1.5")
    return out, meta, a, b


def test_training_only_vector_is_absent_from_the_exported_graph(exported):
    import onnx
    out, meta, _, b = exported
    assert X.training_only_leaks(out / X.STAGE_B_FILE) == []
    graph = onnx.load(str(out / X.STAGE_B_FILE)).graph
    for init in graph.initializer:
        assert not any(name in init.name for name in TRAINING_ONLY)
        arr = onnx.numpy_helper.to_array(init)
        assert not (arr.shape == (SMALL_B.d_audio,) and np.allclose(arr, 0.37))
    assert "no_audio_embed" in b.state_dict()                   # ...but it is trained


def test_exported_signatures_match_the_shipped_model(exported):
    out, meta, _, _ = exported
    shipped = infer.default_model_dir()
    for name in (X.STAGE_A_FILE, X.STAGE_B_FILE):
        new, old = X.graph_signature(out / name), X.graph_signature(shipped / name)
        assert list(new["inputs"]) == list(old["inputs"])
        assert list(new["outputs"]) == list(old["outputs"])
        for io in ("inputs", "outputs"):
            for key, shape in new[io].items():
                # same rank and the same dynamic axes; only the layer width
                # of this miniature model differs
                assert len(shape) == len(old[io][key]), key
                assert [isinstance(d, str) for d in shape] == \
                    [isinstance(d, str) for d in old[io][key]], key
    assert list(X.graph_signature(out / X.STAGE_B_FILE)["inputs"]) == list(INPUT_NAMES)
    assert list(X.graph_signature(out / X.STAGE_A_FILE)["outputs"]) == list(OUTPUT_NAMES)
    assert meta["version"] == "1.5" and meta["display_name"] == "StepForge Flow v1.5"
    assert meta["stage_b"]["inputs"] == list(INPUT_NAMES)


def test_exported_stage_b_matches_pytorch_eval(exported):
    out, meta, _, b = exported
    assert meta["stage_b"]["parity_max_abs_diff"] <= X.PARITY_ATOL
    assert X.verify_stage_b(b, out / X.STAGE_B_FILE) <= X.PARITY_ATOL
    # the runtime loads it and never needs torch for it
    sess = X._session(out / X.STAGE_B_FILE)
    ctx, scal, tok, cv = _b_inputs(n=7, batch=1)
    got = sess.run(None, {"ctx": ctx.numpy(), "scalars": scal.numpy(),
                          "tokens": tok.numpy(), "cond": cv.numpy()})[0]
    with torch.no_grad():
        assert np.allclose(got, b.eval()(ctx, scal, tok, cv).numpy(), atol=X.PARITY_ATOL)


def test_a_pre_v15_checkpoint_still_loads(tmp_path):
    net = ChoreoNet(SMALL_B)
    state = {k: v for k, v in net.state_dict().items() if k not in TRAINING_ONLY}
    torch.save({"model": state, "config": SMALL_B.to_dict()}, tmp_path / "old.pt")
    model, _ = X.load_checkpoint(tmp_path / "old.pt", "b")
    assert isinstance(model, ChoreoNet)
    state.pop("head.bias")
    torch.save({"model": state, "config": SMALL_B.to_dict()}, tmp_path / "broken.pt")
    with pytest.raises(RuntimeError, match="does not fit"):
        X.load_checkpoint(tmp_path / "broken.pt", "b")


# --------------------------------------------------------------------------- #
# the trainers end to end, tiny
# --------------------------------------------------------------------------- #
def test_two_phase_schedule_shape():
    lr = [TA.lr_two_phase(s, 100, 50, 3e-4, 5e-5, 1e-6, 10) for s in range(151)]
    assert lr[0] == pytest.approx(3e-5) and lr[9] == pytest.approx(3e-4)
    assert lr[10] == pytest.approx(3e-4) and lr[100] == pytest.approx(5e-5)
    assert lr[150] == pytest.approx(1e-6)
    assert all(a >= b - 1e-12 for a, b in zip(lr[10:], lr[11:]))        # never rises again


def test_stage_a_trains_on_a_mixed_corpus_and_validates_on_real_songs(root, tmp_path):
    out = tmp_path / "a"
    TA.main(["--corpus-root", str(root), "--out", str(out), "--phase1-epochs", "1",
             "--phase2-epochs", "1", "--epoch-items", "8", "--batch", "4", "--workers", "0",
             "--amp", "off", "--warmup", "2", "--crop", str(8 * C.G), "--d-model", "32",
             "--layers", "1", "--heads", "2", "--sub16-weight", "1.45", "--sub12-weight",
             "1.3", "--phase-margin-weight", "0.1", "--phase-shift-prob", "1.0",
             "--device", "cpu"])
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in rows] == [1, 2]
    for r in rows:
        for key in ("val_ap", "val_f1_hard16", "val_phase_acc", "val_val_score",
                    "val_stems_ap", "train_phase"):
            assert np.isfinite(r[key]), key
        assert r["val_val_score"] == pytest.approx(
            0.5 * r["val_ap"] + 0.25 * r["val_f1_hard16"] + 0.25 * r["val_phase_acc"])
        assert r["val_n_hard16"] > 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["select"] == "val_score" and "aug_mild" in cfg["data"]["sources"]
    assert (out / "best.pt").exists() and (out / "last.pt").exists()


def test_stage_b_trains_with_audio_dropout_and_logs_the_style_probe(root, tmp_path):
    out = tmp_path / "b"
    TB.main(["--corpus-root", str(root), "--out", str(out), "--phase1-epochs", "1",
             "--phase2-epochs", "1", "--epoch-items", "8", "--batch", "4", "--workers", "0",
             "--amp", "off", "--warmup", "2", "--crop", str(8 * C.G), "--d-model", "32",
             "--layers", "1", "--heads", "2", "--d-audio", "16", "--p-drop-audio", "0.5",
             "--real-aug-share", "0.3", "--probe-every", "1", "--device", "cpu"])
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in rows] == [1, 2]
    for r in rows:
        for name in ("cross", "jack", "fsw"):
            for key in (f"val_probe_mae_{name}", f"val_probe_{name}_lo", f"val_probe_{name}_hi"):
                assert 0.0 <= r[key] <= 1.0, key
        assert np.isfinite(r["val_ce"])
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert "no_audio_embed" in state["model"]


def test_the_style_probe_sampler_draws_rare_moves_and_argmax_never_does():
    """Most-confident-first decoding is mode-seeking: a move the model gives
    10 % can never come out, so nothing a slider does to it can be seen. The
    probe therefore samples the way the app does."""
    n, rare = 400, 3

    def model(ctx, scalars, tokens, cond, pad):
        logits = torch.full((*tokens.shape, C.N_TOKENS), -1e9)
        logits[..., 0] = math.log(0.9)
        logits[..., rare] = math.log(0.1)
        return logits

    batch = {"tokens": torch.zeros(2, n, dtype=torch.long), "pad": torch.ones(2, n, dtype=torch.bool),
             "class_mask": torch.ones(2, n, C.N_TOKENS, dtype=torch.bool),
             "ctx": None, "scalars": None, "cond": None}
    greedy = TB.sample_unmask(model, batch, tau=0.0)
    assert (greedy == rare).sum() == 0 and (greedy == 0).all()
    gen = torch.Generator().manual_seed(0)
    drawn = TB.sample_unmask(model, batch, tau=1.0, generator=gen, order="random")
    assert set(drawn.unique().tolist()) == {0, rare}
    assert 0.06 < (drawn == rare).float().mean().item() < 0.14
    with pytest.raises(ValueError):
        TB.sample_unmask(model, batch, order="alphabetical")
