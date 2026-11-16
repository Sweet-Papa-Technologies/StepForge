"""REQ-SFN-06 — leak-free corpus ingestion and PyTorch -> ONNX parity."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from beatforge.adapters.stepmania.neural import codec as C
from beatforge.adapters.stepmania.neural import grid_features as GF
from beatforge.adapters.stepmania.neural.train import corpus

SSC = """#VERSION:0.83;
#TITLE:{title};
#ARTIST:{artist};
#MUSIC:{music};
#OFFSET:-0.250;
#BPMS:0.000={bpm:.3f};
#STOPS:;
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Easy;
#METER:3;
#NOTES:
{easy};
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Hard;
#METER:8;
#NOTES:
{hard};
#NOTEDATA:;
#STEPSTYPE:dance-double;
#DIFFICULTY:Hard;
#METER:8;
#NOTES:
00000000
00000000
00000000
00000000
;
"""
EASY_BAR = "1000\n0000\n0001\n0000"
HARD_BAR = "1000\n0100\n0010\n0001\n1000\n0010\n0100\n0001"


def _clicks(bpm: float, dur_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(int(dur_s * 22050), np.float32)
    period = 60.0 / bpm / 2                       # clicks on 8ths
    t = 0.25
    while t < dur_s - 0.05:
        i = int(t * 22050)
        y[i:i + 60] += np.hanning(120)[60:].astype(np.float32) * rng.standard_normal(60)
        t += period
    return 0.6 * y / np.abs(y).max()


def _song(root, name, title, artist, bpm, seed, audio=None):
    d = root / name
    d.mkdir(parents=True)
    wav = d / f"{name}.wav"
    if audio is None:
        sf.write(wav, _clicks(bpm, 34.0, seed), 22050)
    else:
        shutil.copyfile(audio, wav)
    bars = int(34.0 * bpm / 60 / 4) - 1
    (d / f"{name}.ssc").write_text(SSC.format(
        title=title, artist=artist, music=wav.name, bpm=bpm,
        easy="\n,\n".join([EASY_BAR] * bars), hard="\n,\n".join([HARD_BAR] * bars)))
    return wav


@pytest.fixture(scope="module")
def mini_corpus(tmp_path_factory):
    """A 3-song pack: A and B are different songs; A2 is A's audio repackaged
    under another title in another folder."""
    root = tmp_path_factory.mktemp("packs")
    out = tmp_path_factory.mktemp("corpus")
    a = _song(root / "PackOne", "alpha", "Alpha", "Artist A", 120.0, 1)
    _song(root / "PackOne", "beta", "Beta", "Artist B", 150.0, 2)
    _song(root / "PackTwo", "alpha_repack", "Totally Different Name", "Someone", 120.0, 9,
          audio=a)
    stats = corpus.compile_corpus(root, out, workers=1, log=lambda *_: None)
    manifest = corpus.build_manifest(out, log=lambda *_: None)
    return out, stats, manifest


def test_corpus_compiles_dance_single_charts_only(mini_corpus):
    out, stats, manifest = mini_corpus
    assert stats == {"ok": 3, "rejected": 0}
    assert manifest["n_songs"] == 3 and manifest["n_charts"] == 6     # no dance-double
    meta, arrays = corpus.load_song(next((out / "songs").glob("*.npz")))
    assert arrays["core"].shape[1] == GF.D_CORE and arrays["core"].dtype == np.float16
    assert len(arrays["core"]) % C.G == 0 == len(arrays["dt_ms"]) % C.G
    assert meta["sync"]["contrast"] > 1.5          # the charts sit on the clicks
    assert abs(meta["sync_shift_applied_s"]) <= 0.02
    for ch in meta["charts"]:
        j = ch["j"]
        raw, rawd = arrays[f"c{j}_raw"], arrays[f"c{j}_rawd"]
        ev = C.ChartEvents(arrays[f"c{j}_tick"], raw[0], raw[1], raw[2], raw[3],
                           rawd[0], rawd[1], arrays[f"c{j}_mines"])
        enc = C.EncodedChart(ch["f_init"], arrays[f"c{j}_tick"], arrays[f"c{j}_tok"],
                             arrays[f"c{j}_flav"], arrays[f"c{j}_da"], arrays[f"c{j}_ds"],
                             arrays[f"c{j}_mines"])
        assert C.decode(enc).equals(ev)             # stored tokens reproduce the rows
        assert arrays[f"c{j}_style"].shape == (10,)
        assert ch["difficulty"] in ("easy", "hard") and ch["lock_violations"] == 0


def test_identical_audio_shares_a_song_id_and_never_crosses_splits(mini_corpus):
    _, _, manifest = mini_corpus
    by_title = {s["title"]: s for s in manifest["songs"]}
    a, a2, b = by_title["Alpha"], by_title["Totally Different Name"], by_title["Beta"]
    assert a["song_id"] == a2["song_id"], "repackaged audio must keep its song_id"
    assert a["split"] == a2["split"]
    assert b["song_id"] != a["song_id"]
    assert manifest["n_song_ids"] == 2
    # the split is a pure function of the song id
    for s in manifest["songs"]:
        assert s["split"] == corpus.split_of(s["song_id"])


def test_song_ids_merge_on_title_and_on_spectral_fingerprint():
    def m(sha, name, dur):
        return {"audio_sha256": sha, "name_key": name, "duration_s": dur}
    rng = np.random.default_rng(0)
    fp = rng.standard_normal((4, 320)).astype(np.float32)
    fp /= np.linalg.norm(fp, axis=1, keepdims=True)
    fp[3] = fp[2] + 0.005 * rng.standard_normal(320)       # a re-encode of song 2
    fp[3] /= np.linalg.norm(fp[3])
    metas = [m("aa", "song|artist", 100.0), m("bb", "song|artist", 131.0),  # same name
             m("cc", "other|x", 90.0), m("dd", "renamed|y", 90.4)]          # same audio
    ids = corpus.assign_song_ids(metas, fp)
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]
    # same length but a different spectrum stays separate
    fp[3] = rng.standard_normal(320)
    fp[3] /= np.linalg.norm(fp[3])
    ids = corpus.assign_song_ids(metas, fp)
    assert ids[2] != ids[3]


def test_splits_are_roughly_85_7_7_and_deterministic():
    ids = [f"s{i:06d}" for i in range(4000)]
    counts = {"train": 0, "val": 0, "test": 0}
    for i in ids:
        counts[corpus.split_of(i)] += 1
    assert 0.82 < counts["train"] / 4000 < 0.88
    assert 0.055 < counts["val"] / 4000 < 0.095
    assert 0.055 < counts["test"] / 4000 < 0.095
    assert [corpus.split_of(i) for i in ids[:50]] == [corpus.split_of(i) for i in ids[:50]]


def test_quality_gate_names_its_reason():
    song = {"n_stops": 0, "bpm_min": 120, "bpm_max": 120, "timesig_44": True,
            "sync": {"contrast": 2.0}}
    chart = {"n_events": 200, "sustained_nps": 5.0, "hold_share": 0.05,
             "hand_share": 0.0, "offgrid_dropped": 0, "mean_cost": 1.5,
             "lock_violations": 0, "meter": 8, "difficulty": "hard"}
    assert corpus.song_verdict(song) is None and corpus.chart_verdict(song, chart) is None
    assert corpus.song_verdict({**song, "n_stops": 9}) == "stops"
    assert corpus.song_verdict({**song, "bpm_max": 400}) == "bpm_range"
    assert corpus.song_verdict({**song, "sync": {"contrast": 1.0}}) == "sync"
    assert corpus.chart_verdict(song, {**chart, "n_events": 10}) == "too_few_notes"
    assert corpus.chart_verdict(song, {**chart, "sustained_nps": 30}) == "nps"
    assert corpus.chart_verdict(song, {**chart, "hold_share": 0.6}) == "hold_share"
    assert corpus.chart_verdict(song, {**chart, "hand_share": 0.1}) == "hands"
    assert corpus.chart_verdict(song, {**chart, "mean_cost": 40.0}) == "discomfort"


def test_dataset_respects_the_manifest_split(mini_corpus):
    pytest.importorskip("torch")
    from beatforge.adapters.stepmania.neural.train import dataset as D
    out, _, manifest = mini_corpus
    seen = {}
    for split in ("train", "val", "test"):
        for s in D.load_corpus(out, split, log=lambda *_: None):
            assert s.song_id not in seen or seen[s.song_id] == split
            seen[s.song_id] = split
    assert len(seen) == manifest["n_song_ids"]
    songs = [s for sp in ("train", "val", "test")
             for s in D.load_corpus(out, sp, log=lambda *_: None)]
    a = D.StageADataset(songs, crop=4 * C.G, train=False)[0]
    assert tuple(a["x"].shape) == (4 * C.G, 64) and tuple(a["cond"].shape) == (19,)
    assert a["onset"].sum() > 0 and a["valid"].all()
    b = D.StageBDataset(songs, train=False)[0]
    n = len(b["tokens"])
    assert tuple(b["ctx"].shape) == (n, 17, 64) and tuple(b["scalars"].shape) == (n, 31)
    batch = D.collate_b([b, D.StageBDataset(songs, train=False)[1]])
    assert batch["pad"].sum() == n + len(D.StageBDataset(songs, train=False)[1]["tokens"])


# --------------------------------------------------------------------------- #
# ONNX
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from beatforge.adapters.stepmania.neural.train import export_onnx as E
    from beatforge.adapters.stepmania.neural.train.choreo_net import ChoreoConfig, ChoreoNet
    from beatforge.adapters.stepmania.neural.train.rhythm_net import RhythmConfig, RhythmNet
    torch.manual_seed(0)
    a = RhythmNet(RhythmConfig(d_model=32, n_layers=1, n_heads=2))
    b = ChoreoNet(ChoreoConfig(d_model=32, n_layers=1, n_heads=2, d_audio=16))
    # AdaLN-Zero makes every block the identity at initialisation; randomise so
    # the parity check exercises attention, convolution and conditioning.
    for model in (a, b):
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.15)
    out = tmp_path_factory.mktemp("models")
    meta = E.export_models(a, b, out, metrics={"note": "random init"})
    return E, a, b, out, meta


def test_tiny_models_export_with_parity(tiny_export):
    E, a, b, out, meta = tiny_export
    assert (out / E.STAGE_A_FILE).exists() and (out / E.STAGE_B_FILE).exists()
    assert meta["stage_a"]["parity_max_abs_diff"] < 1e-4
    assert meta["stage_b"]["parity_max_abs_diff"] < 1e-4
    on_disk = json.loads((out / E.META_FILE).read_text())
    assert on_disk["codec_version"] == C.CODEC_VERSION == "1.0"
    assert on_disk["opset"] >= 17
    assert on_disk["features"]["d_in"] == 64 and on_disk["features"]["ticks_per_measure"] == 48
    assert on_disk["metrics"] == {"note": "random init"}
    assert on_disk["trained_with_stems"] is False


def test_onnx_graphs_accept_dynamic_batch_and_length(tiny_export):
    import torch
    E, a, b, out, _ = tiny_export
    sa, sb = E._session(out / E.STAGE_A_FILE), E._session(out / E.STAGE_B_FILE)
    assert [o.name for o in sa.get_outputs()] == [
        "onset_logit", "jump_logit", "hold_type_logits", "hold_dur_logits",
        "mine_logit", "hidden_state"]
    assert [i.name for i in sb.get_inputs()] == ["ctx", "scalars", "tokens", "cond"]
    for batch, ticks in ((1, 48), (3, 240), (1, 1536)):
        x, c = E.stage_a_inputs(batch, ticks, a.cfg, seed=ticks)
        got = sa.run(None, {"x": x.numpy(), "cond": c.numpy()})
        with torch.no_grad():
            ref = a(x, c)
        assert got[0].shape == (batch, ticks) and got[2].shape == (batch, ticks, 6)
        for r, o in zip(ref, got):
            np.testing.assert_allclose(o, r.numpy(), atol=1e-4)
    for batch, events in ((1, 1), (2, 64), (1, 700)):
        ins = E.stage_b_inputs(batch, events, b.cfg, seed=events)
        got = sb.run(None, {n: t.numpy() for n, t in zip(("ctx", "scalars", "tokens",
                                                           "cond"), ins)})[0]
        with torch.no_grad():
            ref = b(*ins).numpy()
        assert got.shape == (batch, events, 20)
        np.testing.assert_allclose(got, ref, atol=1e-4)


def test_export_refuses_to_write_meta_when_parity_fails(tiny_export, tmp_path, monkeypatch):
    E, a, b, _, _ = tiny_export
    monkeypatch.setattr(E, "PARITY_ATOL", -1.0)       # nothing can pass
    with pytest.raises(AssertionError, match="parity"):
        E.export_models(a, b, tmp_path)
    assert not (tmp_path / E.META_FILE).exists()
