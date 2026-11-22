"""v1.5c: SpecAugment on the rhythm network's input. Only the acoustic channels
are ever masked, inside the stated bounds, and never outside training."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beatforge.adapters.stepmania.neural import codec as C  # noqa: E402
from beatforge.adapters.stepmania.neural import grid_features as GF  # noqa: E402
from beatforge.adapters.stepmania.neural.train import dataset as D  # noqa: E402
from beatforge.adapters.stepmania.neural.train import train_stage_a as TA  # noqa: E402

import synth_helpers as H  # noqa: E402

quiet = dict(log=lambda *_: None)
ACOUSTIC = slice(0, 54)
TIMING = slice(54, 64)


@pytest.fixture(scope="module")
def songs(tmp_path_factory):
    root = H.make_corpus_root(tmp_path_factory.mktemp("v15c"), train=2, val=1)
    return D.load_corpus(root / "real", "train", **quiet), D.load_corpus(root / "real", "val", **quiet)


def test_the_metric_timing_channels_are_the_last_ten():
    assert (GF.METRIC.start, GF.METRIC.stop) == (54, 64)
    assert D.SPEC_ACOUSTIC == 54


def _runs(mask: np.ndarray) -> list:
    """Lengths of the runs of True in a 1-D bool array."""
    out, n = [], 0
    for v in mask:
        if v:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


@pytest.mark.parametrize("seed", range(40))
def test_masks_stay_inside_their_bounds_and_off_the_timing_channels(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 2.0, (4 * C.G, 64)).astype(np.float32)   # no zero anywhere
    y = D.spec_augment(x.copy(), np.random.default_rng(1000 + seed))
    assert np.array_equal(y[:, TIMING], x[:, TIMING])             # never touched
    zero = y[:, ACOUSTIC] == 0.0
    assert zero.any()
    assert np.array_equal(y[:, ACOUSTIC][~zero], x[:, ACOUSTIC][~zero])   # masked, not altered
    # whole channels: one or two bands, 2-5 channels each (two may touch or overlap)
    bands = _runs(zero.all(axis=0))
    assert 1 <= len(bands) <= 2 and 2 <= sum(bands) <= 10 and min(bands) >= 2
    # whole ticks: two spans of 4-12 ticks (they may overlap into one)
    spans = _runs(zero.all(axis=1))
    assert 1 <= len(spans) <= 2 and 4 <= sum(spans) <= 24 and min(spans) >= 4
    # nothing else is masked
    assert (zero == (zero.all(axis=0)[None, :] | zero.all(axis=1)[:, None])).all()


def test_only_a_training_dataset_masks_and_only_when_asked(songs):
    train, val = songs
    plain = D.StageADataset(train, crop=4 * C.G, train=True, stems="off")
    masked = D.StageADataset(train, crop=4 * C.G, train=True, stems="off", spec_augment=True)
    assert not plain.spec_augment and masked.spec_augment
    differs = 0
    for epoch in range(3):
        plain.epoch = masked.epoch = epoch
        for i in range(len(plain)):
            a, b = plain[i], masked[i]
            # the same crop, the same labels, the same conditioning
            for k in ("cond", "valid", "onset", "jump", "hold_type", "hold_dur", "mine"):
                assert torch.equal(a[k], b[k]), k
            assert torch.equal(a["x"][:, TIMING], b["x"][:, TIMING])
            changed = (a["x"] != b["x"])
            assert not changed[:, TIMING].any()
            assert (b["x"][changed] == 0).all()                  # masking only ever zeroes
            differs += int(changed.any())
    assert differs >= 3 * len(plain) - 1

    # evaluation never masks, whatever it is handed
    off = D.StageADataset(val, crop=4 * C.G, train=False, stems="off")
    on = D.StageADataset(val, crop=4 * C.G, train=False, stems="off", spec_augment=True)
    assert not on.spec_augment
    for i in range(len(off)):
        assert torch.equal(off[i]["x"], on[i]["x"])


def test_the_trainer_flag_reaches_the_training_set_only(songs, tmp_path, monkeypatch):
    root = H.make_corpus_root(tmp_path, train=2, val=1)
    ap_args = ["--corpus-root", str(root), "--out", str(tmp_path / "run"), "--crop", str(4 * C.G)]

    def parse(extra):
        captured = {}

        def fake_build(args, log=print):
            captured["args"] = args
            raise SystemExit(0)

        monkeypatch.setattr(TA, "build_training_set", fake_build)
        with pytest.raises(SystemExit):
            TA.main(ap_args + extra)
        return captured["args"]

    assert parse([]).spec_augment is False and parse([]).save_every == 0
    args = parse(["--spec-augment", "--save-every", "5"])
    assert args.spec_augment is True and args.save_every == 5
    monkeypatch.undo()
    train, _, _ = TA.build_training_set(args, log=lambda *_: None)
    assert all(ds.spec_augment for ds in train.parts.values())
