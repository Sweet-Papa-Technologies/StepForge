"""REQ-SFN-05 (1) — the Stage A / Stage B training losses. PyTorch-only; skipped
cleanly where torch is not installed (the runtime does not need it)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beatforge.adapters.stepmania.neural import codec as C  # noqa: E402
from beatforge.adapters.stepmania.neural.train import losses as L  # noqa: E402
from beatforge.adapters.stepmania.neural.train.choreo_net import (  # noqa: E402
    ChoreoConfig, ChoreoNet, class_mask, mask_tokens)
from beatforge.adapters.stepmania.neural.train.rhythm_net import (  # noqa: E402
    RhythmConfig, RhythmNet)


def spike(pos: int, measures: int = 1, height: float = 1.0) -> "torch.Tensor":
    x = torch.zeros(1, 48 * measures)
    x[0, pos] = height
    return x


@pytest.mark.parametrize("shift", range(1, 11))
def test_wasserstein_gradient_is_smooth_and_nonzero_for_a_shifted_spike(shift):
    """A predicted note `shift` ticks late: the loss equals the shift and its
    gradient is finite, non-zero, and says 'move mass earlier'."""
    y = spike(12)
    logit = torch.full((1, 48), -12.0)        # a clean spike on a silent measure
    logit[0, 12 + shift] = 6.0
    logit.requires_grad_(True)
    loss = L.wasserstein_1d_measure_loss(torch.sigmoid(logit), y)
    loss.backward()
    g = logit.grad[0]
    assert torch.isfinite(loss) and torch.isfinite(g).all()
    assert g.abs().sum() > 0
    assert loss.item() == pytest.approx(shift, rel=0.15)
    # mass AT the true tick reduces the loss; mass even further from it than
    # the misplaced spike increases it — the gradient points toward the note
    assert g[12] < 0
    assert g[12 + shift + 6] > 0
    assert g[12] < g[12 + shift + 6]


def test_wasserstein_grows_with_distance_where_bce_does_not():
    y = spike(12)
    w1, bce = [], []
    for shift in (1, 4, 9):
        p = spike(12 + shift, height=0.999) + 1e-4
        w1.append(L.wasserstein_1d_measure_loss(p, y).item())
        bce.append(torch.nn.functional.binary_cross_entropy(p, y).item())
    assert w1[0] < w1[1] < w1[2]
    assert max(bce) - min(bce) < 1e-6          # BCE cannot tell near from far


def test_wasserstein_is_zero_on_a_match_and_skips_empty_measures():
    y = torch.zeros(1, 96)
    y[0, [0, 12, 24, 36]] = 1                   # measure 0 has notes, measure 1 none
    assert L.wasserstein_1d_measure_loss(y.clone(), y).item() == pytest.approx(0, abs=1e-5)
    p = y.clone()
    p[0, 60] = 1.0                              # stray mass in the EMPTY measure
    assert L.wasserstein_1d_measure_loss(p, y).item() == pytest.approx(0, abs=1e-5)
    with pytest.raises(ValueError):
        L.wasserstein_1d_measure_loss(torch.zeros(1, 50), torch.zeros(1, 50))


def test_cardinality_loss_is_zero_at_the_exact_count():
    y = torch.zeros(2, 96)
    y[0, [0, 12, 24]] = 1
    y[1, [3, 50, 51, 52]] = 1
    valid = torch.ones_like(y, dtype=torch.bool)
    # right COUNT per measure, wrong places: cardinality is satisfied
    p = torch.zeros(2, 96)
    p[0, [1, 13, 25]] = 1
    p[1, [40, 60, 70, 80]] = 1
    assert L.cardinality_loss(p, y, valid).item() == pytest.approx(0, abs=1e-6)
    p[0, 30] = 1                                # one extra note in one of 4 measures
    assert L.cardinality_loss(p, y, valid).item() == pytest.approx(0.25)
    # padded ticks are ignored
    valid[0, 48:] = False
    p2 = y.clone()
    p2[0, 70] = 1
    assert L.cardinality_loss(p2, y, valid).item() == pytest.approx(0, abs=1e-6)


def test_focal_loss_down_weights_easy_examples():
    y = torch.tensor([[1.0, 0.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    easy = L.focal_bce(torch.tensor([[6.0, -6.0]]), y, valid)
    hard = L.focal_bce(torch.tensor([[-6.0, 6.0]]), y, valid)
    assert easy.item() < 1e-6 and hard.item() > 1.0
    # alpha = 0.75 weights a missed note three times a false alarm
    miss = L.focal_bce(torch.tensor([[-6.0]]), torch.tensor([[1.0]]), valid[:, :1])
    alarm = L.focal_bce(torch.tensor([[6.0]]), torch.tensor([[0.0]]), valid[:, :1])
    assert miss.item() == pytest.approx(3 * alarm.item(), rel=1e-4)


def test_stage_a_loss_backpropagates_through_every_head():
    torch.manual_seed(0)
    model = RhythmNet(RhythmConfig(d_model=32, n_layers=1, n_heads=2, dropout=0.0))
    b, t = 2, 96
    batch = {"onset": (torch.rand(b, t) < 0.15).float(),
             "valid": torch.ones(b, t, dtype=torch.bool)}
    batch["jump"] = batch["onset"] * (torch.rand(b, t) < 0.3)
    batch["hold_type"] = (batch["onset"] * torch.randint(0, 6, (b, t))).long()
    batch["hold_dur"] = torch.randint(0, 10, (b, t))
    batch["mine"] = (torch.rand(b, t) < 0.02).float()
    out = model(torch.randn(b, t, 64), torch.rand(b, 19), batch["valid"])
    assert [tuple(o.shape) for o in out] == [(b, t), (b, t), (b, t, 6), (b, t, 10),
                                             (b, t), (b, t, 32)]
    loss, parts = L.stage_a_loss(out, batch)
    assert set(parts) == {"focal", "card", "w1", "jump", "hold_type", "hold_dur", "mine"}
    assert torch.isfinite(loss)
    loss.backward()
    for head in (model.onset, model.jump, model.hold_type, model.hold_dur, model.mine):
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0


def test_ergo_table_matches_the_codec_and_is_mirror_free():
    """C_ergo lives in token space, which is already mirror-invariant, so the
    symmetry that matters is in the cost table it is built from: swapping left
    and right everywhere leaves every cost unchanged."""
    table = L.ergo_table("cpu")
    assert tuple(table.shape) == (C.N_DT, 20, 20)
    assert table.min() >= 0 and table.max() <= 1
    np.testing.assert_allclose(table.numpy(), C.ergo_matrix())
    t = C.tables()
    ms = np.array([C.mirror_state(s) for s in range(C.N_STATES)])
    assert np.array_equal(t["cost"], t["cost"][:, [0, 2, 1, 3]][:, :, ms])


def test_stage_b_loss_prefers_comfortable_neighbours():
    """Same cross-entropy target, two beliefs about a masked note: the one that
    puts its mass on a double-step pays a larger ergonomic penalty."""
    tokens = torch.tensor([[C.DOWN, C.HOME, C.UP]])          # ALT, ALT, ALT
    pad = torch.ones(1, 3, dtype=torch.bool)
    masked = torch.tensor([[False, True, False]])
    batch = {"tokens": tokens, "pad": pad, "class_mask": class_mask(tokens >= 8),
             "dtb": torch.zeros(1, 3, dtype=torch.long)}       # fast gaps

    def loss_for(tok: int):
        logits = torch.zeros(1, 3, 20)
        logits[0, 1, tok] = 8.0
        return L.stage_b_loss(logits, batch, masked, lambda_ergo=1.0)[1]

    comfy, awkward = loss_for(C.HOME), loss_for(4 + C.HOME)   # ALT vs SAME (double-step)
    assert awkward["ergo"] > comfy["ergo"] + 0.3
    assert comfy["acc"] == 1.0 and awkward["acc"] == 0.0


def test_stage_b_loss_only_scores_legal_tokens_at_masked_notes():
    torch.manual_seed(1)
    model = ChoreoNet(ChoreoConfig(d_model=32, n_layers=1, n_heads=2, d_audio=16,
                                   dropout=0.0))
    b, n = 3, 24
    tokens = torch.randint(0, 20, (b, n))
    pad = torch.ones(b, n, dtype=torch.bool)
    pad[2, 15:] = False
    inp, masked = mask_tokens(tokens, pad, torch.Generator().manual_seed(0))
    assert not masked[~pad].any() and masked.any(dim=1).all()
    assert (inp[masked] == C.MASK).all() and (inp[~masked] == tokens[~masked]).all()
    batch = {"tokens": tokens, "pad": pad, "class_mask": class_mask(tokens >= 8),
             "dtb": torch.randint(0, C.N_DT, (b, n))}
    logits = model(torch.randn(b, n, 17, 64), torch.randn(b, n, 31), inp,
                   torch.rand(b, 19), pad)
    assert tuple(logits.shape) == (b, n, 20)
    loss, parts = L.stage_b_loss(logits, batch, masked)
    assert torch.isfinite(loss) and 0 <= parts["acc"] <= 1
    loss.backward()
    # a huge logit on an ILLEGAL class (a jump token at a single-step note)
    # cannot change the loss: the class is fixed by Stage A
    logits2 = logits.detach().clone()
    logits2[~batch["class_mask"]] += 50.0
    loss2, _ = L.stage_b_loss(logits2, batch, masked)
    assert loss2.item() == pytest.approx(loss.item(), rel=1e-5)


def test_mask_schedule_spans_light_to_heavy_masking():
    tokens = torch.zeros(400, 64, dtype=torch.long)
    pad = torch.ones_like(tokens, dtype=torch.bool)
    _, masked = mask_tokens(tokens, pad, torch.Generator().manual_seed(3))
    rate = masked.float().mean(1)
    assert rate.min() < 0.1 and rate.max() > 0.95
    assert 0.55 < rate.mean().item() < 0.72          # E[cos(pi u / 2)] = 2/pi


def test_blocks_start_as_identity_and_models_are_the_spec_size():
    a, b = RhythmNet(), ChoreoNet()
    assert 7e6 < a.n_params < 11e6, a.n_params        # "~8M"
    assert 12e6 < b.n_params < 17e6, b.n_params       # "~14M"
    a.eval()
    x, c = torch.randn(1, 48, 64), torch.rand(1, 19)
    with torch.no_grad():
        h0 = a.out_norm(a.in_proj(x))
        h = a(x, c)[5]
    torch.testing.assert_close(h, h0)                 # AdaLN-Zero: blocks are no-ops
