"""losses.py — Stage A composite rhythm loss and the Stage B masked-diffusion
objective (spec §6.2, §6.4, REQ-SFN-05).

Stage A:  L = Focal(p, y) + l_card * |sum_m p - sum_m y| + l_W1 * W1_m(p, y) + L_aux

Plain BCE on a sparse grid is satisfied by a blur: a note predicted one tick
late is exactly as wrong as one predicted a beat late. The cardinality term
makes each measure hold the right NUMBER of notes and the Wasserstein term the
right PLACEMENT within the measure, with a gradient that grows with how far a
note is off — which is the signal BCE cannot give.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..codec import G, N_TOKENS, ergo_matrix

LAMBDA_CARD = 0.15
LAMBDA_W1 = 0.50
LAMBDA_MINE = 0.25
LAMBDA_ERGO = 0.05
FOCAL_ALPHA, FOCAL_GAMMA = 0.75, 2.0


def focal_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
              alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA) -> torch.Tensor:
    """Focal binary cross-entropy, averaged over `valid` positions."""
    logits, target = logits.float(), target.float()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    loss = w * (1 - pt).pow(gamma) * ce
    v = valid.float()
    return (loss * v).sum() / v.sum().clamp_min(1.0)


def _measures(x: torch.Tensor) -> torch.Tensor:
    b, t = x.shape
    if t % G:
        raise ValueError(f"sequence length {t} is not a whole number of measures")
    return x.reshape(b, t // G, G)


def cardinality_loss(p: torch.Tensor, y: torch.Tensor, valid: torch.Tensor
                     ) -> torch.Tensor:
    """Mean over real measures of |predicted note count - true note count|."""
    v = _measures(valid.float())
    pm = (_measures(p.float()) * v).sum(-1)
    ym = (_measures(y.float()) * v).sum(-1)
    real = (v.sum(-1) > 0).float()
    return ((pm - ym).abs() * real).sum() / real.sum().clamp_min(1.0)


def wasserstein_1d_measure_loss(p: torch.Tensor, y: torch.Tensor,
                                valid: torch.Tensor | None = None,
                                eps: float = 1e-6) -> torch.Tensor:
    """Per-measure 1-D Wasserstein distance between the predicted and true note
    distributions: sum over the 48 ticks of |CDF_m(p) - CDF_m(y)|, averaged
    over measures that contain at least one true note."""
    pm, ym = _measures(p.float()), _measures(y.float())
    if valid is not None:
        v = _measures(valid.float())
        pm, ym = pm * v, ym * v
    cdf_p = pm.cumsum(-1) / (eps + pm.sum(-1, keepdim=True))
    cdf_y = ym.cumsum(-1) / (eps + ym.sum(-1, keepdim=True))
    w1 = (cdf_p - cdf_y).abs().sum(-1)
    has = (ym.sum(-1) > 0).float()
    return (w1 * has).sum() / has.sum().clamp_min(1.0)


def stage_a_loss(out: tuple, batch: dict, lambda_card: float = LAMBDA_CARD,
                 lambda_w1: float = LAMBDA_W1, lambda_mine: float = LAMBDA_MINE):
    """Composite Stage A loss. `out` is RhythmNet's output tuple; `batch` holds
    onset / jump / hold_type / hold_dur / mine targets and the `valid` mask.
    Returns (total, {part: float})."""
    onset_logit, jump_logit, hold_type, hold_dur, mine_logit = out[:5]
    y, valid = batch["onset"].float(), batch["valid"]
    p = torch.sigmoid(onset_logit.float())
    focal = focal_bce(onset_logit, y, valid)
    card = cardinality_loss(p, y, valid)
    w1 = wasserstein_1d_measure_loss(p, y, valid)

    at = (y > 0) & valid                         # auxiliaries only where a note is
    n_at = at.float().sum().clamp_min(1.0)
    jump = (F.binary_cross_entropy_with_logits(
        jump_logit.float(), batch["jump"].float(), reduction="none") * at).sum() / n_at
    ht = (F.cross_entropy(hold_type.float().transpose(1, 2), batch["hold_type"],
                          reduction="none") * at).sum() / n_at
    held = at & (batch["hold_type"] > 0)
    hd = (F.cross_entropy(hold_dur.float().transpose(1, 2), batch["hold_dur"],
                          reduction="none") * held).sum() / held.float().sum().clamp_min(1.0)
    mine = focal_bce(mine_logit, batch["mine"].float(), valid)
    total = (focal + lambda_card * card + lambda_w1 * w1 + jump + ht + hd
             + lambda_mine * mine)
    parts = {"focal": focal, "card": card, "w1": w1, "jump": jump, "hold_type": ht,
             "hold_dur": hd, "mine": mine}
    return total, {k: float(v.detach()) for k, v in parts.items()}


_ERGO = None


def ergo_table(device) -> torch.Tensor:
    """C_ergo as a tensor (N_DT, 20, 20) on `device` (cached)."""
    global _ERGO
    if _ERGO is None or _ERGO.device != torch.device(device):
        _ERGO = torch.from_numpy(ergo_matrix()).to(device)
    return _ERGO


def stage_b_loss(logits: torch.Tensor, batch: dict, masked: torch.Tensor,
                 lambda_ergo: float = LAMBDA_ERGO):
    """Masked-token cross-entropy + the expected ergonomic penalty.

    Cross-entropy is taken only at masked notes and only over the tokens legal
    for that note's class (single vs jump). The penalty couples neighbours: it
    is P_i^T C_ergo(dt) P_{i+1}, with P the model's distribution at a masked
    note and the known token's one-hot elsewhere.
    """
    tokens, pad = batch["tokens"], batch["pad"]
    legal = batch["class_mask"]
    z = logits.float().masked_fill(~legal, -1e9)
    ce = F.cross_entropy(z.transpose(1, 2), tokens, reduction="none")
    n_masked = masked.float().sum().clamp_min(1.0)
    ce_loss = (ce * masked).sum() / n_masked

    probs = torch.softmax(z, dim=-1)
    known = F.one_hot(tokens, N_TOKENS).to(probs.dtype)
    p = torch.where(masked[..., None], probs, known)
    c = ergo_table(logits.device)[batch["dtb"][:, 1:]]            # (B, N-1, 20, 20)
    pair = torch.einsum("bni,bnij,bnj->bn", p[:, :-1], c, p[:, 1:])
    both = pad[:, 1:] & pad[:, :-1] & (masked[:, 1:] | masked[:, :-1])
    ergo = (pair * both).sum() / both.float().sum().clamp_min(1.0)

    with torch.no_grad():
        acc = ((z.argmax(-1) == tokens) & masked).float().sum() / n_masked
    total = ce_loss + lambda_ergo * ergo
    return total, {"ce": float(ce_loss.detach()), "ergo": float(ergo.detach()),
                   "acc": float(acc)}
