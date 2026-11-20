"""losses.py — Stage A composite rhythm loss and the Stage B masked-diffusion
objective (spec §6.2, §6.4, REQ-SFN-05).

Stage A:  L = Focal(p, y) + l_card * |sum_m p - sum_m y| + l_W1 * W1_m(p, y) + L_aux

Plain BCE on a sparse grid is satisfied by a blur: a note predicted one tick
late is exactly as wrong as one predicted a beat late. The cardinality term
makes each measure hold the right NUMBER of notes and the Wasserstein term the
right PLACEMENT within the measure, with a gradient that grows with how far a
note is off — which is the signal BCE cannot give.

Two optional Stage A terms, both free at inference (v1.5):

* a positive-class weight on 16th and triplet ticks of dense charts. A small
  model learns that a 16th is rarely a note and stops proposing them at all;
  the weight makes a missed 16th cost more than a missed quarter note, while
  the cardinality and Wasserstein terms keep the note COUNT per measure honest.
* a beat-phase margin: asked for a low-tier chart, the network should put its
  onset logit on the quarter notes and not on the off-beat 8ths, by a margin.
  That contrast is exactly what the decoder reads to decide whether a beat
  grid is seated half a beat out (`pipeline.verify_phase`), so the loss trains
  the statistic the runtime depends on.
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
#: v1.5 values of the optional terms (the function defaults leave them off).
SUB16_WEIGHT, SUB12_WEIGHT = 1.45, 1.30
DENSE_METER_NORM = 0.45          # the weights apply at or above this meter / 15
LOW_METER_NORM = 0.25            # the phase margin applies at or below this
PHASE_MARGIN, LAMBDA_PHASE = 2.0, 0.10
PHASE_CONTRAST_MARGIN = 1.5
TICKS_PER_BEAT = G // 4


def focal_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
              alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA,
              pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Focal binary cross-entropy, averaged over `valid` positions.
    `pos_weight` (same shape) scales the loss at positive targets only."""
    logits, target = logits.float(), target.float()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    if pos_weight is not None:
        w = w * torch.where(target > 0, pos_weight.float(), torch.ones_like(w))
    loss = w * (1 - pt).pow(gamma) * ce
    v = valid.float()
    return (loss * v).sum() / v.sum().clamp_min(1.0)


def subdivision_weight(n_ticks: int, meter_norm: torch.Tensor, w16: float = SUB16_WEIGHT,
                       w12: float = SUB12_WEIGHT,
                       dense: float = DENSE_METER_NORM) -> torch.Tensor:
    """(B, T) positive-class weights: `w16` on 16th ticks (t % 3 == 0, not an
    8th) and `w12` on triplet ticks (t % 4 == 0, not a quarter) of charts whose
    `meter_norm` (B,) is at least `dense`; 1 everywhere else. Tick 0 of the
    sequence must be the start of a beat, which every training crop is."""
    t = torch.arange(n_ticks, device=meter_norm.device)
    row = torch.ones(n_ticks, device=meter_norm.device)
    row = torch.where((t % 3 == 0) & (t % 6 != 0), torch.full_like(row, w16), row)
    row = torch.where((t % 4 == 0) & (t % TICKS_PER_BEAT != 0), torch.full_like(row, w12), row)
    on = (meter_norm.float() >= dense)[:, None]
    return torch.where(on, row[None, :], torch.ones_like(row)[None, :])


def phase_score(onset_logit: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B,) mean onset logit on the quarter notes minus the mean on the
    off-beat 8ths, over valid ticks."""
    z, v = onset_logit.float(), valid.float()
    t = torch.arange(z.shape[1], device=z.device) % TICKS_PER_BEAT
    q = ((t == 0).float()[None, :] * v)
    e = ((t == TICKS_PER_BEAT // 2).float()[None, :] * v)
    return ((z * q).sum(1) / q.sum(1).clamp_min(1.0)
            - (z * e).sum(1) / e.sum(1).clamp_min(1.0))


def phase_eligible(batch: dict, low: float = LOW_METER_NORM) -> torch.Tensor:
    """(B,) bool: low-tier items whose own notes sit more on the quarter notes
    than on the off-beats — the crops the margin can honestly be asked of."""
    y, v = batch["onset"].float(), batch["valid"].float()
    t = torch.arange(y.shape[1], device=y.device) % TICKS_PER_BEAT
    on_q = (y * v * (t == 0).float()[None, :]).sum(1)
    on_e = (y * v * (t == TICKS_PER_BEAT // 2).float()[None, :]).sum(1)
    meter = batch["cond"][:, 0].float()
    return (meter >= 0) & (meter <= low) & (on_q > on_e)


def phase_margin_loss(onset_logit: torch.Tensor, valid: torch.Tensor,
                      eligible: torch.Tensor, margin: float = PHASE_MARGIN) -> torch.Tensor:
    """Mean over eligible items of max(0, margin - phase_score). Zero, with a
    zero gradient, for an item whose score already clears the margin."""
    gap = F.relu(margin - phase_score(onset_logit, valid))
    e = eligible.float()
    return (gap * e).sum() / e.sum().clamp_min(1.0)


def phase_contrast_loss(score_true: torch.Tensor, score_shifted: torch.Tensor,
                        margin: float = PHASE_CONTRAST_MARGIN) -> torch.Tensor:
    """The same items on the true grid and on a grid moved half a beat: the
    true grid's phase score must beat the moved one's by `margin`."""
    if score_true.numel() == 0:
        return score_true.new_zeros(())
    return F.relu(margin - (score_true - score_shifted)).mean()


def shift_half_beat(batch: dict, direction: int = 1, n_audio: int = 54) -> dict:
    """The batch as it would look on a grid seated half a beat out: the audio
    channels (the first `n_audio`; the metric block stays) and every target
    move `direction` * 6 ticks, and the ticks that wrapped around are marked
    invalid. The notes still sit on the same sounds — now on the off-beats."""
    k = direction * (TICKS_PER_BEAT // 2)
    out = dict(batch)
    x = batch["x"]
    out["x"] = torch.cat([torch.roll(x[..., :n_audio], k, dims=1), x[..., n_audio:]], dim=-1)
    for name in ("onset", "jump", "hold_type", "hold_dur", "mine"):
        out[name] = torch.roll(batch[name], k, dims=1)
    valid = torch.roll(batch["valid"], k, dims=1).clone()
    if k > 0:
        valid[:, :k] = False
    else:
        valid[:, k:] = False
    out["valid"] = valid
    return out


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
                 lambda_w1: float = LAMBDA_W1, lambda_mine: float = LAMBDA_MINE,
                 sub16_weight: float = 1.0, sub12_weight: float = 1.0,
                 lambda_phase: float = 0.0, phase_margin: float = PHASE_MARGIN):
    """Composite Stage A loss. `out` is RhythmNet's output tuple; `batch` holds
    onset / jump / hold_type / hold_dur / mine targets and the `valid` mask
    (and `cond`, when either optional term is on). Returns (total, {part: float})."""
    onset_logit, jump_logit, hold_type, hold_dur, mine_logit = out[:5]
    y, valid = batch["onset"].float(), batch["valid"]
    p = torch.sigmoid(onset_logit.float())
    pos_weight = None
    if sub16_weight != 1.0 or sub12_weight != 1.0:
        pos_weight = subdivision_weight(y.shape[1], batch["cond"][:, 0], sub16_weight,
                                        sub12_weight)
    focal = focal_bce(onset_logit, y, valid, pos_weight=pos_weight)
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
    if lambda_phase > 0:
        phase = phase_margin_loss(onset_logit, valid, phase_eligible(batch), phase_margin)
        total = total + lambda_phase * phase
        parts["phase"] = phase
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
