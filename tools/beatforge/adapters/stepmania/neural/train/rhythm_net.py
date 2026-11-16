"""rhythm_net.py — Stage A: RhythmNet, a bidirectional Conformer over the
48-tick grid (spec §6.1).

    X (B, T, 64), cond (B, 19)
      -> onset_logit (B, T)            is there a note on this tick
         jump_logit (B, T)             ...and is it a 2-panel jump
         hold_type_logits (B, T, 6)    NONE / HOLD_ALT / HOLD_SAME / HOLD_BOTH /
                                       ROLL_ALT / ROLL_SAME
         hold_dur_logits (B, T, 10)    freeze length bucket
         mine_logit (B, T)
         hidden_state (B, T, d_model)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from ..codec import HOLD_DUR_BUCKETS, N_FLAVORS
from ..conditioning import COND_DIM
from ..grid_features import D_IN
from .modules import (AdaLNZero, CondEmbed, ConvModule, FeedForward,
                      RotaryAttention, count_params)


@dataclass
class RhythmConfig:
    d_in: int = D_IN
    cond_dim: int = COND_DIM
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    ffn_mult: int = 4
    conv_kernel: int = 7
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class ConformerBlock(nn.Module):
    def __init__(self, cfg: RhythmConfig):
        super().__init__()
        d = cfg.d_model
        self.ada_attn, self.attn = AdaLNZero(d), RotaryAttention(d, cfg.n_heads, cfg.dropout)
        self.ada_conv, self.conv = AdaLNZero(d), ConvModule(d, cfg.conv_kernel, cfg.dropout)
        self.ada_ffn, self.ffn = AdaLNZero(d), FeedForward(d, cfg.ffn_mult, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, c, pad_mask=None):
        h, a = self.ada_attn(x, c)
        x = x + a * self.drop(self.attn(h, pad_mask))
        h, a = self.ada_conv(x, c)
        x = x + a * self.conv(h, pad_mask)
        h, a = self.ada_ffn(x, c)
        return x + a * self.drop(self.ffn(h))


class RhythmNet(nn.Module):
    def __init__(self, cfg: RhythmConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or RhythmConfig()
        d = cfg.d_model
        self.in_proj = nn.Sequential(nn.Linear(cfg.d_in, d), nn.SiLU(), nn.Linear(d, d))
        self.cond = CondEmbed(cfg.cond_dim, d)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))
        self.out_norm = nn.LayerNorm(d)
        self.onset = nn.Linear(d, 1)
        self.jump = nn.Linear(d, 1)
        self.hold_type = nn.Linear(d, N_FLAVORS)
        self.hold_dur = nn.Linear(d, len(HOLD_DUR_BUCKETS))
        self.mine = nn.Linear(d, 1)
        # notes are sparse: start the onset/mine heads at a low prior so the
        # first steps are not spent un-learning p = 0.5 everywhere
        nn.init.constant_(self.onset.bias, -2.5)
        nn.init.constant_(self.mine.bias, -5.0)
        nn.init.constant_(self.jump.bias, -2.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                pad_mask: torch.Tensor | None = None):
        c = self.cond(cond)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, c, pad_mask)
        h = self.out_norm(h)
        return (self.onset(h).squeeze(-1), self.jump(h).squeeze(-1), self.hold_type(h),
                self.hold_dur(h), self.mine(h).squeeze(-1), h)

    @property
    def n_params(self) -> int:
        return count_params(self)


OUTPUT_NAMES = ("onset_logit", "jump_logit", "hold_type_logits", "hold_dur_logits",
                "mine_logit", "hidden_state")
