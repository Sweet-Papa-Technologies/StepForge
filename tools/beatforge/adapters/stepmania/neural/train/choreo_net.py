"""choreo_net.py — Stage B: ChoreoNet, a masked discrete diffusion transformer
over the active notes only (spec §6.4).

    ctx     (B, N, K, 64)  beat-warped audio around each note (event_features)
    scalars (B, N, 31)     per-note timing / kind features
    tokens  (B, N)         foot-relative tokens, 20 = [MASK]
    cond    (B, 19)
      -> logits (B, N, 20)

Stage B has its own small audio encoder instead of reading Stage A's hidden
state (the spec allows either). That keeps the two stages independently
trainable and retrainable — a new RhythmNet does not invalidate a ChoreoNet —
and the eight transformer layers contextualise the per-note audio across the
whole phrase anyway.

`no_audio_embed` is a TRAINING-ONLY vector that stands in for the audio
encoder's output when a sequence has no audio (synthetic step patterns) or
when its audio is deliberately withheld (`drop_audio`). It teaches the network
to choreograph from rhythm and style alone without ever being fed silence as
if it were music. Inference always passes real audio and never `drop_audio`,
so the exported graph does not contain the vector at all.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..codec import MASK, N_TOKENS
from ..conditioning import COND_DIM
from ..event_features import CTX_K, N_SCALARS
from ..grid_features import D_IN
from .modules import AdaLNZero, CondEmbed, FeedForward, RotaryAttention, count_params


@dataclass
class ChoreoConfig:
    d_in: int = D_IN
    ctx_k: int = CTX_K
    n_scalars: int = N_SCALARS
    cond_dim: int = COND_DIM
    d_model: int = 320
    n_layers: int = 8
    n_heads: int = 8
    ffn_mult: int = 4
    d_audio: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class AudioStem(nn.Module):
    """Two conv layers over each note's K-tick audio window -> one vector."""

    def __init__(self, cfg: ChoreoConfig):
        super().__init__()
        self.c1 = nn.Conv1d(cfg.d_in, cfg.d_audio, 5, padding=2)
        self.c2 = nn.Conv1d(cfg.d_audio, cfg.d_audio, 5, padding=2)
        self.out = nn.Linear(2 * cfg.d_audio, cfg.d_audio)
        self.centre = cfg.ctx_k // 2

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        b, n, k, d = ctx.shape
        y = ctx.reshape(b * n, k, d).transpose(1, 2)
        y = F.silu(self.c1(y))
        y = F.silu(self.c2(y))
        # the note's own tick, plus the average of its surroundings
        y = torch.cat([y[:, :, self.centre], y.mean(dim=2)], dim=1)
        return self.out(y).reshape(b, n, -1)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ChoreoConfig):
        super().__init__()
        d = cfg.d_model
        self.ada_attn, self.attn = AdaLNZero(d), RotaryAttention(d, cfg.n_heads, cfg.dropout)
        self.ada_ffn, self.ffn = AdaLNZero(d), FeedForward(d, cfg.ffn_mult, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, c, pad_mask=None):
        h, a = self.ada_attn(x, c)
        x = x + a * self.drop(self.attn(h, pad_mask))
        h, a = self.ada_ffn(x, c)
        return x + a * self.drop(self.ffn(h))


class ChoreoNet(nn.Module):
    def __init__(self, cfg: ChoreoConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ChoreoConfig()
        d = cfg.d_model
        self.audio = AudioStem(cfg)
        self.tok = nn.Embedding(N_TOKENS + 1, d)
        self.in_proj = nn.Sequential(
            nn.Linear(cfg.d_audio + cfg.n_scalars, d), nn.SiLU(), nn.Linear(d, d))
        self.cond = CondEmbed(cfg.cond_dim, d)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_TOKENS)
        self.no_audio_embed = nn.Parameter(torch.zeros(cfg.d_audio))
        nn.init.normal_(self.no_audio_embed, std=0.02)

    def forward(self, ctx: torch.Tensor, scalars: torch.Tensor, tokens: torch.Tensor,
                cond: torch.Tensor, pad_mask: torch.Tensor | None = None,
                drop_audio: torch.Tensor | None = None):
        """`drop_audio` (B,) bool, training only: sequences whose audio is
        replaced by `no_audio_embed`."""
        c = self.cond(cond)
        audio = self.audio(ctx)
        if drop_audio is not None:
            audio = torch.where(drop_audio[:, None, None],
                                self.no_audio_embed.to(audio.dtype), audio)
        h = self.in_proj(torch.cat([audio, scalars], dim=-1)) + self.tok(tokens)
        for blk in self.blocks:
            h = blk(h, c, pad_mask)
        return self.head(self.out_norm(h))

    @property
    def n_params(self) -> int:
        return count_params(self)


def class_mask(is_jump: torch.Tensor) -> torch.Tensor:
    """(B, N) bool -> (B, N, 20) bool: the tokens legal for each note. Stage A
    has already decided single vs jump, so Stage B only ever chooses within
    that class — the jump count can never drift between the two stages."""
    jump_tok = torch.arange(N_TOKENS, device=is_jump.device) >= 8
    return jump_tok[None, None, :] == is_jump[:, :, None]


def mask_tokens(tokens: torch.Tensor, pad_mask: torch.Tensor,
                generator: torch.Generator | None = None):
    """Cosine-schedule masking (spec §6.4). Per sequence u ~ U(0,1) and mask
    rate cos(pi*u/2); each real token is masked independently at that rate, and
    at least one always is. Returns (masked tokens, bool mask of masked slots)."""
    b, n = tokens.shape
    u = torch.rand(b, 1, device=tokens.device, generator=generator)
    rate = torch.cos(0.5 * torch.pi * u)
    hit = (torch.rand(b, n, device=tokens.device, generator=generator) < rate) & pad_mask
    none = ~hit.any(dim=1)
    if none.any():
        first = pad_mask.float().argmax(dim=1)
        hit[none, first[none]] = pad_mask[none, first[none]]
    return torch.where(hit, torch.full_like(tokens, MASK), tokens), hit


INPUT_NAMES = ("ctx", "scalars", "tokens", "cond")
#: Parameters that exist for training and must not reach an exported graph.
TRAINING_ONLY = ("no_audio_embed",)
