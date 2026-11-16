"""modules.py — building blocks shared by RhythmNet and ChoreoNet: rotary
self-attention and AdaLN-Zero conditioning. PyTorch only."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_angles(n: torch.Tensor | int, dim: int, device, dtype) -> tuple:
    """cos/sin tables (n, dim/2) for rotary position embedding."""
    half = dim // 2
    inv = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / half))
    pos = torch.arange(n, device=device, dtype=torch.float32)
    ang = pos[:, None] * inv[None, :]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, N, D). Rotate consecutive halves by the position angle."""
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class RotaryAttention(nn.Module):
    """Bidirectional multi-head self-attention with RoPE on queries and keys."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0 and (d_model // n_heads) % 2 == 0
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        """x: (B, N, D). pad_mask: (B, N) bool, True = real position."""
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        cos, sin = rope_angles(n, d // self.h, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        mask = None if pad_mask is None else pad_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class CondEmbed(nn.Module):
    """The 19-D conditioning vector -> one embedding shared by every block."""

    def __init__(self, cond_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, d_model), nn.SiLU(),
                                 nn.Linear(d_model, d_model), nn.SiLU())

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm with a zero-initialised residual gate (as in DiT).

        h     = LayerNorm(x) * (1 + gamma(c)) + beta(c)
        out   = x + alpha(c) * f(h)

    The projection that produces (gamma, beta, alpha) starts at zero, so every
    block is the identity at initialisation and the conditioning is learned
    into the network rather than fought against.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, 3 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """Returns (modulated x, residual gate alpha); c is (B, D)."""
        gamma, beta, alpha = self.proj(c)[:, None, :].chunk(3, dim=-1)
        return self.norm(x) * (1.0 + gamma) + beta, alpha


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, mult * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(mult * d_model, d_model))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    """Conformer convolution module: pointwise expand + GLU, depthwise conv,
    norm, swish, pointwise project."""

    def __init__(self, d_model: int, kernel: int = 7, dropout: float = 0.0):
        super().__init__()
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, groups=d_model)
        self.norm = nn.GroupNorm(1, d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        y = x.transpose(1, 2)
        if pad_mask is not None:
            y = y * pad_mask[:, None, :].to(y.dtype)
        a, g = self.pw1(y).chunk(2, dim=1)
        y = a * torch.sigmoid(g)
        y = F.silu(self.norm(self.dw(y)))
        return self.drop(self.pw2(y)).transpose(1, 2)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
