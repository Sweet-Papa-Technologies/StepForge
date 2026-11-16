"""train_common.py — the loop plumbing both trainers share: device/AMP setup,
warmup + cosine schedule, resumable checkpoints and a JSONL log."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch


def pick_device(name: str = "auto") -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    # CUDA or CPU. Apple's MPS backend is never auto-selected: masked
    # scaled-dot-product attention returned NaN losses on it (torch 2.9), so it
    # has to be asked for by name.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_for(device: torch.device, amp: str):
    """`amp`: 'bf16', 'fp16' or 'off'. Mixed precision only on CUDA."""
    if device.type != "cuda" or amp == "off":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def lr_at(step: int, total: int, base: float, warmup: int, floor: float = 0.02) -> float:
    """Linear warmup, then cosine decay to `floor * base`."""
    if step < warmup:
        return base * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


def param_groups(model: torch.nn.Module, weight_decay: float) -> list:
    """Weight decay on matrices only — not on biases, norms or embeddings."""
    decay, plain = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and "tok" not in name else plain).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": plain, "weight_decay": 0.0}]


class RunDir:
    """Output directory of a training run: config, log, last/best checkpoints."""

    def __init__(self, path, config: dict):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "config.json").write_text(json.dumps(config, indent=2))
        self.log_path = self.path / "log.jsonl"
        self.t0 = time.time()

    def log(self, **row) -> None:
        row["elapsed_s"] = round(time.time() - self.t0, 1)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print("[train] " + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()), flush=True)

    def save(self, name: str, **state) -> None:
        tmp = self.path / f"{name}.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.path / f"{name}.pt")

    def load(self, name: str = "last"):
        p = self.path / f"{name}.pt"
        return torch.load(p, map_location="cpu", weights_only=False) if p.exists() else None


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
