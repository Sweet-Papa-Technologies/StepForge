"""train_stage_b.py — train ChoreoNet (spec §7).

    python -m beatforge.adapters.stepmania.neural.train.train_stage_b \
        --corpus build/neural_corpus --out build/neural_runs/stage_b

AdamW 3e-4, weight decay 1e-2, cosine decay, the notes of a 32-measure window
(at most 512) per item, batch 64. Resumable from `last.pt` in --out.

Validation reports, on held-out songs:
  ce / acc        cross-entropy and accuracy at masked notes, cosine mask rate
  acc_full        accuracy with EVERY note masked (the unconditional prior)
  acc_half        accuracy with half the notes masked
and an ergonomic legality check on charts sampled from the model by 10-step
unmasking, with no Viterbi safety net:
  gen_cost        mean foot-model discomfort per transition
  gen_hard        share of transitions that are physically illegal (spins)
  gen_dstep       share that are unforced double-steps
compared against the same numbers for the human charts (`ref_*`).
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import codec as C
from . import train_common as tc
from .choreo_net import ChoreoConfig, ChoreoNet, class_mask, mask_tokens
from .dataset import CROP_TICKS, MAX_EVENTS, StageBDataset, collate_b, load_corpus
from .losses import LAMBDA_ERGO, stage_b_loss


@torch.no_grad()
def sample_unmask(model, batch: dict, steps: int = 10, tau: float = 0.0,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """MaskGIT-style parallel decoding: start fully masked and commit the most
    confident tokens on a cosine schedule (spec §6.5, without the Viterbi)."""
    pad, legal = batch["pad"], batch["class_mask"]
    tokens = torch.full_like(batch["tokens"], C.MASK)
    n_real = pad.sum(1)
    for s in range(1, steps + 1):
        logits = model(batch["ctx"], batch["scalars"], tokens, batch["cond"], pad).float()
        logits = logits.masked_fill(~legal, -1e9)
        temp = tau * (1 - s / steps)
        if temp > 0:
            g = -torch.log(-torch.log(torch.rand(logits.shape, device=logits.device,
                                                 generator=generator).clamp_min(1e-9)))
            cand = (logits / temp + g).argmax(-1)
        else:
            cand = logits.argmax(-1)
        conf = torch.log_softmax(logits, -1).gather(-1, cand[..., None]).squeeze(-1)
        known = (tokens != C.MASK)
        conf = torch.where(known, torch.full_like(conf, 1e9), conf)   # keep commitments
        conf = conf.masked_fill(~pad, -1e9)
        keep = torch.floor(n_real * (1 - math.cos(math.pi * s / (2 * steps)))).long()
        keep = torch.where(torch.tensor(s == steps, device=keep.device), n_real, keep)
        order = conf.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(order.shape[1], device=order.device)
                      .expand_as(order))
        commit = (rank < keep[:, None]) & pad
        tokens = torch.where(commit, torch.where(known, tokens, cand),
                             torch.full_like(tokens, C.MASK))
    return tokens


def path_stats(tokens: np.ndarray, dtb: np.ndarray, pad: np.ndarray) -> dict:
    """Foot-model cost of token sequences replayed from rest (no hold locks)."""
    t = C.tables()
    cost, n, hard, dstep = 0.0, 0, 0, 0
    for seq, dts, m in zip(tokens, dtb, pad):
        s = C.start_state(C.LEFT)
        for tok, d in zip(seq[m], dts[m]):
            c = int(t["cost"][d, 0, s, tok])
            cost += min(c, C.HARD_COST)
            hard += c >= C.HARD_COST
            dstep += (C.DSTEP_COST[d] >= 60) and (4 <= tok < 8) and (c >= C.DSTEP_COST[d])
            s = int(t["next"][s, tok])
            n += 1
    n = max(1, n)
    return {"cost": cost / n, "hard": hard / n, "dstep": dstep / n}


@torch.no_grad()
def evaluate(model, loader, device, amp: str, args, gen_batches: int = 4) -> dict:
    model.eval()
    g = torch.Generator(device=device).manual_seed(1234)
    sums, n = {}, 0
    full_ok = full_n = half_ok = half_n = 0
    gen, ref = [], []
    for bi, batch in enumerate(loader):
        batch = tc.to_device(batch, device)
        batch["class_mask"] = class_mask(batch["tokens"] >= 8)
        pad, tokens = batch["pad"], batch["tokens"]
        inp, masked = mask_tokens(tokens, pad, g)
        with tc.autocast_for(device, amp):
            logits = model(batch["ctx"], batch["scalars"], inp, batch["cond"], pad)
        loss, parts = stage_b_loss(logits, batch, masked, args.lambda_ergo)
        parts["loss"] = float(loss)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
        for rate, name in ((1.0, "full"), (0.5, "half")):
            hit = (torch.rand(tokens.shape, device=device, generator=g) < rate) & pad
            inp2 = torch.where(hit, torch.full_like(tokens, C.MASK), tokens)
            with tc.autocast_for(device, amp):
                lg = model(batch["ctx"], batch["scalars"], inp2, batch["cond"], pad).float()
            ok = int(((lg.masked_fill(~batch["class_mask"], -1e9).argmax(-1) == tokens)
                      & hit).sum())
            if name == "full":
                full_ok, full_n = full_ok + ok, full_n + int(hit.sum())
            else:
                half_ok, half_n = half_ok + ok, half_n + int(hit.sum())
        if bi < gen_batches:
            with tc.autocast_for(device, amp):
                sampled = sample_unmask(model, batch, tau=0.0)
            args_np = (batch["dtb"].cpu().numpy(), pad.cpu().numpy())
            gen.append(path_stats(sampled.cpu().numpy(), *args_np))
            ref.append(path_stats(tokens.cpu().numpy(), *args_np))
    res = {k: v / max(1, n) for k, v in sums.items()}
    res.update(acc_full=full_ok / max(1, full_n), acc_half=half_ok / max(1, half_n))
    for name, rows in (("gen", gen), ("ref", ref)):
        for k in ("cost", "hard", "dstep"):
            res[f"{name}_{k}"] = float(np.mean([r[k] for r in rows])) if rows else 0.0
    model.train()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--crop", type=int, default=CROP_TICKS)
    ap.add_argument("--max-events", type=int, default=MAX_EVENTS)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu | mps")
    ap.add_argument("--lambda-ergo", type=float, default=LAMBDA_ERGO)
    ap.add_argument("--d-model", type=int, default=320)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-audio", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-songs", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--reencode", action="store_true")
    args = ap.parse_args(argv)

    tc.seed_everything(args.seed)
    device = tc.pick_device(args.device)
    cfg = ChoreoConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                       d_audio=args.d_audio, dropout=args.dropout)
    run = tc.RunDir(args.out, {"stage": "b", "model": cfg.to_dict(), "args": vars(args)})
    train_ds = StageBDataset(load_corpus(args.corpus, "train", args.max_songs,
                                         args.reencode), args.crop, args.max_events,
                             True, args.seed)
    val_ds = StageBDataset(load_corpus(args.corpus, "val", args.max_songs,
                                       args.reencode), args.crop, args.max_events,
                           False, args.seed)
    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
              collate_fn=collate_b)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, **kw)

    model = ChoreoNet(cfg).to(device)
    opt = torch.optim.AdamW(tc.param_groups(model, args.weight_decay), lr=args.lr,
                            betas=(0.9, 0.98))
    steps_per_epoch = max(1, len(train_dl))
    total = args.epochs * steps_per_epoch
    step, start_epoch, best = 0, 0, float("inf")
    state = run.load("last")
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, start_epoch, best = state["step"], state["epoch"] + 1, state["best"]
        print(f"[train] resumed at epoch {start_epoch}, step {step}", flush=True)
    print(f"[train] ChoreoNet {model.n_params / 1e6:.2f}M params on {device}; "
          f"{len(train_ds)} train / {len(val_ds)} val charts; "
          f"{steps_per_epoch} steps/epoch x {args.epochs}", flush=True)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        train_ds.epoch = epoch
        t0, acc, n_acc = time.time(), {}, 0
        for batch in train_dl:
            for g in opt.param_groups:
                g["lr"] = tc.lr_at(step, total, args.lr, args.warmup)
            batch = tc.to_device(batch, device)
            batch["class_mask"] = class_mask(batch["tokens"] >= 8)
            inp, masked = mask_tokens(batch["tokens"], batch["pad"])
            with tc.autocast_for(device, args.amp):
                logits = model(batch["ctx"], batch["scalars"], inp, batch["cond"],
                               batch["pad"])
            loss, parts = stage_b_loss(logits, batch, masked, args.lambda_ergo)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            step += 1
            parts["loss"] = float(loss.detach())
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v
            n_acc += 1
            if step % 200 == 0:
                print(f"[train] e{epoch} step {step}/{total} loss "
                      f"{acc['loss'] / n_acc:.4f} acc {acc['acc'] / n_acc:.3f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} "
                      f"{n_acc / (time.time() - t0):.1f} it/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        val = evaluate(model, val_dl, device, args.amp, args)
        run.log(epoch=epoch, step=step,
                **{f"train_{k}": v / max(1, n_acc) for k, v in acc.items()},
                **{f"val_{k}": v for k, v in val.items()},
                epoch_s=round(time.time() - t0, 1))
        ckpt = dict(model=model.state_dict(), config=cfg.to_dict(), epoch=epoch,
                    step=step, val=val, args=vars(args))
        if val["ce"] < best:
            best = val["ce"]
            run.save("best", **ckpt)
        run.save("last", opt=opt.state_dict(), best=best, **ckpt)
        if args.max_steps and step >= args.max_steps:
            break
    print(f"[train] done; best val ce {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
