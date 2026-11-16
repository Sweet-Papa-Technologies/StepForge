"""train_stage_a.py — train RhythmNet (spec §7).

    python -m beatforge.adapters.stepmania.neural.train.train_stage_a \
        --corpus build/neural_corpus --out build/neural_runs/stage_a

AdamW 3e-4, weight decay 1e-2, 1000-step warmup then cosine, mixed precision,
random 32-measure crops, batch 32, gradient clip 1.0. Resumable: re-running
with the same --out picks up from `last.pt`.

Validation reports, on fixed crops of held-out SONGS:
  ap          average precision of the onset probability over every tick
  r_prec      precision when each chart keeps exactly as many ticks as it has
              notes — the number that matters, since inference picks the
              threshold by note budget, not by probability
  f1          best F1 over a global threshold sweep
  count_err   mean |predicted - true| notes per measure
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import train_common as tc
from .dataset import CROP_TICKS, StageADataset, load_corpus
from .losses import LAMBDA_CARD, LAMBDA_W1, stage_a_loss
from .rhythm_net import RhythmConfig, RhythmNet


def average_precision(score: np.ndarray, target: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    t = target[order] > 0
    n_pos = int(t.sum())
    if n_pos == 0:
        return 0.0
    tp = np.cumsum(t)
    return float((tp[t] / (np.nonzero(t)[0] + 1)).sum() / n_pos)


@torch.no_grad()
def evaluate(model, loader, device, amp: str, args) -> dict:
    model.eval()
    sums, n_batches = {}, 0
    scores, targets, rprec, jump_ok, jump_n = [], [], [], 0, 0
    ht_ok = ht_n = 0
    count_err, n_meas = 0.0, 0
    for batch in loader:
        batch = tc.to_device(batch, device)
        with tc.autocast_for(device, amp):
            out = model(batch["x"], batch["cond"], batch["valid"])
        loss, parts = stage_a_loss(out, batch, args.lambda_card, args.lambda_w1)
        parts["loss"] = float(loss)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        n_batches += 1
        p = torch.sigmoid(out[0].float())
        y, valid = batch["onset"], batch["valid"]
        for i in range(len(p)):
            v = valid[i]
            pi, yi = p[i][v], y[i][v]
            n = int(yi.sum())
            if n > 0:
                top = torch.topk(pi, n).indices
                rprec.append(float(yi[top].mean()))
            scores.append(pi.cpu().numpy().astype(np.float32))
            targets.append(yi.cpu().numpy().astype(np.uint8))
        at = (y > 0) & valid
        jump_ok += int((((out[1] > 0) == (batch["jump"] > 0)) & at).sum())
        jump_n += int(at.sum())
        ht_ok += int(((out[2].argmax(-1) == batch["hold_type"]) & at).sum())
        ht_n += int(at.sum())
        vm = valid.float().reshape(len(p), -1, 48)
        pm = (p * valid).reshape(len(p), -1, 48).sum(-1)
        ym = (y * valid).reshape(len(p), -1, 48).sum(-1)
        real = vm.sum(-1) > 0
        count_err += float(((pm - ym).abs() * real).sum())
        n_meas += int(real.sum())
    s, t = np.concatenate(scores), np.concatenate(targets)
    best_f1 = 0.0
    for th in np.linspace(0.05, 0.95, 19):
        tp = float(((s >= th) & (t > 0)).sum())
        prec, rec = tp / max(1.0, float((s >= th).sum())), tp / max(1.0, float(t.sum()))
        best_f1 = max(best_f1, 2 * prec * rec / max(1e-9, prec + rec))
    res = {k: v / max(1, n_batches) for k, v in sums.items()}
    res.update(ap=average_precision(s, t), r_prec=float(np.mean(rprec)) if rprec else 0.0,
               f1=best_f1, count_err=count_err / max(1, n_meas),
               jump_acc=jump_ok / max(1, jump_n), hold_type_acc=ht_ok / max(1, ht_n))
    model.train()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--crop", type=int, default=CROP_TICKS)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu | mps")
    ap.add_argument("--lambda-card", type=float, default=LAMBDA_CARD)
    ap.add_argument("--lambda-w1", type=float, default=LAMBDA_W1)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-songs", type=int, help="debug: cap songs per split")
    ap.add_argument("--max-steps", type=int, help="debug: stop after this many steps")
    ap.add_argument("--reencode", action="store_true",
                    help="re-run the codec on the stored raw rows at load time")
    args = ap.parse_args(argv)

    tc.seed_everything(args.seed)
    device = tc.pick_device(args.device)
    cfg = RhythmConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                       dropout=args.dropout)
    run = tc.RunDir(args.out, {"stage": "a", "model": cfg.to_dict(), "args": vars(args)})
    train_ds = StageADataset(load_corpus(args.corpus, "train", args.max_songs,
                                         args.reencode), args.crop, True, args.seed)
    val_ds = StageADataset(load_corpus(args.corpus, "val", args.max_songs,
                                       args.reencode), args.crop, False, args.seed)
    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, **kw)

    model = RhythmNet(cfg).to(device)
    opt = torch.optim.AdamW(tc.param_groups(model, args.weight_decay), lr=args.lr,
                            betas=(0.9, 0.98))
    steps_per_epoch = max(1, len(train_dl))
    total = args.epochs * steps_per_epoch
    step, start_epoch, best = 0, 0, -1.0
    state = run.load("last")
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, start_epoch, best = state["step"], state["epoch"] + 1, state["best"]
        print(f"[train] resumed at epoch {start_epoch}, step {step}", flush=True)
    print(f"[train] RhythmNet {model.n_params / 1e6:.2f}M params on {device}; "
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
            with tc.autocast_for(device, args.amp):
                out = model(batch["x"], batch["cond"], batch["valid"])
            loss, parts = stage_a_loss(out, batch, args.lambda_card, args.lambda_w1)
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
                      f"{acc['loss'] / n_acc:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{n_acc / (time.time() - t0):.1f} it/s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        val = evaluate(model, val_dl, device, args.amp, args)
        row = {"epoch": epoch, "step": step,
               **{f"train_{k}": v / max(1, n_acc) for k, v in acc.items()},
               **{f"val_{k}": v for k, v in val.items()},
               "epoch_s": round(time.time() - t0, 1)}
        run.log(**row)
        ckpt = dict(model=model.state_dict(), config=cfg.to_dict(), epoch=epoch,
                    step=step, val=val, args=vars(args))
        if val["r_prec"] > best:
            best = val["r_prec"]
            run.save("best", **ckpt)
        run.save("last", opt=opt.state_dict(), best=best, **ckpt)
        if args.max_steps and step >= args.max_steps:
            break
    print(f"[train] done; best val r_prec {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
