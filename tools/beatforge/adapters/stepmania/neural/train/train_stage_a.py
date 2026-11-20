"""train_stage_a.py — train RhythmNet (spec §7).

    python -m beatforge.adapters.stepmania.neural.train.train_stage_a \
        --corpus build/neural_corpus --out build/neural_runs/stage_a

AdamW 3e-4, weight decay 1e-2, 1000-step warmup then cosine, mixed precision,
random 32-measure crops, batch 32, gradient clip 1.0. Resumable: re-running
with the same --out picks up from `last.pt`.

With `--corpus-root` (a root of source folders, see `neural/synth`) training
is a two-phase curriculum over a mix of sources:

    phase 1  broad: human charts, their acoustic variants, synthesized grooves,
             thinned low-tier charts            (lr -> --lr-mid, cosine)
    phase 2  polish on the real distribution    (--lr-mid -> --lr-end, cosine)

plus the two v1.5 loss terms (`--sub16-weight`, `--phase-margin-weight`) and
half-beat-shifted copies of low-tier crops (`--phase-shift-prob`): the audio
and the notes move together off the grid's beats, the network is asked to
follow them, and its phase score there must fall below the true grid's.
Validation is always the `val` split of `real/`, scored the way the app runs
(no stem channels); the stem-fed score is logged beside it every few epochs.
Checkpoints are then chosen by

    val_score = 0.50 AP + 0.25 F1 on the 16th notes of Hard/Challenge charts
                + 0.25 beat-phase accuracy at Beginner

Validation reports, on fixed crops of held-out SONGS:
  ap          average precision of the onset probability over every tick
  r_prec      precision when each chart keeps exactly as many ticks as it has
              notes — the number that matters, since inference picks the
              threshold by note budget, not by probability
  f1          best F1 over a global threshold sweep
  count_err   mean |predicted - true| notes per measure
  f1_hard16   F1 on 16th-note ticks alone, over Hard / Challenge charts, each
              chart keeping as many ticks as it has notes
  phase_acc   share of validation songs whose Beginner onset probability is
              better gathered on the quarter notes of the true grid than of
              the grid moved half a beat (`phase_gain`: the mean difference in
              nats, negative = right; `phase_gain_worst`: the least sure song)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import codec as C
from .. import conditioning as cond
from .. import grid_features as GF
from . import losses as L
from . import train_common as tc
from .dataset import (CROP_TICKS, P_DROP_STEMS, MixedDataset, StageADataset, load_corpus,
                      load_sources)
from .losses import LAMBDA_CARD, LAMBDA_W1, stage_a_loss
from .rhythm_net import RhythmConfig, RhythmNet

#: Share of each source in a training batch, per phase. `aug_mild` is the
#: stem-balance subset of `aug_stem` (no pitch, tempo or codec change).
PHASE1_SHARES = {"real": 0.35, "aug_stem": 0.30, "synth_groove": 0.20, "synth_thin": 0.15}
PHASE2_SHARES = {"real": 0.55, "aug_mild": 0.30, "synth_thin": 0.10, "synth_groove": 0.05}
MILD_KINDS = ("remix",)
STAGE_A_SOURCES = ("aug_stem", "synth_groove", "synth_thin")
DENSE_TIERS = ("hard", "challenge")


def val_score(m: dict) -> float:
    return 0.50 * m["ap"] + 0.25 * m["f1_hard16"] + 0.25 * m["phase_acc"]


@torch.no_grad()
def phase_eval(model, songs: list, device, amp: str, crop: int = CROP_TICKS) -> dict:
    """The decoder's half-beat question, asked of every validation song with
    the grid as charted and with it moved half a beat (pipeline.verify_phase's
    onset statistic, on features without stems)."""
    cv = torch.from_numpy(cond.cond19(cond.resolve_style("beginner")))
    half = C.TICKS_PER_BEAT // 2

    def contrast(core, dt) -> float:
        x = GF.assemble(core, dt)
        t = len(x)
        n = int(np.ceil(t / crop))
        buf = np.zeros((n, crop, x.shape[1]), np.float32)
        valid = np.zeros((n, crop), bool)
        for i in range(n):
            part = x[i * crop:(i + 1) * crop]
            buf[i, :len(part)] = part
            valid[i, :len(part)] = True
        xb, vb = torch.from_numpy(buf).to(device), torch.from_numpy(valid).to(device)
        with tc.autocast_for(device, amp):
            z = model(xb, cv.to(device).expand(n, -1), vb)[0]
        p = torch.sigmoid(z.float()).cpu().numpy().reshape(-1)[:t]
        k = np.arange(t) % C.TICKS_PER_BEAT
        return float(np.log((p[k == 0].sum() + 1e-9) / (p[k == half].sum() + 1e-9)))

    gains = []
    for s in songs:
        if s.core is None or s.n_ticks < 8 * C.G:
            continue
        gains.append(contrast(s.core[half:], s.dt_ms[half:]) - contrast(s.core, s.dt_ms))
    if not gains:
        return {"phase_acc": 0.0, "phase_gain": 0.0, "phase_gain_worst": 0.0}
    g = np.array(gains)
    return {"phase_acc": float((g < 0).mean()), "phase_gain": float(g.mean()),
            "phase_gain_worst": float(g.max())}


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
    tp16 = pred16 = true16 = 0
    is16 = None
    for batch in loader:
        batch = tc.to_device(batch, device)
        with tc.autocast_for(device, amp):
            out = model(batch["x"], batch["cond"], batch["valid"])
        # the plain loss, whatever the training terms: comparable across runs
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
                if float(batch["cond"][i, 0]) >= L.DENSE_METER_NORM:
                    if is16 is None or len(is16) != p.shape[1]:
                        k = torch.arange(p.shape[1], device=p.device)
                        is16 = (k % 3 == 0) & (k % 6 != 0)
                    sel = torch.zeros_like(yi, dtype=torch.bool)
                    sel[top] = True
                    m16 = is16[v]
                    tp16 += int((sel & (yi > 0) & m16).sum())
                    pred16 += int((sel & m16).sum())
                    true16 += int(((yi > 0) & m16).sum())
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
               jump_acc=jump_ok / max(1, jump_n), hold_type_acc=ht_ok / max(1, ht_n),
               f1_hard16=2 * tp16 / max(1, pred16 + true16), n_hard16=true16)
    model.train()
    return res


def lr_two_phase(step: int, steps1: int, steps2: int, lr: float, lr_mid: float,
                 lr_end: float, warmup: int) -> float:
    """Warmup, cosine from `lr` to `lr_mid` over phase 1, then cosine from
    `lr_mid` to `lr_end` over phase 2."""
    if step < warmup:
        return lr * (step + 1) / warmup
    if step < steps1:
        t = (step - warmup) / max(1, steps1 - warmup)
        return lr_mid + (lr - lr_mid) * 0.5 * (1 + np.cos(np.pi * t))
    t = min(1.0, (step - steps1) / max(1, steps2))
    return lr_end + (lr_mid - lr_end) * 0.5 * (1 + np.cos(np.pi * t))


def build_training_set(args, log=print):
    """-> (train dataset, validation songs, description). One folder trains as
    before; a corpus root trains on the scheduled mix of its sources."""
    if not args.corpus_root:
        train = StageADataset(load_corpus(args.corpus, "train", args.max_songs,
                                          args.reencode), args.crop, True, args.seed,
                              args.p_drop_stems)
        val = load_corpus(args.corpus, "val", args.max_songs, args.reencode)
        return train, val, {"sources": {"real": len(train)}}
    src = load_sources(args.corpus_root, STAGE_A_SOURCES, args.max_songs, args.reencode, log)

    def ds(songs):
        return StageADataset(songs, args.crop, True, args.seed, args.p_drop_stems)

    parts = {name: ds(src[name]) for name in ("real", *STAGE_A_SOURCES) if name in src}
    if "aug_stem" in src:
        parts["aug_mild"] = ds([s for s in src["aug_stem"] if s.kind in MILD_KINDS])
    items = args.epoch_items or int(round(len(parts["real"]) / PHASE1_SHARES["real"]))
    train = MixedDataset(parts, [(0, PHASE1_SHARES), (args.phase1_epochs, PHASE2_SHARES)],
                         items, args.seed)
    desc = {"sources": {k: len(v) for k, v in parts.items()}, "epoch_items": items,
            "phase1": train.shares(0), "phase2": train.shares(args.phase1_epochs)}
    return train, src["val"], desc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="one compiled corpus folder")
    ap.add_argument("--corpus-root", help="a root of source folders (real/, aug_stem/, ...)")
    ap.add_argument("--out", "--checkpoint-dir", dest="out", required=True)
    ap.add_argument("--epochs", type=int, default=50,
                    help="single-phase length (ignored with --corpus-root)")
    ap.add_argument("--phase1-epochs", type=int, default=65)
    ap.add_argument("--phase2-epochs", type=int, default=30)
    ap.add_argument("--epoch-items", type=int,
                    help="training items per epoch of a mixed corpus (default: "
                         "real charts / their phase-1 share)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-mid", type=float, default=5e-5, help="lr where phase 1 ends")
    ap.add_argument("--lr-end", type=float, default=1e-6, help="lr where phase 2 ends")
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
    ap.add_argument("--p-drop-stems", type=float, default=P_DROP_STEMS)
    ap.add_argument("--sub16-weight", type=float, default=1.0,
                    help=f"positive weight on 16th ticks of dense charts (v1.5: {L.SUB16_WEIGHT})")
    ap.add_argument("--sub12-weight", type=float, default=1.0,
                    help=f"...and on triplet ticks (v1.5: {L.SUB12_WEIGHT})")
    ap.add_argument("--phase-margin-weight", type=float, default=0.0,
                    help=f"weight of the beat-phase margin terms (v1.5: {L.LAMBDA_PHASE})")
    ap.add_argument("--phase-shift-prob", type=float, default=0.0,
                    help="share of eligible low-tier crops also trained half a beat "
                         "off the grid (v1.5: 0.25)")
    ap.add_argument("--phase-shift-weight", type=float, default=0.5,
                    help="weight of the ordinary loss on those shifted crops")
    ap.add_argument("--select", choices=["auto", "r_prec", "val_score"], default="auto",
                    help="checkpoint metric (auto: val_score with --corpus-root)")
    ap.add_argument("--stems-val-every", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-songs", type=int, help="debug: cap songs per split")
    ap.add_argument("--max-steps", type=int, help="debug: stop after this many steps")
    ap.add_argument("--reencode", action="store_true",
                    help="re-run the codec on the stored raw rows at load time")
    args = ap.parse_args(argv)
    if bool(args.corpus) == bool(args.corpus_root):
        ap.error("give exactly one of --corpus and --corpus-root")
    mixed = bool(args.corpus_root)
    select = args.select if args.select != "auto" else ("val_score" if mixed else "r_prec")
    epochs = args.phase1_epochs + args.phase2_epochs if mixed else args.epochs

    tc.seed_everything(args.seed)
    device = tc.pick_device(args.device)
    cfg = RhythmConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                       dropout=args.dropout)
    train_ds, val_songs, desc = build_training_set(args)
    run = tc.RunDir(args.out, {"stage": "a", "model": cfg.to_dict(), "args": vars(args),
                               "data": desc, "select": select})
    # validation the way the app runs: no stem channels
    val_ds = StageADataset(val_songs, args.crop, False, args.seed, stems="off")
    has_stems = any(s.stems is not None for s in val_songs)
    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, args.batch, shuffle=not mixed, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, **kw)
    val_stems_dl = (DataLoader(StageADataset(val_songs, args.crop, False, args.seed,
                                             stems="on"), args.batch, shuffle=False, **kw)
                    if has_stems else None)

    model = RhythmNet(cfg).to(device)
    opt = torch.optim.AdamW(tc.param_groups(model, args.weight_decay), lr=args.lr,
                            betas=(0.9, 0.98))
    steps_per_epoch = max(1, len(train_dl))
    total = epochs * steps_per_epoch
    steps1 = args.phase1_epochs * steps_per_epoch
    step, start_epoch, best = 0, 0, -1.0
    state = run.load("last")
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, start_epoch, best = state["step"], state["epoch"] + 1, state["best"]
        print(f"[train] resumed at epoch {start_epoch}, step {step}", flush=True)
    print(f"[train] RhythmNet {model.n_params / 1e6:.2f}M params on {device}; "
          f"{len(train_ds)} train items / {len(val_ds)} val charts; "
          f"{steps_per_epoch} steps/epoch x {epochs}; select on {select}; "
          f"data {desc}", flush=True)

    def lr_now(st: int) -> float:
        if mixed:
            return lr_two_phase(st, steps1, total - steps1, args.lr, args.lr_mid,
                                args.lr_end, args.warmup)
        return tc.lr_at(st, total, args.lr, args.warmup)

    gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    loss_kw = dict(sub16_weight=args.sub16_weight, sub12_weight=args.sub12_weight,
                   lambda_phase=args.phase_margin_weight)
    model.train()
    for epoch in range(start_epoch, epochs):
        train_ds.epoch = epoch
        t0, acc, n_acc = time.time(), {}, 0
        for batch in train_dl:
            for g in opt.param_groups:
                g["lr"] = lr_now(step)
            batch = tc.to_device(batch, device)
            with tc.autocast_for(device, args.amp):
                out = model(batch["x"], batch["cond"], batch["valid"])
            loss, parts = stage_a_loss(out, batch, args.lambda_card, args.lambda_w1,
                                       **loss_kw)
            if args.phase_shift_prob > 0:
                pick = (L.phase_eligible(batch).cpu()
                        & (torch.rand(len(batch["x"]), generator=gen)
                           < args.phase_shift_prob)).to(device)
                if bool(pick.any()):
                    direction = 1 if float(torch.rand(1, generator=gen)) < 0.5 else -1
                    moved = L.shift_half_beat({k: v[pick] for k, v in batch.items()},
                                              direction)
                    with tc.autocast_for(device, args.amp):
                        out2 = model(moved["x"], moved["cond"], moved["valid"])
                    # follow the music onto the off-beats...
                    loss2, _ = stage_a_loss(out2, moved, args.lambda_card, args.lambda_w1)
                    # ...and let the phase score say that is where it went
                    contrast = L.phase_contrast_loss(
                        L.phase_score(out[0][pick], batch["valid"][pick]),
                        L.phase_score(out2[0], moved["valid"]))
                    loss = (loss + args.phase_shift_weight * float(pick.float().mean()) * loss2
                            + args.phase_margin_weight * contrast)
                    parts["shift"] = float(loss2.detach())
                    parts["contrast"] = float(contrast.detach())
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
        val.update(phase_eval(model, val_songs, device, args.amp, args.crop))
        val["val_score"] = val_score(val)
        row = {"epoch": epoch, "step": step, "phase": 2 if mixed and epoch >= args.phase1_epochs
               else 1,
               **{f"train_{k}": v / max(1, n_acc) for k, v in acc.items()},
               **{f"val_{k}": v for k, v in val.items()}}
        last_epoch = epoch == epochs - 1 or (args.max_steps and step >= args.max_steps)
        if val_stems_dl is not None and (epoch % args.stems_val_every == 0 or last_epoch):
            vs = evaluate(model, val_stems_dl, device, args.amp, args)
            row.update({f"val_stems_{k}": vs[k] for k in ("ap", "r_prec", "f1", "f1_hard16")})
        row["epoch_s"] = round(time.time() - t0, 1)
        run.log(**row)
        ckpt = dict(model=model.state_dict(), config=cfg.to_dict(), epoch=epoch,
                    step=step, val=val, args=vars(args))
        if val[select] > best:
            best = val[select]
            run.save("best", **ckpt)
        run.save("last", opt=opt.state_dict(), best=best, **ckpt)
        if args.max_steps and step >= args.max_steps:
            break
    print(f"[train] done; best val {select} {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
