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

With `--corpus-root` (a root of source folders, see `neural/synth`) training
is a two-phase curriculum: human charts, programmatic step patterns laid over
real audio (`synth_style_grounded`) and the same kind of pattern with no audio
at all (`synth_style`, fed through the network's "no audio" vector). On a
share of the human charts the audio is withheld too (`--p-drop-audio`), so
that vector means "decide from rhythm and style", not "this one is synthetic".
Phase 2 is mostly human charts at a low learning rate.

Every `--probe-every` epochs a style probe samples the first validation
windows with chosen style sliders and measures what came out:
  probe_mae_*     |requested - generated| share with each window asked for its
                  own human chart's crossover / jack / footswitch shares
  probe_*_lo/hi   generated share with that one slider set low, then high
The probe samples the way the app does (the model's own distribution, notes
committed in random order). The v1.5 training run used most-confident-first
decoding here, which never writes a crossover or a footswitch at all, so the
probe columns in that run's log say nothing.
Stage B can only steer what it decides — which foot, which panel. How many
notes, jumps and streams there are is Stage A's, and is measured end to end.
  probe_*_delta   hi minus lo, with the other two footing sliders held at 0
  probe_sens      how much of the asked-for change the three sliders deliver,
                  each capped at 1, averaged (1.0 = all three follow)

A corpus root with `synth_cf/` (see `synth/counterfactual.py`) trains the
v1.5b mix instead: human charts plus the same rhythms re-stepped in several
footing styles, drawn in pairs of two variants of one rhythm, with the three
footing sliders always visible on them. `--select ce_probe` then keeps the
checkpoint with the best validation cross-entropy among those whose sliders
work (`best.pt`), and the plain best cross-entropy as `best_ce.pt`.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import codec as C
from .. import conditioning as cond
from . import train_common as tc
from .choreo_net import ChoreoConfig, ChoreoNet, class_mask, mask_tokens
from .dataset import (CROP_TICKS, MAX_EVENTS, P_DROP_STEMS, GroupedStageBDataset,
                      MixedDataset, StageBDataset, collate_b, load_corpus, load_sources)
from .losses import LAMBDA_ERGO, stage_b_loss
from .train_stage_a import lr_two_phase

PHASE1_SHARES = {"real": 0.55, "synth_style_grounded": 0.30, "synth_style": 0.15}
PHASE2_SHARES = {"real_polish": 0.75, "synth_style_grounded": 0.20, "synth_style": 0.05}
#: v1.5b: human charts, the same songs' rhythms re-stepped in other footing
#: styles (`synth_cf`), and re-stepped synthetic stream rhythms with no audio.
CF_PHASE1_SHARES = {"real": 0.45, "synth_cf": 0.45, "synth_cf_stream": 0.10}
CF_PHASE2_SHARES = {"real_polish": 0.65, "synth_cf": 0.30, "synth_cf_stream": 0.05}
MIXES = {"v15": (PHASE1_SHARES, PHASE2_SHARES), "v15b": (CF_PHASE1_SHARES, CF_PHASE2_SHARES)}
STAGE_B_SOURCES = ("aug_stem", "synth_groove", "synth_style_grounded", "synth_style",
                   "synth_cf", "synth_cf_stream")
#: Style sliders Stage B itself can act on: (name, index in the style vector,
#: low request, high request, the change between the two that counts as
#: following the slider).
PROBE_FIELDS = (("cross", cond.INDEX["crossover_share"], 0.0, 0.20, 0.08),
                ("jack", cond.INDEX["jack_share"], 0.0, 0.10, 0.04),
                ("fsw", cond.INDEX["footswitch_share"], 0.0, 0.16, 0.06))
PROBE_WINDOWS = 50
#: Checkpoint score with `--select ce_probe`: validation cross-entropy plus
#: this much for sliders that do nothing (0 when all three move as far as
#: PROBE_FIELDS asks). Large against the epoch-to-epoch wobble of the
#: cross-entropy, small against a model that got worse at human charts.
SENS_WEIGHT = 0.05


def measured_shares(tokens: np.ndarray, pad: np.ndarray) -> np.ndarray:
    """(B, 3) crossover / jack / footswitch share of each generated sequence,
    measured exactly as `conditioning.compute_style_vector` measures a chart."""
    out = np.zeros((len(tokens), 3), np.float32)
    for i, (seq, m) in enumerate(zip(tokens, pad)):
        tok = seq[m].astype(np.uint8)
        n = len(tok)
        if n == 0:
            continue
        zero = np.zeros(n, np.int32)
        enc = C.EncodedChart(C.LEFT, np.arange(n, dtype=np.int32) * 6, tok,
                             np.zeros(n, np.uint8), zero, zero)
        c = cond.compute_style_vector(enc, np.arange(n) * 0.25, 8)
        out[i] = c[[f[1] for f in PROBE_FIELDS]]
    return out


@torch.no_grad()
def style_probe(model, loader, device, amp: str, windows: int = PROBE_WINDOWS) -> dict:
    """Does the network do what the sliders ask? See the module docstring."""
    model.eval()
    batches, n = [], 0
    for batch in loader:
        take = min(len(batch["tokens"]), windows - n)
        batches.append({k: v[:take] for k, v in batch.items()})
        n += take
        if n >= windows:
            break
    res = {}

    def run(edit=None) -> np.ndarray:
        got = []
        # the same draws for every request, so lo/hi differ by the request only
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        for b in batches:
            b = tc.to_device(b, device)
            b["class_mask"] = class_mask(b["tokens"] >= 8)
            if edit is not None:
                b["cond"] = edit(b["cond"].clone())
            with tc.autocast_for(device, amp):
                # the app's own sampling; most-confident-first decoding never
                # produces a crossover whatever is asked (see sample_unmask)
                sampled = sample_unmask(model, b, tau=1.0, generator=gen, order="random")
            got.append(measured_shares(sampled.cpu().numpy(), b["pad"].cpu().numpy()))
        return np.concatenate(got)

    own = run()
    asked = np.concatenate([b["cond"][:, [f[1] for f in PROBE_FIELDS]].numpy()
                            for b in batches])
    sens = []
    for j, (name, idx, lo, hi, want) in enumerate(PROBE_FIELDS):
        res[f"probe_mae_{name}"] = float(np.abs(own[:, j] - np.maximum(asked[:, j], 0)).mean())

        def setter(value, idx=idx):
            def edit(c):
                # one slider moves; the other two footing sliders are held at
                # zero, so what is measured is this slider and nothing else
                for _, other, *_ in PROBE_FIELDS:
                    c[:, other] = value if other == idx else 0.0
                    c[:, cond.STYLE_DIM + other] = 1.0    # mark the slider as given
                return c
            return edit
        res[f"probe_{name}_lo"] = float(run(setter(lo))[:, j].mean())
        res[f"probe_{name}_hi"] = float(run(setter(hi))[:, j].mean())
        res[f"probe_{name}_delta"] = res[f"probe_{name}_hi"] - res[f"probe_{name}_lo"]
        sens.append(float(np.clip(res[f"probe_{name}_delta"] / want, 0.0, 1.0)))
    res["probe_sens"] = float(np.mean(sens))
    model.train()
    return res


@torch.no_grad()
def sample_unmask(model, batch: dict, steps: int = 10, tau: float = 0.0,
                  generator: torch.Generator | None = None,
                  order: str = "confidence") -> torch.Tensor:
    """Parallel decoding from a fully masked sequence on a cosine schedule
    (spec §6.5, without the Viterbi).

    order="confidence" is MaskGIT: the most confident tokens are committed
    first and the temperature is annealed to zero. It is mode-seeking — a rare
    move (a crossover, a footswitch) is never the most confident token, so it
    is re-drawn until it comes out common — which makes it right for the
    legality check and useless for asking what the sliders do.

    order="random" is what the app's decoder does (`infer.unmask`): every
    candidate is drawn from the model's own distribution at temperature `tau`
    and notes are committed in random order, i.e. plain ancestral sampling of
    the masked model."""
    if order not in ("confidence", "random"):
        raise ValueError(f"order must be 'confidence' or 'random', not {order!r}")
    pad, legal = batch["pad"], batch["class_mask"]
    tokens = torch.full_like(batch["tokens"], C.MASK)
    n_real = pad.sum(1)
    for s in range(1, steps + 1):
        logits = model(batch["ctx"], batch["scalars"], tokens, batch["cond"], pad).float()
        logits = logits.masked_fill(~legal, -1e9)
        temp = tau if order == "random" else tau * (1 - s / steps)
        if temp > 0:
            g = -torch.log(-torch.log(torch.rand(logits.shape, device=logits.device,
                                                 generator=generator).clamp_min(1e-9)))
            cand = (logits / temp + g).argmax(-1)
        else:
            cand = logits.argmax(-1)
        if order == "random":
            conf = torch.rand(cand.shape, device=logits.device, generator=generator)
        else:
            conf = torch.log_softmax(logits, -1).gather(-1, cand[..., None]).squeeze(-1)
        known = (tokens != C.MASK)
        conf = torch.where(known, torch.full_like(conf, 1e9), conf)   # keep commitments
        conf = conf.masked_fill(~pad, -1e9)
        keep = torch.floor(n_real * (1 - math.cos(math.pi * s / (2 * steps)))).long()
        keep = torch.where(torch.tensor(s == steps, device=keep.device), n_real, keep)
        by_conf = conf.argsort(dim=1, descending=True)
        rank = torch.empty_like(by_conf)
        rank.scatter_(1, by_conf, torch.arange(by_conf.shape[1], device=by_conf.device)
                      .expand_as(by_conf))
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


def build_training_set(args, log=print):
    """-> (train dataset, validation songs, description)."""
    def ds(songs, p_drop_audio=0.0):
        return StageBDataset(songs, args.crop, args.max_events, True, args.seed,
                             p_drop_stems=args.p_drop_stems, p_drop_audio=p_drop_audio)

    if not args.corpus_root:
        train = ds(load_corpus(args.corpus, "train", args.max_songs, args.reencode))
        val = load_corpus(args.corpus, "val", args.max_songs, args.reencode)
        return train, val, {"sources": {"real": len(train)}}
    src = load_sources(args.corpus_root, STAGE_B_SOURCES, args.max_songs, args.reencode, log)
    human = list(src["real"])
    if args.real_aug_share > 0 and "aug_stem" in src:
        human += src["aug_stem"]
    parts = {"real": ds(human, args.p_drop_audio),
             "real_polish": ds(human, args.p_drop_audio2)}
    mix = args.mix
    if mix == "auto":
        mix = "v15b" if "synth_cf" in src or "synth_cf_stream" in src else "v15"
    phase1, phase2 = MIXES[mix]
    for name in ("synth_cf", "synth_cf_stream"):
        if name in src and mix == "v15b":
            parts[name] = GroupedStageBDataset(
                src[name], args.crop, args.max_events, True, args.seed,
                p_drop_stems=args.p_drop_stems)
    if args.real_aug_share > 0 and "aug_stem" in src:
        # the variants outnumber the songs: weight them down to the asked share
        n_real = sum(len(s.charts) for s in src["real"])
        n_aug = sum(len(s.charts) for s in src["aug_stem"])
        w = args.real_aug_share / (1 - args.real_aug_share) * n_real / max(1, n_aug)
        for name in ("real", "real_polish"):
            for c in parts[name].charts:
                if c.song.source == "aug_stem":
                    c.weight = w
    for name in ("synth_style_grounded", "synth_style"):
        if name in src and mix == "v15":
            parts[name] = ds(src[name])
    items = args.epoch_items or int(round(sum(len(s.charts) for s in src["real"])
                                          / phase1["real"]))
    items += items % 2                      # pairs (see MixedDataset.pick)
    train = MixedDataset(parts, [(0, phase1), (args.phase1_epochs, phase2)],
                         items, args.seed)
    desc = {"sources": {k: len(v) for k, v in parts.items()}, "epoch_items": items,
            "mix": mix, "phase1": train.shares(0), "phase2": train.shares(args.phase1_epochs),
            "rhythm_groups": {k: len(v.groups) for k, v in parts.items()
                              if getattr(v, "groups", None)}}
    return train, src["val"], desc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="one compiled corpus folder")
    ap.add_argument("--corpus-root", help="a root of source folders (real/, synth_style/, ...)")
    ap.add_argument("--out", "--checkpoint-dir", dest="out", required=True)
    ap.add_argument("--epochs", type=int, default=60,
                    help="single-phase length (ignored with --corpus-root)")
    ap.add_argument("--phase1-epochs", type=int, default=100)
    ap.add_argument("--phase2-epochs", type=int, default=45)
    ap.add_argument("--epoch-items", type=int)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-mid", type=float, default=5e-5, help="lr where phase 1 ends")
    ap.add_argument("--lr-end", type=float, default=1e-6, help="lr where phase 2 ends")
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
    ap.add_argument("--p-drop-audio", "--p-drop-h", dest="p_drop_audio", type=float,
                    default=0.0, help="share of human charts trained with their audio "
                                      "withheld, phase 1 (v1.5: 0.20)")
    ap.add_argument("--p-drop-audio2", type=float, default=0.10, help="...in phase 2")
    ap.add_argument("--p-drop-stems", type=float, default=P_DROP_STEMS,
                    help="share of training items with the stem channels zeroed "
                         "(1.0: train as if the corpus had no stems)")
    ap.add_argument("--real-aug-share", type=float, default=0.0,
                    help="share of the human-chart draws taken from acoustic variants")
    ap.add_argument("--probe-every", type=int, default=5)
    ap.add_argument("--mix", choices=["auto", "v15", "v15b"], default="auto",
                    help="which sources train (auto: v15b when the corpus root has "
                         "counterfactual charts)")
    ap.add_argument("--select", choices=["auto", "ce", "ce_probe"], default="auto",
                    help="what picks best.pt: validation cross-entropy, or that plus a "
                         "charge for style sliders that do nothing (auto: ce_probe with "
                         "counterfactual charts; needs --probe-every 1)")
    ap.add_argument("--mirror-dir",
                    help="also copy checkpoints, log and config here (a mounted drive): "
                         "best.pt whenever it improves, everything at the end of each "
                         "phase, last.pt every --mirror-every epochs")
    ap.add_argument("--mirror-every", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=320)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-audio", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-songs", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--reencode", action="store_true")
    args = ap.parse_args(argv)
    if bool(args.corpus) == bool(args.corpus_root):
        ap.error("give exactly one of --corpus and --corpus-root")
    mixed = bool(args.corpus_root)
    epochs = args.phase1_epochs + args.phase2_epochs if mixed else args.epochs

    tc.seed_everything(args.seed)
    device = tc.pick_device(args.device)
    cfg = ChoreoConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                       d_audio=args.d_audio, dropout=args.dropout)
    train_ds, val_songs, desc = build_training_set(args)
    select = args.select
    if select == "auto":
        select = "ce_probe" if desc.get("mix") == "v15b" else "ce"
    if select == "ce_probe" and args.probe_every != 1:
        print("[train] --select ce_probe: probing every epoch", flush=True)
        args.probe_every = 1
    run = tc.RunDir(args.out, {"stage": "b", "model": cfg.to_dict(), "args": vars(args),
                               "data": desc, "select": select}, mirror=args.mirror_dir)
    # validation the way the app runs: real audio, no stem channels
    val_ds = StageBDataset([_without_stems(s) for s in val_songs], args.crop,
                           args.max_events, False, args.seed)
    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
              collate_fn=collate_b)
    train_dl = DataLoader(train_ds, args.batch, shuffle=not mixed, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, **kw)

    model = ChoreoNet(cfg).to(device)
    opt = torch.optim.AdamW(tc.param_groups(model, args.weight_decay), lr=args.lr,
                            betas=(0.9, 0.98))
    steps_per_epoch = max(1, len(train_dl))
    total = epochs * steps_per_epoch
    steps1 = args.phase1_epochs * steps_per_epoch
    step, start_epoch, best, best_ce = 0, 0, float("inf"), float("inf")
    state = run.load("last")
    if state:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, start_epoch, best = state["step"], state["epoch"] + 1, state["best"]
        best_ce = state.get("best_ce", best)
        print(f"[train] resumed at epoch {start_epoch}, step {step}", flush=True)
    print(f"[train] ChoreoNet {model.n_params / 1e6:.2f}M params on {device}; "
          f"{len(train_ds)} train items / {len(val_ds)} val charts; "
          f"{steps_per_epoch} steps/epoch x {epochs}; data {desc}", flush=True)

    def lr_now(st: int) -> float:
        if mixed:
            return lr_two_phase(st, steps1, total - steps1, args.lr, args.lr_mid,
                                args.lr_end, args.warmup)
        return tc.lr_at(st, total, args.lr, args.warmup)

    model.train()
    for epoch in range(start_epoch, epochs):
        train_ds.epoch = epoch
        t0, acc, n_acc = time.time(), {}, 0
        for batch in train_dl:
            for g in opt.param_groups:
                g["lr"] = lr_now(step)
            batch = tc.to_device(batch, device)
            batch["class_mask"] = class_mask(batch["tokens"] >= 8)
            inp, masked = mask_tokens(batch["tokens"], batch["pad"])
            with tc.autocast_for(device, args.amp):
                logits = model(batch["ctx"], batch["scalars"], inp, batch["cond"],
                               batch["pad"], batch["drop_audio"])
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
        last_epoch = epoch == epochs - 1 or (args.max_steps and step >= args.max_steps)
        if args.probe_every and (epoch % args.probe_every == 0 or last_epoch):
            val.update(style_probe(model, val_dl, device, args.amp))
        run.log(epoch=epoch, step=step,
                phase=2 if mixed and epoch >= args.phase1_epochs else 1,
                **{f"train_{k}": v / max(1, n_acc) for k, v in acc.items()},
                **{f"val_{k}": v for k, v in val.items()},
                epoch_s=round(time.time() - t0, 1))
        ckpt = dict(model=model.state_dict(), config=cfg.to_dict(), epoch=epoch,
                    step=step, val=val, args=vars(args))
        score = val["ce"]
        if select == "ce_probe":
            score += SENS_WEIGHT * (1.0 - val.get("probe_sens", 0.0))
        if score < best:
            best = score
            run.save("best", **ckpt)
            run.sync("best")
        if select == "ce_probe" and val["ce"] < best_ce:
            best_ce = val["ce"]
            run.save("best_ce", **ckpt)
        run.save("last", opt=opt.state_dict(), best=best, best_ce=best_ce, **ckpt)
        if last_epoch or (mixed and epoch == args.phase1_epochs - 1):
            run.sync("best", "best_ce", "last")          # end of a phase
        elif args.mirror_every and (epoch + 1) % args.mirror_every == 0:
            run.sync("last")                             # something to resume from
        if args.max_steps and step >= args.max_steps:
            break
    print(f"[train] done; best {'score' if select == 'ce_probe' else 'val ce'} {best:.4f}",
          flush=True)


def _without_stems(song):
    """A view of a song whose stem channels read as absent."""
    from .dataset import Song
    out = Song()
    for name in Song.__slots__:
        setattr(out, name, getattr(song, name, None))
    out.stems = None
    out.charts = []
    for c in song.charts:
        d = type(c)()
        for name in type(c).__slots__:
            setattr(d, name, getattr(c, name))
        d.song = out
        out.charts.append(d)
    return out


if __name__ == "__main__":
    main()
