"""train_critic.py — the choreography critic (v1.6 Phase 3; see `neural/critic.py`).

    python3 -m beatforge.adapters.stepmania.neural.train.train_critic \
        --corpus-root <corpus>/real --out <run dir>

Positives are the human charts of the corpus. Negatives are the SAME charts —
same ticks, same jumps, same freezes — with only the arrows of their single
notes damaged, one kind of damage at a time:

    loop       a span rewritten as a 2-, 3- or 4-arrow loop
    random     arrows of a span replaced at random, or doubled into jacks
               (uncomfortable footing)
    unbalanced one panel favoured, or one panel emptied
    shuffle    the arrow sequence slid along the song, or its 4-bar blocks
               reordered: every figure survives, its place in the music does not
    uncallback where a repeated bar brought back the arrows of the bar it
               repeats, those arrows changed
    smooth     the other side: jacks, same-foot steps and crossovers of a span
               replaced by plain alternating steps. More comfortable than the
               human chart, and not what a person wrote: without it the critic
               learns that the smoothest sample is always the best one

Each damaged chart is read back by the codec's annotator, as the human chart
was, so both sides are footed the same way. A two-layer MLP on
`critic.chart_features` learns to tell them apart; the validation songs of the
manifest are never trained or selected on. Writes `stage_3_critic.onnx`
(normalization inside the graph), `critic_meta.json` and `features.npz`.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

from .. import codec as C
from .. import critic as CR
from .. import event_features as EF
from .. import grid_features as GF
from .. import structure as ST

KINDS = ("loop", "random", "unbalanced", "shuffle", "uncallback", "smooth")
#: Share of the eligible single notes a damaged span covers.
SPAN = (0.3, 1.0)
MIN_NOTES = 24
BLOCK_BARS = 4


def _span(elig: np.ndarray, rng) -> np.ndarray:
    """A contiguous run of the eligible notes."""
    m = len(elig)
    k = max(8, int(round(m * rng.uniform(*SPAN))))
    k = min(k, m)
    a = int(rng.integers(0, m - k + 1))
    return elig[a:a + k]


def damage(kind: str, panel: np.ndarray, elig: np.ndarray, bar: np.ndarray, rng,
           link_pairs=None):
    """A damaged copy of `panel` (the panel of every note; only `elig` notes
    may change), or None when this chart cannot carry this kind."""
    p = panel.copy()
    if len(elig) < 16:
        return None
    if kind == "loop":
        idx = _span(elig, rng)
        period = int(rng.integers(2, 5))
        pat = [int(rng.integers(0, 4))]
        while len(pat) < period:
            q = int(rng.integers(0, 4))
            if q != pat[-1] and not (len(pat) == period - 1 and q == pat[0]):
                pat.append(q)
        p[idx] = np.asarray(pat)[np.arange(len(idx)) % period]
    elif kind == "random":
        idx = _span(elig, rng)
        if rng.random() < 0.5:
            hit = idx[rng.random(len(idx)) < rng.uniform(0.3, 0.8)]
            p[hit] = rng.integers(0, 4, len(hit))
        else:                                              # jacks: repeat the arrow before
            for i in idx[rng.random(len(idx)) < rng.uniform(0.15, 0.4)]:
                if i > 0:
                    p[i] = p[i - 1]
    elif kind == "unbalanced":
        idx = _span(elig, rng)
        fav = int(rng.integers(0, 4))
        if rng.random() < 0.5:
            rate = rng.uniform(0.3, 0.6)
            for i in idx:
                if rng.random() < rate and (i == 0 or p[i - 1] != fav):
                    p[i] = fav
        else:                                              # one panel emptied
            hit = idx[p[idx] == fav]
            p[hit] = (fav + rng.integers(1, 4, len(hit))) % 4
    elif kind == "shuffle":
        seq = p[elig]
        if rng.random() < 0.5 or bar[elig].max() // BLOCK_BARS < 3:
            if len(seq) < 40:
                return None
            seq = np.roll(seq, int(rng.integers(16, len(seq) - 16)))
        else:
            blk = bar[elig] // BLOCK_BARS
            parts = [seq[blk == b] for b in np.unique(blk)]
            order = rng.permutation(len(parts))
            if (order == np.arange(len(parts))).all():
                order = np.roll(order, 1)
            seq = np.concatenate([parts[i] for i in order])
        p[elig] = seq
    elif kind == "uncallback":
        if link_pairs is None or not len(link_pairs[0]):
            return None
        later, earlier = link_pairs
        ok = np.isin(later, elig) & (p[later] == p[earlier])
        if ok.sum() < 6:
            return None
        hit = later[ok]
        p[hit] = (p[hit] + rng.integers(1, 4, len(hit))) % 4
    else:
        raise ValueError(kind)
    return None if (p == panel).all() else p


def smooth(panel: np.ndarray, elig: np.ndarray, rng, read):
    """Panels with the jacks of a span moved to another arrow, then the steps
    the annotator (`read(panels) -> tokens`) still takes with the same foot or
    crossed over moved too. None when the chart has too few of either to
    matter."""
    if len(elig) < 16:
        return None
    idx = _span(elig, rng)
    rate = rng.uniform(0.6, 1.0)
    p = panel.copy()
    n = len(p)

    def move(i):
        near = {int(p[i - 1]) if i > 0 else -1, int(p[i + 1]) if i + 1 < n else -1}
        free = [q for q in range(4) if q != p[i] and q not in near]
        if free:
            p[i] = free[int(rng.integers(0, len(free)))]

    moved = 0
    for i in idx:
        if i > 0 and p[i] == p[i - 1] and rng.random() < rate:
            move(i)
            moved += 1
    tok = np.asarray(read(p), np.int64)
    for i in idx:
        if (4 <= tok[i] < 8 or tok[i] == C.CROSS) and rng.random() < rate:
            move(i)
            moved += 1
    return p if moved >= max(4, n // 50) and (p != panel).any() else None


def chart_rows(z, meta: dict, entry: dict, tau_sim: float, seed: int, negatives: int = 1):
    """Feature rows of one song: (features, label, kind, difficulty) per chart
    and per damaged copy."""
    x = GF.assemble(z["core"].astype(np.float32), z["dt_ms"])
    st = ST.analyze_structure(x, tau_sim=tau_sim)
    rows = []
    for ce, cm in zip(entry["charts"], meta["charts"]):
        if not ce["keep"] or cm["difficulty"] not in CR.DIFFICULTIES:
            continue
        j = cm["j"]
        tick = z[f"c{j}_tick"].astype(np.int64)
        if len(tick) < MIN_NOTES:
            continue
        tms = z[f"c{j}_tms"].astype(np.float64)
        raw, rawd = z[f"c{j}_raw"], z[f"c{j}_rawd"]
        tok, flav = z[f"c{j}_tok"], z[f"c{j}_flav"]
        is_jump = tok >= 8
        held = EF.active_holds(tick, flav, z[f"c{j}_da"], z[f"c{j}_ds"]) > 0
        prep = CR.prepare(x, tick, tms, is_jump, cm["difficulty"], st.motif_links,
                          st.novelty, held)
        rows.append((CR.chart_features(prep, tok, int(cm["f_init"])), 1, "human",
                     cm["difficulty"]))
        p0 = raw[0].astype(np.int64)
        elig = np.nonzero((raw[1] < 0) & (raw[2] == 0) & ~held)[0]
        bar = tick // C.G
        rng = np.random.default_rng(zlib.crc32(f"{entry['key']}:{j}:{seed}".encode()))
        for kind in KINDS:
            for _ in range(negatives):
                if kind == "smooth":
                    p = smooth(p0, elig, rng, lambda q: C.encode(C.ChartEvents(
                        z[f"c{j}_tick"], q.astype(np.int8), raw[1], raw[2], raw[3],
                        rawd[0], rawd[1], z[f"c{j}_mines"]), tms)[0].token)
                else:
                    p = damage(kind, p0, elig, bar, rng, (prep.link_later, prep.link_earlier))
                if p is None:
                    continue
                ev = C.ChartEvents(z[f"c{j}_tick"], p.astype(np.int8), raw[1], raw[2], raw[3],
                                   rawd[0], rawd[1], z[f"c{j}_mines"])
                enc, _ = C.encode(ev, tms)
                rows.append((CR.chart_features(prep, enc.token, int(enc.f_init)), 0, kind,
                             cm["difficulty"]))
    return rows


def build(corpus_root: Path, tau_sim: float, seed: int, negatives: int, log=print) -> dict:
    manifest = json.loads((corpus_root / "manifest.json").read_text())
    out = {k: [] for k in ("x", "y", "kind", "difficulty", "split", "song")}
    t0 = time.time()
    entries = [s for s in manifest["songs"]
               if s["split"] in ("train", "val") and any(c["keep"] for c in s["charts"])]
    for n, entry in enumerate(entries, 1):
        with np.load(corpus_root / "songs" / f"{entry['key']}.npz", allow_pickle=False) as z:
            if "core" not in z.files:
                continue
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            for f, y, kind, diff in chart_rows(z, meta, entry, tau_sim, seed, negatives):
                out["x"].append(f)
                out["y"].append(y)
                out["kind"].append(kind)
                out["difficulty"].append(diff)
                out["split"].append(entry["split"])
                out["song"].append(entry["key"])
        if n % 25 == 0 or n == len(entries):
            log(f"[critic-data] {n}/{len(entries)} songs, {len(out['y'])} rows ({time.time() - t0:.0f}s)")
    return {k: np.asarray(v) for k, v in out.items()}


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Probability that a positive outranks a negative (ties count half)."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(score, kind="stable")
    s = np.asarray(score)[order]
    rank = np.empty(len(s), np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rank[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = int(y.sum())
    return float((rank[y].sum() - pos * (pos + 1) / 2.0) / (pos * (len(y) - pos)))


def train(data: dict, out: Path, seed: int = 0, hidden=(48, 24), epochs: int = 400,
          drop_features=(), log=print) -> dict:
    import torch
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, y = data["x"].astype(np.float32), data["y"].astype(np.float32)
    is_train, is_val = data["split"] == "train", data["split"] == "val"
    songs = np.unique(data["song"][is_train])
    held = set(rng.permutation(songs)[:max(1, len(songs) * 15 // 100)].tolist())
    sel = is_train & np.isin(data["song"], list(held))        # model selection only
    fit = is_train & ~sel
    mu, sd = x[fit].mean(0), x[fit].std(0)
    keep = np.ones(x.shape[1], np.float32)
    for f in drop_features:
        keep[CR.INDEX[f]] = 0.0
    scale = (keep / np.maximum(sd, 1e-6)).astype(np.float32) * (sd > 1e-6)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mu", torch.tensor(mu))
            self.register_buffer("scale", torch.tensor(scale))
            h1, h2 = hidden
            self.body = nn.Sequential(nn.Linear(x.shape[1], h1), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1))

        def forward(self, features):
            return self.body((features - self.mu) * self.scale).squeeze(-1)

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    xf, yf = torch.tensor(x[fit]), torch.tensor(y[fit])
    pos_w = torch.tensor(float((yf == 0).sum() / max(1.0, float((yf == 1).sum()))))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    xs, ys = torch.tensor(x[sel]), y[sel]
    best, best_state = -1.0, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xf))
        for a in range(0, len(xf), 256):
            idx = perm[a:a + 256]
            opt.zero_grad()
            lossf(net(xf[idx]), yf[idx]).backward()
            opt.step()
        if ep % 10 == 9:
            net.eval()
            with torch.no_grad():
                a_sel = auc(net(xs).numpy(), ys)
            if a_sel > best:
                best, best_state = a_sel, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "stage_3_critic.onnx"
    torch.onnx.export(net, (torch.zeros(1, x.shape[1]),), str(path), input_names=["features"],
                      output_names=["score"], dynamic_axes={"features": {0: "batch"},
                                                            "score": {0: "batch"}},
                      opset_version=17, dynamo=False)
    score = CR.Critic(path).score(x)
    with torch.no_grad():
        parity = float(np.abs(net(torch.tensor(x)).numpy() - score).max())
    report = {"features": list(CR.FEATURES), "dropped_features": list(drop_features),
              "hidden": list(hidden), "params": int(sum(p.numel() for p in net.parameters())),
              "onnx_bytes": path.stat().st_size, "parity_max_abs_diff": parity,
              "rows": {"fit": int(fit.sum()), "selection": int(sel.sum()), "val": int(is_val.sum())},
              "auc_selection": round(best, 4), "auc_val": round(auc(score[is_val], y[is_val]), 4),
              "auc_val_by_kind": {}, "auc_val_by_difficulty": {}}
    human = is_val & (data["kind"] == "human")
    for kind in KINDS:
        m = human | (is_val & (data["kind"] == kind))
        report["auc_val_by_kind"][kind] = {"auc": round(auc(score[m], y[m]), 4),
                                           "negatives": int((is_val & (data["kind"] == kind)).sum())}
    for d in CR.DIFFICULTIES:
        m = is_val & (data["difficulty"] == d)
        report["auc_val_by_difficulty"][d] = round(auc(score[m], y[m]), 4)
    # what each feature says on its own, and what the score loses without it
    single = {}
    for i, f in enumerate(CR.FEATURES):
        a = auc(x[is_val, i], y[is_val])
        xp = x[is_val].copy()
        xp[:, i] = rng.permutation(xp[:, i])
        single[f] = {"alone": round(max(a, 1 - a), 3),
                     "drop_when_shuffled": round(report["auc_val"] - auc(CR.Critic(path).score(xp), y[is_val]), 4)}
    report["features_on_val"] = single
    (out / "critic_meta.json").write_text(json.dumps(report, indent=1))
    log(f"[critic] val AUC {report['auc_val']} (selection {report['auc_selection']}), "
        f"{report['params']} params, {report['onnx_bytes']} bytes")
    log("[critic] by kind: " + "  ".join(f"{k} {v['auc']}" for k, v in report["auc_val_by_kind"].items()))
    return report


def train_typical(data: dict, out: Path, band: float = 1.0, drop_features=(), log=print) -> dict:
    """The other kind of critic: no negatives at all. Per difficulty, the mean
    and spread of every feature over the human training charts; a chart's
    score is minus the sum of how far each feature lies outside `band`
    standard deviations of that mean, squared. Two-sided by construction: the
    best-scored sample is the most ordinary one, not the most extreme.
    Callback features count only where bars are linked, section features only
    where there is a section boundary. Written as the same ONNX interface."""
    import torch
    from torch import nn
    x, y = data["x"].astype(np.float32), data["y"].astype(np.float32)
    is_train, is_val = data["split"] == "train", data["split"] == "val"
    human = is_train & (data["kind"] == "human")
    nf = x.shape[1]
    d0 = CR.INDEX["is_beginner"]
    mu, sd = np.zeros((5, nf), np.float32), np.ones((5, nf), np.float32)
    cb = np.asarray([f.startswith("cb_") and f != "cb_chance" for f in CR.FEATURES])
    sec = np.asarray([f in ("section_change", "section_contrast") for f in CR.FEATURES])
    w = np.ones(nf, np.float32)
    w[CR.CONTEXT] = 0.0
    for f in drop_features:
        w[CR.INDEX[f]] = 0.0
    for d in range(5):
        rows = x[human & (x[:, d0 + d] > 0)]
        linked, sect = rows[rows[:, CR.INDEX["has_links"]] > 0], rows[rows[:, CR.INDEX["has_sections"]] > 0]
        for i in range(nf):
            r = linked[:, i] if cb[i] else sect[:, i] if sec[i] else rows[:, i]
            if len(r) >= 10:
                mu[d, i], sd[d, i] = r.mean(), max(float(r.std()), 1e-3)
            else:
                w[i] = w[i] if len(rows) == 0 else w[i]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            for name, v in (("mu", mu), ("sd", sd), ("w", w), ("cb", cb.astype(np.float32)),
                            ("sec", sec.astype(np.float32))):
                self.register_buffer(name, torch.tensor(v))

        def forward(self, features):
            onehot = features[:, d0:d0 + 5]
            z = (features - onehot @ self.mu) / (onehot @ self.sd)
            pen = torch.relu(z.abs() - band) ** 2 * self.w
            links = features[:, CR.INDEX["has_links"]:CR.INDEX["has_links"] + 1]
            sects = features[:, CR.INDEX["has_sections"]:CR.INDEX["has_sections"] + 1]
            gate = 1.0 - self.cb * (1.0 - links) - self.sec * (1.0 - sects)
            return -(pen * gate).sum(dim=1)

    net = Net().eval()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "stage_3_critic.onnx"
    torch.onnx.export(net, (torch.zeros(1, nf),), str(path), input_names=["features"],
                      output_names=["score"], dynamic_axes={"features": {0: "batch"},
                                                            "score": {0: "batch"}},
                      opset_version=17, dynamo=False)
    score = CR.Critic(path).score(x)
    report = {"kind": "typical", "band_sd": band, "features": list(CR.FEATURES),
              "dropped_features": list(drop_features), "params": int(mu.size + sd.size),
              "onnx_bytes": path.stat().st_size,
              "rows": {"fit": int(human.sum()), "val": int(is_val.sum())},
              "auc_val": round(auc(score[is_val], y[is_val]), 4), "auc_val_by_kind": {}}
    hv = is_val & (data["kind"] == "human")
    for kind in KINDS:
        m = hv | (is_val & (data["kind"] == kind))
        report["auc_val_by_kind"][kind] = {"auc": round(auc(score[m], y[m]), 4),
                                           "negatives": int((is_val & (data["kind"] == kind)).sum())}
    (out / "critic_meta.json").write_text(json.dumps(report, indent=1))
    log(f"[critic] typical: val AUC {report['auc_val']}, {report['onnx_bytes']} bytes")
    log("[critic] by kind: " + "  ".join(f"{k} {v['auc']}" for k, v in report["auc_val_by_kind"].items()))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau-sim", type=float, default=0.70, help="the decoder's callback threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--negatives", type=int, default=2, help="damaged copies per kind and chart")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--drop-feature", action="append", default=[])
    ap.add_argument("--typical", action="store_true",
                    help="no negatives: score closeness to the human charts' own feature ranges")
    ap.add_argument("--features-from", help="a features.npz of another run to train on")
    ap.add_argument("--reuse-features", action="store_true",
                    help="train on <out>/features.npz from an earlier run")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "features.npz"
    if a.features_from:
        data = dict(np.load(a.features_from, allow_pickle=False))
    elif a.reuse_features and cache.exists():
        data = dict(np.load(cache, allow_pickle=False))
    else:
        data = build(Path(a.corpus_root), a.tau_sim, a.seed, a.negatives)
        np.savez_compressed(cache, **data)
    if a.typical:
        train_typical(data, out, drop_features=a.drop_feature)
    else:
        train(data, out, seed=a.seed, epochs=a.epochs, drop_features=a.drop_feature)


if __name__ == "__main__":
    main()
