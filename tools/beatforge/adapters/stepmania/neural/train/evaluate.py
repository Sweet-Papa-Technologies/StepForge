"""evaluate.py — score an EXPORTED model on held-out songs through the same
runtime path the app uses (onnxruntime + infer.py; no PyTorch).

    python -m beatforge.adapters.stepmania.neural.train.evaluate \
        --corpus build/neural_corpus --model-dir models/stepforge_flow \
        --split test --max-charts 400 --out models/stepforge_flow/eval_test.json

For each held-out human chart it generates a chart for the same audio at the
same difficulty slot and the human chart's own sustained NPS, then reports:

  rhythm    onset precision / recall / F1 of generated note ticks against the
            human chart's (exact tick, and within one 16th), plus the same for
            a no-model baseline that places notes on the strongest onset-flux
            peaks of the audio at the same density — what "just follow the
            transients" scores, so the model's number means something
  style     generated vs human shares of jumps, freezes, crossovers, jacks,
            and the foot-model cost per note
  legality  spins, moved-while-holding, unforced double-steps below Hard —
            all of which must be zero
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

from ...grammar import BUDGETS
from .. import codec as C
from .. import conditioning as cond
from .. import grid_features as GF
from .. import infer

TIERS = tuple(BUDGETS)


def _prf(pred: np.ndarray, true: np.ndarray, tol: int) -> tuple:
    if len(pred) == 0 or len(true) == 0:
        return 0.0, 0.0, 0.0
    used = np.zeros(len(true), bool)
    hit = 0
    for p in pred:                                  # greedy one-to-one matching
        j = int(np.searchsorted(true, p))
        best = None
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(true) and not used[k] and abs(int(true[k]) - int(p)) <= tol:
                if best is None or abs(int(true[k]) - int(p)) < abs(int(true[best]) - int(p)):
                    best = k
        if best is not None:
            used[best] = True
            hit += 1
    prec, rec = hit / len(pred), hit / len(true)
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def flux_baseline(x: np.ndarray, tick_times, difficulty: str, style, duration_s: float
                  ) -> np.ndarray:
    """Notes on the strongest spectral-flux peaks, decoded by the SAME tier
    rules and density target as the model."""
    flux = x[:, GF.FLUX].sum(1).astype(np.float64)
    z = (flux - flux.mean()) / (flux.std() + 1e-9)
    t = len(x)
    logits = {"onset": z.astype(np.float32), "jump": np.full(t, -9.0, np.float32),
              "hold_type": np.tile(np.array([9, 0, 0, 0, 0, 0], np.float32), (t, 1)),
              "hold_dur": np.zeros((t, 10), np.float32), "mine": np.full(t, -9.0, np.float32)}
    return infer.decode_rhythm(logits, tick_times, difficulty, style, duration_s).tick


def evaluate(corpus_dir, model_dir, split: str = "test", max_charts: int = 400,
             seed: int = 0, tau: float = infer.DEFAULT_TAU, int8: bool = False,
             cond_mode: str = "tier", log=print) -> dict:
    corpus_dir = Path(corpus_dir)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    engine = infer.StepManiaNeuralEngine(model_dir, int8=int8)
    items = [(s, c) for s in manifest["songs"] if s["split"] == split
             for c in s["charts"] if c["keep"] and c["difficulty"] in TIERS]
    rng = np.random.default_rng(seed)
    if len(items) > max_charts:
        items = [items[i] for i in np.sort(rng.choice(len(items), max_charts, False))]
    rows = []
    t0 = time.time()
    cache_key, cache = None, None
    for n, (s, c) in enumerate(items, 1):
        if cache_key != s["key"]:
            with np.load(corpus_dir / "songs" / f"{s['key']}.npz", allow_pickle=False) as z:
                cache = {k: z[k] for k in z.files}
            cache_key = s["key"]
        z = cache
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        dt = z["dt_ms"].astype(np.float64)
        tick_times = meta["tick0_time_s"] + np.concatenate([[0.0], np.cumsum(dt[:-1])]) / 1000.0
        x = GF.assemble(z["core"], z["dt_ms"])
        j, diff = c["j"], c["difficulty"]
        h_tick = z[f"c{j}_tick"].astype(np.int64)
        h_enc = C.EncodedChart(0, h_tick, z[f"c{j}_tok"], z[f"c{j}_flav"], z[f"c{j}_da"],
                               z[f"c{j}_ds"], z[f"c{j}_mines"])
        h_style = z[f"c{j}_style"].astype(np.float32)
        nps = float(min(h_style[1] * 16.0, BUDGETS[diff].max_nps_4s))
        if cond_mode == "human":
            # every slider set to what the human charter did: does the model
            # follow the sliders? (clamped to the tier, as the CLI would)
            style = cond.clamp_to_budget(cond.training_style(h_style), diff)
        else:
            # the default user-facing situation: a tier and a density, style free
            style = cond.tier_only_vector(diff)
            style[1] = nps / 16.0
            style[0] = cond.effective_meter(nps) / 15.0
        chart = engine.generate(x, tick_times, diff, style, meta["duration_s"],
                                seed=zlib.crc32(f"{s['key']}:{j}".encode()), tau=tau)
        g_tick = chart.events.tick.astype(np.int64)
        b_tick = flux_baseline(x, tick_times, diff, style, meta["duration_s"])
        # compare on the ticks this tier is allowed to use at all
        legal = np.isin(h_tick % 12, infer._SUBDIVISIONS[diff])
        h_legal = h_tick[legal]
        g_enc = chart.encoded
        times_s = tick_times[np.minimum(g_tick, len(tick_times) - 1)]
        g_style = cond.compute_style_vector(g_enc, times_s, c["meter"]) if len(g_enc) \
            else np.zeros(10, np.float32)
        st = infer.chart_stats(chart, tick_times)
        row = {"difficulty": diff, "n_human": int(len(h_tick)), "n_gen": int(len(g_tick)),
               "human_legal_share": float(legal.mean()) if len(legal) else 0.0,
               "f1_0": _prf(g_tick, h_legal, 0)[2], "f1_3": _prf(g_tick, h_legal, 3)[2],
               "p_0": _prf(g_tick, h_legal, 0)[0], "r_0": _prf(g_tick, h_legal, 0)[1],
               "base_f1_0": _prf(b_tick, h_legal, 0)[2],
               "base_f1_3": _prf(b_tick, h_legal, 3)[2],
               "nps_target": nps, "nps_gen": float(g_style[1] * 16.0),
               "spins": st.get("spins", 0), "lock_violations": st.get("lock_violations", 0),
               "double_steps": st.get("double_steps", 0),
               "relaxed": chart.stats.get("choreo", {}).get("relaxed", 0),
               "gen_cost": st.get("mean_cost", 0.0)}
        for name, i in (("jump", 3), ("hold", 4), ("cross", 5), ("jack", 6), ("fsw", 7)):
            row[f"h_{name}"], row[f"g_{name}"] = float(h_style[i]), float(g_style[i])
        row["h_cost"] = float(c.get("mean_cost", 0.0)) if "mean_cost" in c else None
        rows.append(row)
        if n % 50 == 0:
            log(f"[eval] {n}/{len(items)} charts, {time.time() - t0:.0f}s")

    def mean(key, sel=None):
        v = [r[key] for r in rows if (sel is None or r["difficulty"] == sel)
             and r[key] is not None]
        return round(float(np.mean(v)), 4) if v else None

    keys = ("f1_0", "f1_3", "p_0", "r_0", "base_f1_0", "base_f1_3", "nps_target",
            "nps_gen", "h_jump", "g_jump", "h_hold", "g_hold", "h_cross", "g_cross",
            "h_jack", "g_jack", "g_fsw", "gen_cost", "human_legal_share")
    summary = {"split": split, "charts": len(rows), "tau": tau, "cond_mode": cond_mode,
               "quantization": engine.model.quantization,
               "seconds_per_chart": round((time.time() - t0) / max(1, len(rows)), 3),
               "overall": {k: mean(k) for k in keys},
               "by_difficulty": {d: {"charts": sum(r["difficulty"] == d for r in rows),
                                     **{k: mean(k, d) for k in keys}} for d in TIERS},
               "legality": {"spins": int(sum(r["spins"] for r in rows)),
                            "lock_violations": int(sum(r["lock_violations"] for r in rows)),
                            "double_steps_below_hard": int(sum(
                                r["double_steps"] for r in rows
                                if r["difficulty"] in ("beginner", "easy", "medium"))),
                            "viterbi_relaxed": int(sum(r["relaxed"] for r in rows))}}
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-charts", type=int, default=400)
    ap.add_argument("--tau", type=float, default=infer.DEFAULT_TAU)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--cond", choices=["tier", "human"], default="tier",
                    help="tier: difficulty + density only (default); human: every "
                         "style slider set from the human chart")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = evaluate(a.corpus, a.model_dir, a.split, a.max_charts, tau=a.tau, int8=a.int8,
                   cond_mode=a.cond)
    text = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(text)
    o = res["overall"]
    print(f"[eval] {res['charts']} {a.split} charts: onset F1 {o['f1_0']} exact / "
          f"{o['f1_3']} within a 16th (flux baseline {o['base_f1_0']} / {o['base_f1_3']}); "
          f"legality {res['legality']}; {res['seconds_per_chart']}s per chart")
    for d, r in res["by_difficulty"].items():
        if r["charts"]:
            print(f"  {d:<10} n={r['charts']:<4} F1 {r['f1_0']} (base {r['base_f1_0']}) "
                  f"nps {r['nps_gen']}/{r['nps_target']} jumps {r['g_jump']}/{r['h_jump']} "
                  f"holds {r['g_hold']}/{r['h_hold']} cross {r['g_cross']}/{r['h_cross']} "
                  f"jack {r['g_jack']}/{r['h_jack']}")


if __name__ == "__main__":
    main()
