"""chart_thinning.py — Beginner and Easy charts thinned from the densest human
chart of each TRAINING song.

    python -m beatforge.adapters.stepmania.neural.synth.chart_thinning \
        --real-dir build/v15_corpus/real --out-dir build/v15_corpus/synth_thin

The low tiers are where a small model is weakest and where there is least to
learn from: one Beginner chart per song, if that. But a human Beginner chart
is very nearly a subset of the same song's Hard chart — measured on the 227
training songs, 92 % of Beginner notes and 90 % of Easy notes (medians) sit on
a tick the densest chart also uses. So a second, different-but-plausible low
tier chart can be made by choosing WHICH of the dense chart's notes to keep:

    salience = 0.45 metric + 0.35 percussive + 0.20 bar energy

`metric` prefers beat 1, then 3, then 2 and 4, then the off-beat 8ths;
`percussive` is the drum stem's onset flux at the note (the percussive flux of
the mix when there are no stems); `bar energy` is the loudness of the bar. The
most salient notes are kept until the chart is as dense as human charts of
that tier are, subject to the tier's rules: quarter notes only at Beginner,
8ths at most at Easy and few of them, a ceiling on notes per 4 s.

Nothing is invented: every kept note is a note of the human chart, on its
tick, on one of its panels. Rests are therefore preserved for free — a silence
in the source cannot gain a note. The spec'd rule "every jump becomes a tap"
is softened: human Beginner / Easy charts in this corpus carry 6-9 % jumps, so
jumps on beats 1 and 3 are kept up to that share and the rest become taps.

Output shards are ATTACHED: charts only, with `parent` naming the real shard
whose audio they sit on and the parent's `song_id`. A chart for a tier the
song already has is marked `companion` (the mixer samples those at half
weight). Only songs `shards.training_parents` lists are thinned.
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np

from ...grammar import BUDGETS
from .. import codec as C
from .. import conditioning as cond
from ..train import corpus
from . import shards
from .audio_augment import chart_events

TIERS = ("beginner", "easy")
GRID = {"beginner": C.TICKS_PER_BEAT, "easy": C.TICKS_PER_BEAT // 2}
#: Share of the source's legal-tick notes a human chart of the tier keeps:
#: (mean, sd, lo, hi), measured on the training songs that have both charts
#: (Beginner: median 0.45, 10-90 % 0.27-0.68; Easy: 0.50, 0.35-0.66).
KEEP_SHARE = {"beginner": (0.47, 0.15, 0.25, 0.70), "easy": (0.50, 0.12, 0.33, 0.68)}
#: Jumps a human chart of the tier carries (corpus means 0.076 / 0.100).
JUMP_SHARE = {"beginner": 0.06, "easy": 0.09}
W_METRIC, W_PERC, W_ENERGY = 0.45, 0.35, 0.20
METRIC_WEIGHT = {0: 1.0, 24: 0.8, 12: 0.7, 36: 0.7}      # tick in bar -> weight
OFFBEAT_WEIGHT = 0.35
NPS_WINDOW_S = 4.0
MIN_HOLD_TICKS = C.TICKS_PER_BEAT
MIN_SOURCE_NOTES = 60
RANK = {d: i for i, d in enumerate(("beginner", "easy", "medium", "hard", "challenge"))}


def _unit(v: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 95.0) -> np.ndarray:
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def salience(ticks: np.ndarray, core: np.ndarray, stems: np.ndarray | None) -> np.ndarray:
    """Per-note salience in [0, 1] from the song's grid features."""
    ticks = np.asarray(ticks, np.int64)
    t = np.clip(ticks, 0, len(core) - 1)
    in_bar = ticks % C.G
    metric = np.array([METRIC_WEIGHT.get(int(k), OFFBEAT_WEIGHT) for k in in_bar])
    if stems is not None:
        flux = np.asarray(stems[:, 1:4], np.float32).sum(1)          # drums
    else:
        flux = np.asarray(core[:, 23:26], np.float32).sum(1)         # percussive part
    perc = _unit(flux, 50.0, 99.0)[t]
    loud = np.asarray(core[:, 0:20], np.float32).mean(1)
    n_bars = int(np.ceil(len(core) / C.G))
    bar = np.array([loud[b * C.G:(b + 1) * C.G].mean() for b in range(n_bars)])
    energy = _unit(bar)[np.minimum(t // C.G, n_bars - 1)]
    return W_METRIC * metric + W_PERC * perc + W_ENERGY * energy


def keep_share(key: str, tier: str, seed: int = 0) -> float:
    """How much of the source to keep for this song and tier (seeded)."""
    mean, sd, lo, hi = KEEP_SHARE[tier]
    rng = np.random.default_rng((seed, zlib.crc32(key.encode()), RANK[tier]))
    return float(np.clip(rng.normal(mean, sd), lo, hi))


def select(ticks: np.ndarray, times_s: np.ndarray, score: np.ndarray, tier: str,
           share: float) -> np.ndarray:
    """Indices (ascending) of the source notes the thinned chart keeps."""
    budget = BUDGETS[tier]
    grid = GRID[tier]
    ticks = np.asarray(ticks, np.int64)
    cand = np.nonzero(ticks % grid == 0)[0]
    if len(cand) == 0:
        return cand
    target = max(1, int(round(share * len(cand))))
    cap = int(np.floor(budget.max_nps_4s * NPS_WINDOW_S))
    max_fine = int(np.floor(budget.fine_frac * target))
    # most salient first; ties go to the earlier note
    order = cand[np.lexsort((ticks[cand], -score[cand]))]
    chosen = np.zeros(len(ticks), bool)
    fine = 0
    for i in order:
        if int(chosen.sum()) >= target:
            break
        off_beat = ticks[i] % C.TICKS_PER_BEAT != 0
        if off_beat and fine >= max_fine:
            continue
        chosen[i] = True
        if not (_window_ok(times_s, chosen, i, cap)
                and _run_ok(ticks, chosen, i, grid, budget.max_run)):
            chosen[i] = False
            continue
        fine += off_beat
    return np.nonzero(chosen)[0]


def _window_ok(times_s: np.ndarray, chosen: np.ndarray, i: int, cap: int) -> bool:
    """With note i chosen, does every 4-second window still hold <= cap notes?"""
    t = times_s[i]
    near = np.nonzero(chosen & (times_s > t - NPS_WINDOW_S) & (times_s < t + NPS_WINDOW_S))[0]
    tt = times_s[near]
    for start in tt[tt <= t]:
        if int(((tt >= start) & (tt < start + NPS_WINDOW_S)).sum()) > cap:
            return False
    return True


def _run_ok(ticks: np.ndarray, chosen: np.ndarray, i: int, grid: int, max_run: int) -> bool:
    """With note i chosen, is every run of notes at the tier's finest spacing
    within the tier's run limit? (A quarter-note run at Beginner is not a run
    in this sense: `max_run` counts notes faster than the beat.)"""
    if grid >= C.TICKS_PER_BEAT:
        return True
    idx = np.nonzero(chosen)[0]
    pos = int(np.searchsorted(idx, i))
    run = 1
    j = pos
    while j > 0 and ticks[idx[j]] - ticks[idx[j - 1]] <= grid:
        run, j = run + 1, j - 1
    j = pos
    while j < len(idx) - 1 and ticks[idx[j + 1]] - ticks[idx[j]] <= grid:
        run, j = run + 1, j + 1
    return run <= max_run + 1


def thin_events(src: C.ChartEvents, keep: np.ndarray, score: np.ndarray, tier: str
                ) -> C.ChartEvents:
    """The kept notes as a chart of `tier`: mines gone, rolls and short holds
    turned into taps, holds and jumps kept only up to what the tier carries."""
    budget = BUDGETS[tier]
    n = len(keep)
    is_jump = src.p1[keep] >= 0
    on_strong = src.tick[keep] % (2 * C.TICKS_PER_BEAT) == 0
    jump_ok = np.zeros(n, bool)
    wanted = np.nonzero(is_jump & on_strong)[0]
    limit = int(np.floor(JUMP_SHARE[tier] * n))
    jump_ok[wanted[np.argsort(-score[keep][wanted], kind="stable")][:limit]] = True

    def hold_len(i: int, col: int) -> int:
        kind = (src.k0, src.k1)[col][i]
        dur = int((src.d0, src.d1)[col][i])
        return dur if (kind == C.HOLD and dur >= MIN_HOLD_TICKS) else 0

    max_holds = min(budget.max_holds, int(np.floor(budget.hold_share[1] * n)))
    held = [(float(score[i]), pos) for pos, i in enumerate(keep)
            if hold_len(i, 0) or (src.p1[i] >= 0 and hold_len(i, 1))]
    hold_pos = {pos for _, pos in sorted(held, reverse=True)[:max_holds]}

    rows, last_panel = [], -1
    for pos, i in enumerate(keep):
        cells = [(int(src.p0[i]), 0)]
        if src.p1[i] >= 0:
            cells.append((int(src.p1[i]), 1))
        if len(cells) == 2 and not jump_ok[pos]:
            # a jump becomes one tap: prefer its freeze, then a fresh panel
            cells.sort(key=lambda c: (-(hold_len(i, c[1]) > 0 and pos in hold_pos),
                                      c[0] == last_panel, c[0]))
            cells = cells[:1]
        out = []
        for panel, col in cells:
            dur = hold_len(i, col) if pos in hold_pos else 0
            out.append((panel, C.HOLD if dur else C.TAP, dur))
        rows.append((int(src.tick[i]), out))
        last_panel = out[0][0] if len(out) == 1 else -1
    return C.ChartEvents.from_rows(rows)


def source_chart(manifest_row: dict, meta: dict, tier: str) -> dict | None:
    """The chart to thin for `tier`: the densest kept chart of the song at
    Medium or above. (An Easy chart is too sparse to choose a Beginner from.)"""
    by_j = {c["j"]: c for c in meta["charts"]}
    best = None
    for c in manifest_row["charts"]:
        if not c["keep"] or RANK.get(c["difficulty"], -1) < RANK["medium"]:
            continue
        if c["n_events"] < MIN_SOURCE_NOTES:
            continue
        if best is None or c["n_events"] > best["n_events"]:
            best = by_j[c["j"]]
    return best


def thin_song(manifest_row: dict, meta: dict, arrays: dict, seed: int = 0) -> tuple | None:
    """One real shard -> (arrays, meta) of its attached `synth_thin` shard, or
    None when the song has no chart dense enough to thin."""
    key = meta["key"]
    core = arrays["core"]
    stems = arrays.get("stems")
    have = {c["difficulty"] for c in manifest_row["charts"] if c["keep"]}
    out, rows = {}, []
    for tier in TIERS:
        src_row = source_chart(manifest_row, meta, tier)
        if src_row is None:
            continue
        j = src_row["j"]
        src = chart_events(arrays, j)
        times_ms = arrays[f"c{j}_tms"].astype(np.float64)
        score = salience(src.tick, core, stems)
        keep = select(src.tick, times_ms / 1000.0, score, tier, keep_share(key, tier, seed))
        if len(keep) < shards.MIN_NOTES:
            continue
        ev = thin_events(src, keep, score, tier)
        nps = cond.sustained_nps(times_ms[keep] / 1000.0)
        lo, hi = BUDGETS[tier].meter_range
        meter = int(np.clip(round(cond.effective_meter(nps)), lo, hi))
        scratch: dict = {}
        row = corpus.pack_chart(scratch, len(rows), ev, times_ms[keep], meter, tier,
                                has_stems=stems is not None, description="thinned")
        if row["lock_violations"] or row["spins"]:
            # a freeze the thinned chart cannot stand on: fall back to taps
            ev = C.ChartEvents.from_rows([(t, [(c[0], C.TAP, 0) for c in cells])
                                          for t, cells in ev.rows()])
            scratch = {}
            row = corpus.pack_chart(scratch, len(rows), ev, times_ms[keep], meter, tier,
                                    has_stems=stems is not None, description="thinned")
        row.update(source_j=j, source_difficulty=src_row["difficulty"],
                   companion=tier in have)
        out.update(scratch)
        rows.append(row)
    if not rows:
        return None
    new = shards.base_meta(
        f"{key}__thin", manifest_row["song_id"], "synth_thin", title=meta["title"],
        artist=meta["artist"], n_ticks=meta["n_ticks"], duration_s=meta["duration_s"],
        has_stems=stems is not None, has_audio=False,
        parent={"source": "real", "key": key})
    new["charts"] = rows
    return out, new


def thin_corpus(real_dir, out_dir, seed: int = 0, limit: int | None = None,
                log=print) -> dict:
    real_dir, out_dir = Path(real_dir), Path(out_dir)
    parents = shards.training_parents(real_dir)
    keys = sorted(parents)[:limit] if limit else sorted(parents)
    written = skipped = 0
    for key in keys:
        meta, arrays = corpus.load_song(real_dir / "songs" / f"{key}.npz")
        res = thin_song(parents[key], meta, arrays, seed)
        if res is None:
            skipped += 1
            continue
        corpus.save_shard(out_dir / "songs" / f"{res[1]['key']}.npz", *res)
        written += 1
    (out_dir / "songs").mkdir(parents=True, exist_ok=True)
    manifest = shards.write_source_manifest(out_dir, "synth_thin", log)
    log(f"[thin] {written} song(s) thinned, {skipped} had no chart dense enough")
    return {"songs": written, "skipped": skipped, "charts": manifest["n_charts_kept"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    print(f"[thin] done: {thin_corpus(a.real_dir, a.out_dir, a.seed, a.limit)}")


if __name__ == "__main__":
    main()
