"""audio_grounder.py — lay programmatic step patterns over real audio.

    python -m beatforge.adapters.stepmania.neural.synth.audio_grounder \
        --corpus-root build/v15_corpus [--share 0.6]

`synth_style` patterns have no audio. Trained on them alone, the step network
would learn that a stream or a crossover run is something that happens in
silence. So a share of the patterns (60 % by default) is seated on a window of
a training song instead — real, acoustically varied, or synthesized — whose
tempo is within 15 % of the pattern's and whose loudness suits it (a stamina
stream goes on a loud passage, a casual pattern on an average one). The step
network then sees genuine audio around every note of the pattern. The other
40 % stay symbolic and reach the network through its "no audio" vector.

The step network (Stage B) has its own audio encoder over the grid features;
it does not read the rhythm network's hidden state. Grounding therefore needs
no trained model: it only chooses a window and re-times the pattern on it.

A grounded pattern is re-encoded on its new timing (the tempo differs a
little, which can change the cheapest footing and the measured density), so
its tokens and its style label are again exactly what the codec measures.
The output shard is ATTACHED to the song it sits on and carries that song's
`song_id`; windows are only ever taken from training-only audio.

`synth_style/manifest.json` is rewritten with a `grounded` flag per pattern,
which is how the loader keeps the two uses of a pattern apart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import codec as C
from ..train import corpus
from . import shards
from .audio_augment import chart_events

AUDIO_SOURCES = ("real", "aug_stem", "synth_groove")
BPM_TOLERANCE = 0.15
CANDIDATES = 8
ATTEMPTS = 3
#: A pattern longer than the song it is seated on is cut at the song's end, as
#: long as this many bars and notes of it are left.
MIN_BARS, MIN_EVENTS = 12, 64
#: Loudness rank (0 = the song's quietest bars, 1 = its loudest) a pattern of
#: each archetype is seated on; None = the loudest window on offer.
ENERGY_TARGET = {"itg_stamina": None, "jumpstream": None, "modern_hard_burst": 0.75,
                 "crossover_flow": 0.60, "tech_footswitch": 0.60, "casual_ddr_flow": 0.45}
#: Stream patterns go on synthesized driving-16th tracks this often when one
#: is in tempo range: that audio has a transient under every 16th.
STREAM_ARCHETYPES = ("itg_stamina", "jumpstream")
GROOVE_PREFERENCE = 0.5


def _audio_index(root: Path, log=print) -> list:
    """Every training-only song with audio: tempo, length and per-bar loudness
    rank. Real songs are restricted to the training split."""
    pool = []
    for source in AUDIO_SOURCES:
        d = root / source
        if not (d / "manifest.json").exists():
            continue
        if source == "real":
            rows = list(shards.training_parents(d).values())
        else:
            rows = json.loads((d / "manifest.json").read_text())["songs"]
        for row in rows:
            with np.load(d / "songs" / f"{row['key']}.npz", allow_pickle=False) as z:
                meta = json.loads(bytes(z["meta"]).decode("utf-8"))
                dt = z["dt_ms"].astype(np.float64)
                loud = z["core"][:, 0:20].astype(np.float32).mean(1)
            n_bars = len(dt) // C.G
            if n_bars < 8:
                continue
            bar = loud[:n_bars * C.G].reshape(n_bars, C.G).mean(1)
            rank = np.argsort(np.argsort(bar)) / max(1, n_bars - 1)
            pool.append({"source": source, "key": row["key"], "song_id": row["song_id"],
                         "bpm": 5000.0 / float(np.median(dt)), "n_bars": n_bars,
                         "rank": rank.astype(np.float32),
                         "tick_times": meta["tick0_time_s"]
                         + np.concatenate([[0.0], np.cumsum(dt[:-1])]) / 1000.0,
                         "has_stems": bool(meta.get("has_stems")),
                         "archetype": meta.get("archetype")})
    log(f"[ground] {len(pool)} training songs with audio to seat patterns on")
    return pool


def _window(seq: dict, pool: list, by_bpm: np.ndarray, rng) -> list:
    """Candidate (song, first bar) seats for one pattern, best first."""
    bars = int(np.ceil(seq["n_ticks"] / C.G))
    ratio = by_bpm / seq["bpm"]
    in_tempo = np.abs(ratio - 1.0) <= BPM_TOLERANCE
    fits = np.array([p["n_bars"] >= bars for p in pool])
    if not (fits & in_tempo).any():
        # no song in tempo range holds the whole pattern: one that holds a
        # usable part of it will do (the overhang is cut, see ground_pattern)
        fits = np.array([p["n_bars"] >= min(bars, MIN_BARS) for p in pool])
    near = np.nonzero(fits & in_tempo)[0]
    if len(near) == 0:                  # nothing in range: the closest tempos
        ok = np.nonzero(fits)[0]
        near = ok[np.argsort(np.abs(np.log(ratio[ok])))[:5]]
    if len(near) == 0:
        return []
    if seq["archetype"] in STREAM_ARCHETYPES and rng.random() < GROOVE_PREFERENCE:
        drive = [i for i in near if pool[i]["archetype"] == "stamina_drive"]
        near = np.array(drive) if drive else near
    target = ENERGY_TARGET.get(seq["archetype"], 0.6)
    seats = []
    for i in rng.choice(near, size=CANDIDATES, replace=True):
        p = pool[int(i)]
        start = int(rng.integers(0, max(0, p["n_bars"] - bars) + 1))
        level = float(p["rank"][start:start + bars].mean())
        seats.append((-level if target is None else abs(level - target), int(i), start))
    return sorted(seats)


def ground_pattern(arrays: dict, meta: dict, seat: dict, start_bar: int) -> tuple | None:
    """One symbolic pattern on one window -> (arrays, meta) of its attached
    shard, or None if the pattern cannot be stood on at this tempo."""
    src = meta["charts"][0]
    ev = chart_events(arrays, src["j"])
    tick = ev.tick.astype(np.int64) + start_bar * C.G
    # whole bars of the song only, and a freeze must end inside them too: the
    # pattern stops at the first note that does not fit
    end = (len(seat["tick_times"]) // C.G) * C.G
    over = np.nonzero(tick + np.maximum(ev.d0, ev.d1) >= end)[0]
    n = int(over[0]) if len(over) else len(tick)
    if n == 0 or n < min(MIN_EVENTS, len(tick)):
        return None
    cut = len(tick) - n
    tick = tick[:n]
    ev = C.ChartEvents(tick.astype(np.int32), ev.p0[:n], ev.p1[:n], ev.k0[:n], ev.k1[:n],
                       ev.d0[:n], ev.d1[:n])
    out: dict = {}
    row = corpus.pack_chart(out, 0, ev, seat["tick_times"][tick] * 1000.0, src["meter"],
                            src["difficulty"], has_stems=seat["has_stems"],
                            description=meta.get("archetype", ""))
    if row["lock_violations"] or row["spins"]:
        return None
    new = shards.base_meta(
        f"{meta['key']}__g", seat["song_id"], "synth_style_grounded",
        title=meta.get("title", ""), artist=meta.get("artist", ""),
        n_ticks=len(seat["tick_times"]), duration_s=float(seat["tick_times"][-1]),
        has_stems=seat["has_stems"], has_audio=False,
        parent={"source": seat["source"], "key": seat["key"]},
        archetype=meta.get("archetype"), pattern=meta["key"], start_tick=start_bar * C.G,
        cut=int(cut),
        bpm_pattern=meta.get("bpm"), bpm_audio=round(seat["bpm"], 3))
    new["charts"] = [row]
    return out, new


def ground_corpus(corpus_root, share: float = 0.6, seed: int = 0, log=print) -> dict:
    root = Path(corpus_root)
    style_dir, out_dir = root / "synth_style", root / "synth_style_grounded"
    manifest = json.loads((style_dir / "manifest.json").read_text())
    pool = _audio_index(root, log)
    if not pool:
        raise ValueError("no training audio under this corpus root to ground on")
    by_bpm = np.array([p["bpm"] for p in pool])
    (out_dir / "songs").mkdir(parents=True, exist_ok=True)
    for old in shards.shard_files(out_dir):
        old.unlink()

    # the same share of every archetype, chosen by seed
    rng = np.random.default_rng(seed)
    by_arch: dict = {}
    for s in manifest["songs"]:
        by_arch.setdefault(s.get("archetype", ""), []).append(s["key"])
    chosen = set()
    for arch in sorted(by_arch):
        keys = sorted(by_arch[arch])
        chosen |= set(rng.permutation(keys)[:int(round(share * len(keys)))].tolist())

    grounded, failed, used = set(), 0, {}
    for s in manifest["songs"]:
        if s["key"] not in chosen:
            continue
        meta, arrays = corpus.load_song(style_dir / "songs" / f"{s['key']}.npz")
        seq = {"n_ticks": meta["n_ticks"], "bpm": float(meta["bpm"]),
               "archetype": meta.get("archetype", "")}
        res = None
        for _, i, start in _window(seq, pool, by_bpm, rng)[:ATTEMPTS]:
            res = ground_pattern(arrays, meta, pool[i], start)
            if res is not None:
                used[pool[i]["source"]] = used.get(pool[i]["source"], 0) + 1
                break
        if res is None:
            failed += 1
            continue
        corpus.save_shard(out_dir / "songs" / f"{res[1]['key']}.npz", *res)
        grounded.add(s["key"])
    for s in manifest["songs"]:
        s["grounded"] = s["key"] in grounded
    manifest["n_grounded"] = len(grounded)
    (style_dir / "manifest.json").write_text(json.dumps(manifest))
    shards.write_source_manifest(out_dir, "synth_style_grounded", log)
    log(f"[ground] {len(grounded)} pattern(s) grounded on {used}; {failed} could not be "
        f"seated and stay symbolic; {len(manifest['songs']) - len(grounded)} symbolic")
    return {"grounded": len(grounded), "failed": failed,
            "symbolic": len(manifest["songs"]) - len(grounded), "on": used}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--share", type=float, default=0.6,
                    help="share of the patterns to ground (the rest stay symbolic)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(f"[ground] done: {ground_corpus(a.corpus_root, a.share, a.seed)}")


if __name__ == "__main__":
    main()
