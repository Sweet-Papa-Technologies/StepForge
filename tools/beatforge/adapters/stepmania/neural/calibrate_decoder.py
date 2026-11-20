"""calibrate_decoder.py — fit the decoder settings that belong to one set of
weights, on songs that have human charts, and store them with the model.

    python -m beatforge.adapters.stepmania.neural.calibrate_decoder \
        --model-dir models/stepforge_flow --tuning-dir /path/to/tuning/songs \
        [--cache-dir build/calibration] [--jobs 6] [--write]

`--tuning-dir` holds one folder per song: an audio file and the human simfile
(.sm / .ssc). The human charts are used for two things only — where their beat
grid sits and how many notes each difficulty has. Keep these songs apart from
any you report results on: a setting fitted on a song flatters it.

Two settings are fitted (both live in `model_meta.json` under `decoder`):

**The half-beat rule.** The beat tracker can seat the grid on the off-beats;
the rhythm model is asked whether it did (`pipeline.verify_phase`). How far to
trust the answer depends on the weights, so three rules are compared on the
tuning songs — the onset statistic alone above a margin (the margin swept),
a vote of three statistics, and the statistic where it is sure with the vote
where it is not — and the one that seats the most grids where the human
charter did is kept. A change from the current rule must fix at least
`--min-gain` grids; a tie keeps what the model has.

**The density scale.** With the density slider left to the model, the decoder
keeps as many notes as the model expects, times a per-difficulty scale. The
scale is set so the MEDIAN chart has as many notes as the human chart of the
same song (within `--tolerance`).

It also reports, without changing anything, how many 16th notes a Hard and a
Challenge chart carry next to the human charts — the check that a model
trained to use them does.

Without `--write` nothing is changed: the fitted values are printed and saved
beside the cache for inspection.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .... import dsp
from ..grammar import ALL_DIFFICULTIES
from . import codec as C
from . import grid_features as GF
from . import infer, pipeline

AUDIO_EXTS = (".ogg", ".mp3", ".wav", ".flac", ".opus")
MARGINS = tuple(round(0.2 + 0.1 * i, 1) for i in range(19))        # 0.2 .. 2.0
SCALE_RANGE = (0.30, 1.50)
MAX_TEMPO_MISMATCH = 0.005
HALF_BEAT_WINDOW = 0.15            # beats: how near 0 or 0.5 a phase must be to count
_ENGINE = None


# --------------------------------------------------------------------------- #
# Tuning songs
# --------------------------------------------------------------------------- #
def find_songs(tuning_dir) -> list:
    """[(name, audio path, simfile path)] for every usable song folder."""
    out = []
    for folder in sorted(p for p in Path(tuning_dir).iterdir() if p.is_dir()):
        sims = sorted(folder.glob("*.ssc")) + sorted(folder.glob("*.sm"))
        audio = sorted(q for q in folder.iterdir() if q.suffix.lower() in AUDIO_EXTS)
        if sims and audio:
            out.append((folder.name, str(max(audio, key=lambda q: q.stat().st_size)),
                        str(sims[0])))
    return out


def human_charts(sim_path: str) -> dict:
    """The human simfile's grid and note ticks: {'bpm', 'offset_s' (time of
    beat 0), 'constant', 'ticks': {difficulty: array}}."""
    import simfile
    from simfile.timing import TimingData
    sf = simfile.open(sim_path, strict=False)
    td = TimingData(sf)
    bpms = [(float(b.beat), float(b.value)) for b in td.bpms]
    stops = [(float(x.beat), float(x.value)) for x in td.stops]
    timing = GF.TimingMap(-float(td.offset), bpms, stops)
    ticks = {}
    for chart in sf.charts:
        if (chart.stepstype or "").strip().lower() != "dance-single":
            continue
        slot = (chart.difficulty or "").strip().lower()
        if slot not in ALL_DIFFICULTIES or slot in ticks:
            continue
        triples, _, _ = C.simfile_chart_to_ticks(chart, timing.bpm_at)
        ev, _ = C.events_from_notes(triples)
        if len(ev) >= 5:
            ticks[slot] = ev.tick.astype(np.int64)
    return {"bpm": bpms[0][1], "offset_s": float(timing.offset_s),
            "constant": timing.is_constant, "ticks": ticks}


def half_beat_off(bpm: float, offset_s: float, human: dict) -> bool | None:
    """Is a grid half a beat from the human one? None when the tempos differ,
    the human grid is not one constant tempo, or the phase is neither."""
    if not human["constant"] or abs(bpm / human["bpm"] - 1.0) > MAX_TEMPO_MISMATCH:
        return None
    phase = ((offset_s - human["offset_s"]) * human["bpm"] / 60.0) % 1.0
    if min(phase, 1.0 - phase) <= HALF_BEAT_WINDOW:
        return False
    if abs(phase - 0.5) <= HALF_BEAT_WINDOW:
        return True
    return None


def sixteenths(ticks: np.ndarray) -> int:
    t = np.asarray(ticks) % C.TICKS_PER_BEAT
    return int(((t % 3 == 0) & (t % 6 != 0)).sum())


# --------------------------------------------------------------------------- #
# Per-song work (runs in worker processes)
# --------------------------------------------------------------------------- #
def _engine(model_dir: str):
    global _ENGINE
    if _ENGINE is None or str(_ENGINE.model.dir) != str(Path(model_dir).expanduser()):
        _ENGINE = infer.StepManiaNeuralEngine(model_dir, threads=1)
    return _ENGINE


def _analysis(name: str, audio: str, cache_dir: str | None) -> dict:
    path = Path(cache_dir) / f"{name}.analysis.json" if cache_dir else None
    if path is not None and path.exists():
        return json.loads(path.read_text())
    an = dsp.analyze_signal(audio)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(an))
    return an


def _evidence(job: tuple) -> dict:
    """The three half-beat statistics of one song on the grid as the beat
    tracker left it, and whether that grid is half a beat from the human one."""
    name, audio, sim, model_dir, cache_dir = job
    engine = _engine(model_dir)
    an = _analysis(name, audio, cache_dir)
    human = human_charts(sim)
    g, align = pipeline._features(audio, an["bpm"], an["offset"], an["duration_s"],
                                  an.get("beat_backend", "local_dsp"), None)
    saved = engine.model.meta.get("decoder")
    engine.model.meta["decoder"] = dict(saved or {}, phase_rule="vote")
    try:
        votes = pipeline.verify_phase(engine, g.core, g.dt_ms)["votes"]
    finally:
        engine.model.meta["decoder"] = saved
    return {"name": name, "truth": half_beat_off(align["bpm"], align["offset"], human),
            "votes": votes}


def _chart(job: tuple) -> dict:
    """Note counts of the charts the model writes for one song, per difficulty,
    against the human charts' — under the decoder block given."""
    name, audio, sim, model_dir, cache_dir, decoder = job
    engine = _engine(model_dir)
    engine.model.meta["decoder"] = decoder
    an = _analysis(name, audio, cache_dir)
    human = human_charts(sim)
    ctx = pipeline.prepare_song(audio, an, log=lambda *a, **k: None, engine=engine)
    charts, _, _ = pipeline.chart_song(ctx, tuple(human["ticks"]), engine, track_id=name)
    rows = {}
    for slot, chart in charts.items():
        rows[slot] = {"n": len(chart.events), "n_human": len(human["ticks"][slot]),
                      "sixteenths": sixteenths(chart.events.tick),
                      "sixteenths_human": sixteenths(human["ticks"][slot])}
    return {"name": name, "charts": rows,
            "half_beat_off": half_beat_off(ctx.align["bpm"], ctx.align["offset"], human)}


def _map(fn, jobs: list, workers: int, label: str, log) -> list:
    t0 = time.time()
    if workers <= 1:
        out = [fn(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(fn, jobs))
    log(f"[calibrate] {label}: {len(jobs)} songs in {time.time() - t0:.0f}s")
    return out


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_phase_rule(rows: list, current: dict, min_gain: int = 2) -> dict:
    """Choose the half-beat rule from per-song evidence. `rows` are
    `_evidence` results; `current` is the model's present decoder block."""
    rows = [r for r in rows if r["truth"] is not None and r["votes"]]

    def errors(rule: str, margin: float) -> int:
        return sum(pipeline.phase_decision(rule, margin, r["votes"]["onset"],
                                           r["votes"]["jump"], r["votes"]["audio"])
                   != r["truth"] for r in rows)

    table = {"vote": errors("vote", 0.0),
             "margin": {str(m): errors("margin", m) for m in MARGINS},
             "margin_vote": {str(m): errors("margin_vote", m) for m in MARGINS}}
    cur_rule = current.get("phase_rule", "margin")
    cur_margin = float(current.get("phase_margin", pipeline.PHASE_MARGIN))
    cur_err = errors(cur_rule, cur_margin)
    best = (cur_err, cur_rule, cur_margin)
    for rule in ("margin", "margin_vote"):
        e = {float(m): v for m, v in table[rule].items()}
        low = min(e.values())
        # the middle of the widest flat stretch at the minimum: the least
        # sensitive margin, not the luckiest one
        runs, start = [], None
        for i, m in enumerate(MARGINS):
            if e[m] == low and start is None:
                start = m
            if e[m] != low and start is not None:
                runs.append((start, MARGINS[i - 1]))
                start = None
        if start is not None:
            runs.append((start, MARGINS[-1]))
        a, b = max(runs, key=lambda r: r[1] - r[0])
        mid = MARGINS[int(np.argmin([abs(m - (a + b) / 2) for m in MARGINS]))]
        if low <= cur_err - min_gain and low < best[0]:
            best = (low, rule, mid)
    if table["vote"] <= cur_err - min_gain and table["vote"] < best[0]:
        best = (table["vote"], "vote", cur_margin)
    return {"songs": len(rows), "off_before": sum(r["truth"] for r in rows),
            "errors": table, "current": {"rule": cur_rule, "margin": cur_margin,
                                         "errors": cur_err},
            "chosen": {"rule": best[1], "margin": best[2], "errors": best[0]}}


def median_ratios(results: list) -> dict:
    out = {}
    for slot in ALL_DIFFICULTIES:
        r = [s["charts"][slot]["n"] / s["charts"][slot]["n_human"] for s in results
             if slot in s["charts"] and s["charts"][slot]["n_human"]]
        if r:
            out[slot] = statistics.median(r)
    return out


def fit_density(jobs: list, decoder: dict, start: dict, workers: int, tolerance: float,
                rounds: int, log) -> tuple:
    """Iterate scale <- scale / median(notes / human notes) per difficulty
    until every median is within `tolerance` of 1. Returns (scale, results of
    the last pass, history)."""
    scale = {d: float(start.get(d, infer.DENSITY_SCALE.get(d, 1.0)))
             for d in ALL_DIFFICULTIES}
    history, results = [], []
    for n in range(rounds + 1):
        block = dict(decoder, density_scale=scale)
        results = _map(_chart, [(*j, block) for j in jobs], workers,
                       f"density pass {n + 1}", log)
        ratios = median_ratios(results)
        history.append({"scale": dict(scale), "median_ratio": ratios})
        log("[calibrate]   " + "  ".join(f"{d} x{scale[d]:.2f} -> {ratios.get(d, 0):.2f}"
                                         for d in ALL_DIFFICULTIES))
        off = {d: r for d, r in ratios.items() if abs(r - 1.0) > tolerance}
        if not off or n == rounds:
            break
        scale = {d: round(float(np.clip(scale[d] / ratios[d], *SCALE_RANGE)), 3)
                 if d in off else scale[d] for d in scale}
    return scale, results, history


def calibrate(model_dir, tuning_dir, cache_dir=None, workers: int = 6,
              tolerance: float = 0.05, rounds: int = 4, min_gain: int = 2,
              write: bool = False, log=print) -> dict:
    model_dir = infer.resolve_model_dir(model_dir)
    meta_path = model_dir / infer.META_FILE
    meta = json.loads(meta_path.read_text())
    current = dict(meta.get("decoder") or {})
    songs = find_songs(tuning_dir)
    if not songs:
        raise SystemExit(f"no song folders with audio and a simfile in {tuning_dir}")
    base = [(n, a, s, str(model_dir), str(cache_dir) if cache_dir else None)
            for n, a, s in songs]
    log(f"[calibrate] {len(songs)} tuning songs, model {model_dir}")

    phase = fit_phase_rule(_map(_evidence, base, workers, "half-beat evidence", log),
                           current, min_gain)
    c = phase["chosen"]
    log(f"[calibrate] grids half a beat off before the check: {phase['off_before']} of "
        f"{phase['songs']}; wrong after it — current rule ({phase['current']['rule']} "
        f"{phase['current']['margin']}): {phase['current']['errors']}, vote: "
        f"{phase['errors']['vote']}, best margin: {min(phase['errors']['margin'].values())}, "
        f"best margin_vote: {min(phase['errors']['margin_vote'].values())} "
        f"-> {c['rule']} (margin {c['margin']}): {c['errors']}")
    decoder = dict(current, phase_rule=c["rule"], phase_margin=c["margin"])

    scale, results, history = fit_density(base, decoder, current.get("density_scale") or {},
                                          workers, tolerance, rounds, log)
    decoder["density_scale"] = scale
    six = {}
    for slot in ("hard", "challenge"):
        rows = [s["charts"][slot] for s in results if slot in s["charts"]]
        if rows:
            six[slot] = {"median": statistics.median(r["sixteenths"] for r in rows),
                         "human_median": statistics.median(r["sixteenths_human"]
                                                           for r in rows)}
            log(f"[calibrate] 16th notes per {slot} chart: median {six[slot]['median']:g} "
                f"(human {six[slot]['human_median']:g})")
    still_off = sum(bool(s["half_beat_off"]) for s in results)
    report = {"model": str(model_dir), "tuning_songs": len(songs), "phase": phase,
              "density": {"scale": scale, "history": history,
                          "median_ratio": history[-1]["median_ratio"]},
              "sixteenths": six, "grids_half_a_beat_off": still_off, "decoder": decoder}
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / "calibration.json").write_text(json.dumps(report, indent=1))
    if write:
        meta["decoder"] = decoder
        meta["calibration"] = {"tuning_songs": len(songs),
                               "phase_errors": c["errors"], "phase_songs": phase["songs"],
                               "median_ratio": {k: round(v, 3) for k, v in
                                                history[-1]["median_ratio"].items()}}
        meta_path.write_text(json.dumps(meta, indent=2))
        log(f"[calibrate] wrote decoder settings to {meta_path}")
    else:
        log(f"[calibrate] fitted (not written; pass --write): {json.dumps(decoder)}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="default: the installed model")
    ap.add_argument("--tuning-dir", required=True)
    ap.add_argument("--cache-dir", help="keeps the per-song audio analysis between runs")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--min-gain", type=int, default=2)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    calibrate(a.model_dir, a.tuning_dir, a.cache_dir, a.jobs, a.tolerance, a.rounds,
              a.min_gain, a.write)


if __name__ == "__main__":
    main()
