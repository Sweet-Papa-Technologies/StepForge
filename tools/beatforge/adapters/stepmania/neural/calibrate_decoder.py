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

**Energy tilt at Beginner and Easy** (`--fit-energy-tilt`, off by default). The
decoder leans a chart's notes toward the song's louder bars, harder if the
chart would otherwise not follow the song's energy (a QA gate). A rhythm
network trained longer is surer of itself and the engine's steepest lean no
longer moves it; if fewer than 98 % of the tuning songs' Beginner and Easy
charts pass the gates, longer ladders are tried and the mildest that restores
the pass rate without costing placement is stored as `decoder.energy_tilt`.

**16th-note figures at Hard** (`--fit-sixteenths`, off by default). The
decoder can promote 16ths between two notes a Hard chart already has
(`infer._promote_sixteenths`); how hard is `decoder.hard_sixteenth_boost`, an
odds multiplier. The smallest value is taken at which the median Hard chart
has `--sixteenth-target` of the median human Hard chart's 16ths — but only if
step placement against the human charts stays within `--sixteenth-f1-loss` of
what it is without the boost and as many charts pass the QA gates. If no
value does both, none is stored.

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
from .. import qa as sm_qa
from ..grammar import ALL_DIFFICULTIES
from . import codec as C
from . import grid_features as GF
from . import infer, pipeline

AUDIO_EXTS = (".ogg", ".mp3", ".wav", ".flac", ".opus")
MARGINS = tuple(round(0.2 + 0.1 * i, 1) for i in range(19))        # 0.2 .. 2.0
SCALE_RANGE = (0.30, 1.50)
MAX_TEMPO_MISMATCH = 0.005
HALF_BEAT_WINDOW = 0.15            # beats: how near 0 or 0.5 a phase must be to count
#: Odds multipliers tried for `decoder.hard_sixteenth_boost`, smallest first.
SIXTEENTH_BOOSTS = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
SIXTEENTH_TARGET = 0.55            # share of the human median to reach
SIXTEENTH_F1_LOSS = 0.005          # placement F1 it may cost at most
PLACEMENT_TOL_S = 0.020
#: Ladders of energy-tilt weights tried for the two lowest tiers
#: (`decoder.energy_tilt`), mildest first; each only extends the engine's own.
LOW_TIER_TILTS = ((1.5, 3.0, 5.0, 8.0), (1.5, 3.0, 6.0, 12.0))
LOW_TIERS = ("beginner", "easy")
GATE_PASS_TARGET = 0.98
TILT_F1_LOSS = 0.005
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


def placement_f1(times: np.ndarray, truth: np.ndarray, tol: float = PLACEMENT_TOL_S
                 ) -> float | None:
    """Step-placement F1 against a human chart of the same song, +-`tol`
    seconds, one note matched to one note, after removing the one constant
    offset between the two charts (their beat 0 need not be the same instant).
    None for charts too short to say."""
    times, truth = np.sort(np.asarray(times, float)), np.sort(np.asarray(truth, float))
    if len(times) < 5 or len(truth) < 5:
        return None
    j = np.clip(np.searchsorted(truth, times), 1, len(truth) - 1)
    near = np.where(np.abs(times - truth[j - 1]) <= np.abs(times - truth[j]),
                    truth[j - 1], truth[j])
    err = times - near
    ok = np.abs(err) <= 0.100
    shift = float(np.median(err[ok])) if ok.sum() >= 10 else 0.0
    t = times - shift
    hits = i = k = 0
    while i < len(t) and k < len(truth):
        d = t[i] - truth[k]
        if abs(d) <= tol:
            hits, i, k = hits + 1, i + 1, k + 1
        elif d < 0:
            i += 1
        else:
            k += 1
    p, r = hits / len(t), hits / len(truth)
    return 2 * p * r / (p + r) if p + r else 0.0


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
    name, audio, sim, model_dir, cache_dir, decoder = job[:6]
    only = job[6] if len(job) > 6 else None            # difficulties to chart
    engine = _engine(model_dir)
    engine.model.meta["decoder"] = decoder
    an = _analysis(name, audio, cache_dir)
    human = human_charts(sim)
    ctx = pipeline.prepare_song(audio, an, log=lambda *a, **k: None, engine=engine)
    slots = tuple(d for d in human["ticks"] if only is None or d in only)
    charts, _, _ = pipeline.chart_song(ctx, slots, engine, track_id=name)
    rows = {}
    for slot, chart in charts.items():
        rows[slot] = {"n": len(chart.events), "n_human": len(human["ticks"][slot]),
                      "sixteenths": sixteenths(chart.events.tick),
                      "sixteenths_human": sixteenths(human["ticks"][slot])}
        if only is not None:
            # the 16th fit also needs to know what a setting costs
            rows[slot]["gates"] = bool(all(sm_qa.chart_metrics(
                infer.to_placements(chart), ctx.analysis, slot)["gates"].values()))
            if human["constant"] and len(chart.events):
                truth = human["offset_s"] + human["ticks"][slot] * (
                    60.0 / human["bpm"] / C.TICKS_PER_BEAT)
                rows[slot]["f1"] = placement_f1(ctx.g.tick_times_sec[chart.events.tick],
                                                truth)
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


def sixteenth_summary(results: list, slot: str = "hard") -> dict:
    rows = [s["charts"][slot] for s in results if slot in s["charts"]]
    f1 = [r["f1"] for r in rows if r.get("f1") is not None]
    return {"charts": len(rows),
            "median": statistics.median(r["sixteenths"] for r in rows) if rows else 0.0,
            "mean": statistics.mean(r["sixteenths"] for r in rows) if rows else 0.0,
            "human_median": statistics.median(r["sixteenths_human"] for r in rows)
            if rows else 0.0,
            "f1": statistics.mean(f1) if f1 else None,
            "gates": statistics.mean(1.0 if r.get("gates") else 0.0 for r in rows)
            if rows else None}


def choose_sixteenth_boost(off: dict, tried: list, target: float,
                           f1_loss: float) -> float | None:
    """`off` is `sixteenth_summary` without a boost, `tried` [(boost, summary)]
    in rising order. The smallest boost that brings the median to `target` of
    the human median while placement and the QA gates hold; None if none."""
    want = target * off["human_median"]
    for boost, s in tried:
        holds = ((off["f1"] is None or s["f1"] is None or s["f1"] >= off["f1"] - f1_loss)
                 and (off["gates"] is None or s["gates"] >= off["gates"] - 1e-9))
        if not holds:
            return None                 # a stronger boost only costs more
        if s["median"] >= want - 1e-9:
            return float(boost)
    return None


def fit_sixteenths(jobs: list, decoder: dict, workers: int, target: float, f1_loss: float,
                   log) -> dict:
    """Try the boosts in rising order on the Hard charts of the tuning songs."""
    block = {k: v for k, v in decoder.items() if k != "hard_sixteenth_boost"}
    off = sixteenth_summary(_map(_chart, [(*j, block, ("hard",)) for j in jobs], workers,
                                 "16th pass, no boost", log))
    log(f"[calibrate]   no boost: median {off['median']:g} 16ths per Hard chart (human "
        f"{off['human_median']:g}), F1 {off['f1']:.4f}, gates {off['gates']:.0%}")
    tried, chosen = [], None
    for boost in SIXTEENTH_BOOSTS:
        s = sixteenth_summary(_map(
            _chart, [(*j, dict(block, hard_sixteenth_boost=boost), ("hard",)) for j in jobs],
            workers, f"16th pass, boost {boost:g}", log))
        tried.append((boost, s))
        log(f"[calibrate]   boost {boost:g}: median {s['median']:g} (mean {s['mean']:.1f}), "
            f"F1 {s['f1']:.4f} ({s['f1'] - off['f1']:+.4f}), gates {s['gates']:.0%}")
        chosen = choose_sixteenth_boost(off, tried, target, f1_loss)
        last_holds = choose_sixteenth_boost(off, tried[-1:], 0.0, f1_loss) is not None
        if chosen is not None or not last_holds:
            break
    return {"off": off, "tried": [{"boost": b, **s} for b, s in tried], "chosen": chosen,
            "target": target, "f1_loss": f1_loss}


def low_tier_summary(results: list) -> dict:
    """QA gate pass rate and mean placement F1 of the Beginner and Easy charts."""
    rows = [s["charts"][d] for s in results for d in LOW_TIERS if d in s["charts"]]
    f1 = [r["f1"] for r in rows if "f1" in r]
    return {"charts": len(rows),
            "gates": statistics.mean(r["gates"] for r in rows) if rows else None,
            "f1": statistics.mean(f1) if f1 else None}


def choose_energy_tilt(off: dict, tried: list, target: float = GATE_PASS_TARGET,
                       f1_loss: float = TILT_F1_LOSS) -> tuple | None:
    """`off` is `low_tier_summary` with the engine's own ladder, `tried`
    [(ladder, summary)] mildest first. None when the engine's ladder already
    passes; else the mildest ladder that reaches `target` of the charts through
    the QA gates, or failing that the one that passes most — never one that
    costs more than `f1_loss` of placement F1 or passes fewer than before."""
    if off["gates"] is None or off["gates"] >= target:
        return None
    ok = [(lad, s) for lad, s in tried
          if s["gates"] > off["gates"]
          and (off["f1"] is None or s["f1"] is None or s["f1"] >= off["f1"] - f1_loss)]
    if not ok:
        return None
    for lad, s in ok:
        if s["gates"] >= target:
            return tuple(lad)
    return tuple(max(ok, key=lambda t: t[1]["gates"])[0])


def fit_energy_tilt(jobs: list, decoder: dict, workers: int, log) -> dict:
    """Does a Beginner / Easy chart of this model still follow the song's
    energy? If fewer than GATE_PASS_TARGET of them pass the QA gates on the
    tuning songs, try longer tilt ladders."""
    block = {k: v for k, v in decoder.items() if k != "energy_tilt"}
    off = low_tier_summary(_map(_chart, [(*j, block, LOW_TIERS) for j in jobs], workers,
                                "low-tier pass, the engine's tilt", log))
    log(f"[calibrate]   engine's tilt: {off['gates']:.1%} of Beginner/Easy charts pass the "
        f"QA gates, F1 {off['f1']:.4f}")
    tried = []
    if off["gates"] < GATE_PASS_TARGET:
        for ladder in LOW_TIER_TILTS:
            tilt = {d: list(ladder) for d in LOW_TIERS}
            s = low_tier_summary(_map(
                _chart, [(*j, dict(block, energy_tilt=tilt), LOW_TIERS) for j in jobs],
                workers, f"low-tier pass, tilt up to {ladder[-1]:g}", log))
            tried.append((ladder, s))
            log(f"[calibrate]   tilt up to {ladder[-1]:g}: {s['gates']:.1%} pass, F1 "
                f"{s['f1']:.4f} ({s['f1'] - off['f1']:+.4f})")
            if s["gates"] >= GATE_PASS_TARGET:
                break
    chosen = choose_energy_tilt(off, tried)
    return {"off": off, "tried": [{"ladder": list(lad), **s} for lad, s in tried],
            "chosen": list(chosen) if chosen else None}


def calibrate(model_dir, tuning_dir, cache_dir=None, workers: int = 6,
              tolerance: float = 0.05, rounds: int = 4, min_gain: int = 2,
              write: bool = False, log=print, fit_sixteenth: bool = False,
              sixteenth_target: float = SIXTEENTH_TARGET,
              sixteenth_f1_loss: float = SIXTEENTH_F1_LOSS,
              fit_tilt: bool = False) -> dict:
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
    burst = None
    if fit_sixteenth:
        burst = fit_sixteenths(base, decoder, workers, sixteenth_target, sixteenth_f1_loss,
                               log)
        decoder.pop("hard_sixteenth_boost", None)
        if burst["chosen"] is not None:
            decoder["hard_sixteenth_boost"] = burst["chosen"]
            log(f"[calibrate] 16th figures at Hard: boost {burst['chosen']:g}")
        else:
            log("[calibrate] 16th figures at Hard: no boost reaches "
                f"{sixteenth_target:.0%} of the human median within {sixteenth_f1_loss} of "
                "placement F1 and the QA gates — none stored")
    tilt = None
    if fit_tilt:
        tilt = fit_energy_tilt(base, decoder, workers, log)
        decoder.pop("energy_tilt", None)
        if tilt["chosen"]:
            decoder["energy_tilt"] = {d: tilt["chosen"] for d in LOW_TIERS}
            log(f"[calibrate] Beginner/Easy energy tilt: ladder {tilt['chosen']}")
        else:
            log("[calibrate] Beginner/Easy energy tilt: the engine's own ladder is kept")
    report = {"model": str(model_dir), "tuning_songs": len(songs), "phase": phase,
              "density": {"scale": scale, "history": history,
                          "median_ratio": history[-1]["median_ratio"]},
              "sixteenths": six, "sixteenth_boost": burst, "energy_tilt": tilt,
              "grids_half_a_beat_off": still_off, "decoder": decoder}
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
    ap.add_argument("--fit-sixteenths", action="store_true",
                    help="also fit decoder.hard_sixteenth_boost (16th figures at Hard)")
    ap.add_argument("--fit-energy-tilt", action="store_true",
                    help="also fit decoder.energy_tilt: a longer tilt ladder for Beginner "
                         "and Easy when their charts fail the density-follows-energy gate")
    ap.add_argument("--sixteenth-target", type=float, default=SIXTEENTH_TARGET,
                    help="share of the human median 16ths per Hard chart to reach")
    ap.add_argument("--sixteenth-f1-loss", type=float, default=SIXTEENTH_F1_LOSS,
                    help="placement F1 the boost may cost at most")
    a = ap.parse_args(argv)
    calibrate(a.model_dir, a.tuning_dir, a.cache_dir, a.jobs, a.tolerance, a.rounds,
              a.min_gain, a.write, fit_sixteenth=a.fit_sixteenths,
              sixteenth_target=a.sixteenth_target, sixteenth_f1_loss=a.sixteenth_f1_loss,
              fit_tilt=a.fit_energy_tilt)


if __name__ == "__main__":
    main()
