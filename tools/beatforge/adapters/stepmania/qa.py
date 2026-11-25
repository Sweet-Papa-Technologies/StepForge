"""qa.py — objective QA metrics + critic + previews (STEPFORGE §9,
REQ-SM-12/13/14)."""
from __future__ import annotations

from . import footflow as ff
from .grammar import BUDGETS, PANELS
from .quantize import Placement
from .timing import beats_to_times, has_tempo_map, timing_of


def chart_metrics(placements: list[Placement], analysis: dict, difficulty: str,
                  *, beat_window: tuple | None = None) -> dict:
    """Objective metrics + gate verdicts for `placements`.

    `beat_window` is `(lo, hi)` in beats, hi exclusive, and names the span of the
    song these placements were taken from. It exists for the CHUNK path, where
    `placements` is one chunk's slice: the ratios are computed over whatever slice
    is handed in, so any gate whose verdict also consults the ANALYSIS has to
    consult the same span or the two halves disagree. See `_usable_sustains`.
    Whole-chart callers omit it.
    """
    b = BUDGETS[difficulty]
    bpm, offset = analysis["bpm"], analysis["offset"]
    # A mine is something to avoid, not a step: it is not on the music, moves
    # no foot and presses no panel. Counted as a step (as it used to be) it
    # was walked through the foot model like an arrow, and a chart with a
    # dozen mines failed the flow-ceiling gate for steps nobody takes.
    placements = [p for p in placements if p.kind != "mine"]
    n = len(placements)
    onset_times = sorted(o["time"] for o in analysis.get("onsets", []))

    # a song with a tempo map (BPM changes, stops) is timed through it; the
    # usual single-tempo song by the expression this has always used
    mapped = has_tempo_map(analysis)
    times = beats_to_times(analysis, [p.beat for p in placements]) if mapped else None
    aligned = grid = 0
    for i, p in enumerate(placements):
        t = float(times[i]) if mapped else offset + p.beat / bpm * 60
        if _near(t, onset_times, 0.035):
            aligned += 1
        else:
            grid += 1
    onset_alignment = aligned / n if n else 1.0

    # panel balance (each panel 18-32%)
    counts = [0, 0, 0, 0]
    for p in placements:
        for c in p.panels:
            counts[c] += 1
    total_cells = sum(counts) or 1
    balance = {PANELS[i]: round(counts[i] / total_cells, 3) for i in range(4)}

    holds = sum(1 for p in placements if p.hold_beats)
    jumps = sum(1 for p in placements if len(p.panels) > 1)

    # foot-flow cost distribution (re-score the realized chart)
    max_cost, mean_cost = _flow_costs(placements, b, analysis.get("bpm"),
                                      timing_of(analysis) if mapped else None)

    # density vs energy (per-bar) Spearman
    spearman = _density_energy(placements, analysis)

    return {
        "notes": n,
        "onset_alignment": round(onset_alignment, 4),
        "grid_notes": grid,
        "panel_balance": balance,
        "hold_share": round(holds / n, 4) if n else 0.0,
        # Both feet pinned at once, as a fraction of charted time. Reported
        # and NOT gated, for the reason jump_share is not: validate.py::
        # cap_two_foot_time bounds it before any metric is taken, so a gate
        # could never fire. It is here because it is the number that predicts
        # whether a chart is miserable, and hold_share demonstrably is not.
        "two_foot_frac": round(_two_foot(placements), 4),
        "jump_share": round(jumps / n, 4) if n else 0.0,
        "flow_cost_max": round(max_cost, 2),
        "flow_cost_mean": round(mean_cost, 2),
        "density_energy_spearman": spearman,
        "gates": _gates(onset_alignment, balance, holds / n if n else 0, b, max_cost,
                        spearman, analysis, beat_window),
    }


def _gates(align, balance, hold_share, b, max_cost, spearman, analysis,
           beat_window=None):
    flat = analysis.get("energy_cv", 0.0) < config_cv()
    # hold-share only applies if the music actually offers sustains usable at this
    # difficulty (long enough, on-grid) — else there's nothing to hold.
    usable = _usable_sustains(analysis, b, beat_window)
    # section-structure: the density-vs-energy gate needs >=2 sections with real
    # energy spread; a uniform loop can't have density "follow" a flat curve.
    sections = analysis.get("sections", [])
    spread = (max((s.get("energy_pct", 0) for s in sections), default=0)
              - min((s.get("energy_pct", 0) for s in sections), default=0))
    structured = len(sections) >= 2 and spread >= 0.15
    return {
        "onset_alignment": align >= 0.90 - 1e-6,
        "panel_balance": all(0.15 - 1e-6 <= v <= 0.35 + 1e-6 for v in balance.values()),
        "hold_share": _hold_share_ok(hold_share, b, usable),
        "flow_ceiling": max_cost < ff.DOUBLE_STEP + ff.JACK + 1e-6,     # no forbidden-tier transition
        "density_energy": (flat or not structured or (spearman is None)
                           or spearman >= _spearman_min() - 1e-6),
    }
    # NB: there is deliberately no `jump_share` gate here. It looks like an
    # obvious omission — jump_share is measured, and docs/model-comparison.md
    # records Flash Lite at 0.267 against 3.6 Flash's 0.046 — but a runtime gate
    # on it could never fire: validate.py::cap_jump_share already DEMOTES surplus
    # jumps to single taps against realize.py::_JUMP_MAX_FRAC (0.03/0.07/0.12)
    # before any metric is computed, so a realized chart is under the cap by
    # construction whatever the model proposed. (Which also dates that 0.267: a
    # `hard` chart cannot measure above 0.12 through the current validate path.)
    # The invariant is worth holding, so it is held where it belongs — as a test,
    # tests/test_cascade.py::test_validate_caps_jump_share_whatever_the_model_did.


def _two_foot(placements) -> float:
    from .validate import two_foot_fraction
    return two_foot_fraction(placements)


def _hold_share_ok(share, b, usable) -> bool:
    """The hold-share verdict: a CEILING, at every scale.

    The band is `(0.0, hi)` at every tier now, so this is a one-sided test and the
    chart/chunk distinction that used to live here is gone with the floor that
    needed it. `grammar.BUDGETS` carries the measurements; the short version is
    that the old floor rejected most known-good charts, including a majority of
    the ones produced by a model trained on human work, so it was measuring its
    own guess rather than the chart.

    A hold pins a foot for its duration — that is the whole intent, and it is a
    ceiling-shaped intent. There is no matching argument for a minimum, at chart
    scale or section scale: a chart of pure taps is ordinary DDR. "This chart
    could use more holds" survives as a critic note, where it can be weighed
    against the song.

    `usable == 0` waives the gate: nothing holdable in the span, nothing to judge.
    Kept because it also short-circuits the ceiling, which is not obviously right
    but is long-standing behaviour and out of scope to change here.
    """
    _, hi_h = b.hold_share
    return usable == 0 or share <= hi_h + 1e-6


def _usable_sustains(analysis, b, beat_window=None) -> int:
    """Sustains this tier could actually hold, inside `beat_window` if given.

    The window is not optional bookkeeping — it is what keeps the hold-share
    waiver honest. `hold_share` is measured over the placements handed to
    `chart_metrics`, which on the chunk path is ONE chunk's slice; without a
    window this count ran over the whole song, so an 8-bar intro with no sustains
    in it was never waived as long as sustains existed anywhere in the track. It
    then failed a floor it had nothing to meet, and the chunk-blame pass charged
    a scoped re-prompt for a chunk with nothing to fix.
    """
    lo = b.hold_len_beats[0]
    w0, w1 = beat_window if beat_window else (float("-inf"), float("inf"))
    return sum(1 for o in analysis.get("onsets", [])
               if w0 <= o["nearest_beat"] < w1
               and o.get("sustain") and o.get("sustain_beats", 0) >= lo - 1.0
               and abs(o["nearest_beat"] - round(o["nearest_beat"] / b.finest_subdiv) * b.finest_subdiv) <= 1e-3
               and abs(o.get("snap_error_ms", 0)) <= 35.0)



def _spearman_min():
    from ... import config
    return getattr(config, "DENSITY_ENERGY_SPEARMAN_MIN", 0.30)


def config_cv():
    from ... import config
    return getattr(config, "DENSITY_GATE_MIN_CV", 0.12)


def _near(t, onset_times, tol):
    for ot in onset_times:
        if abs(ot - t) <= tol + 1e-9:
            return True
        if ot - t > tol:
            break
    return False


def _flow_costs(placements, b, bpm: float | None = None, timing=None):
    """(worst, mean) footflow cost of a chart as its own foot labels play it.
    With `bpm`, each step is told how long ago the previous one was, so a slow
    repeat of the same arrow is not scored as a jack (footflow.JACK_FREE_S);
    `timing` (a TimingMap) does that for a song with more than one tempo."""
    if len(placements) < 2:
        return 0.0, 0.0
    state = ff.REST_STATE
    costs = []
    prev_beat = None
    for p in placements:
        if bpm is None or prev_beat is None:
            dt_s = None
        elif timing is not None:
            dt_s = float(timing.time_at(p.beat) - timing.time_at(prev_beat))
        else:
            dt_s = (p.beat - prev_beat) * 60.0 / bpm
        if len(p.panels) > 1:
            c, state = ff.jump_cost(state, p.panels[0], p.panels[1], b)
        else:
            foot = p.meta.get("foot", ff.LEFT)
            c, state = ff.step(state, foot, p.panels[0], b,
                              p.meta.get("movement", "static"), 0.0, dt_s=dt_s)
        costs.append(max(0.0, c))
        prev_beat = p.beat
    return max(costs), sum(costs) / len(costs)


def render_previews(placements, analysis, audio_path, out_stem):
    """REQ-SM-14 human-auditable previews: a notefield PNG (matplotlib, if
    available) + an assist-tick render (ffmpeg) — the fastest ear-check that the
    steps land on the music."""
    import shutil
    import subprocess
    from pathlib import Path
    out = {}
    bpm, offset = analysis["bpm"], analysis["offset"]
    times = [offset + p.beat / bpm * 60 for p in placements]

    png = Path(str(out_stem) + ".notefield.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 3))
        for p, t in zip(placements, times):
            for c in p.panels:
                if p.hold_beats:
                    ax.plot([t, t + p.hold_beats / bpm * 60], [c, c], color="#ff2d95", lw=4)
                else:
                    ax.scatter([t], [c], color="#2de2e6", s=16)
        for s in analysis.get("sections", []):
            ax.axvline(offset + s["start_bar"] * 4 / bpm * 60, color="#b14cff", ls="--", lw=0.6)
        ax.set_yticks(range(4)); ax.set_yticklabels(list(PANELS)); ax.set_xlabel("s")
        fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
        out["png"] = str(png)
    except Exception:
        pass

    click = Path(str(out_stem) + ".assist.ogg")
    if shutil.which("ffmpeg") and times:
        ticks = "+".join(f"(lt(abs(t-{t:.3f}),0.015))*sin(2*PI*2000*t)" for t in times[:500])
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", audio_path,
                 "-f", "lavfi", "-t", str(analysis["duration_s"]), "-i", f"aevalsrc='{ticks}':s=44100",
                 "-filter_complex", "[0:a]volume=0.8[a];[1:a]volume=0.5[b];[a][b]amix=inputs=2:duration=first[o]",
                 "-map", "[o]", "-c:a", "libvorbis", "-q:a", "5", str(click)],
                check=True, capture_output=True, timeout=120)
            out["assist"] = str(click)
        except Exception:
            pass
    return out


def _density_energy(placements, analysis):
    energy = analysis.get("energy_curve", [])
    if not energy or len(energy) < 3:
        return None
    meter = analysis.get("meter", 4)
    counts = [0] * len(energy)
    for p in placements:
        bar = int(p.beat // meter)
        if 0 <= bar < len(counts):
            counts[bar] += 1
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(counts, energy)
        return round(float(rho), 4) if rho == rho else None
    except Exception:
        return None
