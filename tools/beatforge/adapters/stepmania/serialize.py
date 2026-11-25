"""serialize.py — write .ssc/.sm via garcia/simfile and package the song folder
(STEPFORGE §7, REQ-SM-07/08). Never hand-formats the file; simfile owns the
grammar, we own the timing/metadata + note objects."""
from __future__ import annotations

import shutil
from pathlib import Path

import simfile
from simfile.sm import SMChart, SMSimfile
from simfile.ssc import SSCChart, SSCSimfile

from .grammar import BUDGETS
from .quantize import to_simfile_notes
from .timing import has_tempo_map, timing_of



#: Song and chart credit (the "step artist" ITGmania shows). It says what made
#: the chart, so a chart shared without its context still discloses that.
CHART_CREDIT = "StepForge (AI-generated)"

def _sample_start(analysis: dict) -> float:
    bpm, offset = analysis["bpm"], analysis["offset"]
    best = max(analysis.get("sections", []),
               key=lambda s: s.get("energy_pct", 0), default=None)
    if not best:
        return 0.0
    if has_tempo_map(analysis):
        return round(float(timing_of(analysis).time_at(best["start_bar"] * 4.0)), 3)
    return round(offset + best["start_bar"] * 4 / bpm * 60, 3)


def timing_tags(analysis: dict) -> tuple:
    """(#BPMS value, #STOPS value or None) for an analysis.

    One tempo — nearly every song — is the single entry this has always
    written. A song with a tempo map (`analysis["tempo_map"]`, see timing.py)
    gets every BPM segment and every stop, three decimals each, which is the
    precision the map was built with."""
    tm = analysis.get("tempo_map")
    if not tm:
        return f"0.000={analysis['bpm']:.3f}", None
    bpms = ",".join(f"{b:.3f}={v:.3f}" for b, v in tm["bpms"])
    stops = ",".join(f"{b:.3f}={s:.3f}" for b, s in tm.get("stops", []))
    return bpms, (stops or None)


def build_simfile(track_meta: dict, analysis: dict, charts_by_diff: dict,
                  ssc: bool = True, banner: str | None = None, background: str | None = None,
                  subdiv: dict | None = None):
    """charts_by_diff: {difficulty_key: (meter, [Placement, ...])}.

    `subdiv` optionally overrides the snapping grid per difficulty (in beats).
    The default is the tier's `finest_subdiv`, which cannot express a triplet;
    the neural engine passes 1/12 (one 48th-grid tick) for the tiers it lets
    use them. Omitted, behaviour is unchanged."""
    sf = SSCSimfile.blank() if ssc else SMSimfile.blank()
    sf.title = track_meta["title"]
    sf.artist = track_meta["artist"]
    sf.music = track_meta["music"]
    sf.offset = f"{-round(analysis['offset'], 3):.3f}"      # #OFFSET = -(beat0 time)
    sf.bpms, stops = timing_tags(analysis)
    if stops:
        sf.stops = stops
    sf.samplestart = f"{_sample_start(analysis):.3f}"
    sf.samplelength = "12.000"
    sf.selectable = "YES"
    sf.credit = CHART_CREDIT
    if banner:
        sf.banner = banner
    if background:
        sf.background = background

    for key, (meter, placements) in charts_by_diff.items():
        budget = BUDGETS[key]
        note_str = str(to_simfile_notes(placements,
                                        (subdiv or {}).get(key, budget.finest_subdiv)))
        chart = (SSCChart if ssc else SMChart).blank()
        chart.stepstype = "dance-single"
        chart.difficulty = budget.sm_difficulty
        chart.meter = str(meter)
        chart.description = CHART_CREDIT          # the step artist in .sm files
        if ssc:
            chart.credit = CHART_CREDIT
        chart.notes = note_str
        sf.charts.append(chart)
    return sf


def write_song_folder(track_meta: dict, analysis: dict, charts_by_diff: dict,
                      audio_src: str, out_dir: Path, formats=("ssc",),
                      subdiv: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(track_meta["music"]).stem
    # pick up song art if present in the folder (assetforge-generated)
    banner = next((p.name for p in (out_dir / f"{base}-banner.png",
                                    out_dir / "banner.png") if p.exists()), None)
    background = next((p.name for p in (out_dir / f"{base}-bg.png",
                                       out_dir / "background.png") if p.exists()), None)
    written = {}
    for fmt in formats:
        sf = build_simfile(track_meta, analysis, charts_by_diff, ssc=(fmt == "ssc"),
                          banner=banner, background=background, subdiv=subdiv)
        path = out_dir / f"{base}.{fmt}"
        path.write_text(str(sf))
        # round-trip referee: simfile must re-open its own output (REQ-SM-11)
        with open(path) as fh:
            reopened = simfile.load(fh)
        assert len(reopened.charts) == len(charts_by_diff), "chart count changed on reload"
        written[fmt] = str(path)
    audio_dst = out_dir / track_meta["music"]
    if Path(audio_src).resolve() != audio_dst.resolve():
        shutil.copyfile(audio_src, audio_dst)
    written["audio"] = str(audio_dst)
    return written
