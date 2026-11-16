"""chunked.py — the chunked designer/critic loop for StepMania.

Whole-song mode asks the model for a chart in one call. Chunked mode asks it for
one SECTION at a time, with all four inputs scoped to that section — its audio,
its inventory rows, its density budget, and a short summary of what earlier
sections decided (docs/chunked-mode-brief.md).

What stays exactly the same, and it is most of the pipeline: the DSP runs once
over the whole track before any of this; the chunk outputs are assembled into a
single design object identical in shape to a whole-song one; and the realizer,
the validators, the density repair, the QA metrics and the gates then run on that
assembled design unchanged (REQ-CHUNK-05). Nothing downstream of `design()` knows
whether a chart was chunked.

The generic half — boundaries, scoping, prior-decision summaries, per-chunk retry
budgets, assembly — lives in `beatforge/chunking.py` and is shared with the Beat
Saber adapter. What is here is the part that is genuinely StepMania's: the prompt
call, and a critic that reads a 4-panel chart.
"""
from __future__ import annotations

import math

from ... import chunking, ledger
from ...chunking import Chunk, ChunkedRun, ChunkError
from .resolve import IntentError


# --------------------------------------------------------------------------- #
# Designing
# --------------------------------------------------------------------------- #
def design_chunked(track_id: str, difficulty: str, analysis: dict, audio: str,
                   client, seed_feedback: str | None = None,
                   chunks: list[Chunk] | None = None) -> ChunkedRun:
    """Design a whole song section by section. Returns the per-chunk run."""
    chunks = chunking.plan_chunks(analysis) if chunks is None else chunks
    if not chunks:
        raise IntentError(
            "chunked mode planned zero chunks for this track — the analysis has no "
            "bars (empty energy curve and no duration), so there is nothing to "
            "design. Set BEATFORGE_CHUNK_MODE=never to chart it whole-song.")

    print(f"    [{difficulty}] chunked: {len(chunks)} section(s) — "
          + ", ".join(f"bars {c.start_bar}-{c.end_bar}" for c in chunks), flush=True)

    def progress(chunk, scoped):
        print(f"    [{difficulty}] chunk {chunk.index + 1}/{len(chunks)} "
              f"(bars {chunk.start_bar}-{chunk.end_bar}) — {len(scoped['notes'])} notes",
              flush=True)

    try:
        return chunking.run_chunked_design(
            analysis, audio, chunks,
            _design_call(track_id, difficulty, client),
            filler=chunking.STEPMANIA_FILLER, seed_feedback=seed_feedback,
            on_progress=progress, validate=_validator(analysis, difficulty),
            fallback=_fallback(difficulty))
    except ChunkError as e:
        # Surfaced as the adapter's own error type so `_design_once`'s existing
        # retry path handles it exactly as it handles a bad whole-song reply.
        raise IntentError(str(e)) from e


def _validator(analysis: dict, difficulty: str):
    """Run StepMania's OWN resolver over a chunk's scoped output.

    This is the difference between "the chunk that broke the rule is re-prompted
    with the exact number it missed" and "the whole song is re-designed after
    assembly". `resolve_intent` is the authority on what a designer may emit —
    the closed texture/movement/crossover vocabularies, the ban on raw times and
    panels, and the per-phrase density band from the plan — so asking it is both
    the earliest and the most faithful check available.

    Measured on a self-hosted Gemma: the first chunk of a run declared 6.0
    notes/bar for bars 0-4 against an allowed 2.52-5.89, which surfaced as an
    `IntentError` from `build_song` and cost every chunk already designed.
    """
    from .resolve import IntentError, resolve_intent

    def check(chunk, scoped):
        try:
            resolve_intent({"notes": scoped["notes"], "phrases": scoped["phrases"]},
                           analysis, difficulty)
        except IntentError as e:
            raise ChunkError(str(e)) from e

    return check


def _fallback(difficulty: str):
    """DSP-only intent for a chunk the model could not design.

    `deterministic_intent` is the existing no-LLM path — it picks on-grid onsets
    that already pass the tier's subdivision and snap gates, and declares one
    phrase per section from the density plan. Handed the chunk's SCOPED analysis,
    it produces intent for those bars and nothing else, in exactly the shape a
    model reply would have taken, so the assembled design does not know the
    difference.

    Worse than a designed section, and it is reported as such. Much better than
    the alternative, which was losing every other section too.
    """
    from .resolve import deterministic_intent

    def build(chunk, scoped_analysis):
        return deterministic_intent(scoped_analysis, difficulty)

    return build


def _design_call(track_id: str, difficulty: str, client):
    """The target-specific half of a chunk call: the prompt, and the ledger stage.

    A chunk retry is billed under `reprompt` like any other retry — and in hybrid
    mode the stage name is also the routing key, so a retry escalates to the heavy
    tier exactly as a whole-song retry does. The ledger line carries the chunk
    index and its bars, which is what makes "the ledger shows the retry scoped to
    that chunk" (REQ-CHUNK-07) something you can read off a cost report rather
    than infer.
    """
    from .design import design_chunk_intent

    def call(chunk, scoped_analysis, chunk_audio, prior_summary, feedback, attempt):
        stage = "designer" if (attempt == 1 and not feedback) else "reprompt"
        with ledger.stage(stage, song=track_id, difficulty=difficulty,
                          attempt=attempt, chunk=chunk.index,
                          chunk_bars=f"{chunk.start_bar}-{chunk.end_bar}"):
            return design_chunk_intent(difficulty, scoped_analysis, chunk,
                                       chunk_audio, client, prior_summary,
                                       feedback=feedback)
    return call


# --------------------------------------------------------------------------- #
# Scoped re-prompt (REQ-CHUNK-07)
# --------------------------------------------------------------------------- #
def chunk_gate_failures(placements: list, analysis: dict, difficulty: str,
                        chunks: list[Chunk], failed_gates: list[str]) -> dict:
    """Which chunks are responsible for the gates the whole chart failed."""
    from . import qa as sm_qa
    from .grammar import BUDGETS

    meter = int(analysis.get("meter", 4) or 4)

    def metrics_fn(slice_, chunk):
        # The chunk's own bars, so a gate that consults the analysis reads the
        # same span the slice was taken from (see qa._usable_sustains).
        window = (chunk.start_bar * meter, chunk.end_bar * meter)
        return sm_qa.chart_metrics(slice_, analysis, difficulty,
                                   beat_window=window)["gates"]

    return chunking.gate_failures_by_chunk(
        placements, analysis, chunks, failed_gates, metrics_fn,
        min_notes_by_gate={"hold_share": _ceiling_gate_min_notes(
            BUDGETS[difficulty].hold_share[1])},
        emergent_gates=_EMERGENT_GATES)


#: Gates where "the chart failed it but no chunk did" means the ARRANGEMENT is at
#: fault, so every chunk hears about it. The rest of the battery is deliberately
#: absent, and the absences are the point:
#:
#:   * `density_energy` — a Spearman over the whole song. Every section can be
#:     individually plausible while the contour between them is flat.
#:   * `flow_ceiling` — a max over transitions, and the transition from the last
#:     note of one chunk to the first of the next belongs to no chunk's slice. A
#:     seam is real and genuinely nobody's locally.
#:
#:   * `hold_share`, `onset_alignment` and `panel_balance` are plain averages
#:     over the chart, and an average cannot exceed a bound that none of its parts
#:     exceed. If the whole is out of budget some part is too, so an empty blame
#:     list means the small-chunk exclusions ate it, not that the arrangement did
#:     it — and blaming everyone would re-prompt sections that are individually
#:     fine. (`hold_share` is here for a second reason too: it briefly had a floor
#:     on charts but not on chunks, which made a floor failure unreproducible
#:     locally BY CONSTRUCTION and blamed all three chunks of every under-holding
#:     chart, every run. The floor is gone — see grammar.BUDGETS — but the general
#:     trap is not: never let a gate the chunk battery cannot reproduce fall
#:     through to "emergent".)
_EMERGENT_GATES = frozenset({"density_energy", "flow_ceiling"})


def _ceiling_gate_min_notes(hi: float) -> int:
    """Smallest slice a share-of-notes CEILING can fairly be read off.

    One hold has to be affordable before the ceiling is allowed to trip, which is
    `1/hi` notes: below it a single held note is itself over budget, and the gate
    stops meaning "too many holds here" and starts meaning "no holds allowed in
    this section at all". Easy's 30% ceiling clears at 4 notes, challenge's 20%
    at 5 — so with the measured ceilings this barely binds above the global
    `min_notes`, which is the correct amount for it to bind.

    Much smaller than the floor-era threshold this replaces (20 and 29), and it
    should be: a ceiling is always satisfiable — zero holds passes it — so the
    only thing to guard against is ratio noise on a tiny slice, not an
    unreachable target.
    """
    return int(math.ceil(1.0 / hi)) if hi > 0 else 0


def reprompt_failing_chunks(run: ChunkedRun, track_id: str, difficulty: str,
                            analysis: dict, audio: str, client,
                            failing: dict, gate_briefs: dict) -> int:
    """Re-design only the chunks named in `failing`, in place. Returns how many."""
    def progress(chunk, gates):
        print(f"    [{difficulty}] chunk {chunk.index + 1} (bars {chunk.start_bar}-"
              f"{chunk.end_bar}) failed {', '.join(gates)} — scoped re-prompt…",
              flush=True)

    def failure(chunk, err):
        print(f"    [{difficulty}] chunk {chunk.index + 1} re-prompt failed ({err}); "
              f"keeping the original", flush=True)

    return chunking.redesign_chunks(
        run, analysis, audio, _design_call(track_id, difficulty, client),
        failing, gate_briefs, on_progress=progress, on_failure=failure,
        validate=_validator(analysis, difficulty))


# --------------------------------------------------------------------------- #
# Chunked critic (REQ-CHUNK-06)
# --------------------------------------------------------------------------- #
def critic_chunked(track_id: str, difficulty: str, analysis: dict, placements: list,
                   audio: str, client, run: ChunkedRun,
                   metrics: dict | None = None) -> dict:
    """Review per chunk, then once over the assembled structure without audio.

    Scoped the same way the designer was: that chunk's audio, that chunk's
    realized notes. The final pass has no audio on purpose — it is judging
    arrangement (does the chart build, does the drop land where the drop is), and
    that question is answered from the structure summary, not by re-listening to
    the whole track, which is the payload chunked mode exists to avoid.

    The return shape is the SAME dict a whole-song critic returns — `score`,
    `verdict`, `issues`, `measured_rho`, `dynamics` — so the QA report, the
    revision loop and the ship threshold need no special-casing (REQ-CHUNK-06).
    Per-chunk detail rides alongside under `chunks`.
    """
    from .design import _dynamics_block

    meter = int(analysis.get("meter", 4) or 4)
    per_chunk = []
    for c in run.chunks:
        slice_ = [p for p in placements if c.owns_beat(p.beat, meter)]
        if not slice_:
            continue
        chunk_audio = run.audio_for(audio, analysis, c)
        with ledger.stage("critic", song=track_id, difficulty=difficulty,
                          chunk=c.index, chunk_bars=f"{c.start_bar}-{c.end_bar}"):
            review = _critic_one(difficulty, analysis, c, slice_, chunk_audio, client)
        review["chunk"] = c.index
        review["bars"] = [c.start_bar, c.end_bar]
        review["notes"] = len(slice_)
        per_chunk.append(review)

    with ledger.stage("critic", song=track_id, difficulty=difficulty,
                      chunk="structure"):
        structure = _critic_structure(difficulty, analysis, run, placements, client,
                                      _dynamics_block(analysis, metrics, difficulty))
    review = chunking.aggregate_critic_scores(per_chunk, structure)
    # The measurement is ours, not the model's — same rule as the whole-song
    # critic: a QA report must never carry a hallucinated correlation.
    if metrics is not None:
        echoed = review.get("measured_rho")
        truth = metrics.get("density_energy_spearman")
        review["measured_rho"] = truth
        if echoed is not None and truth is not None and abs(float(echoed) - truth) > 0.01:
            review["measured_rho_echoed_by_critic"] = echoed
    return review


def _critic_one(difficulty: str, analysis: dict, chunk: Chunk, placements: list,
                chunk_audio: str, client) -> dict:
    """One chunk's critic call.

    Times are rendered RELATIVE TO THE CLIP the model is hearing. Absolute song
    times would be a trap: the audio starts at the chunk's lead-in, so a chart
    line reading `104.2s` against a 26-second clip invites the model to conclude
    the notes are nowhere near the music.
    """
    from .grammar import PANELS

    bpm, offset = analysis["bpm"], analysis["offset"]
    t0, _ = chunking.chunk_window(analysis, chunk)

    def t(beat):
        return round(offset + beat / bpm * 60 - t0, 3)

    rows = []
    for p in sorted(placements, key=lambda p: p.beat)[:600]:
        panels = "".join(PANELS[c] for c in p.panels)
        if p.hold_beats:
            rows.append(f"{t(p.beat):.3f}: HOLD {panels} until "
                        f"{t(p.beat + p.hold_beats):.3f}")
        elif len(p.panels) > 1:
            rows.append(f"{t(p.beat):.3f}: JUMP {panels}")
        else:
            rows.append(f"{t(p.beat):.3f}: tap {panels}")
    role = "/".join(chunk.roles) if chunk.roles else "section"
    prompt = (
        f"You are a StepMania (ITG) chart critic. You are HEARING one section of a "
        f"song — bars {chunk.start_bar}-{chunk.end_bar} ({role}), plus a bar of "
        f"lead-in/lead-out — and reading the realized {difficulty} steps for it. "
        f"Times are SECONDS FROM THE START OF THE CLIP you are hearing, not from "
        f"the start of the song. Each line is '<time_s>: EVENT panels'. A 'HOLD X "
        f"until T' occupies panel X until T — a later 'tap X' after that time is "
        f"legal, not a conflict. The validator already guarantees no note lands on "
        f"a held panel and jumps are <=2 panels. You did NOT design this — judge it "
        f"fresh: do the steps land on the music, is the foot flow comfortable, does "
        f"the density suit this section's energy?\n"
        f"Steps:\n" + "\n".join(rows) + "\n\n"
        f"Reply ONLY JSON: {{\"score\": <0-10>, \"verdict\": \"<one line>\", "
        f"\"issues\": [{{\"where\":\"..\",\"problem\":\"..\"}}]}}")
    return client.generate_json(prompt, audio_path=chunk_audio)


def _critic_structure(difficulty: str, analysis: dict, run: ChunkedRun,
                      placements: list, client, dynamics: str) -> dict:
    """The one whole-chart pass: arrangement only, no audio.

    This is the check no per-chunk review can make, because it is precisely the
    thing a section-at-a-time designer is at risk of getting wrong — whether the
    sections add up to a song. It is handed the measured per-section note density
    and the measured energy so its judgement cites numbers rather than an
    impression of a track it has not heard in one piece.
    """
    meter = int(analysis.get("meter", 4) or 4)
    energy = analysis.get("energy_curve") or []
    rows = []
    for c, d in zip(run.chunks, run.decisions):
        n = sum(1 for p in placements if c.owns_beat(p.beat, meter))
        seg = energy[c.start_bar:c.end_bar]
        e = sum(seg) / len(seg) if seg else 0.0
        rows.append(
            f"  bars {c.start_bar:>3}-{c.end_bar:<3} ({'/'.join(c.roles) or '?':<6}) "
            f"energy {e:.2f}  {n / max(1, c.bars):.1f} notes/bar  "
            f"texture {d.texture or '-'}  movement {d.movement or '-'}")
    prompt = (
        f"You are a StepMania (ITG) chart critic reviewing the ARRANGEMENT of a "
        f"{difficulty} chart that was designed one section at a time. You are not "
        f"hearing the audio for this pass and you are not judging individual steps "
        f"— judge whether the sections add up to one song: does density track the "
        f"measured energy across the whole track, does the chart build and release "
        f"in the right places, do repeated sections feel related, are the seams "
        f"between sections abrupt?\n"
        f"Sections (measured):\n" + "\n".join(rows) + "\n"
        + dynamics + "\n\n"
        f"Reply ONLY JSON: {{\"score\": <0-10>, \"verdict\": \"<one line>\", "
        f"\"measured_rho\": <the rho value above, or null>, "
        f"\"dynamics\": \"<one line on whether density tracks energy>\", "
        f"\"issues\": [{{\"where\":\"bars a-b\",\"problem\":\"..\"}}]}}")
    return client.generate_json(prompt)
