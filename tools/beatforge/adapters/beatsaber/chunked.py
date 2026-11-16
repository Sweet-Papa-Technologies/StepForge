"""chunked.py — the chunked designer/critic loop for Beat Saber.

The Beat Saber half of CHUNKED MODE (docs/chunked-mode-brief.md): the mapper is
asked for one SECTION at a time, with the audio, the onset inventory and the
prior-decision summary all scoped to that section, and the results assembled into
a single design object identical in shape to a whole-song one — so the parity
realizer, the validators, the swing simulator and the QA metrics run on it
unchanged (REQ-CHUNK-05).

The generic half lives in `beatforge/chunking.py` and is shared with the
StepMania adapter. What is here is the part that is genuinely Beat Saber's: the
prompt call, and a critic that reads a per-hand swing sequence.

One Beat Saber-specific wrinkle, and it is the reason `_scope` exists: an `arc`
carries a `tail_ref` as well as a `ref`. The head decides which chunk owns the
arc, but the tail can legitimately land in the next chunk's bars, and dropping
such an arc would silently delete notes the model deliberately placed. So the
generic scoping (which only reads `ref`) is kept, and the tail is checked
separately against the inventory the chunk was actually shown.
"""
from __future__ import annotations

from ... import chunking, ledger
from ...chunking import Chunk, ChunkedRun, ChunkError
from .resolve import IntentError

#: What the mapper is told when a gate fails in its section. Addressed to a
#: designer that declares INTENT, not geometry — "take fewer onsets here" is
#: actionable, "nps_in_budget: False" is not.
GATE_BRIEFS = {
    "onset_alignment":
        "Too many notes do not sit on a real onset. Choose notes from the onset "
        "inventory by id; stop filling grid lines that no transient supports.",
    "hand_balance":
        "The two hands are used too unevenly here. Set `hand_balance` to even and "
        "alternate which hand takes the accents.",
    "no_unforced_resets":
        "This section forces parity resets the player is not warned about. Ease "
        "the `tech` setting and take fewer awkwardly-spaced onsets; use a "
        "`bomb_reset` if you genuinely want to break parity.",
    "nps_in_budget":
        "This section is denser than the tier allows. Take fewer onsets — the "
        "strongest ones — and let the quiet bars breathe.",
    "arc_chain_share":
        "The share of arcs and chains is outside this tier's budget. Adjust how "
        "many sustained onsets you mark as `arc`/`chain`.",
    "precision":
        "Notes are landing off the tier's finest legal subdivision. Only take "
        "onsets whose beat sits on that grid.",
    "simulator_clean":
        "The swing simulator cannot play this section cleanly. Simplify the "
        "texture and the hand alternation through the fast passages.",
    "density_energy":
        "Note density does not follow the song's energy. Raise density into the "
        "loud bars and let the quiet ones breathe.",
}


# --------------------------------------------------------------------------- #
# Designing
# --------------------------------------------------------------------------- #
def design_chunked(track_id: str, difficulty: str, analysis: dict, audio: str,
                   client, seed_feedback: str | None = None,
                   chunks: list[Chunk] | None = None) -> ChunkedRun:
    """Map a whole song section by section. Returns the per-chunk run."""
    chunks = chunking.plan_chunks(analysis) if chunks is None else chunks
    if not chunks:
        raise IntentError(
            "chunked mode planned zero chunks for this track — the analysis has no "
            "bars (empty energy curve and no duration), so there is nothing to "
            "map. Set BEATFORGE_CHUNK_MODE=never to map it whole-song.")

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
            filler=chunking.BEATSABER_FILLER, seed_feedback=seed_feedback,
            on_progress=progress, validate=_validator(analysis, difficulty),
            fallback=_fallback(difficulty))
    except ChunkError as e:
        raise IntentError(str(e)) from e


def _validator(analysis: dict, difficulty: str):
    """Run Beat Saber's OWN resolver over a chunk's scoped output.

    `resolve_intent` is the authority on what a mapper may emit — the closed
    density/movement/tech/hand_balance vocabularies, the ban on coordinates and
    cut directions, and the `tail_ref` an arc needs. Asking it here means a chunk
    that breaks one of those rules is re-prompted on its own, instead of the
    error surfacing after assembly as a whole-song `IntentError` that costs every
    chunk already designed (REQ-CHUNK-07).
    """
    from .resolve import IntentError as BsIntentError
    from .resolve import resolve_intent

    def check(chunk, scoped):
        try:
            resolve_intent({"notes": scoped["notes"], "phrases": scoped["phrases"]},
                           analysis, difficulty)
        except BsIntentError as e:
            raise ChunkError(str(e)) from e

    return check


def _fallback(difficulty: str):
    """DSP-only intent for a chunk the mapper could not design.

    `deterministic_intent` is the existing no-LLM path. Handed the chunk's SCOPED
    analysis it produces intent for those bars alone, in the shape a model reply
    would have taken, so the assembled design does not know the difference.
    Worse than a mapped section, and reported as such — much better than losing
    every other section with it.
    """
    from .resolve import deterministic_intent

    def build(chunk, scoped_analysis):
        return deterministic_intent(scoped_analysis, difficulty)

    return build


def _design_call(track_id: str, difficulty: str, client):
    """The target-specific half of a chunk call: the prompt and the ledger stage."""
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
def chunk_gate_failures(objects: list, analysis: dict, difficulty: str,
                        chunks: list[Chunk], failed_gates: list[str]) -> dict:
    """Which chunks are responsible for the gates the whole map failed."""
    from . import qa as bs_qa

    def metrics_fn(slice_, chunk):
        # `chunk` is unused: no Beat Saber gate consults the analysis span, so
        # there is nothing here to keep in step with the slice. The stepmania
        # adapter's metrics_fn is the case that does.
        return bs_qa.chart_metrics(slice_, analysis, difficulty)["gates"]

    return chunking.gate_failures_by_chunk(objects, analysis, chunks,
                                           failed_gates, metrics_fn)


def reprompt_failing_chunks(run: ChunkedRun, track_id: str, difficulty: str,
                            analysis: dict, audio: str, client,
                            failing: dict) -> int:
    """Re-map only the chunks named in `failing`, in place. Returns how many."""
    def progress(chunk, gates):
        print(f"    [{difficulty}] chunk {chunk.index + 1} (bars {chunk.start_bar}-"
              f"{chunk.end_bar}) failed {', '.join(gates)} — scoped re-prompt…",
              flush=True)

    def failure(chunk, err):
        print(f"    [{difficulty}] chunk {chunk.index + 1} re-prompt failed ({err}); "
              f"keeping the original", flush=True)

    return chunking.redesign_chunks(
        run, analysis, audio, _design_call(track_id, difficulty, client),
        failing, GATE_BRIEFS, on_progress=progress, on_failure=failure,
        validate=_validator(analysis, difficulty))


# --------------------------------------------------------------------------- #
# Chunked critic (REQ-CHUNK-06)
# --------------------------------------------------------------------------- #
def critic_chunked(track_id: str, difficulty: str, analysis: dict, objects: list,
                   audio: str, client, run: ChunkedRun) -> dict:
    """Review per chunk (that chunk's audio, that chunk's swings), then once over
    the assembled map's structure without audio.

    Same return shape as the whole-song critic — `score`, `verdict`, `issues` —
    so the revision loop, the ship threshold and the QA report are unchanged.
    """
    meter = int(analysis.get("meter", 4) or 4)
    per_chunk = []
    for c in run.chunks:
        slice_ = [o for o in objects if c.owns_beat(o.beat, meter)]
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
        structure = _critic_structure(difficulty, analysis, run, objects, client)
    return chunking.aggregate_critic_scores(per_chunk, structure)


def _critic_one(difficulty: str, analysis: dict, chunk: Chunk, objects: list,
                chunk_audio: str, client) -> dict:
    """One chunk's critic call. Times are RELATIVE TO THE CLIP being heard —
    absolute song times against a 26-second clip would read as wildly misaligned."""
    from .grammar import COLOR_HAND
    from .parity import direction_parity

    bpm, offset = analysis["bpm"], analysis["offset"]
    t0, _ = chunking.chunk_window(analysis, chunk)
    _DIRNAME = {0: "up", 1: "down", 2: "left", 3: "right", 4: "up-left",
                5: "up-right", 6: "down-left", 7: "down-right", 8: "dot"}

    def t(beat):
        return round(offset + beat / bpm * 60 - t0, 3)

    rows = []
    for o in sorted(objects, key=lambda o: o.beat)[:600]:
        if o.kind == "bomb":
            rows.append(f"{t(o.beat):.3f}: BOMB ({o.x},{o.y})")
            continue
        hand = COLOR_HAND[o.color]
        par = direction_parity(o.direction)
        kind = o.kind.upper() if o.kind != "note" else "note"
        rows.append(f"{t(o.beat):.3f}: {hand} {kind} {_DIRNAME[o.direction]} "
                    f"[{par}] @({o.x},{o.y})")
    role = "/".join(chunk.roles) if chunk.roles else "section"
    prompt = (
        f"You are a Beat Saber (Standard) map critic. You are HEARING one section "
        f"of a song — bars {chunk.start_bar}-{chunk.end_bar} ({role}), plus a bar "
        f"of lead-in/lead-out — and reading the realized {difficulty} swings for "
        f"it. Times are SECONDS FROM THE START OF THE CLIP you are hearing, not "
        f"from the start of the song. Each line is '<time_s>: <hand> <kind> "
        f"<cut-direction> [parity] @(x,y)'. Smooth play ALTERNATES "
        f"forehand/backhand per hand; a reset (two same-parity swings in a row) is "
        f"bad unless a BOMB telegraphs it. You did NOT design this — judge it "
        f"fresh: do the notes land on the music, does the swing flow comfortably, "
        f"does the density suit this section's energy?\n"
        f"Map:\n" + "\n".join(rows) + "\n\n"
        f"Reply ONLY JSON: {{\"score\": <0-10>, \"verdict\": \"<one line>\", "
        f"\"issues\": [{{\"where\":\"..\",\"problem\":\"..\"}}]}}")
    return client.generate_json(prompt, audio_path=chunk_audio)


def _critic_structure(difficulty: str, analysis: dict, run: ChunkedRun,
                      objects: list, client) -> dict:
    """The one whole-map pass: arrangement only, no audio. This is the check no
    per-chunk review can make — whether the sections add up to a song."""
    meter = int(analysis.get("meter", 4) or 4)
    energy = analysis.get("energy_curve") or []
    rows = []
    for c, d in zip(run.chunks, run.decisions):
        n = sum(1 for o in objects if c.owns_beat(o.beat, meter))
        seg = energy[c.start_bar:c.end_bar]
        e = sum(seg) / len(seg) if seg else 0.0
        rows.append(
            f"  bars {c.start_bar:>3}-{c.end_bar:<3} ({'/'.join(c.roles) or '?':<6}) "
            f"energy {e:.2f}  {n / max(1, c.bars):.1f} notes/bar  "
            f"feel {d.texture or '-'}  movement {d.movement or '-'}")
    prompt = (
        f"You are a Beat Saber map critic reviewing the ARRANGEMENT of a "
        f"{difficulty} map that was designed one section at a time. You are not "
        f"hearing the audio for this pass and you are not judging individual "
        f"swings — judge whether the sections add up to one song: does density "
        f"track the measured energy, does the map build and release in the right "
        f"places, do repeated sections feel related, are the seams abrupt?\n"
        f"Sections (measured):\n" + "\n".join(rows) + "\n\n"
        f"Reply ONLY JSON: {{\"score\": <0-10>, \"verdict\": \"<one line>\", "
        f"\"issues\": [{{\"where\":\"bars a-b\",\"problem\":\"..\"}}]}}")
    return client.generate_json(prompt)
