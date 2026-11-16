"""chunking.py — CHUNKED PROCESSING MODE (docs/chunked-mode-brief.md).

A chunk is a **scoped designer call**: one slice of the song, with everything
scoped down to that slice — the audio for that slice, the onset inventory rows
for that slice, the density budget for that slice, and a short summary of what
previous chunks decided. Scope all four together; a call that trims the audio but
still ships the full-song inventory is not a chunked call.

This module is TARGET-AGNOSTIC. Both designer surfaces (StepMania, Beat Saber)
emit the same intent shape — `{"design_notes", "notes":[{"ref":..}],
"phrases":[{"start_bar","end_bar",..}]}` — so boundary computation, scoping,
prior-decision summaries and assembly live here once, and each adapter only
supplies its own prompt text.

What it does NOT do, on purpose:

  * It does not touch the DSP. Analysis runs ONCE over the whole track before any
    model call, and tempo, offset, beat grid, section boundaries, per-bar energy,
    the density plan and the onset IDs are global values that travel into each
    chunk as numbers (REQ-CHUNK-03). Nothing is recomputed per chunk and no ID is
    ever renumbered.
  * It does not change the output contract. `assemble()` returns a design object
    identical in shape to a whole-song one, so the realizer, the validators, the
    repair passes and the QA metrics run unchanged on it (REQ-CHUNK-05).
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field

from . import config

#: Reference forms a designer may use for a note. Mirrors the adapters'
#: resolvers (`adapters/*/resolve.py`) — kept here so a chunk can be scoped
#: before the target-specific resolver ever sees the design.
_GRID_RE = re.compile(r"^grid:(\d+(?:\.\d+)?)$")
_ONSET_RE = re.compile(r"^[pms]\d+$", re.IGNORECASE)


class ChunkError(ValueError):
    """A chunk's output could not be used as-is (bad ref, no notes at all).

    Raised per chunk so the retry is scoped to that chunk (REQ-CHUNK-07) rather
    than throwing away every other chunk that came back fine.
    """


# --------------------------------------------------------------------------- #
# Chunk geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chunk:
    """One slice of the song, in BARS.

    Bars are absolute and global — the same numbering the sections, the density
    plan and the energy curve use — so a chunk never has to be translated back
    into song coordinates and an off-by-one cannot silently shift a section.

    `start_bar`/`end_bar` (end exclusive) are the bars this chunk OWNS: the only
    bars its notes are kept from. `ctx_start_bar`/`ctx_end_bar` add the overlap
    the model may see and hear but must not chart (REQ-CHUNK-02).
    """
    index: int
    start_bar: int
    end_bar: int
    ctx_start_bar: int
    ctx_end_bar: int
    #: Section roles the owned bars overlap, in bar order ("intro", "drop", ...).
    #: Carried for the prior-decision summary and for repeat detection.
    roles: tuple = ()

    @property
    def bars(self) -> int:
        return self.end_bar - self.start_bar

    def owns_bar(self, bar: int) -> bool:
        return self.start_bar <= bar < self.end_bar

    def owns_beat(self, beat: float, meter: int) -> bool:
        return self.owns_bar(int(beat // meter))

    def label(self) -> str:
        return f"chunk {self.index + 1} (bars {self.start_bar}-{self.end_bar})"


def bar_seconds(analysis: dict) -> float:
    bpm = float(analysis.get("bpm") or 0.0)
    meter = int(analysis.get("meter", 4) or 4)
    return (60.0 / bpm) * meter if bpm > 0 else 0.0


def total_bars(analysis: dict) -> int:
    """How many bars the analysis covers.

    The energy curve is the authority — `dsp.structure_energy` builds it with one
    entry per bar and the sections index into it — so using it keeps chunk bars,
    section bars and density-plan bars in the same coordinate system by
    construction. The duration fallback is for hand-built fixtures.
    """
    ec = analysis.get("energy_curve") or []
    if ec:
        return len(ec)
    bs = bar_seconds(analysis)
    if bs <= 0:
        return 0
    return max(1, int(math.ceil((float(analysis.get("duration_s", 0.0))
                                 - float(analysis.get("offset", 0.0))) / bs)))


def should_chunk(analysis: dict, mode: str | None = None,
                 max_chunk_seconds: float | None = None) -> bool:
    """REQ-CHUNK-08. `auto` (the default) chunks when the track exceeds
    `max_chunk_seconds` — or, unset, the active backend's threshold, which is far
    longer on Argon (config.auto_chunk_threshold); `always` and `never` are the
    explicit overrides."""
    mode = (mode or config.CHUNK_MODE or "auto").strip().lower()
    if mode == "never":
        return False
    if mode == "always":
        return True
    if mode != "auto":
        # An unrecognized value must not silently pick a mode the user did not
        # ask for. `auto` is the documented default, so fall back to it loudly
        # enough to be findable in a log but without failing the run.
        print(f"[chunk] unknown chunk_mode {mode!r}; using 'auto'")
    limit = config.auto_chunk_threshold() if max_chunk_seconds is None else max_chunk_seconds
    return float(analysis.get("duration_s", 0.0)) > limit


def plan_chunks(analysis: dict, *, max_chunk_seconds: float | None = None,
                min_chunk_seconds: float | None = None,
                overlap_bars: int | None = None) -> list[Chunk]:
    """REQ-CHUNK-01/02. Section-aligned chunk boundaries, with overlap.

    Boundaries come from `dsp.structure_energy`'s section list, which already
    computes musically-aligned edges — inventing fixed-duration cuts would put a
    boundary in the middle of a phrase, which is the one place a chunk seam is
    audible. From there, two corrections, in this order:

      1. MERGE a section shorter than `min_chunk_seconds` into its neighbour. A
         3-second stab is not worth a round trip and carries no context to design
         against. Merging first means the split below sees the real spans.
      2. SPLIT a span longer than `max_chunk_seconds` into `ceil(len/max)` roughly
         equal parts. Every part boundary is a bar line, so the seam still lands
         on the grid even where no section boundary was available.

    Every boundary is therefore a section boundary or a bar line, no chunk
    exceeds `max_chunk_seconds` (unless one single bar does, which nothing can
    fix), and the result is a pure function of the analysis — the same input
    reproduces the same chunks.
    """
    max_s = config.MAX_CHUNK_SECONDS if max_chunk_seconds is None else max_chunk_seconds
    min_s = config.MIN_CHUNK_SECONDS if min_chunk_seconds is None else min_chunk_seconds
    overlap = config.CHUNK_OVERLAP_BARS if overlap_bars is None else overlap_bars
    overlap = max(0, int(overlap))

    n_bars = total_bars(analysis)
    if n_bars <= 0:
        return []
    bs = bar_seconds(analysis)
    if bs <= 0:
        return []

    spans = _section_spans(analysis, n_bars)
    spans = _merge_short(spans, bs, min_s)
    spans = _split_long(spans, bs, max_s)

    sections = analysis.get("sections") or []
    out: list[Chunk] = []
    for i, (s0, s1) in enumerate(spans):
        out.append(Chunk(
            index=i, start_bar=s0, end_bar=s1,
            ctx_start_bar=max(0, s0 - overlap),
            ctx_end_bar=min(n_bars, s1 + overlap),
            roles=tuple(_roles_for(sections, s0, s1)),
        ))
    return out


def _section_spans(analysis: dict, n_bars: int) -> list[tuple[int, int]]:
    """Section bar spans, clamped to the analyzed bars and made to tile [0,n).

    A gap between two sections (or a track whose sections stop short of the last
    bar) would leave bars no chunk owns, and an unowned bar is an uncharted bar.
    So the spans are closed up here rather than trusted.
    """
    raw = []
    for s in analysis.get("sections") or []:
        a, b = int(s.get("start_bar", 0)), int(s.get("end_bar", 0))
        a, b = max(0, min(a, n_bars)), max(0, min(b, n_bars))
        if b > a:
            raw.append((a, b))
    raw.sort()
    if not raw:
        return [(0, n_bars)]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for a, b in raw:
        a = max(a, cursor)
        if b <= a:
            continue                      # fully swallowed by the previous span
        if a > cursor:                    # close a hole with a span of its own
            spans.append((cursor, a))
        spans.append((a, b))
        cursor = b
    if cursor < n_bars:
        spans.append((cursor, n_bars))
    return spans


def _merge_short(spans: list[tuple[int, int]], bs: float, min_s: float
                 ) -> list[tuple[int, int]]:
    """Fold any span under `min_s` into the adjacent one (REQ-CHUNK-01).

    Merges into the PREVIOUS span where there is one, so a short intro attaches
    forward to the section it leads into rather than orphaning it.
    """
    if not spans:
        return spans
    min_bars = max(1, int(math.ceil(min_s / bs))) if bs > 0 else 1
    out: list[list[int]] = [list(spans[0])]
    for a, b in spans[1:]:
        if (b - a) < min_bars:
            out[-1][1] = b                # merge forward into the previous chunk
        elif (out[-1][1] - out[-1][0]) < min_bars:
            out[-1][1] = b                # previous was short: absorb into this one
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _split_long(spans: list[tuple[int, int]], bs: float, max_s: float
                ) -> list[tuple[int, int]]:
    """Cut any span over `max_s` into roughly equal parts at BAR boundaries."""
    if bs <= 0 or max_s <= 0:
        return spans
    max_bars = max(1, int(math.floor(max_s / bs)))
    out: list[tuple[int, int]] = []
    for a, b in spans:
        n = b - a
        if n <= max_bars:
            out.append((a, b))
            continue
        parts = int(math.ceil(n / max_bars))
        # Equal-ish parts, NOT max_bars-sized ones plus whatever is left. A
        # 33-bar span at a 16-bar ceiling is 11+11+11, not 16+16+1 — greedy
        # packing leaves a one-bar tail, and a one-bar chunk is a model call with
        # no musical context to design against.
        edges = [a + round(k * n / parts) for k in range(parts + 1)]
        edges[0], edges[-1] = a, b
        for k in range(parts):
            if edges[k + 1] > edges[k]:
                out.append((edges[k], edges[k + 1]))
    return out


def _roles_for(sections: list, s0: int, s1: int) -> list[str]:
    """Section roles this bar span overlaps, in bar order, de-duplicated."""
    roles: list[str] = []
    for s in sections:
        a, b = int(s.get("start_bar", 0)), int(s.get("end_bar", 0))
        if b > s0 and a < s1:
            role = str(s.get("role_guess") or "section")
            if role not in roles:
                roles.append(role)
    return roles


def chunk_brief(chunk: Chunk, analysis: dict, verb: str = "Design") -> str:
    """The prose statement of one chunk's assignment.

    States the owned bars, the context bars, and the seconds each maps to. The
    seconds matter: a model that hears 24 seconds of audio has to be told which
    part of the song it is hearing, because everything else in the prompt is in
    absolute bar numbers and it has no way to line the two up otherwise.

    Shared by both targets — the assignment is the same idea in each, and the two
    briefs drifting apart is exactly how one target would quietly stop discarding
    overlap notes.
    """
    t0, t1 = chunk_window(analysis, chunk)
    bs = bar_seconds(analysis)
    offset = float(analysis.get("offset", 0.0))
    own0 = offset + chunk.start_bar * bs
    own1 = offset + chunk.end_bar * bs
    role = "/".join(chunk.roles) if chunk.roles else "section"
    lines = [
        f"{verb} section {chunk.index + 1} of this song: **bars {chunk.start_bar}"
        f"-{chunk.end_bar}** ({role}), which is {own0:.1f}s-{own1:.1f}s of the "
        f"track. Every note you emit must fall in those bars.",
    ]
    lead_in = chunk.start_bar - chunk.ctx_start_bar
    lead_out = chunk.ctx_end_bar - chunk.end_bar
    if lead_in or lead_out:
        lines.append(
            f"The audio you are hearing and the inventory below run from bar "
            f"{chunk.ctx_start_bar} to bar {chunk.ctx_end_bar} ({t0:.1f}s-{t1:.1f}s) "
            f"— that is {lead_in} bar(s) of lead-in and {lead_out} bar(s) of "
            f"lead-out, given to you as CONTEXT so you can hear how this section "
            f"is entered and left. Notes in those bars are discarded.")
    if chunk.index == 0:
        lines.append("This is the first section of the song.")
    return "\n".join(lines)


def chunk_window(analysis: dict, chunk: Chunk) -> tuple[float, float]:
    """The chunk's CONTEXT window in seconds — what its audio slice covers.

    Derived from the global grid (offset + bar * bar_seconds), so the slice edges
    land on the same bar lines the inventory was cut at and the model's ear and
    its onset rows agree about where the slice starts.
    """
    bs = bar_seconds(analysis)
    offset = float(analysis.get("offset", 0.0))
    dur = float(analysis.get("duration_s", 0.0))
    t0 = max(0.0, offset + chunk.ctx_start_bar * bs)
    t1 = offset + chunk.ctx_end_bar * bs
    return (t0, min(t1, dur) if dur > 0 else t1)


# --------------------------------------------------------------------------- #
# Scoping the four inputs
# --------------------------------------------------------------------------- #
def slice_analysis(analysis: dict, chunk: Chunk) -> dict:
    """The analysis a chunk's designer call sees.

    Scoped: the onset inventory rows, the sections, and the density plan — all
    cut to the chunk's CONTEXT bars. Not scoped, and deliberately: bpm, offset,
    meter, duration and the per-bar energy curve are global facts (REQ-CHUNK-03)
    that travel as numbers so a model hearing 20 seconds of the song still places
    notes on the whole song's grid. The energy curve in particular stays whole so
    its list index remains the absolute bar number.

    Onset IDs are NEVER renumbered. A chunk receives a subset of rows carrying
    exactly the ids the whole-song run would have used, which is what makes the
    assembled design's references resolvable against the same analysis.
    """
    meter = int(analysis.get("meter", 4) or 4)
    b0 = chunk.ctx_start_bar * meter
    b1 = chunk.ctx_end_bar * meter

    out = dict(analysis)
    out["onsets"] = [o for o in analysis.get("onsets") or []
                     if b0 <= float(o.get("nearest_beat", -1)) < b1]
    out["sections"] = _clip_sections(analysis.get("sections") or [],
                                     chunk.ctx_start_bar, chunk.ctx_end_bar)
    plan = analysis.get("density_plan") or {}
    if plan:
        out["density_plan"] = _slice_plan(plan, chunk)
    # The chunk's own identity, for any surface that wants to state the
    # assignment in prose. Additive: nothing downstream reads it unless asked.
    out["chunk"] = {
        "index": chunk.index, "start_bar": chunk.start_bar, "end_bar": chunk.end_bar,
        "context_start_bar": chunk.ctx_start_bar, "context_end_bar": chunk.ctx_end_bar,
        "roles": list(chunk.roles),
    }
    return out


def _clip_sections(sections: list, lo: int, hi: int) -> list:
    """Sections overlapping [lo, hi), clipped to it but keeping ABSOLUTE bars.

    Clipping rather than filtering matters at the seams: a chunk that starts
    mid-section would otherwise be handed a section claiming to begin 40 bars
    before anything it can see.
    """
    out = []
    for s in sections:
        a, b = int(s.get("start_bar", 0)), int(s.get("end_bar", 0))
        if b <= lo or a >= hi:
            continue
        c = dict(s)
        c["start_bar"], c["end_bar"] = max(a, lo), min(b, hi)
        out.append(c)
    return out


def _slice_plan(plan: dict, chunk: Chunk) -> dict:
    """The chunk's slice of the density plan (REQ-CHUNK-03).

    The per-bar targets and the per-section bands are the SAME NUMBERS the
    whole-song plan carries — this filters, it never recomputes. That is what
    stops a model who only hears one section from bending the global energy
    contour, and therefore what keeps the Spearman gate meaningful under
    chunking.
    """
    out = dict(plan)
    out["per_bar"] = [e for e in plan.get("per_bar") or []
                      if chunk.ctx_start_bar <= int(e.get("bar", -1)) < chunk.ctx_end_bar]
    out["per_section"] = _clip_sections(plan.get("per_section") or [],
                                        chunk.ctx_start_bar, chunk.ctx_end_bar)
    return out


def onset_lookup(analysis: dict) -> dict:
    """Onset id -> onset, tolerant of dropped zero-padding and casing.

    Mirrors `adapters/stepmania/resolve.py::_onset_lookup`. Duplicated rather
    than imported because this module must not depend on an adapter — core
    cannot import a target, and both targets need this.
    """
    lut: dict[str, dict] = {}
    for o in analysis.get("onsets") or []:
        oid = str(o.get("id", ""))
        if not oid:
            continue
        lut[oid] = o
        lut[oid.lower()] = o
        m = re.match(r"^([pmsPMS])0*(\d+)$", oid)
        if m:
            lut[f"{m.group(1).lower()}{int(m.group(2))}"] = o
    return lut


def beat_for_ref(ref, lut: dict) -> float:
    """Beat position of a designer reference, or raise `ChunkError`.

    Only two forms exist, in both targets: an onset id from the inventory, or
    `grid:<beat>`. Anything else is the model inventing a coordinate, which the
    adapters' resolvers reject too — catching it here just means the retry is
    scoped to the one chunk that did it.
    """
    if not isinstance(ref, str):
        raise ChunkError(f"note ref {ref!r} is not a string")
    gm = _GRID_RE.match(ref)
    if gm:
        return float(gm.group(1))
    if _ONSET_RE.match(ref):
        o = lut.get(ref) or lut.get(ref.lower())
        if o is None:
            raise ChunkError(
                f"note ref '{ref}' is not in the onset inventory you were shown. "
                f"Use the EXACT ids from the inventory rows above — they are "
                f"assigned over the whole song, so they are not consecutive here.")
        return float(o["nearest_beat"])
    raise ChunkError(
        f"note ref '{ref}' is neither an onset id from the inventory nor "
        f"grid:<beat>. Never write a time, a row or a second.")


def scope_design(design, analysis: dict, chunk: Chunk,
                 lut: dict | None = None) -> dict:
    """Keep only what this chunk OWNS (REQ-CHUNK-02).

    Overlap bars are context, never output: a note the model placed in a lead-in
    or lead-out bar is discarded here, because that bar belongs to the adjacent
    chunk and will be designed (or already was) by the call that owns it. That is
    what makes "a note in an overlap region appears exactly once in the assembled
    design, attributed to the chunk that owns its bar" true by construction
    rather than by a de-duplication pass afterwards.

    Phrases are clipped to the owned range instead of dropped, so a phrase the
    model declared across the seam still governs the part of it we keep.
    """
    if not isinstance(design, dict):
        raise ChunkError("chunk output is not a JSON object")
    lut = onset_lookup(analysis) if lut is None else lut
    meter = int(analysis.get("meter", 4) or 4)

    notes_in = design.get("notes")
    if not isinstance(notes_in, list):
        raise ChunkError("chunk output has no `notes` array")

    kept, seen = [], set()
    for n in notes_in:
        if not isinstance(n, dict):
            raise ChunkError("every note must be a JSON object")
        beat = beat_for_ref(n.get("ref"), lut)
        if not chunk.owns_beat(beat, meter):
            continue                      # overlap context — not this chunk's to emit
        key = (str(n.get("ref")), round(beat, 4))
        if key in seen:
            continue                      # the model listed the same onset twice
        seen.add(key)
        kept.append(n)

    phrases = []
    for ph in design.get("phrases") or []:
        if not isinstance(ph, dict):
            continue
        a, b = int(ph.get("start_bar", 0)), int(ph.get("end_bar", 0))
        a, b = max(a, chunk.start_bar), min(b, chunk.end_bar)
        if b <= a:
            continue
        c = dict(ph)
        c["start_bar"], c["end_bar"] = a, b
        phrases.append(c)
    phrases.sort(key=lambda p: (p["start_bar"], p["end_bar"]))

    return {"design_notes": str(design.get("design_notes") or "").strip(),
            "notes": kept, "phrases": phrases}


# --------------------------------------------------------------------------- #
# Cross-chunk continuity (REQ-CHUNK-04)
# --------------------------------------------------------------------------- #
@dataclass
class ChunkDecision:
    """What one designed chunk decided — the whole of its cross-chunk memory.

    Decisions only. No audio, no inventory rows: those are what chunking exists
    to keep out of the context, and letting them back in through the summary
    would undo the entire mechanism.
    """
    index: int
    start_bar: int
    end_bar: int
    roles: tuple
    texture: str | None
    movement: str | None
    crossover: str | None
    notes_per_bar: float
    intent: str = ""

    def line(self) -> str:
        role = "/".join(self.roles) if self.roles else "section"
        bits = [f"bars {self.start_bar}-{self.end_bar} ({role}):"]
        if self.texture:
            bits.append(f"texture {self.texture}")
        if self.movement:
            bits.append(f"movement {self.movement}")
        if self.crossover:
            bits.append(f"crossover {self.crossover}")
        bits.append(f"{self.notes_per_bar:.1f} notes/bar")
        line = "  - " + " ".join(bits[:1]) + " " + ", ".join(bits[1:])
        if self.intent:
            line += f" — {self.intent[:80]}"
        return line


def summarize_chunk(chunk: Chunk, scoped: dict) -> ChunkDecision:
    """Reduce one designed chunk to the handful of facts the next ones need.

    The dominant phrase wins the texture/movement/crossover slots — a chunk
    usually declares one, and where it declares several the one covering the most
    bars is the one another chunk would be echoing.
    """
    phrases = scoped.get("phrases") or []
    dominant = max(phrases, key=lambda p: p["end_bar"] - p["start_bar"],
                   default=None) if phrases else None
    bars = max(1, chunk.bars)
    return ChunkDecision(
        index=chunk.index, start_bar=chunk.start_bar, end_bar=chunk.end_bar,
        roles=chunk.roles,
        texture=(dominant or {}).get("texture"),
        movement=(dominant or {}).get("movement"),
        crossover=(dominant or {}).get("crossover"),
        notes_per_bar=len(scoped.get("notes") or []) / bars,
        intent=str((dominant or {}).get("intent") or "").strip(),
    )


def _approx_tokens(text: str) -> int:
    """Tokens, near enough for a budget check. ~4 characters per token is the
    standard rule of thumb and this is a ceiling, not an invoice."""
    return (len(text) + 3) // 4


def prior_summary_text(decisions: list[ChunkDecision], chunk: Chunk,
                       token_budget: int | None = None) -> str:
    """REQ-CHUNK-04. The compact prior-decision block for chunk N's prompt.

    Contains a line per previously designed chunk: its bars, its role, the
    texture/movement/crossover it chose and the notes-per-bar it realized. This
    is the one thing genuinely lost by chunking — a chorus echoing an earlier
    chorus's pattern — so it is stated explicitly rather than hoped for.

    Under budget pressure, a prior chunk whose ROLE matches this chunk's is kept
    ahead of a merely more recent one: "the last drop played like this" is worth
    more to a drop than "the intro played like this", which is exactly the repeat
    relationship the requirement names.
    """
    budget = (config.CHUNK_SUMMARY_TOKEN_BUDGET if token_budget is None
              else token_budget)
    if not decisions:
        return ("(this is the first chunk of the song — nothing has been designed "
                "yet)")

    repeats = [d for d in decisions if set(d.roles) & set(chunk.roles)]
    repeat_idx = {d.index for d in repeats}
    # Priority: role-matching chunks first (most recent of them first), then the
    # rest most-recent-first. Rendered back in bar order so the model reads the
    # song forwards.
    ranked = (sorted(repeats, key=lambda d: -d.index)
              + sorted((d for d in decisions if d.index not in repeat_idx),
                       key=lambda d: -d.index))

    head = ("Already designed, earlier in this same song. Keep the chart coherent "
            "with these decisions — repeat a section's feel when the music repeats "
            "it, and contrast when it does not:\n")
    chosen: list[ChunkDecision] = []
    for d in ranked:
        candidate = chosen + [d]
        body = "\n".join(x.line() for x in sorted(candidate, key=lambda x: x.index))
        if _approx_tokens(head + body) > budget and chosen:
            break
        chosen.append(d)

    lines = [d.line() for d in sorted(chosen, key=lambda d: d.index)]
    if repeats:
        matched = ", ".join(f"bars {d.start_bar}-{d.end_bar}" for d in repeats
                            if d in chosen)
        if matched:
            lines.append(f"  NOTE: {matched} share this chunk's section role "
                         f"({'/'.join(chunk.roles) or 'section'}) — this is a "
                         f"REPEAT. Echo that pattern rather than inventing a new one.")
    dropped = len(decisions) - len(chosen)
    if dropped > 0:
        lines.append(f"  (+{dropped} earlier section(s) omitted to stay inside the "
                     f"summary budget)")
    return head + "\n".join(lines)


# --------------------------------------------------------------------------- #
# Assembly (REQ-CHUNK-05)
# --------------------------------------------------------------------------- #
#: Phrase fields a filler span gets when no chunk declared one. Each target has
#: its own CLOSED vocabulary and a value from the wrong one is a hard validation
#: error, so the caller states which — there is no safe cross-target default.
#: Deliberately carries no `density`: a phrase may omit it (the deterministic
#: repair pass still enforces the plan on the realized chart), whereas an
#: invented number would be checked against the plan's band and could be
#: rejected outright.
STEPMANIA_FILLER = {"texture": "steps", "movement": "static", "crossover": "none",
                    "jump_density": "accents"}
BEATSABER_FILLER = {"density": "steady", "movement": "static", "tech": "flowy",
                    "hand_balance": "even", "emphasis": "kick+snare"}


def assemble(scoped: list[dict], analysis: dict, chunks: list[Chunk],
             filler: dict | None = None) -> dict:
    """Merge scoped chunk outputs into ONE design object.

    The result is identical in shape to a whole-song design, which is the whole
    point: the realizer, the validators, the repair passes and the QA metrics run
    on it unchanged. Notes come out in beat order with no duplicate onset
    references; phrases tile the charted bars in bar order with no overlaps and
    no gaps.
    """
    lut = onset_lookup(analysis)
    notes: list[dict] = []
    seen_refs: set = set()
    for s in scoped:
        for n in s.get("notes") or []:
            ref = str(n.get("ref"))
            if ref in seen_refs:
                # Ownership already makes this unreachable — two chunks cannot own
                # the same bar, so they cannot both keep a note at the same beat.
                # It is enforced anyway because the cost of being wrong is a
                # duplicate note in the shipped chart, and the cost of the check
                # is one set lookup.
                continue
            seen_refs.add(ref)
            notes.append(n)
    notes.sort(key=lambda n: beat_for_ref(n.get("ref"), lut))

    phrases: list[dict] = []
    for s in scoped:
        phrases.extend(s.get("phrases") or [])
    phrases = _tile_phrases(phrases, chunks, filler or STEPMANIA_FILLER)

    parts = []
    for s, c in zip(scoped, chunks):
        note = (s.get("design_notes") or "").strip()
        if note:
            parts.append(f"[bars {c.start_bar}-{c.end_bar}] {note}")
    return {"design_notes": "\n".join(parts), "notes": notes, "phrases": phrases,
            "chunked": True, "chunk_count": len(chunks)}


def _tile_phrases(phrases: list[dict], chunks: list[Chunk], filler: dict) -> list[dict]:
    """Make the phrase list a clean tiling: sorted, non-overlapping, gapless.

    A model that declared overlapping phrases, or left a couple of bars
    undeclared, must not be able to produce an assembled design the whole-song
    path would never have emitted. Overlaps are trimmed to the earlier phrase;
    holes get a neutral filler phrase from the target's own vocabulary.
    """
    if not chunks:
        return sorted(phrases, key=lambda p: (p["start_bar"], p["end_bar"]))
    lo = min(c.start_bar for c in chunks)
    hi = max(c.end_bar for c in chunks)

    ordered = sorted((p for p in phrases if p.get("end_bar", 0) > p.get("start_bar", 0)),
                     key=lambda p: (p["start_bar"], p["end_bar"]))
    out: list[dict] = []
    cursor = lo
    for ph in ordered:
        a, b = int(ph["start_bar"]), int(ph["end_bar"])
        if b <= cursor:
            continue                      # entirely covered already
        if a > cursor:
            out.append(_filler_phrase(cursor, a, filler))
        c = dict(ph)
        c["start_bar"], c["end_bar"] = max(a, cursor), b
        out.append(c)
        cursor = b
    if cursor < hi:
        out.append(_filler_phrase(cursor, hi, filler))
    return out


def _filler_phrase(a: int, b: int, filler: dict) -> dict:
    """A bar span no chunk declared. Neutral, closed-vocabulary values only."""
    return {"start_bar": a, "end_bar": b, **filler,
            "intent": "continuation (no phrase declared for these bars)"}


# --------------------------------------------------------------------------- #
# The run: designing a song chunk by chunk
# --------------------------------------------------------------------------- #
@dataclass
class ChunkedRun:
    """Everything one chunked design produced, kept PER CHUNK.

    Per chunk rather than collapsed into the assembled design as soon as it is
    built, because a scoped re-prompt has to be able to replace one chunk's
    output and re-assemble around it (REQ-CHUNK-07). Collapse early and there is
    nothing left to replace but the whole song.
    """
    chunks: list
    filler: dict
    scoped: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    audio: dict = field(default_factory=dict)     # chunk index -> sliced wav path
    calls: int = 0                                # model calls made, incl. retries

    def design(self, analysis: dict) -> dict:
        """The assembled whole-song design (REQ-CHUNK-05)."""
        return assemble(self.scoped, analysis, self.chunks, self.filler)

    def replace(self, index: int, scoped: dict) -> None:
        self.scoped[index] = scoped
        self.decisions[index] = summarize_chunk(self.chunks[index], scoped)

    def fallback_bars(self) -> list:
        """Bar ranges that were charted from the DSP because the model could not
        design them. Empty on a healthy run; surfaced in the QA report so a
        degraded section is never a silent one."""
        return [[c.start_bar, c.end_bar]
                for c, s in zip(self.chunks, self.scoped)
                if s.get("fallback")]

    def audio_for(self, audio_path: str, analysis: dict, chunk: Chunk) -> str:
        """This chunk's sliced wav, cut once per run and reused.

        Slicing the audio is the point of the exercise — it is what keeps the KV
        cache small enough to stay in VRAM — so a failure here is fatal rather
        than quietly falling back to the whole track, which would send exactly
        the payload chunked mode exists to avoid.
        """
        path = self.audio.get(chunk.index)
        if path is None:
            path = slice_audio(audio_path, analysis, chunk)
            self.audio[chunk.index] = path
        return path


def run_chunked_design(analysis: dict, audio_path: str, chunks: list, design_call,
                       *, filler: dict, seed_feedback: str | None = None,
                       max_attempts: int | None = None, on_progress=None,
                       validate=None, fallback=None) -> ChunkedRun:
    """Design a song section by section. Returns the per-chunk run.

    `design_call(chunk, scoped_analysis, chunk_audio, prior_summary, feedback,
    attempt) -> raw model output` is the ONLY target-specific part: it owns the
    prompt text and the ledger stage. Everything here — boundaries, scoping,
    prior-decision summaries, per-chunk retry budgets, assembly — is the same for
    every target, and lives here so the two cannot drift.

    `validate(chunk, scoped)` is the other one: an optional hook that runs the
    TARGET's own resolver over the chunk's scoped output and raises `ChunkError`
    if it would be rejected. Without it, a design error the resolver catches — an
    out-of-vocabulary texture, a phrase declaring a density outside the plan's
    band — only surfaces later, after assembly, as a whole-song `IntentError`
    that re-designs every chunk. With it, the chunk that made the mistake is
    re-prompted on its own and told the exact number it missed.

    `fallback(chunk, scoped_analysis)` is the last one, and the difference between
    a finished song and a lost one: when a chunk exhausts its retry budget, it
    supplies intent for those bars WITHOUT the model — the DSP-only path each
    target already has. See `_design_one_chunk`.

    `seed_feedback` (a critic's issues, say) goes to EVERY chunk: it is a critique
    of the chart as a whole and there is no honest way to decide which section a
    line like "the drop does not hit" belongs to. A GATE failure, which can be
    attributed to bars, goes through `redesign_chunks` instead.
    """
    run = ChunkedRun(chunks=list(chunks), filler=dict(filler))
    for chunk in run.chunks:
        summary = prior_summary_text(run.decisions, chunk)
        scoped = _design_one_chunk(run, analysis, audio_path, chunk, summary,
                                   design_call, seed_feedback, max_attempts,
                                   validate, fallback)
        run.scoped.append(scoped)
        run.decisions.append(summarize_chunk(chunk, scoped))
        if on_progress:
            on_progress(chunk, scoped)
    return run


def _design_one_chunk(run: ChunkedRun, analysis: dict, audio_path: str, chunk: Chunk,
                      prior_summary: str, design_call, feedback: str | None,
                      max_attempts: int | None, validate=None, fallback=None) -> dict:
    """One chunk, against its OWN retry budget (REQ-CHUNK-07).

    Retries here are for output the pipeline could not USE — an onset id that is
    not in the inventory, a missing `notes` array, a reply that would not parse,
    a reply that scoped down to nothing. One bad slice must not cost the song the
    chunks already paid for, which is exactly what a song-level retry budget
    would do.

    **And when the retries run out, the song still finishes.** A small local model
    can make the same mistake every time it is asked — the measured case was a
    Gemma emitting `{"kind": "hold"}` with no duration on both attempts, at chunk
    4 of 4, which threw away three chunks of paid-for work and the whole run with
    it. If the caller supplied a `fallback`, those bars are charted from the DSP
    instead: real onsets, no model judgement, and a loud warning saying so. That
    is a worse chart for one section; failing was a worse chart for all of them.
    """
    attempts = max(1, config.CHUNK_MAX_ATTEMPTS if max_attempts is None else max_attempts)
    scoped_analysis = slice_analysis(analysis, chunk)
    chunk_audio = run.audio_for(audio_path, analysis, chunk)
    note = feedback
    last: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            raw = design_call(chunk, scoped_analysis, chunk_audio, prior_summary,
                              note, attempt)
        except ValueError as e:
            # An unparseable reply. `_parse_json` already salvages one that merely
            # stopped early, so reaching here means the reply was genuinely
            # malformed — which is a property of THIS call, not of the song, and
            # so is retried against this chunk's budget. Before this, the
            # JSONDecodeError travelled all the way out of `build_song` and took
            # every already-designed chunk with it.
            run.calls += 1
            last = e
            note = (f"Your previous reply for these bars could not be read as JSON "
                    f"({str(e)[:200]}). Reply with ONLY the JSON object — no prose, "
                    f"no code fences — and keep it short enough to finish.")
            continue
        run.calls += 1
        try:
            scoped = scope_design(raw, analysis, chunk)
            if validate is not None:
                validate(chunk, scoped)
        except ChunkError as e:
            last = e
            # The error text carries the specifics — which ref was unknown, which
            # band a density missed — so the note states them and adds only the
            # one instruction that is true of every chunk retry.
            note = (f"{e}\nFix that and resubmit, emitting notes only for bars "
                    f"{chunk.start_bar}-{chunk.end_bar}.")
            continue
        if not scoped["notes"]:
            last = ChunkError(f"{chunk.label()} came back with no notes in its own bars")
            note = (f"Your reply contained no notes inside bars {chunk.start_bar}-"
                    f"{chunk.end_bar}. The lead-in/lead-out context bars do not "
                    f"count — chart the assigned bars themselves. Resubmit.")
            continue
        return scoped

    if fallback is None:
        raise ChunkError(f"{chunk.label()} failed after {attempts} attempt(s): {last}")

    print(f"[chunk] {chunk.label()} could not produce usable intent after "
          f"{attempts} attempt(s) ({last}). Charting those bars from the DSP "
          f"instead — real onsets, but WITHOUT the model's musical judgement. "
          f"The rest of the song is unaffected.", file=sys.stderr)
    try:
        scoped = scope_design(fallback(chunk, slice_analysis(analysis, chunk)),
                              analysis, chunk)
    except ChunkError as e:
        # The deterministic path is supposed to be incapable of this. If it
        # manages it anyway, that is a bug in the fallback rather than a model
        # problem, and hiding it behind an empty chunk would be worse than
        # failing — so report both causes.
        raise ChunkError(
            f"{chunk.label()} failed after {attempts} attempt(s) ({last}), and the "
            f"deterministic fallback for those bars was rejected too: {e}") from e
    scoped["fallback"] = "deterministic"
    return scoped


def redesign_chunks(run: ChunkedRun, analysis: dict, audio_path: str, design_call,
                    failing: dict, gate_briefs: dict, *, on_progress=None,
                    on_failure=None, validate=None) -> int:
    """Re-design only the chunks named in `failing`, in place. Returns how many.

    Each re-designed chunk still sees the prior-decision summary of the chunks
    BEFORE it, so a repair cannot quietly break the continuity the first pass
    established. A chunk whose repair fails outright keeps its original output —
    a failed repair costs a call, never a chart.

    No `fallback` here, deliberately: the chunk being repaired already HAS usable
    intent from the first pass, and that is a better thing to keep than a
    deterministic replacement. `on_failure` reports it and the original stands.
    """
    repaired = 0
    for index in sorted(failing):
        chunk = run.chunks[index]
        gates = failing[index]
        brief = ("The chart you contributed these bars to failed these objective "
                 "gates, and this section is where they fail. Fix them while "
                 "keeping every budget rule:\n"
                 + "\n".join(f"- {g}: {gate_briefs.get(g, 'out of budget')}"
                             for g in gates))
        if on_progress:
            on_progress(chunk, gates)
        try:
            scoped = _design_one_chunk(run, analysis, audio_path, chunk,
                                       prior_summary_text(run.decisions[:index], chunk),
                                       design_call, brief, None, validate)
        except ChunkError as e:
            if on_failure:
                on_failure(chunk, e)
            continue
        run.replace(index, scoped)
        repaired += 1
    return repaired


def gate_failures_by_chunk(placements: list, analysis: dict, chunks: list,
                           failed_gates: list, metrics_fn, min_notes: int = 4,
                           min_notes_by_gate: dict | None = None,
                           emergent_gates: frozenset | set | None = None) -> dict:
    """Which chunks are responsible for the gates the whole chart failed.

    Measured, not guessed: `metrics_fn(slice, chunk)` re-runs the target's own QA
    battery over each chunk's placements, so a gate is attributed to a chunk only
    if that chunk's slice fails it too. A chunk with too few notes for a metric to
    mean anything is skipped rather than blamed.

    `min_notes_by_gate` raises that floor for named gates, and a RATIO gate needs
    it: a share-of-notes budget is a whole-chart figure, and on a small slice it
    quantizes hard enough to stop meaning what it says. Targets derive the number
    from the budget rather than picking one — see the stepmania adapter's
    `chunk_gate_failures`.

    Which is half the fix. The other half is that a target may also want a gate
    JUDGED differently on a slice than on a chart, and that belongs in the
    target's `metrics_fn`, not here — `hold_share` is a band on a chart and a
    ceiling on a chunk, for reasons that are about music rather than about
    chunking (see stepmania `qa._hold_share_ok`).

    A globally-failed gate that no single chunk fails locally may be an EMERGENT
    failure — density-vs-energy is the obvious case, since the correlation is
    taken over the whole song and every section can be individually plausible
    while the contour between them is flat. Those are reported against every
    chunk that was ELIGIBLE to be measured, because the honest answer to "which
    section caused it" is "the arrangement of all of them" — and a chunk too
    small to measure is no more able to fix an emergent failure than a local one.

    `emergent_gates` NAMES the gates that reading applies to, and naming them is
    not optional bookkeeping. "No chunk failed it" has a second cause: a gate the
    chunk battery does not judge the same way the chart battery does, which
    CANNOT be reproduced locally however wrong the chart is. `hold_share` is
    exactly that — a band on a chart, a ceiling on a chunk — so a chart under the
    floor blamed every chunk, every run, and each one was asked to fix a section
    that was individually fine. Treating "unreproducible" as "emergent" turns one
    chart-level failure into N scoped re-prompts that cannot converge.

    `None` means every gate, which is the old behaviour and is kept for targets
    whose chunk battery is identical to their chart battery.

    A gate that is neither reproduced locally nor named emergent is dropped, as
    is one no chunk was eligible for: "not attributable at chunk scale" is a
    different answer from "everyone", and it correctly leaves the failure to the
    chart-level path and the QA report.

    `placements` items need only a `.beat`; both targets' realized objects have one.
    """
    meter = int(analysis.get("meter", 4) or 4)
    per_gate_min = min_notes_by_gate or {}
    # Measure each eligible chunk ONCE — the battery does not depend on which
    # gate is being attributed, and it is the expensive half of this loop.
    measured: list = []                       # (chunk, note_count, gates)
    for c in chunks:
        slice_ = [p for p in placements if c.owns_beat(p.beat, meter)]
        if len(slice_) < min_notes:
            continue
        measured.append((c, len(slice_), metrics_fn(slice_, c)))

    out: dict = {}
    for gate in failed_gates:
        need = max(min_notes, per_gate_min.get(gate, 0))
        eligible = [(c, g) for c, n, g in measured if n >= need]
        if not eligible:
            continue
        blamed = [c.index for c, g in eligible if gate in g and not g[gate]]
        if not blamed:
            if emergent_gates is not None and gate not in emergent_gates:
                continue
            blamed = [c.index for c, _ in eligible]
        for i in blamed:
            out.setdefault(i, []).append(gate)
    return out


# --------------------------------------------------------------------------- #
# Critic aggregation (REQ-CHUNK-06)
# --------------------------------------------------------------------------- #
def _score_of(review: dict) -> float:
    """A model's score as a float. A missing or non-numeric score is 0, not a
    crash: the aggregate has to survive one chunk answering badly."""
    try:
        return float((review or {}).get("score", 0))
    except (TypeError, ValueError):
        return 0.0


def aggregate_critic_scores(per_chunk: list, structure: dict) -> dict:
    """Fold per-chunk reviews plus the structural pass into ONE review.

    The result has the same keys a whole-song critic returns — `score`,
    `verdict`, `issues` — so the revision loop, the ship threshold and the QA
    report need no special-casing (REQ-CHUNK-06). Per-chunk detail rides
    alongside under `chunks`, so both the aggregate and the per-chunk scores are
    recorded, which is what the requirement asks for.

    Note-weighted across chunks, so a 4-bar tag cannot outvote the drop, and then
    the structural pass counts as one average-sized chunk. Both halves matter: a
    chart of individually good sections that do not add up is not a good chart,
    and neither is a well-shaped arrangement of badly-placed notes.
    """
    scores, weights, issues = [], [], []
    for r in per_chunk:
        scores.append(_score_of(r))
        weights.append(max(1, int(r.get("notes", 1))))
        bars = r.get("bars") or [0, 0]
        for i in r.get("issues") or []:
            where = str(i.get("where", "")).strip()
            issues.append({**i, "where": f"bars {bars[0]}-{bars[1]}"
                                         + (f" · {where}" if where else "")})

    s_score = _score_of(structure)
    for i in (structure or {}).get("issues") or []:
        issues.append({**i, "where": str(i.get("where") or "arrangement")})

    if scores:
        mean_w = sum(weights) / len(weights)
        total = sum(s * w for s, w in zip(scores, weights)) + s_score * mean_w
        score = total / (sum(weights) + mean_w)
    else:
        score = s_score

    verdicts = [str(r.get("verdict", "")).strip() for r in per_chunk if r.get("verdict")]
    return {
        "score": round(score, 2),
        "verdict": str((structure or {}).get("verdict")
                       or (verdicts[0] if verdicts else "")),
        "dynamics": (structure or {}).get("dynamics"),
        "measured_rho": (structure or {}).get("measured_rho"),
        "issues": issues[:20],
        "chunked": True,
        "chunks": [{"chunk": r.get("chunk"), "bars": r.get("bars"),
                    "notes": r.get("notes"), "score": r.get("score"),
                    "verdict": r.get("verdict")} for r in per_chunk],
        "structure_score": s_score,
    }


# --------------------------------------------------------------------------- #
# Audio slicing
# --------------------------------------------------------------------------- #
def slice_audio(audio_path: str, analysis: dict, chunk: Chunk,
                out_dir=None) -> str:
    """Write (and cache) the chunk's audio window as a wav, returning its path.

    soundfile rather than ffmpeg: it is already a hard dependency (the DSP reads
    every track through it), so slicing works on a machine with no ffmpeg, and
    reading a frame range avoids decoding the whole track per chunk. The output
    is wav because every audio-capable serving stack decodes it without argument;
    the LLM clients re-encode/downsample from here as their endpoint requires.

    Cached on (track, chunk, window): slicing is deterministic, so an existing
    slice is the same slice, and a five-difficulty song then pays for the cut
    once instead of five times.
    """
    import soundfile as sf

    t0, t1 = chunk_window(analysis, chunk)
    if t1 <= t0:
        raise ChunkError(f"{chunk.label()} has an empty audio window")

    out_dir = config.CHUNK_AUDIO_DIR if out_dir is None else out_dir
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem
    out = out_dir / f"{stem}.c{chunk.index:02d}.{int(t0 * 1000)}-{int(t1 * 1000)}.wav"
    if out.exists() and out.stat().st_size > 0:
        return str(out)

    info = sf.info(str(audio_path))
    sr = info.samplerate
    start = max(0, int(round(t0 * sr)))
    stop = min(info.frames, int(round(t1 * sr)))
    if stop <= start:
        raise ChunkError(
            f"{chunk.label()} maps to samples {start}..{stop} of "
            f"{Path(audio_path).name}, which is empty — the analysis and the audio "
            f"file disagree about the track's length.")
    data, _ = sf.read(str(audio_path), start=start, stop=stop, always_2d=True)
    # Write to a temp name and rename, so a run interrupted mid-write can never
    # leave a truncated slice behind for the next run to send to a model.
    tmp = out.with_suffix(".part.wav")
    sf.write(str(tmp), data, sr, subtype="PCM_16")
    tmp.replace(out)
    return str(out)


__all__ = [
    "BEATSABER_FILLER", "STEPMANIA_FILLER",
    "Chunk", "ChunkDecision", "ChunkError", "ChunkedRun",
    "aggregate_critic_scores", "assemble",
    "bar_seconds", "beat_for_ref", "chunk_brief", "chunk_window",
    "gate_failures_by_chunk",
    "onset_lookup", "plan_chunks", "prior_summary_text", "redesign_chunks",
    "run_chunked_design", "scope_design", "should_chunk", "slice_analysis",
    "slice_audio", "summarize_chunk", "total_bars",
]
