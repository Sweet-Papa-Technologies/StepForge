"""test_chunking.py — CHUNKED PROCESSING MODE, the offline unit half
(docs/chunked-mode-brief.md §7 "Unit (offline)").

No network, no model, no Colab: everything here is pure function over an analysis
dict. The end-to-end path — a stubbed designer, the assembled design through the
real realizer, and a scoped re-prompt — is `test_chunked_stepforge.py`.

Requirement coverage, so a failure here names the thing it broke:
  REQ-CHUNK-01  section-aligned boundaries, max split, min merge, bar snapping
  REQ-CHUNK-02  overlap is context-only; a note appears exactly once
  REQ-CHUNK-03  globals preserved; onset IDs stable across modes
  REQ-CHUNK-04  prior-decision summary, repeat handling, token budget
  REQ-CHUNK-05  assembly: shape, coverage, no gaps/overlaps/duplicate refs
  REQ-CHUNK-08  auto/always/never, and the shipped default
"""
from __future__ import annotations

import json

import pytest

from beatforge import chunking, config

from conftest import CHUNK_BAR_S as BAR_S, CHUNK_BPM as BPM, CHUNK_METER as METER

# --------------------------------------------------------------------------- #
# REQ-CHUNK-01 — boundaries
# --------------------------------------------------------------------------- #
def test_boundaries_follow_sections(chunk_analysis):
    """Boundaries come from the DSP section list, not from fixed-duration cuts."""
    an = chunk_analysis()
    chunks = chunking.plan_chunks(an, max_chunk_seconds=30, min_chunk_seconds=8,
                                  overlap_bars=1)
    assert [(c.start_bar, c.end_bar) for c in chunks] == [(0, 4), (4, 12), (12, 16)]


def test_boundaries_tile_every_bar_with_no_gap_or_overlap(chunk_analysis):
    an = chunk_analysis()
    chunks = chunking.plan_chunks(an)
    assert chunks[0].start_bar == 0
    assert chunks[-1].end_bar == chunking.total_bars(an)
    for a, b in zip(chunks, chunks[1:]):
        assert a.end_bar == b.start_bar


def test_long_section_is_split_at_bar_boundaries_under_the_ceiling(chunk_analysis):
    """A 40-bar (80s) section at max 30s must come out as several chunks, each
    under the ceiling, each edge on a bar line."""
    an = chunk_analysis(n_bars=40, sections=[
        {"name": "S0", "start_bar": 0, "end_bar": 40, "energy_pct": 0.6,
         "role_guess": "drop"}])
    chunks = chunking.plan_chunks(an, max_chunk_seconds=30, min_chunk_seconds=8,
                                  overlap_bars=1)
    assert len(chunks) > 1
    for c in chunks:
        assert c.bars * BAR_S <= 30 + 1e-9, f"{c.label()} exceeds max_chunk_seconds"
        assert float(c.start_bar).is_integer() and float(c.end_bar).is_integer()


def test_short_section_is_merged_into_its_neighbour(chunk_analysis):
    """A 2-bar (4s) stab under an 8s floor must not become its own call."""
    an = chunk_analysis(n_bars=16, sections=[
        {"name": "S0", "start_bar": 0, "end_bar": 8, "energy_pct": 0.4,
         "role_guess": "intro"},
        {"name": "S1", "start_bar": 8, "end_bar": 10, "energy_pct": 0.5,
         "role_guess": "build"},                       # 2 bars = 4s, under the floor
        {"name": "S2", "start_bar": 10, "end_bar": 16, "energy_pct": 0.9,
         "role_guess": "drop"},
    ])
    chunks = chunking.plan_chunks(an, max_chunk_seconds=30, min_chunk_seconds=8,
                                  overlap_bars=1)
    assert (8, 10) not in [(c.start_bar, c.end_bar) for c in chunks]
    assert all(c.bars * BAR_S >= 8 - 1e-9 for c in chunks)


def test_boundaries_reproduce_exactly_for_the_same_input(chunk_analysis):
    """`Accept:` chunk boundaries reproduce exactly for the same analysis input."""
    an = chunk_analysis(n_bars=37)
    a = chunking.plan_chunks(an)
    b = chunking.plan_chunks(chunk_analysis(n_bars=37))
    assert [(c.start_bar, c.end_bar, c.ctx_start_bar, c.ctx_end_bar) for c in a] == \
           [(c.start_bar, c.end_bar, c.ctx_start_bar, c.ctx_end_bar) for c in b]


def test_sections_that_do_not_reach_the_last_bar_still_tile(chunk_analysis):
    """A section list stopping short would otherwise leave bars no chunk owns —
    and an unowned bar is an uncharted bar."""
    an = chunk_analysis(n_bars=16, sections=[
        {"name": "S0", "start_bar": 0, "end_bar": 10, "energy_pct": 0.5,
         "role_guess": "intro"}])
    chunks = chunking.plan_chunks(an)
    assert chunks[-1].end_bar == 16


# --------------------------------------------------------------------------- #
# REQ-CHUNK-02 — overlap is context only
# --------------------------------------------------------------------------- #
def test_overlap_widens_context_but_not_ownership(chunk_analysis):
    an = chunk_analysis()
    c = chunking.plan_chunks(an, overlap_bars=1)[1]      # bars 4-12
    assert (c.start_bar, c.end_bar) == (4, 12)
    assert (c.ctx_start_bar, c.ctx_end_bar) == (3, 13)
    assert c.owns_bar(4) and c.owns_bar(11)
    assert not c.owns_bar(3) and not c.owns_bar(12)


def test_overlap_is_clamped_at_the_ends_of_the_song(chunk_analysis):
    an = chunk_analysis()
    chunks = chunking.plan_chunks(an, overlap_bars=2)
    assert chunks[0].ctx_start_bar == 0
    assert chunks[-1].ctx_end_bar == chunking.total_bars(an)


def test_the_scoped_inventory_includes_overlap_bars(chunk_analysis):
    """The model must SEE the transition it is not allowed to chart."""
    an = chunk_analysis()
    c = chunking.plan_chunks(an, overlap_bars=1)[1]      # owns 4-12, context 3-13
    sliced = chunking.slice_analysis(an, c)
    beats = [o["nearest_beat"] for o in sliced["onsets"]]
    assert min(beats) < c.start_bar * METER      # lead-in present
    assert max(beats) >= c.end_bar * METER       # lead-out present
    assert all(3 * METER <= b < 13 * METER for b in beats)


def test_a_note_in_an_overlap_bar_is_discarded(chunk_analysis):
    """`Accept:` a note falling in an overlap region appears exactly once in the
    assembled design, attributed to the chunk that OWNS its bar."""
    an = chunk_analysis()
    chunks = chunking.plan_chunks(an, overlap_bars=1)
    c0, c1 = chunks[0], chunks[1]                # 0-4 and 4-12
    # Both chunks reference p016 (beat 16 = bar 4), which only c1 owns.
    d0 = chunking.scope_design(
        {"notes": [{"ref": "p000"}, {"ref": "p016"}], "phrases": []}, an, c0)
    d1 = chunking.scope_design(
        {"notes": [{"ref": "p016"}, {"ref": "p020"}], "phrases": []}, an, c1)
    assert [n["ref"] for n in d0["notes"]] == ["p000"]
    assert [n["ref"] for n in d1["notes"]] == ["p016", "p020"]
    design = chunking.assemble([d0, d1], an, [c0, c1])
    refs = [n["ref"] for n in design["notes"]]
    assert refs.count("p016") == 1


# --------------------------------------------------------------------------- #
# REQ-CHUNK-03 — globals preserved, IDs stable
# --------------------------------------------------------------------------- #
def test_globals_travel_unchanged_into_every_chunk(chunk_analysis):
    an = chunk_analysis()
    for c in chunking.plan_chunks(an):
        s = chunking.slice_analysis(an, c)
        assert s["bpm"] == an["bpm"]
        assert s["offset"] == an["offset"]
        assert s["meter"] == an["meter"]
        assert s["duration_s"] == an["duration_s"]
        # The energy curve stays WHOLE on purpose: its list index is the absolute
        # bar number, and slicing it would silently renumber every bar.
        assert s["energy_curve"] == an["energy_curve"]


def test_onset_ids_are_never_renumbered_per_chunk(chunk_analysis):
    """`Accept:` a test asserts ID stability across modes."""
    an = chunk_analysis()
    whole = {o["id"]: o["nearest_beat"] for o in an["onsets"]}
    seen = {}
    for c in chunking.plan_chunks(an):
        for o in chunking.slice_analysis(an, c)["onsets"]:
            assert o["id"] in whole, f"{o['id']} is not a whole-song id"
            assert o["nearest_beat"] == whole[o["id"]]
            seen[o["id"]] = o["nearest_beat"]
    # Every onset inside the charted bars is offered to some chunk.
    assert seen == whole


def test_the_density_plan_is_filtered_not_recomputed(chunk_analysis):
    an = chunk_analysis()
    plan = an["density_plan"]
    by_bar = {e["bar"]: e for e in plan["per_bar"]}
    for c in chunking.plan_chunks(an):
        sliced = chunking.slice_analysis(an, c)["density_plan"]
        assert sliced["tiers"] == plan["tiers"]
        for e in sliced["per_bar"]:
            assert e == by_bar[e["bar"]], "a chunk's per-bar target was recomputed"
            assert c.ctx_start_bar <= e["bar"] < c.ctx_end_bar


def test_sections_are_clipped_to_the_chunk_but_keep_absolute_bars(chunk_analysis):
    an = chunk_analysis()
    c = chunking.plan_chunks(an, overlap_bars=1)[1]      # context 3-13
    for s in chunking.slice_analysis(an, c)["sections"]:
        assert s["start_bar"] >= 3 and s["end_bar"] <= 13


# --------------------------------------------------------------------------- #
# REQ-CHUNK-04 — prior-decision summary
# --------------------------------------------------------------------------- #
def _decision(index, a, b, roles, texture="steps", npb=6.0):
    return chunking.ChunkDecision(
        index=index, start_bar=a, end_bar=b, roles=tuple(roles), texture=texture,
        movement="drift_L_to_R", crossover="light", notes_per_bar=npb,
        intent="keep it moving")


def test_the_first_chunk_gets_no_prior_summary(chunk_analysis):
    an = chunk_analysis()
    c = chunking.plan_chunks(an)[0]
    assert "first chunk" in chunking.prior_summary_text([], c)


def test_chunk_n_sees_summaries_for_chunks_1_to_n_minus_1(chunk_analysis):
    """`Accept:` the prompt for chunk N contains summaries for chunks 1..N-1."""
    an = chunk_analysis()
    chunks = chunking.plan_chunks(an)
    decisions = [_decision(0, 0, 4, ["intro"]), _decision(1, 4, 12, ["drop"])]
    text = chunking.prior_summary_text(decisions, chunks[2])
    assert "bars 0-4" in text and "bars 4-12" in text
    assert "6.0 notes/bar" in text


def test_a_repeated_section_role_is_flagged_as_a_repeat():
    """`Accept:` a section labelled as a repeat of an earlier section receives
    that earlier section's summary."""
    later = chunking.Chunk(index=3, start_bar=20, end_bar=28,
                           ctx_start_bar=19, ctx_end_bar=29, roles=("drop",))
    decisions = [_decision(0, 0, 4, ["intro"]),
                 _decision(1, 4, 12, ["drop"], texture="jumpstream"),
                 _decision(2, 12, 20, ["break"])]
    text = chunking.prior_summary_text(decisions, later)
    assert "REPEAT" in text
    assert "jumpstream" in text, "the earlier drop's texture must survive"


def test_the_summary_stays_inside_its_token_budget():
    """`Accept:` summary size stays within the configured token budget."""
    decisions = [_decision(i, i * 4, i * 4 + 4, ["build"]) for i in range(60)]
    later = chunking.Chunk(index=60, start_bar=240, end_bar=248,
                           ctx_start_bar=239, ctx_end_bar=249, roles=("drop",))
    text = chunking.prior_summary_text(decisions, later, token_budget=120)
    assert len(text) // 4 <= 120 + 40, "budget blown"
    assert "omitted to stay inside the summary budget" in text


def test_the_summary_carries_decisions_only_never_inventory():
    decisions = [_decision(0, 0, 4, ["intro"])]
    later = chunking.Chunk(index=1, start_bar=4, end_bar=12, ctx_start_bar=3,
                           ctx_end_bar=13, roles=("drop",))
    text = chunking.prior_summary_text(decisions, later)
    assert "p0" not in text and "strength" not in text


# --------------------------------------------------------------------------- #
# REQ-CHUNK-05 — assembly
# --------------------------------------------------------------------------- #
def _scoped(chunk, refs, phrase_bars=None):
    a, b = phrase_bars or (chunk.start_bar, chunk.end_bar)
    return {"design_notes": f"section {chunk.index}",
            "notes": [{"ref": r, "kind": "tap"} for r in refs],
            "phrases": [{"start_bar": a, "end_bar": b, "texture": "steps",
                         "movement": "static", "crossover": "none",
                         "jump_density": "accents"}]}


def test_assembly_has_the_whole_song_shape(chunk_analysis):
    """`Accept:` an assembled chunked design is the same shape as a whole-song one."""
    an = chunk_analysis()
    chunks = chunking.plan_chunks(an)
    scoped = [_scoped(c, [f"p{c.start_bar * METER:03d}"]) for c in chunks]
    design = chunking.assemble(scoped, an, chunks)
    assert set(design) >= {"design_notes", "notes", "phrases"}
    assert isinstance(design["notes"], list) and isinstance(design["phrases"], list)
    json.dumps(design)              # must survive the same serialization


def test_assembled_notes_are_in_beat_order_with_no_duplicate_refs(chunk_analysis):
    an = chunk_analysis()
    chunks = chunking.plan_chunks(an)
    scoped = [_scoped(c, [f"p{b:03d}" for b in
                          range(c.start_bar * METER, c.end_bar * METER, 2)])
              for c in chunks]
    design = chunking.assemble(scoped, an, chunks)
    lut = chunking.onset_lookup(an)
    beats = [chunking.beat_for_ref(n["ref"], lut) for n in design["notes"]]
    assert beats == sorted(beats)
    refs = [n["ref"] for n in design["notes"]]
    assert len(refs) == len(set(refs))


def test_assembled_phrases_tile_the_charted_bars(chunk_analysis):
    """`Accept:` bar coverage is complete and non-overlapping."""
    an = chunk_analysis()
    chunks = chunking.plan_chunks(an)
    # Chunk 1 deliberately under-declares (leaves bars 8-12 with no phrase).
    scoped = [_scoped(chunks[0], ["p000"]),
              _scoped(chunks[1], ["p016"], phrase_bars=(4, 8)),
              _scoped(chunks[2], ["p048"])]
    phrases = chunking.assemble(scoped, an, chunks)["phrases"]
    assert phrases[0]["start_bar"] == chunks[0].start_bar
    assert phrases[-1]["end_bar"] == chunks[-1].end_bar
    for a, b in zip(phrases, phrases[1:]):
        assert a["end_bar"] == b["start_bar"], "gap or overlap between phrases"


def test_a_phrase_spanning_the_seam_is_clipped_to_the_owning_chunk(chunk_analysis):
    an = chunk_analysis()
    chunks = chunking.plan_chunks(an)
    c1 = chunks[1]                                   # owns 4-12
    scoped = chunking.scope_design(
        {"notes": [{"ref": "p016"}],
         "phrases": [{"start_bar": 0, "end_bar": 40, "texture": "stream"}]}, an, c1)
    assert scoped["phrases"][0]["start_bar"] == 4
    assert scoped["phrases"][0]["end_bar"] == 12


def test_filler_phrases_use_the_targets_own_vocabulary(chunk_analysis):
    """A Beat Saber filler carrying StepMania's `texture` would be rejected by the
    Beat Saber phrase validator, and vice versa."""
    an = chunk_analysis()
    chunks = chunking.plan_chunks(an)
    scoped = [{"design_notes": "", "notes": [{"ref": "p000"}], "phrases": []}]
    bs = chunking.assemble(scoped, an, chunks, filler=chunking.BEATSABER_FILLER)
    assert bs["phrases"][0]["density"] == "steady"
    assert "texture" not in bs["phrases"][0]
    sm = chunking.assemble(scoped, an, chunks, filler=chunking.STEPMANIA_FILLER)
    assert sm["phrases"][0]["texture"] == "steps"
    # A filler must never invent a density NUMBER — it would be checked against
    # the plan's band and could be rejected outright.
    assert "density" not in sm["phrases"][0]


def test_a_bad_ref_is_a_chunk_error_not_a_silent_drop(chunk_analysis):
    an = chunk_analysis()
    c = chunking.plan_chunks(an)[0]
    with pytest.raises(chunking.ChunkError):
        chunking.scope_design({"notes": [{"ref": "p999"}], "phrases": []}, an, c)
    with pytest.raises(chunking.ChunkError):
        chunking.scope_design({"notes": [{"ref": "1.5s"}], "phrases": []}, an, c)
    with pytest.raises(chunking.ChunkError):
        chunking.scope_design({"phrases": []}, an, c)


def test_grid_refs_resolve_and_survive_assembly(chunk_analysis):
    an = chunk_analysis()
    c = chunking.plan_chunks(an)[1]                  # owns bars 4-12 = beats 16-48
    scoped = chunking.scope_design(
        {"notes": [{"ref": "grid:20"}, {"ref": "grid:4"}], "phrases": []}, an, c)
    assert [n["ref"] for n in scoped["notes"]] == ["grid:20"]


# --------------------------------------------------------------------------- #
# REQ-CHUNK-08 — mode switch
# --------------------------------------------------------------------------- #
def test_auto_chunks_only_above_the_ceiling(chunk_analysis):
    assert chunking.should_chunk(chunk_analysis(n_bars=40), "auto", max_chunk_seconds=30)
    assert not chunking.should_chunk(chunk_analysis(n_bars=8), "auto", max_chunk_seconds=30)


def test_always_and_never_override_the_length(chunk_analysis):
    short = chunk_analysis(n_bars=4)
    long = chunk_analysis(n_bars=60)
    assert chunking.should_chunk(short, "always")
    assert not chunking.should_chunk(long, "never")


def test_the_shipped_default_is_chunked(chunk_analysis):
    """`Accept:` a fresh install generates in chunked mode with no configuration."""
    assert config.CHUNK_MODE == "auto"
    assert chunking.should_chunk(chunk_analysis(n_bars=60)) is True


def test_an_unknown_mode_falls_back_to_auto_rather_than_guessing(chunk_analysis):
    assert chunking.should_chunk(chunk_analysis(n_bars=60), "banana") is True
    assert chunking.should_chunk(chunk_analysis(n_bars=4), "banana") is False


# --------------------------------------------------------------------------- #
# Audio slicing
# --------------------------------------------------------------------------- #
def test_the_audio_slice_covers_the_context_window(chunk_analysis, tmp_path, click_wav):
    import soundfile as sf

    an = chunk_analysis()
    path = click_wav(bpm=BPM, dur_s=an["duration_s"])
    c = chunking.plan_chunks(an, overlap_bars=1)[1]
    out = chunking.slice_audio(path, an, c, out_dir=tmp_path)
    t0, t1 = chunking.chunk_window(an, c)
    info = sf.info(out)
    assert info.frames / info.samplerate == pytest.approx(t1 - t0, abs=0.02)


def test_slicing_is_cached_and_deterministic(chunk_analysis, tmp_path, click_wav):
    an = chunk_analysis()
    path = click_wav(bpm=BPM, dur_s=an["duration_s"])
    c = chunking.plan_chunks(an)[0]
    a = chunking.slice_audio(path, an, c, out_dir=tmp_path)
    mtime = __import__("os").stat(a).st_mtime_ns
    b = chunking.slice_audio(path, an, c, out_dir=tmp_path)
    assert a == b
    assert __import__("os").stat(b).st_mtime_ns == mtime, "re-cut an existing slice"


def test_a_chunk_beyond_the_end_of_the_audio_fails_loudly(chunk_analysis, tmp_path, click_wav):
    """A short audio file against a long analysis must not silently produce an
    empty clip the model would hear as silence."""
    an = chunk_analysis()
    path = click_wav(bpm=BPM, dur_s=4.0)             # 4s of audio, 32s of analysis
    c = chunking.plan_chunks(an)[-1]                 # bars 12-16 = 24s-32s
    with pytest.raises(chunking.ChunkError):
        chunking.slice_audio(path, an, c, out_dir=tmp_path)


# --------------------------------------------------------------------------- #
# REQ-CHUNK-07 — which chunk a failed gate is charged to
# --------------------------------------------------------------------------- #
class _Note:
    """`gate_failures_by_chunk` needs only a `.beat`; both targets' objects have one."""

    def __init__(self, beat):
        self.beat = beat


def _blame(counts, failed, gates, **kw):
    """Run the blame pass over one chunk per entry in `counts`, holding that many
    notes, against a stub battery that returns `gates` for every slice."""
    meter = 4
    chunks = [chunking.Chunk(index=i, start_bar=i * 4, end_bar=(i + 1) * 4,
                             ctx_start_bar=i * 4, ctx_end_bar=(i + 1) * 4)
              for i in range(len(counts))]
    notes = [_Note(i * 4 * meter + j * 0.25)
             for i, n in enumerate(counts) for j in range(n)]
    return chunking.gate_failures_by_chunk(
        notes, {"meter": meter}, chunks, failed,
        lambda slice_, chunk: gates, **kw)


def test_a_ratio_gate_does_not_blame_a_chunk_too_small_to_satisfy_it():
    """A share-of-notes budget is a whole-chart figure and quantizes hard on a
    slice: at a 10-20% band a 12-note chunk passes on exactly 2 holds. Blaming it
    buys a re-prompt that is arithmetically unable to succeed."""
    failing = _blame([12, 40], ["hold_share"], {"hold_share": False},
                     min_notes_by_gate={"hold_share": 20})
    assert 0 not in failing, "blamed a chunk too small to hit the band"
    assert failing[1] == ["hold_share"]


def test_a_gate_no_chunk_can_be_measured_for_is_dropped_not_smeared():
    """"Not attributable at chunk scale" is a different answer from "everyone".
    The emergent-failure fallback must not turn an ineligible gate into a
    re-prompt for every chunk — the failure belongs to the chart-level path."""
    assert _blame([8, 8], ["hold_share"], {"hold_share": False},
                  min_notes_by_gate={"hold_share": 20}) == {}


def test_an_emergent_failure_blames_every_measurable_chunk():
    """A gate the chart fails but no chunk fails locally is the arrangement's
    fault, so every chunk that could be measured hears about it."""
    assert _blame([40, 40], ["density_energy"], {"density_energy": True}) == {
        0: ["density_energy"], 1: ["density_energy"]}


def test_the_emergent_fallback_skips_chunks_too_small_to_measure():
    """A chunk with no measurable slice is no more able to fix an emergent
    failure than a local one, so it is not charged for it either."""
    assert _blame([2, 40], ["density_energy"], {"density_energy": True}) == {
        1: ["density_energy"]}


def test_a_gate_the_chunk_battery_cannot_reproduce_is_not_called_emergent():
    """The regression this exists to stop: `hold_share` is a band on a chart and
    a ceiling on a chunk, so a chart under the FLOOR is unreproducible locally by
    construction. Read as "emergent", that blamed every chunk of every
    under-holding chart and asked each to fix a section that was individually
    fine — three scoped re-prompts per run, none of which could converge."""
    assert _blame([25, 36, 51], ["hold_share"], {"hold_share": True},
                  emergent_gates={"density_energy"}) == {}


def test_naming_a_gate_emergent_still_smears_it():
    """The reading is not gone, just no longer the default for every gate."""
    assert _blame([25, 36], ["density_energy"], {"density_energy": True},
                  emergent_gates={"density_energy"}) == {
        0: ["density_energy"], 1: ["density_energy"]}


def test_a_locally_reproduced_failure_is_blamed_whatever_its_emergent_status():
    """Naming gates emergent must not change attribution that was MEASURED — it
    only decides what happens when nothing was."""
    assert _blame([25, 36], ["hold_share"], {"hold_share": False},
                  emergent_gates={"density_energy"}) == {
        0: ["hold_share"], 1: ["hold_share"]}


def test_omitting_emergent_gates_keeps_the_old_smear_for_every_gate():
    """Targets whose chunk battery matches their chart battery need no list."""
    assert _blame([25, 36], ["anything"], {"anything": True}) == {
        0: ["anything"], 1: ["anything"]}
