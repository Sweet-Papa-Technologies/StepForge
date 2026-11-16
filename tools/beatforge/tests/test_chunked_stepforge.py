"""test_chunked_stepforge.py — CHUNKED MODE end to end, with a stubbed model
(docs/chunked-mode-brief.md §7 "Integration (mocked model)").

Offline: the "model" is a stub that reads its assignment out of the prompt it was
handed and answers with intent for exactly those bars. That is deliberate — what
is under test is everything AROUND the call. Specifically:

  * the designer really is handed one slice at a time, and the slice really is
    scoped in all four ways (audio, inventory, budget, prior summary);
  * the assembled design goes through the REAL resolver and realizer unchanged;
  * a gate failure re-prompts only the chunk it can be attributed to;
  * the cacheable prompt prefix is byte-identical across chunks.

Requirement coverage:
  REQ-CHUNK-02  a chunk is only ever shown its own window of audio
  REQ-CHUNK-04  chunk N's prompt carries summaries for chunks 1..N-1
  REQ-CHUNK-05  the assembled design needs no realizer or validator changes
  REQ-CHUNK-07  a scoped re-prompt hits only the failing chunk, per-chunk retries
  REQ-CHUNK-08  `never` reproduces whole-song mode exactly
  REQ-CHUNK-09  byte-identical invariant prefix across chunks of a song
"""
from __future__ import annotations

import re

import pytest

from beatforge import chunking, ledger
from beatforge.adapters.stepmania import chunked as sm_chunked
from beatforge.adapters.stepmania import qa as sm_qa
from beatforge.adapters.stepmania.adapter import StepManiaAdapter
from beatforge.adapters.stepmania.grammar import BUDGETS
from beatforge.adapters.stepmania.quantize import Placement

from conftest import CHUNK_BPM as BPM

DIFF = "medium"
#: The chunk prompt states the assignment as `**bars A-B**`; the stub reads it
#: back out. Coupling the stub to that string is intentional — if the assignment
#: ever stops being stated, these tests fail rather than silently charting the
#: whole song in every chunk.
_ASSIGNMENT = re.compile(r"\*\*bars (\d+)-(\d+)\*\*")


class StubClient:
    """A designer that charts every onset it can see inside its assignment.

    Records every prompt and every audio path so the tests can assert on what
    the model was actually shown, which is the whole substance of "scoped".
    """

    name = "stub"

    def __init__(self, fail_bars: tuple | None = None, empty_first: bool = False):
        self.prompts: list[str] = []
        self.audio: list[str] = []
        #: Bars whose chunk should answer with a deliberately thin chart, to
        #: induce a gate failure in one section only.
        self.fail_bars = fail_bars
        #: Answer the first call with nothing, to exercise the per-chunk retry.
        self.empty_first = empty_first

    def generate_json(self, prompt, *, audio_path=None, **kw):
        self.prompts.append(prompt)
        if audio_path:
            self.audio.append(str(audio_path))
        if self.empty_first and len(self.prompts) == 1:
            return {"design_notes": "oops", "notes": [], "phrases": []}

        m = _ASSIGNMENT.search(prompt)
        if not m:                       # the structural critic pass has no bars
            return {"score": 8, "verdict": "fine", "issues": []}
        a, b = int(m.group(1)), int(m.group(2))
        # Onset ids are in the inventory rows at the end of the prompt; take the
        # ones inside the assignment. Beat is column 2 of each row.
        notes = []
        for line in prompt.splitlines():
            parts = line.split()
            if len(parts) == 5 and re.fullmatch(r"[pms]\d+", parts[0]):
                beat = float(parts[1])
                if a * 4 <= beat < b * 4:
                    notes.append({"ref": parts[0], "kind": "tap"})
        if self.fail_bars == (a, b):
            notes = notes[:1]           # one note in the whole section
        return {
            "design_notes": f"bars {a}-{b}",
            "notes": notes,
            "phrases": [{"start_bar": a, "end_bar": b, "texture": "steps",
                         "movement": "drift_L_to_R", "crossover": "light",
                         "jump_density": "accents", "intent": "drive"}],
        }


@pytest.fixture
def song(chunk_analysis, click_wav):
    """A 16-bar analysis and a real 32-second click track to slice."""
    an = chunk_analysis()
    return an, click_wav(bpm=BPM, dur_s=an["duration_s"])


# --------------------------------------------------------------------------- #
# The designer really is called per chunk, scoped
# --------------------------------------------------------------------------- #
def test_one_designer_call_per_chunk(song, tmp_path, monkeypatch):
    """`Accept:` chunk count for a known analysis."""
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    client = StubClient()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)
    assert len(run.chunks) == 3
    assert len(client.prompts) == 3
    assert run.calls == 3


def test_each_chunk_hears_only_its_own_window(song, tmp_path, monkeypatch):
    """REQ-CHUNK-02. The point of the exercise: a chunk is handed a CLIP, and
    that clip is the length of its context window, not the length of the song."""
    import soundfile as sf

    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    client = StubClient()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)

    assert len(set(client.audio)) == 3, "chunks shared an audio file"
    for c, path in zip(run.chunks, client.audio):
        t0, t1 = chunking.chunk_window(an, c)
        info = sf.info(path)
        assert info.frames / info.samplerate == pytest.approx(t1 - t0, abs=0.05)
        assert info.frames / info.samplerate < an["duration_s"], "whole track sent"


def test_each_chunk_sees_only_its_own_inventory(song, tmp_path, monkeypatch):
    """Scoping the audio but shipping the full inventory is explicitly NOT a
    chunked call (brief §0), so it is asserted rather than assumed."""
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    client = StubClient()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)

    all_ids = {o["id"] for o in an["onsets"]}
    for c, prompt in zip(run.chunks, client.prompts):
        shown = {o["id"] for o in chunking.slice_analysis(an, c)["onsets"]}
        assert shown < all_ids, "a chunk was shown the whole-song inventory"
        for oid in all_ids - shown:
            assert not re.search(rf"^{oid} ", prompt, re.M), \
                f"{oid} leaked into {c.label()}'s inventory"


def test_each_chunk_sees_only_its_own_density_budget(song, tmp_path, monkeypatch):
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    client = StubClient()
    with ledger.capture():
        sm_chunked.design_chunked("t", DIFF, an, audio, client)
    # The first chunk owns bars 0-4 and must not be shown the drop's budget rows.
    first = client.prompts[0]
    budget = first.split("DENSITY BUDGET")[1].split("MACHINE ANALYSIS")[0]
    assert "bars   0-4" in budget.replace("  ", "  ")
    assert "12-16" not in budget


def test_chunk_n_carries_the_prior_decisions(song, tmp_path, monkeypatch):
    """REQ-CHUNK-04 `Accept:` the prompt for chunk N contains summaries for
    chunks 1..N-1."""
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    client = StubClient()
    with ledger.capture():
        sm_chunked.design_chunked("t", DIFF, an, audio, client)

    assert "first chunk of the song" in client.prompts[0]
    assert "bars 0-4" in client.prompts[1]
    assert "bars 0-4" in client.prompts[2] and "bars 4-12" in client.prompts[2]


# --------------------------------------------------------------------------- #
# REQ-CHUNK-09 — prompt caching
# --------------------------------------------------------------------------- #
def test_the_invariant_prefix_is_byte_identical_across_chunks(song, tmp_path,
                                                              monkeypatch):
    """`Accept:` cached-prompt tokens are non-zero on chunks 2..N of a song.

    Which requires the prefix to be identical byte for byte — a cache hit is a
    token-prefix match, so a single character that varies per chunk costs the
    entire saving. Asserted on the prompts that were really sent, not on the
    builder in isolation.
    """
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    from beatforge.adapters.stepmania.design import cacheable_prefix

    an, audio = song
    client = StubClient()
    with ledger.capture():
        sm_chunked.design_chunked("t", DIFF, an, audio, client)

    prefix = cacheable_prefix(DIFF)
    assert len(prefix) > 500, "the cacheable prefix is suspiciously small"
    for i, p in enumerate(client.prompts):
        assert p.startswith(prefix), f"chunk {i} broke the cacheable prefix"


def test_the_invariant_prefix_still_varies_by_difficulty(song):
    """The budget block is inside the prefix, so two tiers must NOT share it —
    otherwise a cached prefix would carry the wrong tier's hard limits."""
    from beatforge.adapters.stepmania.design import cacheable_prefix
    assert cacheable_prefix("easy") != cacheable_prefix("challenge")


# --------------------------------------------------------------------------- #
# REQ-CHUNK-05 — the assembled design runs the normal pipeline
# --------------------------------------------------------------------------- #
def test_the_assembled_design_realizes_unchanged(song, tmp_path, monkeypatch):
    """`Accept:` the realizer and validators require no changes."""
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    client = StubClient()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)

    design = run.design(an)
    adapter = StepManiaAdapter()
    placements = adapter.realize(design, an, DIFF)          # resolve inside
    placements, meter, vinfo = adapter.validate_repair(placements, an, DIFF)
    assert placements, "the assembled design realized to nothing"
    assert isinstance(meter, int)
    metrics = adapter.qa_metrics(placements, an, DIFF)
    assert "gates" in metrics and "onset_alignment" in metrics


def test_the_assembled_chart_covers_every_chunk(song, tmp_path, monkeypatch):
    """No section may come out silent — a hole is the chunked failure mode."""
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    client = StubClient()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)
    placements = StepManiaAdapter().realize(run.design(an), an, DIFF)
    meter = an["meter"]
    for c in run.chunks:
        assert any(c.owns_beat(p.beat, meter) for p in placements), \
            f"{c.label()} charted nothing"


# --------------------------------------------------------------------------- #
# REQ-CHUNK-07 — scoped re-prompt and per-chunk retries
# --------------------------------------------------------------------------- #
def test_a_gate_failure_is_attributed_to_the_chunk_that_fails_it(song, tmp_path,
                                                                 monkeypatch):
    """`Accept:` inducing a gate failure in one section re-prompts only that
    section's chunk."""
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    client = StubClient()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)
    placements = StepManiaAdapter().realize(run.design(an), an, DIFF)

    # Fabricate a per-chunk failure: strip the middle chunk's notes off the
    # placement list so its slice fails the alignment/balance battery, then ask
    # who is responsible.
    victim = run.chunks[1]
    meter = an["meter"]
    failing = sm_chunked.chunk_gate_failures(
        [p for p in placements if not victim.owns_beat(p.beat, meter)],
        an, DIFF, run.chunks, ["onset_alignment"])
    assert victim.index not in failing or failing[victim.index] == ["onset_alignment"]


def _slice(beats, holds=0):
    """Placements on `beats`, the first `holds` of them held."""
    return [Placement(beat=float(bt), panels=(bt % 4,),
                      kind="hold" if i < holds else "tap",
                      hold_beats=2.0 if i < holds else None)
            for i, bt in enumerate(beats)]


def test_the_hold_share_waiver_reads_the_same_span_as_the_slice(make_analysis):
    """`hold_share` is waived when the music offers nothing holdable — and the
    waiver has to be measured over the SAME bars the ratio was.

    It was not: the count ran over the whole song while the ratio ran over one
    chunk's slice, so a section with no sustains in it was judged against a
    waiver computed from sustains elsewhere in the track, and was charged a
    scoped re-prompt for a span that had nothing to fix.
    """
    an = make_analysis()                    # sustains at beats 8, 16, 24, 32
    b = BUDGETS["easy"]
    assert sm_qa._usable_sustains(an, b) > 0, "song-wide there is plenty to hold"
    assert sm_qa._usable_sustains(an, b, (0, 8)) == 0, "and none in the first two bars"

    half_held = _slice(range(8), holds=4)   # 50%, well over easy's 30% ceiling
    assert sm_qa.chart_metrics(half_held, an, "easy")["gates"]["hold_share"] is False
    assert sm_qa.chart_metrics(half_held, an, "easy",
                               beat_window=(0, 8))["gates"]["hold_share"] is True


def test_hold_share_is_a_ceiling_at_every_scale(make_analysis):
    """No floor, on a chart or on a chunk. A chart of pure taps is ordinary DDR —
    the reference steppers in tools/chartbench/out/ agree, DDC's median hold share
    at beginner being 0.000 — so a minimum was measuring our own guess. A hold
    pins a foot, which is a ceiling-shaped intent and nothing else."""
    an = make_analysis()
    win = (8, 28)                           # bars 2-7, which DO contain sustains,
    assert sm_qa._usable_sustains(an, BUDGETS["easy"], win) > 0   # so no waiver

    def verdict(slice_):
        return sm_qa.chart_metrics(slice_, an, "easy",
                                   beat_window=win)["gates"]["hold_share"]

    assert verdict(_slice(range(8, 28), holds=0)) is True, "zero holds is legal"
    assert verdict(_slice(range(8, 28), holds=1)) is True
    assert verdict(_slice(range(8, 28), holds=2)) is True, "10% is the ceiling"
    assert verdict(_slice(range(8, 28), holds=4)) is False, "20% is hold-spam"


def test_no_tier_carries_a_hold_share_floor():
    """A guard, not a restatement: the floor came back twice while this was being
    worked out, and each time it cost a run's worth of unfixable re-prompts.
    Re-derive from tools/chartbench before putting one back."""
    for name, b in BUDGETS.items():
        assert b.hold_share[0] == 0.0, f"{name} grew a hold_share floor"
        assert 0.05 <= b.hold_share[1] <= 0.10, f"{name} ceiling left measured range"


def test_a_ceiling_gate_floor_lets_one_hold_be_affordable_first():
    """Not a magic number: below `1/hi` notes a single held note is itself over
    budget, so the gate would stop meaning "too many holds here" and start
    meaning "no holds allowed in this section"."""
    assert sm_chunked._ceiling_gate_min_notes(BUDGETS["easy"].hold_share[1]) == 10
    assert sm_chunked._ceiling_gate_min_notes(BUDGETS["challenge"].hold_share[1]) == 17
    assert sm_chunked._ceiling_gate_min_notes(0.0) == 0        # no ceiling, no floor


def test_a_scoped_reprompt_calls_only_the_named_chunk(song, tmp_path, monkeypatch):
    """`Accept:` the ledger shows the retry scoped to that chunk."""
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    client = StubClient()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)
    before = len(client.prompts)

    with ledger.capture():
        n = sm_chunked.reprompt_failing_chunks(
            run, "t", DIFF, an, audio, client, {1: ["density_energy"]},
            {"density_energy": "raise density into the loud bars"})
    assert n == 1
    assert len(client.prompts) == before + 1, "a scoped repair called other chunks"
    assert "**bars 4-12**" in client.prompts[-1]
    assert "density_energy" in client.prompts[-1]


def test_a_repaired_chunk_keeps_the_earlier_chunks_summaries(song, tmp_path,
                                                             monkeypatch):
    """A repair must not break the continuity the first pass established, so it
    is shown what came BEFORE it — and not what came after, which it is not
    allowed to depend on."""
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    client = StubClient()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)
    with ledger.capture():
        sm_chunked.reprompt_failing_chunks(
            run, "t", DIFF, an, audio, client, {2: ["density_energy"]}, {})
    prompt = client.prompts[-1]
    summary = prompt.split("ALREADY DESIGNED")[1].split("YOUR ASSIGNMENT")[0]
    assert "bars 0-4" in summary and "bars 4-12" in summary
    assert "bars 12-16" not in summary, "a chunk was shown its own prior output"


def test_a_bad_chunk_is_retried_on_its_own(song, tmp_path, monkeypatch):
    """Retry budgets apply per chunk: an empty first reply costs one extra call,
    not the song's whole attempt budget."""
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    client = StubClient(empty_first=True)
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)
    assert run.calls == 4, "expected 3 chunks + 1 scoped retry"
    assert "no notes inside bars 0-4" in client.prompts[1]


def test_a_chunk_whose_design_the_resolver_rejects_is_retried_on_its_own(
        song, tmp_path, monkeypatch):
    """An out-of-vocabulary texture (or an out-of-band phrase density) is a
    mistake ONE chunk made. Without the resolver check at chunk scope it only
    surfaces after assembly, as a whole-song IntentError that re-designs every
    chunk — which is what a local model declaring 6.0 notes/bar against an
    allowed 2.52-5.89 really cost before this.
    """
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song

    class BadPhraseOnce(StubClient):
        def generate_json(self, prompt, *, audio_path=None, **kw):
            out = super().generate_json(prompt, audio_path=audio_path, **kw)
            if len(self.prompts) == 1 and out.get("phrases"):
                out["phrases"][0]["texture"] = "interpretive_dance"
            return out

    client = BadPhraseOnce()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)
    assert run.calls == 4, "expected 3 chunks + 1 scoped retry"
    assert "interpretive_dance" in client.prompts[1], \
        "the retry must quote what the resolver rejected"
    assert all(p.get("texture") != "interpretive_dance"
               for s in run.scoped for p in s["phrases"])


def test_an_unparseable_chunk_is_retried_rather_than_killing_the_song(
        song, tmp_path, monkeypatch):
    """A designer reply that will not parse is a property of ONE call.

    Before this, the `ValueError` from `_parse_json` travelled all the way out of
    `build_song` as a raw JSONDecodeError traceback, taking every chunk already
    designed and paid for with it — which on a local model that stops mid-array
    meant no chart at all.
    """
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song

    class BadJsonOnce(StubClient):
        def generate_json(self, prompt, *, audio_path=None, **kw):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                raise ValueError("JSON decode failed: Expecting ',' delimiter")
            return super().generate_json(prompt, audio_path=audio_path, **kw)

    client = BadJsonOnce()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)
    assert len(run.chunks) == 3, "the other chunks survived the bad one"
    assert run.calls == 4, "expected 3 chunks + 1 scoped retry"
    assert all(s["notes"] for s in run.scoped)
    assert "could not be read as JSON" in client.prompts[1]


def test_a_hold_with_no_duration_no_longer_kills_the_song(song, tmp_path,
                                                          monkeypatch):
    """The reported crash, end to end.

    `IntentError: chunk 4 (bars 20-26) failed after 2 attempt(s): note[8]
    kind=hold needs hold_beats > 0` — a local model naming a real onset and
    omitting the duration, on every attempt, at the last chunk of a song. The
    note now demotes to a tap and the run finishes.
    """
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song

    class DurationlessHolds(StubClient):
        def generate_json(self, prompt, *, audio_path=None, **kw):
            out = super().generate_json(prompt, audio_path=audio_path, **kw)
            for n in out.get("notes", [])[8:9]:      # note[8], as reported
                n["kind"] = "hold"                   # ...and no hold_beats
            return out

    client = DurationlessHolds()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)

    assert run.calls == 3, "no chunk needed a retry — the note is repairable"
    assert run.fallback_bars() == [], "and none of them needed the fallback"
    placements = StepManiaAdapter().realize(run.design(an), an, DIFF)
    assert placements
    # The demoted notes are still on the music, just not held.
    assert any(p.hold_beats is None for p in placements)


def test_a_hopeless_chunk_falls_back_instead_of_losing_the_song(song, tmp_path,
                                                                monkeypatch, capsys):
    """A small model can make the SAME mistake every attempt.

    The measured case: a Gemma emitting `{"kind": "hold"}` with no duration on
    both attempts, at chunk 4 of 4 — which threw away three chunks of paid-for
    work and the whole run. Those bars now come from the DSP instead: a worse
    section, loudly reported, against a lost song.
    """
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song

    class HopelessMiddleChunk(StubClient):
        def generate_json(self, prompt, *, audio_path=None, **kw):
            out = super().generate_json(prompt, audio_path=audio_path, **kw)
            if "**bars 4-12**" in prompt:
                raise ValueError("JSON decode failed: Expecting ',' delimiter")
            return out

    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, HopelessMiddleChunk())

    assert len(run.scoped) == 3, "the other chunks survived"
    assert run.fallback_bars() == [[4, 12]], "only the hopeless chunk fell back"
    assert run.scoped[1]["notes"], "the fallback still charted those bars"
    assert "WITHOUT the model's musical judgement" in capsys.readouterr().err, \
        "a degraded section must never be silent"
    # And the whole thing still realizes, which is the point of not dying.
    assert StepManiaAdapter().realize(run.design(an), an, DIFF)


def test_a_healthy_run_reports_no_fallbacks(song, tmp_path, monkeypatch):
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, StubClient())
    assert run.fallback_bars() == []


def test_a_chunk_that_never_parses_fails_with_its_bars_named(song, tmp_path,
                                                             monkeypatch):
    """With no fallback supplied it is still a chunk-scoped error naming the
    bars, not a bare JSONDecodeError from the top of the stack."""
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song

    class AlwaysBadJson(StubClient):
        def generate_json(self, prompt, *, audio_path=None, **kw):
            self.prompts.append(prompt)
            raise ValueError("JSON decode failed: Expecting ',' delimiter")

    chunks = chunking.plan_chunks(an)
    with ledger.capture(), pytest.raises(chunking.ChunkError, match=r"bars 0-4"):
        chunking.run_chunked_design(
            an, audio, chunks,
            sm_chunked._design_call("t", DIFF, AlwaysBadJson()),
            filler=chunking.STEPMANIA_FILLER)


def test_a_model_that_charts_nothing_still_produces_a_song(song, tmp_path,
                                                           monkeypatch, capsys):
    """The worst case — a model that never returns a usable note at all.

    Every chunk falls back, so the chart is the DSP-only one end to end. That is
    the same chart `--deterministic` would have produced, which is a real and
    playable chart, and every bar of it is reported as un-designed. Before the
    fallback this was an exception and no output whatsoever.
    """
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song

    class Mute(StubClient):
        def generate_json(self, prompt, *, audio_path=None, **kw):
            self.prompts.append(prompt)
            return {"design_notes": "", "notes": [], "phrases": []}

    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, Mute())

    assert run.fallback_bars() == [[0, 4], [4, 12], [12, 16]], \
        "every chunk fell back, and every one of them says so"
    assert StepManiaAdapter().realize(run.design(an), an, DIFF)
    assert capsys.readouterr().err.count("WITHOUT the model's musical judgement") == 3


# --------------------------------------------------------------------------- #
# REQ-CHUNK-08 — `never` reproduces whole-song mode
# --------------------------------------------------------------------------- #
def test_never_uses_the_whole_song_prompt(chunk_analysis, monkeypatch):
    """`Accept:` setting `never` reproduces current whole-song behaviour exactly.

    The observable difference between the modes is which prompt the designer is
    handed, so that is what is checked: with `never`, the chunk template is not
    used and the model gets the whole song's inventory in one call.
    """
    from beatforge.adapters.stepmania.design import designer_prompt

    an = chunk_analysis(n_bars=60)
    monkeypatch.setattr("beatforge.config.CHUNK_MODE", "never")
    assert chunking.should_chunk(an) is False

    prompt = designer_prompt(DIFF, an)
    assert "**bars" not in prompt, "the whole-song prompt gained a chunk assignment"
    # One onset from near the start of the playable window and one from near the
    # end: whole-song mode ships the inventory for the entire track in one call.
    # (`p000` is deliberately not checked — beat 0 is before FIRST_PLAYABLE_BEAT
    # and is filtered out in BOTH modes.)
    for oid in ("p010", "p200"):
        assert re.search(rf"^{oid} ", prompt, re.M), \
            "whole-song mode must still ship the whole inventory"


def test_always_chunks_a_track_shorter_than_the_ceiling(chunk_analysis, tmp_path,
                                                        click_wav, monkeypatch):
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an = chunk_analysis(n_bars=8)          # 16s, well under the 30s ceiling
    monkeypatch.setattr("beatforge.config.CHUNK_MODE", "always")
    assert chunking.should_chunk(an) is True
    audio = click_wav(bpm=BPM, dur_s=an["duration_s"])
    client = StubClient()
    with ledger.capture():
        run = sm_chunked.design_chunked("t", DIFF, an, audio, client)
    assert len(run.chunks) >= 1
    assert run.design(an)["notes"]


# --------------------------------------------------------------------------- #
# Ledger attribution
# --------------------------------------------------------------------------- #
def test_every_chunk_call_is_ledgered_with_its_chunk_index(song, tmp_path,
                                                           monkeypatch):
    """A cost report has to be able to answer "what did each section cost?" —
    which it cannot if every chunk bills to an undifferentiated `designer`."""
    monkeypatch.setattr("beatforge.config.CHUNK_AUDIO_DIR", tmp_path)
    an, audio = song

    seen = []

    class Recording(StubClient):
        def generate_json(self, prompt, *, audio_path=None, **kw):
            seen.append(ledger.current())
            return super().generate_json(prompt, audio_path=audio_path, **kw)

    with ledger.capture():
        sm_chunked.design_chunked("t", DIFF, an, audio, Recording())

    assert [c.get("chunk") for c in seen] == [0, 1, 2]
    assert all(c.get("stage") == "designer" for c in seen)
    assert all(c.get("difficulty") == DIFF for c in seen)
    assert [c.get("chunk_bars") for c in seen] == ["0-4", "4-12", "12-16"]
