"""test_intent_demotion.py — a malformed DECORATION demotes; a bad REFERENCE does not.

Both adapters' resolvers already forgave one class of designer slip and hard-failed
its identical twin, three lines apart:

    if kind not in KINDS:            # unknown kind -> tap, "rather than failing
        kind = "tap"                 #                 the whole chart"
    ...
    if kind in ("hold", "roll"):     # ...but a hold with no duration -> raise,
        raise IntentError(...)       #    which fails the whole chart

Measured cost of that inconsistency: a self-hosted Gemma emitted
`{"kind": "hold"}` with no `hold_beats` on both attempts at chunk 4 of 4, and the
`IntentError` took the three already-designed chunks with it.

The line these tests pin is where forgiveness stops. A note's REF is its
identity — a hallucinated one means the model is not reading the inventory, and
that stays an error worth re-prompting. Everything hung off the ref is trimming:
get it wrong and the note survives as a plain tap/note, on the same onset, with
the loss visible to the hold_share / arc_chain_share gates.
"""
from __future__ import annotations

import pytest

from beatforge.adapters.beatsaber.resolve import IntentError as BsIntentError
from beatforge.adapters.beatsaber.resolve import resolve_intent as bs_resolve
from beatforge.adapters.stepmania.resolve import IntentError as SmIntentError
from beatforge.adapters.stepmania.resolve import resolve_intent as sm_resolve

SM_DIFF = "medium"
BS_DIFF = "hard"


def _sm(notes, analysis):
    return sm_resolve({"notes": notes, "phrases": []}, analysis, SM_DIFF)


def _bs(notes, analysis):
    return bs_resolve({"notes": notes, "phrases": []}, analysis, BS_DIFF)


# --------------------------------------------------------------------------- #
# StepMania — hold/roll without a usable duration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    {},                                  # the measured case: no hold_beats at all
    {"hold_beats": 0},
    {"hold_beats": -2},
    {"hold_beats": None},
    {"hold_beats": "4"},                 # a string, not a number
    {"hold_beats": True},                # bool is an int subclass; must not pass
])
@pytest.mark.parametrize("kind", ["hold", "roll"])
def test_a_hold_without_a_usable_duration_becomes_a_tap(chunk_analysis, kind, bad):
    an = chunk_analysis()
    out = _sm([{"ref": "p016", "kind": kind, **bad}], an)
    assert len(out) == 1, "the note must survive — it is on a real onset"
    assert out[0].kind == "tap"
    assert out[0].hold_beats is None
    assert out[0].beat == 16.0, "and it must stay exactly where the music is"


def test_a_valid_hold_is_left_alone(chunk_analysis):
    an = chunk_analysis()
    out = _sm([{"ref": "p016", "kind": "hold", "hold_beats": 2}], an)
    assert out[0].kind == "hold"
    assert out[0].hold_beats == 2.0


def test_one_bad_hold_does_not_take_the_other_notes_with_it(chunk_analysis):
    """The whole point: a chart of 200 good notes is not worth discarding over
    one missing field."""
    an = chunk_analysis()
    out = _sm([{"ref": "p016", "kind": "tap"},
               {"ref": "p017", "kind": "hold"},          # the offender
               {"ref": "p018", "kind": "tap"}], an)
    assert [n.kind for n in out] == ["tap", "tap", "tap"]
    assert len(out) == 3


def test_a_hallucinated_ref_is_still_an_error(chunk_analysis):
    """Where forgiveness stops. An id that is not in the inventory means the
    model is not reading it, and no demotion can recover the note's identity."""
    an = chunk_analysis()
    with pytest.raises(SmIntentError, match="not in onset inventory"):
        _sm([{"ref": "p999", "kind": "tap"}], an)


def test_a_raw_time_is_still_an_error(chunk_analysis):
    """SACRED-02: timing truth comes from the DSP, never from the model."""
    an = chunk_analysis()
    with pytest.raises(SmIntentError, match="raw-time"):
        _sm([{"ref": "p016", "kind": "tap", "time": 12.34}], an)


# --------------------------------------------------------------------------- #
# Beat Saber — arc without a tail, chain with bad slices
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    {},                                      # no tail_ref at all
    {"tail_ref": "p999"},                    # a tail that is not in the inventory
    {"tail_ref": "p016"},                    # a tail at/behind its own head
    {"tail_ref": "p008"},
])
def test_an_arc_without_a_usable_tail_becomes_a_note(chunk_analysis, bad):
    an = chunk_analysis()
    out = _bs([{"ref": "p016", "hand": "left", "kind": "arc", **bad}], an)
    assert len(out) == 1
    assert out[0].kind == "note"
    assert out[0].tail_beat is None
    assert out[0].beat == 16.0


def test_a_valid_arc_is_left_alone(chunk_analysis):
    an = chunk_analysis()
    out = _bs([{"ref": "p016", "hand": "left", "kind": "arc",
                "tail_ref": "p020"}], an)
    assert out[0].kind == "arc"
    assert out[0].tail_beat == 20.0


@pytest.mark.parametrize("bad", [0, 1, 9, "3", None, True])
def test_a_chain_with_bad_slices_becomes_a_note(chunk_analysis, bad):
    an = chunk_analysis()
    out = _bs([{"ref": "p016", "hand": "right", "kind": "chain",
                "slices": bad}], an)
    assert out[0].kind == "note"
    assert out[0].slices is None


def test_a_valid_chain_is_left_alone(chunk_analysis):
    an = chunk_analysis()
    out = _bs([{"ref": "p016", "hand": "right", "kind": "chain", "slices": 3}], an)
    assert out[0].kind == "chain"
    assert out[0].slices == 3


def test_a_hallucinated_head_ref_is_still_an_error(chunk_analysis):
    an = chunk_analysis()
    with pytest.raises(BsIntentError, match="not in onset inventory"):
        _bs([{"ref": "p999", "hand": "left", "kind": "note"}], an)


def test_a_bad_hand_is_still_an_error(chunk_analysis):
    """`hand` is not trimming — it decides which saber plays the note, and
    guessing would put a swing on the wrong colour."""
    an = chunk_analysis()
    with pytest.raises(BsIntentError, match="hand"):
        _bs([{"ref": "p016", "hand": "third", "kind": "note"}], an)
