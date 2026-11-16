"""test_json_salvage.py — recovering a designer reply that stopped part-way.

**The failure this exists for**, measured against a self-hosted Gemma 4 12B on a
chunk prompt: `finish_reason: "stop"` after 1,664 completion tokens, mid-way
through the `notes` array, the reply ending on a bare `{"ref": "`. The output cap
was 16,384 and nowhere near reached — the model simply emitted its end token
early, which is what small models do on a long repetitive array. So no
`max_tokens` change fixes it, and a plain "your JSON was invalid" re-prompt tends
to stop in the same place.

What the model HAD written was valid and real: onset ids taken from the inventory
it was shown. The salvage keeps those and drops the half-written element. The
rule that makes it safe is that it only ever DELETES — it never inserts a value,
repairs a delimiter, or completes a partial object. So a salvaged design can be
short, and cannot be wrong.
"""
from __future__ import annotations

import json

import pytest

from beatforge.vertex import (_parse_json, json_retry_note, looks_truncated,
                              salvage_truncated_json)

#: The shape the local model really produced, cut where it really cut.
TRUNCATED = """{
  "design_notes": "driving four-on-the-floor",
  "notes": [
    {"ref": "p271", "kind": "tap"},
    {"ref": "m304", "kind": "tap"},
    {"ref": \""""


# --------------------------------------------------------------------------- #
# Detecting the two different failures
# --------------------------------------------------------------------------- #
def test_a_cut_off_reply_is_recognised_as_truncated():
    assert looks_truncated(TRUNCATED)
    assert looks_truncated('{"notes": [{"ref": "p1"}')
    assert looks_truncated('{"a": "unterminated')


def test_a_complete_reply_is_not_called_truncated():
    assert not looks_truncated('{"notes": []}')
    assert not looks_truncated('{"a": "has } and ] inside a string"}')
    assert not looks_truncated('  {"a": 1}  ')


def test_malformed_is_not_mistaken_for_truncated():
    """A real syntax error with balanced brackets must keep the syntax advice —
    telling that model to "write less" would thin the chart for no reason."""
    malformed = '{"notes": [{"ref": "p1" "kind": "tap"}]}'   # missing comma
    assert not looks_truncated(malformed)
    with pytest.raises(ValueError, match="JSON decode failed"):
        _parse_json(malformed)


# --------------------------------------------------------------------------- #
# Salvage
# --------------------------------------------------------------------------- #
def test_the_complete_elements_survive_and_the_partial_one_does_not():
    obj = salvage_truncated_json(TRUNCATED)
    assert obj is not None
    assert [n["ref"] for n in obj["notes"]] == ["p271", "m304"]
    assert obj["design_notes"] == "driving four-on-the-floor"


def test_salvage_never_completes_a_half_written_element():
    """A note cut off after its `ref` is DROPPED, not resurrected without its
    `kind` — inventing the missing half is how a salvage starts being wrong."""
    obj = salvage_truncated_json(
        '{"notes": [{"ref": "p1", "kind": "tap"}, {"ref": "p2"')
    assert obj is not None
    assert [n["ref"] for n in obj["notes"]] == ["p1"]
    assert all("kind" in n for n in obj["notes"])


def test_a_reply_with_nothing_complete_is_not_salvaged():
    """Stopping inside the very first value leaves nothing worth keeping, and
    guessing at it would be fabrication."""
    assert salvage_truncated_json('{"design_notes": "the intro should') is None
    assert salvage_truncated_json('{"notes": [') is None


def test_salvage_closes_every_open_container():
    obj = salvage_truncated_json(
        '{"a": {"b": [{"c": 1}, {"c": 2}, {"c": ')
    assert obj == {"a": {"b": [{"c": 1}, {"c": 2}]}}


def test_salvage_is_not_confused_by_brackets_inside_strings():
    obj = salvage_truncated_json(
        '{"notes": [{"ref": "p1", "why": "the [drop} hits"}, {"ref": "p2')
    assert obj is not None
    assert [n["ref"] for n in obj["notes"]] == ["p1"]
    assert obj["notes"][0]["why"] == "the [drop} hits"


def test_salvage_is_not_confused_by_escaped_quotes():
    obj = salvage_truncated_json(
        r'{"notes": [{"ref": "p1", "why": "a \"big\" hit"}, {"ref": "p2')
    assert obj is not None
    assert [n["ref"] for n in obj["notes"]] == ["p1"]


def test_salvage_declines_rather_than_guessing_on_junk():
    assert salvage_truncated_json("not json at all") is None
    assert salvage_truncated_json('}}]}') is None


# --------------------------------------------------------------------------- #
# Through the real parser
# --------------------------------------------------------------------------- #
def test_the_parser_salvages_instead_of_raising(capsys):
    obj = _parse_json(TRUNCATED)
    assert len(obj["notes"]) == 2
    assert obj["_truncated"] is True, "a salvaged reply must be identifiable"
    assert "stopped mid-reply" in capsys.readouterr().err, "must not be silent"


def test_a_truncated_reply_wrapped_in_prose_is_still_salvaged():
    """Local models prepend chatter, and a truncated reply has no closing brace
    for the usual outermost-{...} extraction to find."""
    obj = _parse_json("Sure! Here is the chart:\n" + TRUNCATED)
    assert len(obj["notes"]) == 2


def test_a_complete_reply_is_untouched_and_unmarked():
    obj = _parse_json('{"notes": [{"ref": "p1", "kind": "tap"}], "phrases": []}')
    assert "_truncated" not in obj
    assert len(obj["notes"]) == 1


def test_a_salvaged_design_is_still_valid_designer_output():
    """The salvaged object has to survive everything downstream, which for a
    chunk means `scope_design` — so `notes` must be a real list of real refs."""
    obj = _parse_json(TRUNCATED)
    assert isinstance(obj["notes"], list)
    assert all(isinstance(n, dict) and "ref" in n for n in obj["notes"])
    json.dumps(obj)          # and it must still serialize into the design dump


# --------------------------------------------------------------------------- #
# The retry note
# --------------------------------------------------------------------------- #
def test_a_truncated_reply_is_asked_for_less_not_for_better_syntax():
    note = json_retry_note(TRUNCATED, ValueError("boom"))
    assert "STOPPED PART-WAY THROUGH" in note
    assert "SHORTER" in note


def test_a_malformed_reply_is_asked_to_fix_its_syntax():
    note = json_retry_note('{"a": 1 "b": 2}', ValueError("expecting ,"))
    assert "could not be parsed as JSON" in note
    assert "SHORTER" not in note
