"""inventory.py — how the onset menu is serialized into a designer prompt.

The onset inventory is 84-95% of every designer prompt (`docs/cost-reduction-plan.md`
§"Prompt composition"), and as JSON objects most of those bytes are the same nine
key names repeated once per onset:

    {"id": "p001", "beat": 0.0, "strength": 0.1159, "bands": {"low": 0.005,
     "mid": 0.897, "high": 0.097}, "sustain": false, "sustain_beats": 0.0}

~145 bytes, of which ~60 are keys and punctuation. This module emits the same
onsets, the same ids and the same fields as fixed-column rows instead — measured
at -75% on the largest inventory in the repo (158,320 B -> 40,188 B for the 1,093
onsets of `challenge` on `Some Will Say`).

This is a TRANSPORT change, not a content change. Every onset that was offered is
still offered — nothing is filtered here — because trimming the menu would touch
SACRED-02 (timing truth comes from the DSP, and the designer can only pick from
what it is shown). The one thing that IS dropped is spurious precision: the source
analysis carries `"strength": 0.1159`, four decimal places the designer cannot act
on, so values are rounded on the way out.

Shared by all three designer surfaces (core, StepMania, Beat Saber) so the row
format and the legend can never drift apart between them.
"""
from __future__ import annotations

# The one-line header that rides with the rows wherever they are used. The fuller
# prose legend lives in each `designer.md`, where it is static and therefore part
# of the cacheable prompt prefix.
HEADER = "# id beat strength low/mid/high sustain_beats"


def rows(onsets: list[dict]) -> str:
    """Serialize an inventory of compacted onsets as whitespace-delimited rows.

    Expects the shape `_compact_analysis()` builds: `id`, `beat`, `strength`,
    `bands{low,mid,high}` and `sustain_beats`. Anything else on the onset (e.g.
    the core designer's tier-usable `sustain` flag) is deliberately NOT emitted —
    callers that need it carry it in the prompt text, where it is stated once
    instead of once per onset.
    """
    out = [HEADER]
    for o in onsets:
        b = o.get("bands") or {}
        out.append(
            f"{o['id']} {float(o['beat']):.3f} {float(o['strength']):.2f} "
            f"{float(b.get('low', 0.0)):.2f}/{float(b.get('mid', 0.0)):.2f}/"
            f"{float(b.get('high', 0.0)):.2f} "
            f"{float(o.get('sustain_beats', 0.0)):.1f}")
    return "\n".join(out)


def legend(sustain_note: str = "") -> str:
    """The static column legend for a `designer.md`. Kept here beside the writer
    so a column can never be renamed in one place and not the other."""
    lines = [
        "Each line is one onset, whitespace-delimited, in the column order:",
        "",
        "    " + HEADER,
        "    p001 0.000 0.12 0.01/0.90/0.10 0.0",
        "",
        "  id             the onset id you reference (`p001`, `m042`, `s003`)",
        "  beat           position on the beat grid, in beats",
        "  strength       onset salience, 0-1",
        "  low/mid/high   spectral band split of that onset, each 0-1",
        "  sustain_beats  length of the harmonic plateau in beats; 0 means this",
        "                 onset is not a sustain candidate and can never be a hold",
    ]
    if sustain_note:
        lines.append("                 " + sustain_note)
    return "\n".join(lines)
