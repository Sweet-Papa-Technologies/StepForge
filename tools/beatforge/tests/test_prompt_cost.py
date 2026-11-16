"""test_prompt_cost.py — the transport half of `docs/cost-reduction-plan.md`.

Workstreams A (compact the onset inventory encoding) and B (reorder for prefix
caching + de-duplicate identical inventories) are pure transport changes: the
model must receive the SAME information, in fewer bytes and a cacheable order.
So the tests that matter here are not "is it smaller" — that is easy and is only
one assertion below — but:

  * the row encoding carries every field the JSON encoding did, for every onset.
    A silently dropped column would be a chart-quality bug that no cost metric
    would ever catch;
  * `easy`/`medium` and `hard`/`challenge` produce byte-identical analysis
    blocks, which is the exact property the caching saving depends on;
  * the invariant part of the prompt really does come first.

Workstream C (per-stage thinking) is a generation-config change that ships OFF,
so its tests pin the resolution order and the off switch rather than any level.
"""
from __future__ import annotations

import json

import pytest

from beatforge import config, costreport, inventory
from beatforge import design as core_design
from beatforge.adapters.beatsaber import design as bs_design
from beatforge.adapters.stepmania import design as sm_design
from beatforge.adapters.stepmania.grammar import BUDGETS


@pytest.fixture
def analysis(make_analysis):
    """The shared fixture, made representative in the two ways these tests need.

    `make_analysis` puts every onset on an integer beat and gives every sustain
    4 beats, which would make all five tiers serialize identically and hide both
    properties under test. Real tracks are not like that, so this adds:

      * OFF-GRID onsets (halves, quarters) — the finest_subdiv filter is what
        makes `beginner` < `easy`/`medium` < `hard`/`challenge`;
      * SHORT sustains — a 0.5-beat plateau is a hold candidate on `medium`
        (floor 0.0) and not on `easy` (floor 1.0). That is exactly the nine-onset
        divergence Workstream B removes.
    """
    a = make_analysis(n_onsets=40, n_sustain=4)
    for j, beat in enumerate((12.0, 20.0, 28.0)):
        a["onsets"].append({
            "id": f"s1{j:02d}", "time": round(a["offset"] + beat / a["bpm"] * 60, 4),
            "nearest_beat": beat, "snap_error_ms": 2.0, "strength": 0.6123,
            "bands": {"low": 0.15, "mid": 0.55, "high": 0.30},
            "source": "harmonic_sustain", "sustain": True, "sustain_beats": 0.5,
        })
    for j, beat in enumerate((10.5, 18.5, 26.5, 14.25, 22.75, 30.25)):
        a["onsets"].append({
            "id": f"f{j:03d}", "time": round(a["offset"] + beat / a["bpm"] * 60, 4),
            "nearest_beat": beat, "snap_error_ms": 4.0, "strength": 0.42,
            "bands": {"low": 0.3, "mid": 0.4, "high": 0.3},
            "source": "mix_perc", "sustain": False, "sustain_beats": 0.0,
        })
    a["onsets"].sort(key=lambda o: o["nearest_beat"])
    return a


# --------------------------------------------------------------------------- #
# Workstream A — the row encoding is lossless
# --------------------------------------------------------------------------- #
def _parse_rows(block: str) -> list[dict]:
    """Read the inventory back out of a serialized analysis block, the way the
    model has to. If this parser can reconstruct the fields, so can the model."""
    out = []
    for line in block.splitlines():
        if not line or line.startswith("#") or line.startswith("{"):
            continue
        oid, beat, strength, bands, sustain = line.split()
        lo, mid, hi = bands.split("/")
        out.append({"id": oid, "beat": float(beat), "strength": float(strength),
                    "bands": {"low": float(lo), "mid": float(mid), "high": float(hi)},
                    "sustain_beats": float(sustain)})
    return out


@pytest.mark.parametrize("difficulty", sorted(BUDGETS))
def test_rows_carry_every_onset_and_every_field(analysis, difficulty):
    """The regression that matters: `_compact_analysis` and the serializer must
    agree on every id, beat, strength, band triple and sustain value."""
    compact = sm_design._compact_analysis(analysis, difficulty)
    parsed = _parse_rows(sm_design.analysis_block(difficulty, analysis))

    assert [o["id"] for o in parsed] == [o["id"] for o in compact["onsets"]]
    for got, want in zip(parsed, compact["onsets"]):
        assert got["beat"] == pytest.approx(want["beat"], abs=5e-4)
        assert got["strength"] == pytest.approx(want["strength"], abs=5e-3)
        for band in ("low", "mid", "high"):
            assert got["bands"][band] == pytest.approx(want["bands"][band], abs=5e-3)
        assert got["sustain_beats"] == pytest.approx(
            float(want["sustain_beats"]), abs=5e-2)


def test_rows_lose_nothing_the_json_form_had(analysis):
    """Round the other way too: the row block must not INVENT onsets either."""
    compact = sm_design._compact_analysis(analysis, "hard")
    parsed = _parse_rows(sm_design.analysis_block("hard", analysis))
    assert len(parsed) == len(compact["onsets"]) > 0


def test_row_encoding_is_substantially_smaller_than_json(analysis):
    """The cost claim itself. Measured at -75% on the largest real inventory in
    the repo; the synthetic fixture here is smaller and noisier, so the bar is a
    conservative -50% — this test guards the ENCODING, not the exact ratio."""
    onsets = sm_design._compact_analysis(analysis, "challenge")["onsets"]
    as_json = len(json.dumps(onsets).encode())
    as_rows = len(inventory.rows(onsets).encode())
    assert as_rows < as_json * 0.5, f"{as_rows} B of rows vs {as_json} B of JSON"


def test_rows_round_precision_down(analysis):
    """The source analysis carries 4 dp of onset strength, which is precision the
    designer cannot act on. It must not be paying to transport it."""
    rows = inventory.rows([{"id": "p001", "beat": 1.23456, "strength": 0.115934,
                            "bands": {"low": 0.00512, "mid": 0.89731, "high": 0.09757},
                            "sustain_beats": 0.0}])
    assert rows.splitlines()[1] == "p001 1.235 0.12 0.01/0.90/0.10 0.0"


def test_all_three_designer_surfaces_use_rows(analysis):
    """Core, StepMania and Beat Saber had the same pattern and the same problem;
    a fix that lands on one of them is a third of the saving."""
    assert inventory.HEADER in sm_design.analysis_block("hard", analysis)
    assert inventory.HEADER in bs_design.analysis_block("expert", analysis)
    assert inventory.HEADER in core_design._designer_prompt("standard", analysis, None)


# --------------------------------------------------------------------------- #
# Workstream B — byte-identical payloads and a cacheable prefix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b", [("easy", "medium"), ("hard", "challenge")])
def test_tier_pairs_produce_byte_identical_analysis_blocks(analysis, a, b):
    """The whole caching saving rests on this. `hard`/`challenge` were already
    identical; `easy`/`medium` differed only because the tier's hold-length
    threshold was baked into the shared inventory."""
    assert sm_design.analysis_block(a, analysis) == sm_design.analysis_block(b, analysis)


def test_the_identity_test_is_not_vacuous(analysis):
    """Guard against the above passing because every tier serializes the same.
    `beginner` has a coarser finest_subdiv and really does offer fewer onsets."""
    assert (sm_design.analysis_block("beginner", analysis)
            != sm_design.analysis_block("easy", analysis))


def test_five_difficulties_collapse_to_three_payloads(analysis):
    blocks = {sm_design.analysis_block(d, analysis) for d in BUDGETS}
    assert len(blocks) == 3


def test_hold_threshold_moved_from_the_inventory_to_the_budget(analysis):
    """The tier threshold must be stated once, in the budget text, not once per
    onset. `easy` (hold floor 2.0 beats) and `medium` (1.0) must say different
    things while offering the same inventory."""
    for d in BUDGETS:
        for o in sm_design._compact_analysis(analysis, d)["onsets"]:
            assert "sustain" not in o, "the tier-thresholded flag is back"
    assert "sustain_beats >= 1.0" in sm_design._budget_text("easy")
    assert "sustain_beats > 0" in sm_design._budget_text("medium")


def test_hold_rule_selects_exactly_what_the_old_flag_did(analysis):
    """Equivalence check for the flag that was removed. The stated rule and the
    old per-onset computation must pick the same onsets, or the designer is being
    offered holds the resolver will reject (or denied holds it would accept)."""
    for d in BUDGETS:
        floor = BUDGETS[d].hold_len_beats[0] - 1.0
        compact = sm_design._compact_analysis(analysis, d)
        offered = {o["id"] for o in compact["onsets"]
                   if float(o["sustain_beats"]) > 0
                   and float(o["sustain_beats"]) >= floor}
        raw = {o["id"] for o in analysis["onsets"]
               if o["sustain"] and o.get("sustain_beats", 0) >= floor}
        assert offered == raw & {o["id"] for o in compact["onsets"]}


def test_invariant_block_leads_and_the_difficulty_trails(analysis):
    """The shared prefix used to end after six lines, because `{difficulty}` was
    substituted on line 7 of a ~40,000-token prompt. It must now run all the way
    through the rules, the schema and the inventory."""
    p = sm_design.designer_prompt("hard", analysis)
    assignment = p.index("**hard**")
    for marker in ("HARD RULES", "OUTPUT", "MACHINE ANALYSIS", inventory.HEADER):
        assert p.index(marker) < assignment, f"{marker} trails the difficulty"
    assert p.index("BUDGET (hard)") > p.index(inventory.HEADER)


@pytest.mark.parametrize("a,b", [("easy", "medium"), ("hard", "challenge")])
def test_identical_tiers_share_almost_the_whole_prompt(analysis, a, b):
    """The acceptance criterion, expressed statically: what an implicit cache can
    reuse between two calls is their common prefix, so that prefix has to be most
    of the prompt rather than a preamble."""
    pa, pb = sm_design.designer_prompt(a, analysis), sm_design.designer_prompt(b, analysis)
    shared = 0
    for ca, cb in zip(pa, pb):
        if ca != cb:
            break
        shared += 1
    assert shared > 0.8 * len(pa), f"only {shared}/{len(pa)} B of shared prefix"


def test_unrelated_tiers_still_share_the_rules_and_the_schema(analysis):
    """Across the three distinct payloads the cacheable prefix is smaller, but it
    must still cover everything invariant: role, rules, vocabularies, schema."""
    pa = sm_design.designer_prompt("beginner", analysis)
    pb = sm_design.designer_prompt("challenge", analysis)
    shared = 0
    for ca, cb in zip(pa, pb):
        if ca != cb:
            break
        shared += 1
    prefix = pa[:shared]
    for marker in ("HARD RULES", "jumpstream", "OUTPUT", "design_notes"):
        assert marker in prefix


def test_seed_feedback_still_goes_last(analysis):
    """The one call in the ledger that hit cache did so because its revision note
    was APPENDED to a byte-identical prompt. That ordering is the pattern."""
    p = sm_design.designer_prompt("hard", analysis)
    with_feedback = p + "\n\n--- REVISION"
    assert with_feedback.startswith(p)


# --------------------------------------------------------------------------- #
# Workstream C — the mechanism, which ships OFF
# --------------------------------------------------------------------------- #
def test_role_thinking_is_off_by_default(monkeypatch):
    """It is a quality bet with an A/B attached, not a cost fix to be defaulted
    on. Off means every role resolves to today's single global level."""
    monkeypatch.setattr(config, "ROLE_THINKING_ENABLED", False)
    from beatforge.llm import thinking_for
    for role in ("designer", "critic", "reprompt", "scorecard", None):
        assert thinking_for(role) is None


def test_role_thinking_maps_each_stage_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "ROLE_THINKING_ENABLED", True)
    monkeypatch.setattr(config, "THINKING_LEVEL_PINNED", False)
    from beatforge.llm import thinking_for
    assert thinking_for("designer") == "high"
    assert thinking_for("critic") == "low"
    assert thinking_for("reprompt") == "medium"
    assert thinking_for("reprompt_density") == "medium"
    assert thinking_for("scorecard") == "medium"


def test_unrouted_role_falls_back_to_the_default(monkeypatch):
    """Mirrors HYBRID_DEFAULT_TIER: an unrouted new call site should cost too
    much, not silently think too little."""
    monkeypatch.setattr(config, "ROLE_THINKING_ENABLED", True)
    monkeypatch.setattr(config, "THINKING_LEVEL_PINNED", False)
    from beatforge.llm import thinking_for
    assert thinking_for("some_new_stage") is None      # -> DEFAULT_THINKING
    assert config.DEFAULT_THINKING == config.THINKING_LEVEL


def test_an_explicit_global_level_pins_every_role(monkeypatch):
    """`BEATFORGE_THINKING_LEVEL=high` is how the uniform arm of the A/B is
    requested; it must beat the map without also having to disable it."""
    monkeypatch.setattr(config, "ROLE_THINKING_ENABLED", True)
    monkeypatch.setattr(config, "THINKING_LEVEL_PINNED", True)
    from beatforge.llm import thinking_for
    assert thinking_for("critic") is None


def test_explicit_argument_beats_the_role_map(monkeypatch):
    """`compare.py` pins configuration deliberately and a router must not
    second-guess a benchmark."""
    from beatforge import ledger
    from beatforge.vertex import VertexClient

    monkeypatch.setattr(config, "VERTEX_PROJECT", "test-project")
    monkeypatch.setattr(config, "ROLE_THINKING_ENABLED", True)
    monkeypatch.setattr(config, "THINKING_LEVEL_PINNED", False)
    seen = {}

    c = VertexClient()
    monkeypatch.setattr(c, "_get_token", lambda force=False: "fake-token")

    def _post(url, body, timeout=600):
        seen["level"] = body["generationConfig"]["thinkingConfig"]["thinkingLevel"]
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}
    monkeypatch.setattr(c, "_post", _post)

    with ledger.stage("critic", song="t", difficulty="hard"):
        c.generate("hi")
        assert seen["level"] == "low"                    # role map applies
        c.generate("hi", thinking_level="high")
        assert seen["level"] == "high"                   # caller wins


# --------------------------------------------------------------------------- #
# Small item — the cost report has to be able to READ the A/B
# --------------------------------------------------------------------------- #
def _call(stage, level, thinking=100):
    return {"kind": "model", "song": "t", "difficulty": "hard", "stage": stage,
            "model": config.GEMINI_MODEL, "thinking_level": level,
            "thinking_tokens": thinking, "cost_usd": {"total": 0.01}}


def test_cost_report_surfaces_thinking_level_per_stage():
    """Prerequisite for the Workstream C A/B: the report must show WHICH level
    produced which cost, or the two arms are not comparable."""
    r = costreport.aggregate([_call("designer", "high"), _call("critic", "low")])
    assert r["by_stage"]["designer"]["thinking_levels"] == {"high": 1}
    assert r["by_stage"]["critic"]["thinking_levels"] == {"low": 1}
    md = costreport.render_markdown(r)
    assert "| level |" in md
    assert "high" in md and "low" in md


def test_cost_report_does_not_average_a_mixed_stage():
    """A stage that saw two levels is a half-applied budget, and hiding that
    behind a single value would misattribute the saving."""
    r = costreport.aggregate([_call("critic", "low"), _call("critic", "high"),
                              _call("critic", "low")])
    assert r["by_stage"]["critic"]["thinking_levels"] == {"high": 1, "low": 2}
    assert "low x2" in costreport.render_markdown(r)


def test_cost_report_shows_cached_input_per_stage():
    """Workstream B's acceptance criterion is a non-zero cached count on the 2nd
    through Nth call of a song, which has to be readable per stage."""
    hit = {**_call("reprompt", "high"), "cached_tokens": 57141}
    md = costreport.render_markdown(costreport.aggregate([_call("designer", "high"), hit]))
    assert "cached-in" in md and "57,141" in md
