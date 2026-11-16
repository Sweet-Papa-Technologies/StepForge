"""The hybrid cascade's safety net: the deterministic referees that decide whether
a cheap first draft ships, and the escalation that re-authors one that fails.

The cascade's premise is "a cheap draft ships only if the deterministic referees
accept it", so the premise is worth exactly as much as the referees are. Two
things are pinned here:

  * `validate_repair` really does neutralize the one failure mode Flash Lite was
    measured to have (jump abuse — 0.267 vs 3.6 Flash's 0.046,
    docs/model-comparison.md), whatever the model proposed. This is why there is
    no runtime jump_share GATE: the repair pass makes one unreachable, and the
    invariant belongs in a test rather than in a branch that can never be taken.
  * A draft the gates reject is re-authored on the heavy tier, and the re-draft
    is kept only if it is actually better.
"""
from __future__ import annotations

import pytest

from beatforge import config, ledger
from beatforge.adapters.stepmania import adapter as sm_adapter
from beatforge.adapters.stepmania import qa as sm_qa
from beatforge.adapters.stepmania.grammar import BUDGETS
from beatforge.adapters.stepmania.quantize import Placement
from beatforge.adapters.stepmania.realize import _JUMP_MAX_FRAC
from beatforge.adapters.stepmania.validate import validate_repair


def _chart(n=100, jump_every=None):
    """A chart of `n` single-panel notes on eighths, with a jump every
    `jump_every` notes. Panels cycle so lane balance stays legal."""
    out = []
    for i in range(n):
        panels = [i % 4]
        if jump_every and i % jump_every == 0:
            panels = [i % 4, (i + 2) % 4]
        out.append(Placement(beat=4.0 + i * 0.5, panels=tuple(panels), kind="tap",
                             hold_beats=None, meta={"foot": i % 2}))
    return out


def _flat_analysis(bpm=120.0):
    return {"bpm": bpm, "offset": 0.1, "duration_s": 240.0, "meter": 4,
            "onsets": [], "sections": [], "energy_curve": [0.5] * 16,
            "energy_cv": 0.2}


# --------------------------------------------------------------------------- #
# The referee that makes a cheap draft safe
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("difficulty", ["beginner", "easy", "medium", "hard", "challenge"])
def test_validate_caps_jump_share_whatever_the_model_did(difficulty):
    """A model that jumps on every third note must not be able to ship a
    jump-spam chart at ANY tier. `cap_jump_share` demotes the surplus to single
    taps, so the realized share lands under realize.py's authored-DDR ceiling.

    This is the load-bearing guarantee behind drafting on the cheap model: Flash
    Lite's measured jump_share of 0.267 cannot reach a simfile.
    """
    an = _flat_analysis()
    rr = validate_repair(_chart(100, jump_every=3), an, difficulty)
    share = sm_qa.chart_metrics(rr.placements, an, difficulty)["jump_share"]
    cap = _JUMP_MAX_FRAC[BUDGETS[difficulty].jumps]
    assert share <= cap + 1e-9, (
        f"{difficulty}: realized jump_share {share} exceeds the {cap} ceiling")


def test_the_cap_demotes_rather_than_deletes():
    """Demotion keeps the note on the music (SACRED-02); deletion would move the
    chart off the audio to fix an ergonomics problem."""
    an = _flat_analysis()
    chart = _chart(100, jump_every=3)
    rr = validate_repair(chart, an, "hard")
    assert any(r[0] == "jump_demoted" for r in rr.repairs)
    kept_beats = {round(p.beat, 3) for p in rr.placements}
    # Every note the NPS/jack passes did not drop is still at its original beat.
    assert kept_beats <= {round(p.beat, 3) for p in chart}


def test_gentle_tiers_get_fewer_jumps_than_free_tiers():
    an = _flat_analysis()
    chart = _chart(100, jump_every=3)
    easy = sm_qa.chart_metrics(validate_repair(chart, an, "easy").placements, an, "easy")
    hard = sm_qa.chart_metrics(validate_repair(_chart(100, jump_every=3), an, "hard").placements,
                               an, "hard")
    assert easy["jump_share"] < hard["jump_share"]


def test_jump_share_is_still_reported():
    """The cap makes a gate unnecessary, not the measurement — a QA report that
    stopped showing jump_share would hide a realizer regression."""
    an = _flat_analysis()
    m = sm_qa.chart_metrics(validate_repair(_chart(100, jump_every=3), an, "hard").placements,
                            an, "hard")
    assert "jump_share" in m
    assert "jump_share" not in m["gates"], (
        "a gate here can never fire; see the note in qa.py::_gates")


# --------------------------------------------------------------------------- #
# The escalation
# --------------------------------------------------------------------------- #
class _TieredClient:
    """Answers with a jump-heavy design on the lite tier and a clean one on the
    heavy tier, so the escalation has something real to improve on. Routing is
    read from the ambient ledger stage, exactly as HybridClient does it."""

    def __init__(self, analysis):
        self.analysis = analysis
        self.stages: list[str] = []

    def _ids(self, keep):
        return [o["id"] for o in self.analysis["onsets"]][:keep]

    def generate_json(self, prompt, *, audio_path=None, **kw):
        stage = ledger.current().get("stage")
        self.stages.append(str(stage))
        lite = stage == "designer"
        return {
            "design_notes": "test",
            "phrases": [{
                "start_bar": 0, "end_bar": 24, "texture": "steps",
                "movement": "static", "crossover": "none",
                # jump_density "free" on every phrase is how a reckless draft
                # produces jump spam; "none" is the well-behaved answer.
                "jump_density": "free" if lite else "none",
                "density": 4.0,
            }],
            "notes": [{"ref": oid, "kind": "tap"} for oid in self._ids(40)],
        }


@pytest.fixture
def hybrid_on(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "hybrid")
    monkeypatch.setattr(config, "HYBRID_CASCADE", True)
    # Whole-song mode, deliberately. What is under test here is the TIER routing
    # — does a cheap draft ship, does a failing one escalate — which is
    # orthogonal to how the draft was assembled. Leaving these on the chunked
    # default would make them slice audio and issue a call per section, so a
    # change to chunking could fail the cascade suite for reasons that have
    # nothing to do with the cascade. The chunked path's own tier routing is
    # covered in test_chunked_stepforge.py.
    monkeypatch.setattr(config, "CHUNK_MODE", "never")


def test_a_clean_lite_draft_is_not_escalated(hybrid_on, make_analysis, monkeypatch):
    """The saving only exists if a passing draft ships as-is. If every draft
    escalated, hybrid mode would be strictly more expensive than single mode."""
    an = make_analysis()
    monkeypatch.setattr(sm_adapter, "_audio_path", lambda t: "audio.ogg")
    calls = []

    def _no_escalation(*a, **k):
        calls.append(a)
        return None

    monkeypatch.setattr(sm_adapter, "_escalate_to_heavy", _no_escalation)
    client = _TieredClient(an)
    # Force the lite draft to be judged clean by stubbing the gates it must clear.
    monkeypatch.setattr(sm_adapter.StepManiaAdapter, "qa_metrics",
                        lambda self, p, a, d: {"gates": {"flow_ceiling": True}})
    with ledger.capture():
        sm_adapter._design_once(sm_adapter.StepManiaAdapter(), "t", "hard", an,
                                False, client, None)
    assert calls == [], "a draft that passed every gate must not be re-authored"
    assert client.stages == ["designer"]


def test_a_failing_lite_draft_escalates_to_the_heavy_tier(hybrid_on, make_analysis,
                                                          monkeypatch):
    an = make_analysis()
    monkeypatch.setattr(sm_adapter, "_audio_path", lambda t: "audio.ogg")
    client = _TieredClient(an)
    with ledger.capture():
        sm_adapter._design_once(sm_adapter.StepManiaAdapter(), "t", "hard", an,
                                False, client, None)
    # The first call was the lite draft; it failed gates, so a second call was
    # made — billed as a `reprompt`, which is both the honest cost bucket and the
    # routing key for the heavy tier.
    assert client.stages[0] == "designer"
    assert "reprompt" in client.stages[1:]


def _gate_sequence(monkeypatch, *verdicts):
    """Stub qa_metrics to return a scripted gate verdict per call, so the
    keep-or-discard rule can be tested without a chart that happens to fail
    exactly the right things."""
    seq = iter(verdicts)
    last = {"v": verdicts[-1]}

    def _fake(self, placements, analysis, difficulty):
        last["v"] = next(seq, last["v"])
        return {"gates": dict(last["v"])}

    monkeypatch.setattr(sm_adapter.StepManiaAdapter, "qa_metrics", _fake)


def test_a_better_heavy_redesign_replaces_the_lite_draft(hybrid_on, make_analysis,
                                                         monkeypatch):
    an = make_analysis()
    monkeypatch.setattr(sm_adapter, "_audio_path", lambda t: "audio.ogg")
    # 1st: the lite draft is judged (2 failures). 2nd: the heavy redesign (0).
    _gate_sequence(monkeypatch,
                   {"flow_ceiling": False, "density_energy": False},
                   {"flow_ceiling": True, "density_energy": True})
    client = _TieredClient(an)
    with ledger.capture():
        out = sm_adapter._design_once(sm_adapter.StepManiaAdapter(), "t", "hard", an,
                                      False, client, None)
    assert out["metrics"]["designed_by_tier"] == "heavy"


def test_a_no_better_heavy_redesign_is_discarded(hybrid_on, make_analysis, monkeypatch):
    """A re-draft that clears no more gates is not an improvement — it is just a
    second bill. The original stands."""
    an = make_analysis()
    monkeypatch.setattr(sm_adapter, "_audio_path", lambda t: "audio.ogg")
    _gate_sequence(monkeypatch,
                   {"flow_ceiling": False, "density_energy": True},
                   {"flow_ceiling": False, "density_energy": True})
    client = _TieredClient(an)
    with ledger.capture():
        out = sm_adapter._design_once(sm_adapter.StepManiaAdapter(), "t", "hard", an,
                                      False, client, None)
    assert "reprompt" in client.stages, "the escalation must still have been tried"
    assert out["metrics"]["designed_by_tier"] == "lite"


def test_no_escalation_when_the_draft_was_written_by_the_heavy_tier(
        make_analysis, monkeypatch):
    """In single-model mode there is no lite draft, so the cascade must be inert —
    otherwise turning hybrid off would silently add a retry to every chart."""
    monkeypatch.setattr(config, "LLM_MODE", "single")
    monkeypatch.setattr(config, "CHUNK_MODE", "never")   # tier routing, not chunking
    an = make_analysis()
    monkeypatch.setattr(sm_adapter, "_audio_path", lambda t: "audio.ogg")
    client = _TieredClient(an)
    with ledger.capture():
        sm_adapter._design_once(sm_adapter.StepManiaAdapter(), "t", "hard", an,
                                False, client, None)
    assert client.stages == ["designer"]


def test_a_failed_escalation_keeps_the_original(hybrid_on, make_analysis, monkeypatch):
    """A model call that errors mid-escalation costs money. It must not also cost
    the chart."""
    an = make_analysis()
    monkeypatch.setattr(sm_adapter, "_audio_path", lambda t: "audio.ogg")

    class _Exploding(_TieredClient):
        def generate_json(self, prompt, *, audio_path=None, **kw):
            if ledger.current().get("stage") != "designer":
                raise RuntimeError("vertex is down")
            return super().generate_json(prompt, audio_path=audio_path, **kw)

    client = _Exploding(an)
    with ledger.capture():
        out = sm_adapter._design_once(sm_adapter.StepManiaAdapter(), "t", "hard", an,
                                      False, client, None)
    assert out["placements"], "the lite draft must survive a failed escalation"
    assert out["metrics"]["designed_by_tier"] == "lite"
