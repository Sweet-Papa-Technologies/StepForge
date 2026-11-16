"""Hybrid model routing — the OPT-IN mode (config.LLM_MODE == "hybrid").

The shipped default is "single": GEMINI_MODEL serves every call. Hybrid was built
to be the default and demoted on its own measurement, so the first thing pinned
here is the default itself — that is the only place the decision is recorded in
code rather than in a comment.

Beyond that, two things have to be true for the cascade in config.py to mean
anything, and both are checked without touching the network:

  1. The ROUTING is what the policy says — the first draft goes to the cheap
     model, every retry and every judge goes to the expensive one, and an
     unrecognized role fails expensive rather than cheap.
  2. The ROUTING and the COST ATTRIBUTION agree. They ride the same ambient
     ledger context on purpose; a call billed to "critic" that was routed as
     something else would make every hybrid number in a cost report a lie.
"""
from __future__ import annotations

import os

import pytest

from beatforge import config, ledger, llm
from beatforge.llm import HybridClient, OpenAICompatClient, make_llm_client
from beatforge.vertex import VertexClient


class _RecordingBase:
    """Stands in for VertexClient: records what it was asked to send and bills a
    ledger line the way the real client does, so tier stamping is exercised."""

    provider = "fake"

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, prompt, *, audio_path=None, system=None, model=None,
                 thinking_level=None, json_out=False, timeout=600):
        self.calls.append({"model": model, "thinking_level": thinking_level})
        ledger.record_model_call(
            provider=self.provider, model=model or "?",
            usage=ledger.usage_from_vertex({}), latency_s=0.0)
        return "{}"

    def generate_json(self, prompt, *, audio_path=None, system=None, model=None,
                      thinking_level=None, timeout=600):
        self.generate(prompt, model=model, thinking_level=thinking_level)
        return {}


@pytest.fixture
def hybrid(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "hybrid")
    monkeypatch.setattr(config, "HYBRID_CASCADE", True)
    base = _RecordingBase()
    return HybridClient(base), base


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("role,attempt,tier", [
    ("designer", 1, "lite"),        # the first draft is the whole saving
    ("designer", 2, "heavy"),       # …and its retry escalates
    ("designer", 3, "heavy"),
    ("reprompt", 1, "heavy"),       # gate/intent repair
    ("reprompt_density", 1, "heavy"),
    ("critic", 1, "heavy"),         # Author != Judge; the judge is never cheaper
    ("scorecard", 1, "heavy"),
])
def test_role_routes_to_expected_tier(monkeypatch, role, attempt, tier):
    monkeypatch.setattr(config, "LLM_MODE", "hybrid")
    monkeypatch.setattr(config, "HYBRID_CASCADE", True)
    assert llm.tier_for(role, attempt) == tier


def test_unknown_role_fails_expensive(monkeypatch):
    """A call site nobody routed must cost too much, not silently design charts
    on the cheap model. Failing open toward quality is the safe direction."""
    monkeypatch.setattr(config, "LLM_MODE", "hybrid")
    assert llm.tier_for("some_new_stage") == "heavy"
    assert llm.tier_for(None) == "heavy"


def test_cascade_off_pins_the_draft_to_heavy(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "hybrid")
    monkeypatch.setattr(config, "HYBRID_CASCADE", False)
    assert llm.tier_for("designer", 1) == "heavy"
    assert llm.tier_for("critic", 1) == "heavy"


def test_single_mode_disables_routing(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "single")
    assert llm.tier_for("designer", 1) is None
    assert llm.tier_for("critic", 1) is None


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #
def test_ambient_stage_drives_the_model(hybrid):
    """No call site passes `role=`. The role is the ledger stage the call is made
    under — this is the seam the whole feature hangs from."""
    client, base = hybrid
    with ledger.capture():
        with ledger.stage("designer", song="t", difficulty="hard", attempt=1):
            client.generate_json("design me a chart")
        with ledger.stage("critic", song="t", difficulty="hard", attempt=1):
            client.generate_json("judge it")
    assert base.calls[0]["model"] == config.HYBRID_LITE_MODEL
    assert base.calls[1]["model"] == config.HYBRID_HEAVY_MODEL


def test_reprompt_stage_escalates(hybrid):
    client, base = hybrid
    with ledger.capture():
        with ledger.stage("reprompt", song="t", difficulty="hard", attempt=2):
            client.generate_json("fix these gate failures")
    assert base.calls[0]["model"] == config.HYBRID_HEAVY_MODEL


def test_no_ambient_stage_is_heavy(hybrid):
    client, base = hybrid
    with ledger.capture():
        client.generate_json("an unattributed call")
    assert base.calls[0]["model"] == config.HYBRID_HEAVY_MODEL


def test_explicit_model_beats_the_router(hybrid):
    """compare.py pins models deliberately to benchmark them. A router that
    overrode that would silently invalidate every comparison in docs/."""
    client, base = hybrid
    with ledger.capture():
        with ledger.stage("designer", attempt=1):
            client.generate_json("probe", model="gemini-3.5-flash")
    assert base.calls[0]["model"] == "gemini-3.5-flash"


def test_thinking_level_follows_the_tier(hybrid, monkeypatch):
    monkeypatch.setattr(config, "HYBRID_LITE_THINKING", "low")
    monkeypatch.setattr(config, "HYBRID_HEAVY_THINKING", "high")
    client, base = hybrid
    with ledger.capture():
        with ledger.stage("designer", attempt=1):
            client.generate_json("draft")
        with ledger.stage("critic", attempt=1):
            client.generate_json("judge")
    assert base.calls[0]["thinking_level"] == "low"
    assert base.calls[1]["thinking_level"] == "high"


# --------------------------------------------------------------------------- #
# Routing vs. cost attribution
# --------------------------------------------------------------------------- #
def test_ledger_records_the_tier(hybrid):
    client, _ = hybrid
    with ledger.capture() as entries:
        with ledger.stage("designer", song="t", difficulty="hard", attempt=1):
            client.generate_json("draft")
        with ledger.stage("critic", song="t", difficulty="hard", attempt=1):
            client.generate_json("judge")
    assert [e["tier"] for e in entries] == ["lite", "heavy"]
    assert [e["stage"] for e in entries] == ["designer", "critic"]
    # Every entry must still satisfy the ledger schema (REQ-R2-COST-01).
    assert all(ledger.validate_entry(e) == [] for e in entries)


def test_annotate_does_not_steal_the_stage():
    """`annotate` exists because `stage()` renames — the tier stamp must not
    overwrite the attribution a caller already established."""
    with ledger.stage("critic", song="t", difficulty="hard", attempt=2):
        with ledger.annotate(tier="heavy"):
            ctx = ledger.current()
    assert ctx["stage"] == "critic"
    assert ctx["attempt"] == 2 and ctx["song"] == "t"
    assert ctx["tier"] == "heavy"


def test_pinned_model_is_billed_as_pinned(hybrid):
    client, _ = hybrid
    with ledger.capture() as entries:
        with ledger.stage("designer", song="t", attempt=1):
            client.generate_json("probe", model="gemini-2.5-pro")
    assert entries[0]["tier"] == "pinned"


# --------------------------------------------------------------------------- #
# The factory
# --------------------------------------------------------------------------- #
def test_single_is_the_shipped_default():
    """Hybrid is OPT-IN. It was built to be the default and demoted on its own
    evidence: the one full-length measurement had it ~17% more expensive with an
    identical quality floor (docs/song-quality-review.md §4). Nothing about that
    conclusion is enforced by code, so the default itself is the assertion — this
    test is what fails if someone flips it back without new numbers.

    Read from the environment-free module default, not from a monkeypatch.
    """
    assert config.LLM_MODE == "single"
    assert os.environ.get("BEATFORGE_LLM_MODE") in (None, "single"), (
        "this test asserts the SHIPPED default; unset BEATFORGE_LLM_MODE to run it")


def test_the_default_mode_uses_one_model_for_every_role():
    """The point of the default: no routing at all, GEMINI_MODEL everywhere."""
    for role in ("designer", "reprompt", "critic", "scorecard"):
        assert llm.tier_for(role, 1) is None


def test_hybrid_wraps_both_gemini_paths_when_selected(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "hybrid")
    monkeypatch.setattr(config, "LLM_BACKEND", "gemini")
    assert isinstance(make_llm_client(), HybridClient)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_BACKEND", "google-genai")
    assert isinstance(make_llm_client(), HybridClient)


def test_single_mode_returns_the_plain_client(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "single")
    monkeypatch.setattr(config, "LLM_BACKEND", "gemini")
    c = make_llm_client()
    assert isinstance(c, VertexClient) and not isinstance(c, HybridClient)


def test_the_heavy_tier_follows_the_single_model_choice():
    """Switching to hybrid must change only WHAT DRAFTS. The model that authors
    retries and judges is still whatever `BEATFORGE_GEMINI_MODEL` selected, so the
    two modes are comparable and the mode toggle introduces exactly one variable."""
    if os.environ.get("BEATFORGE_HYBRID_HEAVY_MODEL"):
        pytest.skip("heavy tier explicitly overridden in this environment")
    assert config.HYBRID_HEAVY_MODEL == config.GEMINI_MODEL


def test_openai_backend_is_never_wrapped(monkeypatch):
    """One server, one loaded model — there is no second tier to route to."""
    monkeypatch.setattr(config, "LLM_MODE", "hybrid")
    monkeypatch.setattr(config, "LLM_BACKEND", "openai")
    assert isinstance(make_llm_client(), OpenAICompatClient)


# --------------------------------------------------------------------------- #
# Both tiers must be real, audio-capable, priced models
# --------------------------------------------------------------------------- #
def test_both_tiers_are_priced_and_verified():
    """An unpriced tier bills at the UNKNOWN_MODEL_FALLBACK rate, which would make
    the cascade's whole justification — that lite is ~5x cheaper — unmeasurable."""
    from beatforge import pricing
    for model in (config.HYBRID_HEAVY_MODEL, config.HYBRID_LITE_MODEL):
        p = pricing.price_for(model)
        assert p.model == model, f"{model} is missing from pricing.TABLE"
        assert p.verified, f"{model} has an unverified rate"
    lite = pricing.price_for(config.HYBRID_LITE_MODEL)
    heavy = pricing.price_for(config.HYBRID_HEAVY_MODEL)
    assert lite.text_in < heavy.text_in and lite.out < heavy.out, (
        "the 'lite' tier is not actually cheaper than the 'heavy' one")
