"""Gemini 4 Argon — the transport's Gemini 4 request shape, the argon backend's
policy, and the cost accounting both depend on. All offline: `_post` is stubbed
at the same seam test_cost.py uses, so what is asserted is exactly the request
body that would have gone over the wire.

The three properties worth pinning, because each one fails SILENTLY:
  * A Gemini 4 call never carries a 3.x-only field (`minimal`, thinkingBudget):
    that is a 400 on every call of a run.
  * A reply longer than one response is followed to the end and billed for
    every hop — not truncated into a half-chart, not billed as one hop.
  * A call is billed at the Flex discount only when the server says it was
    served at Flex.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from beatforge import argon, chunking, config, ledger, llm, pricing
from beatforge.vertex import (GEMINI4_HOP_TOKENS, GEMINI4_MAX_OUTPUT_TOKENS,
                              VertexClient, VertexError, gemini4_max_hops)


def _resp(text="{}", *, prompt=1000, audio=0, out=50, thinking=10, cached=0,
          finish="STOP", token=None, traffic=None, cache_audio=None):
    details = [{"modality": "TEXT", "tokenCount": prompt - audio}]
    if audio:
        details.append({"modality": "AUDIO", "tokenCount": audio})
    um = {"promptTokenCount": prompt, "candidatesTokenCount": out,
          "thoughtsTokenCount": thinking, "promptTokensDetails": details}
    if cached:
        um["cachedContentTokenCount"] = cached
    if cache_audio is not None:
        um["cacheTokensDetails"] = [{"modality": "AUDIO", "tokenCount": cache_audio}]
    if traffic:
        um["trafficType"] = traffic
    cand = {"content": {"role": "model", "parts": [{"text": text}]},
            "finishReason": finish}
    if token:
        cand["continuationToken"] = token
    return {"candidates": [cand], "usageMetadata": um}


class _Wire:
    """Stands in for VertexClient._post: records every (url, body) and replays
    canned responses (or raises them, if an entry is an exception)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.sent: list[tuple[str, dict]] = []

    def __call__(self, url, body, timeout=600):
        self.sent.append((url, json.loads(json.dumps(body))))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    monkeypatch.setattr(config, "VERTEX_PROJECT", "test-project")
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config, "ARGON_TRANSPORT", "auto")
    monkeypatch.setattr(config, "ARGON_SERVICE_TIER", "standard")
    monkeypatch.setattr(config, "ARGON_CRITIC_MODEL", "")


def _vertex(monkeypatch, wire):
    c = VertexClient()
    monkeypatch.setattr(c, "_get_token", lambda force=False: "t")
    monkeypatch.setattr(c, "_post", wire)
    return c


def _argon(monkeypatch, wire, **kw):
    c = argon.make_client(**kw)
    monkeypatch.setattr(c, "_get_token", lambda force=False: "t")
    monkeypatch.setattr(c, "_post", wire)
    return c


# --------------------------------------------------------------------------- #
# Transport: the Gemini 4 request shape
# --------------------------------------------------------------------------- #
def test_gemini4_goes_to_the_global_v1beta1_endpoint(monkeypatch):
    """Argon is not served from us-central1, and long decode is v1beta1-only."""
    wire = _Wire(_resp())
    _vertex(monkeypatch, wire).generate("hi", model="gemini-4-argon")
    url = wire.sent[0][0]
    assert "aiplatform.googleapis.com/v1beta1/" in url
    assert "/locations/global/" in url and "us-central1" not in url


@pytest.mark.parametrize("asked,sent", [
    ("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high"),
    ("HIGH", "high"), ("bogus", "medium")])
def test_gemini4_never_receives_a_thinking_level_it_rejects(monkeypatch, asked, sent):
    """`minimal` (offered by the desktop's global level) and thinkingBudget are
    both 400s on Gemini 4. The ledger records what actually ran."""
    wire = _Wire(_resp())
    with ledger.capture() as rows:
        _vertex(monkeypatch, wire).generate("hi", model="gemini-4-argon",
                                            thinking_level=asked)
    tc = wire.sent[0][1]["generationConfig"]["thinkingConfig"]
    assert tc == {"thinkingLevel": sent}
    assert rows[0]["thinking_level"] == sent


def test_gemini4_asks_for_the_long_decode_budget(monkeypatch):
    wire = _Wire(_resp())
    _vertex(monkeypatch, wire).generate("hi", model="gemini-4-argon")
    cap = wire.sent[0][1]["generationConfig"]["maxOutputTokens"]
    assert cap == config.ARGON_MAX_OUTPUT_TOKENS > GEMINI4_HOP_TOKENS


def test_gemini3_requests_are_byte_for_byte_unchanged(monkeypatch):
    """No new field on the 3.x path: every existing benchmark ran without one."""
    wire = _Wire(_resp())
    _vertex(monkeypatch, wire).generate("hi", model="gemini-3.8-flash",
                                        thinking_level="minimal")
    body = wire.sent[0][1]
    assert body["generationConfig"] == {"thinkingConfig": {"thinkingLevel": "minimal"}}
    assert set(body) == {"contents", "generationConfig"}


# --------------------------------------------------------------------------- #
# Long decode
# --------------------------------------------------------------------------- #
def test_a_long_reply_is_followed_across_hops_and_reassembled(monkeypatch):
    wire = _Wire(
        _resp('{"notes": [1, ', finish="CONTINUATION", token="tok1",
              prompt=60000, out=20000, thinking=12768),
        _resp("", finish="CONTINUATION", token="tok2",              # thinking-only hop
              prompt=60000, cached=59000, out=0, thinking=32768),
        _resp('2, 3]}', prompt=60000, cached=59000, out=900, thinking=100))
    with ledger.capture() as rows:
        text = _vertex(monkeypatch, wire).generate("hi", model="gemini-4-argon")
    assert json.loads(text) == {"notes": [1, 2, 3]}
    # Each hop resends the SAME body, plus the token at the top level.
    first = wire.sent[0][1]
    assert "continuationToken" not in first
    assert wire.sent[1][1] == {**first, "continuationToken": "tok1"}
    assert wire.sent[2][1] == {**first, "continuationToken": "tok2"}
    # One ledger row for the one logical call, billed for EVERY hop.
    assert len(rows) == 1
    r = rows[0]
    assert r["detail"] == {"long_decode_hops": 3}
    assert r["input_tokens"] == 180000 and r["cached_tokens"] == 118000
    assert r["output_tokens"] == 20900 and r["thinking_tokens"] == 45636


def test_the_full_million_token_budget_is_usable(monkeypatch):
    """Argon allows 1,048,576 output tokens per call. A fixed hop cap would have
    stopped a 1M budget at ~393k; the cap has to scale with the budget."""
    monkeypatch.setattr(config, "ARGON_MAX_OUTPUT_TOKENS", GEMINI4_MAX_OUTPUT_TOKENS)
    n = GEMINI4_MAX_OUTPUT_TOKENS // GEMINI4_HOP_TOKENS            # 32 full hops
    hops = [_resp("x", finish="CONTINUATION", token=f"t{i}", out=GEMINI4_HOP_TOKENS,
                  thinking=0) for i in range(n - 1)] + [_resp("y")]
    wire = _Wire(*hops)
    text = _vertex(monkeypatch, wire).generate("hi", model="gemini-4-argon")
    assert len(wire.sent) == n and text == "x" * (n - 1) + "y"
    assert wire.sent[0][1]["generationConfig"]["maxOutputTokens"] == GEMINI4_MAX_OUTPUT_TOKENS
    assert gemini4_max_hops(GEMINI4_MAX_OUTPUT_TOKENS) >= n


def test_a_budget_above_the_model_limit_is_clamped_not_sent(monkeypatch):
    monkeypatch.setattr(config, "ARGON_MAX_OUTPUT_TOKENS", 5_000_000)
    wire = _Wire(_resp())
    _vertex(monkeypatch, wire).generate("hi", model="gemini-4-argon")
    assert wire.sent[0][1]["generationConfig"]["maxOutputTokens"] == GEMINI4_MAX_OUTPUT_TOKENS


def test_a_never_ending_continuation_is_still_cut_off(monkeypatch, capsys):
    wire = _Wire(_resp("x", finish="CONTINUATION", token="again"))
    _vertex(monkeypatch, wire).generate("hi", model="gemini-4-argon")
    assert len(wire.sent) == gemini4_max_hops(config.ARGON_MAX_OUTPUT_TOKENS)
    assert "still generating" in capsys.readouterr().err


def test_a_hop_boundary_inside_a_token_does_not_corrupt_the_json(monkeypatch):
    """The cap falls wherever it falls. A separator at the seam would turn
    "p017" into "p01\n7" and an id into invalid JSON."""
    wire = _Wire(_resp('{"notes": [{"ref": "p01', finish="CONTINUATION", token="t"),
                 _resp('7", "kind": "tap"}], "density": 12', finish="CONTINUATION",
                       token="u"),
                 _resp('.5}'))
    text = _vertex(monkeypatch, wire).generate("hi", model="gemini-4-argon")
    assert json.loads(text) == {"notes": [{"ref": "p017", "kind": "tap"}],
                                "density": 12.5}


def test_max_tokens_ends_the_loop_and_says_so(monkeypatch, capsys):
    """MAX_TOKENS carries a token too, but resuming with the same budget is a 400."""
    wire = _Wire(_resp('{"a": 1}', finish="MAX_TOKENS", token="t"))
    _vertex(monkeypatch, wire).generate("hi", model="gemini-4-argon")
    assert len(wire.sent) == 1
    assert "BEATFORGE_ARGON_MAX_OUTPUT_TOKENS" in capsys.readouterr().err


def test_an_endpoint_without_long_decode_falls_back_to_one_response(monkeypatch, capsys):
    refusal = VertexError("HTTP 400 from Vertex:\nmaxOutputTokens must be <= 32768 "
                          "for non-streaming requests")
    wire = _Wire(refusal, _resp('{"ok": true}'))
    c = _vertex(monkeypatch, wire)
    assert json.loads(c.generate("hi", model="gemini-4-argon")) == {"ok": True}
    assert wire.sent[1][1]["generationConfig"]["maxOutputTokens"] == GEMINI4_HOP_TOKENS
    assert "no long decode" in capsys.readouterr().err
    # ...and it is remembered, so later calls do not pay the 400 again.
    c.generate("again", model="gemini-4-argon")
    assert wire.sent[-1][1]["generationConfig"]["maxOutputTokens"] == GEMINI4_HOP_TOKENS


def test_an_unrelated_400_is_not_mistaken_for_a_long_decode_refusal(monkeypatch):
    wire = _Wire(VertexError("HTTP 400 from Vertex:\nInvalid audio part"))
    with pytest.raises(VertexError, match="Invalid audio"):
        _vertex(monkeypatch, wire).generate("hi", model="gemini-4-argon")
    assert len(wire.sent) == 1


def test_a_hop_that_fails_still_bills_the_hops_before_it(monkeypatch):
    wire = _Wire(_resp("{", finish="CONTINUATION", token="t", prompt=5000, out=32000),
                 VertexError("HTTP 500 from Vertex: boom"))
    with ledger.capture() as rows, pytest.raises(VertexError):
        _vertex(monkeypatch, wire).generate("hi", model="gemini-4-argon")
    assert rows[0]["error"] and rows[0]["output_tokens"] == 32000


# --------------------------------------------------------------------------- #
# Cached audio is not billed twice
# --------------------------------------------------------------------------- #
def test_cached_audio_is_billed_once_at_the_cached_rate():
    """A critic prompt is mostly audio. When the cache hit covers it, the old
    arithmetic clamped text to 0 and billed the audio at full rate ON TOP of the
    cached count. VertexClient puts audio first, so a prefix hit covers it."""
    u = ledger.usage_from_vertex(_resp(prompt=7000, audio=5800, cached=5800))
    assert u["audio_tokens"] == 0 and u["text_tokens"] == 1200
    assert u["cached_tokens"] == 5800
    assert u["audio_tokens"] + u["text_tokens"] + u["cached_tokens"] == 7000


def test_the_server_cache_breakdown_wins_over_the_inference():
    u = ledger.usage_from_vertex(_resp(prompt=10000, audio=3000, cached=6000,
                                       cache_audio=2000))
    assert u["audio_tokens"] == 1000 and u["text_tokens"] == 3000


def test_without_a_cache_hit_nothing_changes():
    u = ledger.usage_from_vertex(_resp(prompt=12000, audio=9000))
    assert (u["audio_tokens"], u["text_tokens"]) == (9000, 3000)


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
def _epoch(day: str) -> float:
    return time.mktime(time.strptime(day + " 12:00", "%Y-%m-%d %H:%M"))


def test_both_launch_rates_end_on_their_published_dates():
    assert pricing.price_for("gemini-4-argon", at=_epoch("2026-11-30")).out == 10.00
    assert pricing.price_for("gemini-4-argon", at=_epoch("2026-12-01")).out == 20.00
    assert pricing.price_for("gemini-3.8-flash", at=_epoch("2026-12-31")).text_in == 0.75
    assert pricing.price_for("gemini-3.8-flash", at=_epoch("2027-01-01")).text_in == 1.50


def test_the_previous_default_keeps_its_rate():
    """Every ledger line written before the 3.8 switch names gemini-3.6-flash."""
    p = pricing.price_for("gemini-3.6-flash")
    assert p.verified and p.model == "gemini-3.6-flash"


def test_a_suffixed_id_bills_as_its_longest_known_prefix():
    assert pricing.price_for("gemini-4-argon-cyber-permissive").model == "gemini-4-argon"


def test_flex_bills_at_half_and_standard_is_untouched():
    std = pricing.cost_usd("gemini-4-argon", text_in=10**6, out=10**6,
                           at=_epoch("2026-11-01"))
    flex = pricing.cost_usd("gemini-4-argon", text_in=10**6, out=10**6, tier="flex",
                            at=_epoch("2026-11-01"))
    assert std["total"] == 12.0 and flex["total"] == 6.0 and flex["tier"] == "flex"
    assert "tier" not in std


# --------------------------------------------------------------------------- #
# The argon backend's policy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stage,attempt,expected", [
    ("designer", 1, "high"),
    ("designer", 2, "medium"),          # chunked mode's retry is still "designer"
    ("reprompt", 1, "medium"),          # a critic revision re-enters as attempt 1
    ("reprompt_density", 1, "medium"),
    ("critic", None, "low"),
    ("audio_probe", None, "low"),
    ("something_new", None, "medium"),  # Argon's own default
])
def test_thinking_follows_the_ledger_stage(monkeypatch, stage, attempt, expected):
    wire = _Wire(_resp())
    c = _argon(monkeypatch, wire)
    with ledger.stage(stage, attempt=attempt):
        c.generate("hi")
    assert wire.sent[0][1]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": expected}


def test_explicit_model_and_thinking_win_over_the_policy(monkeypatch):
    """compare.py pins both deliberately; a policy must not second-guess it."""
    wire = _Wire(_resp())
    c = _argon(monkeypatch, wire)
    with ledger.stage("designer", attempt=1):
        c.generate("hi", model="gemini-3.8-flash", thinking_level="low")
    url, body = wire.sent[0]
    assert "/models/gemini-3.8-flash:" in url
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}


def test_a_separate_critic_model_judges_only(monkeypatch):
    monkeypatch.setattr(config, "ARGON_CRITIC_MODEL", "gemini-3.8-flash")
    wire = _Wire(_resp())
    c = _argon(monkeypatch, wire)
    with ledger.stage("designer", attempt=1):
        c.generate("design")
    with ledger.stage("critic"):
        c.generate("judge")
    assert "/models/gemini-4-argon:" in wire.sent[0][0]
    assert "/models/gemini-3.8-flash:" in wire.sent[1][0]


def test_the_short_call_site_timeout_is_raised(monkeypatch):
    seen = []
    c = _argon(monkeypatch, _Wire(_resp()))
    monkeypatch.setattr(c, "_post", lambda u, b, timeout=600: seen.append(timeout) or _resp())
    c.generate("hi", timeout=600)
    assert seen == [config.ARGON_TIMEOUT]


def test_json_and_its_repair_retry_both_go_through_the_policy(monkeypatch):
    wire = _Wire(_resp("not json"), _resp('{"ok": 1}'))
    c = _argon(monkeypatch, wire)
    with ledger.stage("critic"):
        assert c.generate_json("hi") == {"ok": 1}
    assert [b["generationConfig"]["thinkingConfig"]["thinkingLevel"]
            for _, b in wire.sent] == ["low", "low"]


# --------------------------------------------------------------------------- #
# Transport selection and Flex
# --------------------------------------------------------------------------- #
def test_auto_prefers_vertex_but_flex_takes_ai_studio(monkeypatch):
    assert argon.resolve_transport() == "vertex"
    monkeypatch.setattr(config, "GEMINI_API_KEY", "k")
    assert argon.resolve_transport() == "vertex"
    assert argon.resolve_transport("flex") == "aistudio"
    monkeypatch.setattr(config, "VERTEX_PROJECT", None)
    assert argon.resolve_transport() == "aistudio"


def test_no_credentials_names_both_ways_to_fix_it(monkeypatch):
    monkeypatch.setattr(config, "VERTEX_PROJECT", None)
    with pytest.raises(argon.ArgonError, match="VERTEX_PROJECT.*GEMINI_API_KEY"):
        argon.resolve_transport()


def test_vertex_never_sends_flex_and_says_why(monkeypatch, capsys):
    wire = _Wire(_resp())
    c = _argon(monkeypatch, wire, transport="vertex", service_tier="flex")
    assert c.service_tier == "standard"
    assert "Flex PayGo excludes it" in capsys.readouterr().err
    c.generate("hi")
    assert "serviceTier" not in wire.sent[0][1]


def _studio(monkeypatch, wire, **kw):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "k")
    return _argon(monkeypatch, wire, transport="aistudio", **kw)


def test_flex_is_requested_but_billed_only_when_confirmed(monkeypatch):
    wire = _Wire(_resp())                                   # server says nothing
    c = _studio(monkeypatch, wire, service_tier="flex")
    with ledger.capture() as rows:
        c.generate("hi")
    assert wire.sent[0][1]["serviceTier"] == "flex"
    assert "generativelanguage.googleapis.com" in wire.sent[0][0]
    assert "service_tier" not in rows[0]                    # billed at Standard

    wire2 = _Wire(_resp(traffic="ON_DEMAND_FLEX"))
    c2 = _studio(monkeypatch, wire2, service_tier="flex")
    with ledger.capture() as rows2:
        c2.generate("hi")
    assert rows2[0]["service_tier"] == "flex"
    assert rows2[0]["cost_usd"]["tier"] == "flex"


def test_a_rejected_tier_field_is_dropped_for_the_rest_of_the_run(monkeypatch, capsys):
    wire = _Wire(VertexError('HTTP 400 from Vertex:\nInvalid JSON payload received. '
                             'Unknown name "serviceTier": Cannot find field.'),
                 _resp('{"ok": 1}'))
    c = _studio(monkeypatch, wire, service_tier="flex")
    assert json.loads(c.generate("hi")) == {"ok": 1}
    assert "serviceTier" not in wire.sent[1][1]
    assert "rejected the Flex" in capsys.readouterr().err
    c.generate("again")
    assert "serviceTier" not in wire.sent[-1][1]


def test_flex_without_capacity_retries_that_call_at_standard(monkeypatch):
    wire = _Wire(VertexError("HTTP 503 from Vertex:\nat capacity"), _resp('{"ok": 1}'))
    c = _studio(monkeypatch, wire, service_tier="flex")
    assert json.loads(c.generate("hi")) == {"ok": 1}
    assert "serviceTier" not in wire.sent[1][1]
    # Only that call: the next one asks for Flex again.
    c.generate("next")
    assert wire.sent[-1][1]["serviceTier"] == "flex"


def test_flex_fallback_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(config, "ARGON_FLEX_FALLBACK", False)
    wire = _Wire(VertexError("HTTP 429 from Vertex:\nrate"))
    c = _studio(monkeypatch, wire, service_tier="flex")
    with pytest.raises(VertexError):
        c.generate("hi")


# --------------------------------------------------------------------------- #
# Factory, routing, chunking
# --------------------------------------------------------------------------- #
def test_the_factory_builds_an_argon_client(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "argon")
    monkeypatch.setattr(config, "LLM_MODE", "single")
    c = llm.make_llm_client()
    assert isinstance(c, argon.ArgonVertexClient) and c.name == "argon"


def test_a_leftover_hybrid_mode_does_not_cascade_on_argon(monkeypatch, capsys):
    """The desktop exports the last routing mode it rendered. On a backend that
    is never wrapped, `lite` would 'escalate' a draft to the same model."""
    monkeypatch.setattr(config, "LLM_BACKEND", "argon")
    monkeypatch.setattr(config, "LLM_MODE", "hybrid")
    assert isinstance(llm.make_llm_client(), argon.ArgonPolicy)
    assert "no effect on the argon backend" in capsys.readouterr().err
    assert llm.tier_for("designer", 1) is None


def test_hybrid_still_routes_on_the_gemini_backends(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "hybrid")
    for backend in ("gemini", "google-genai"):
        monkeypatch.setattr(config, "LLM_BACKEND", backend)
        assert llm.tier_for("designer", 1) == "lite"


def _song(seconds):
    return {"duration_s": seconds}


def test_auto_charts_a_whole_song_in_one_call_on_argon(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MODE", "auto")
    monkeypatch.setattr(config, "LLM_BACKEND", "gemini")
    assert chunking.should_chunk(_song(180))
    monkeypatch.setattr(config, "LLM_BACKEND", "argon")
    assert not chunking.should_chunk(_song(180))
    assert chunking.should_chunk(_song(config.ARGON_WHOLE_SONG_MAX_SECONDS + 1))


def test_an_explicit_chunk_mode_still_wins_on_argon(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "argon")
    assert chunking.should_chunk(_song(60), mode="always")
    assert not chunking.should_chunk(_song(3600), mode="never")


def test_argon_check_reports_missing_credentials_as_a_verdict(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(config.REPO_ROOT / "tools"),
           "HOME": str(tmp_path), "BEATFORGE_LLM_BACKEND": "argon"}
    p = subprocess.run([sys.executable, "-m", "beatforge", "argon-check"],
                       capture_output=True, text=True, env=env, timeout=120)
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert p.returncode == 1 and out["ok"] is False
    assert "GEMINI_API_KEY" in out["error"]


def test_a_missing_requests_package_names_the_fix(monkeypatch):
    """google-auth no longer installs `requests`, and the desktop venv was built
    from a bare `google-auth` line — the first Vertex call died in a traceback."""
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", None)
    c = VertexClient()
    with pytest.raises(VertexError, match=r"google-auth\[requests\]"):
        c._get_token()
