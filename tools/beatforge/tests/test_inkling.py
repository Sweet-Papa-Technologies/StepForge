"""Inkling-on-Modal provider tests (REQ-INK-01..08).

All offline: a local mock stands in for the Managed Endpoint and records what it
received, so the two things most likely to be wrong — the two-header auth and
whether the audio actually goes out whole — are checked on the wire rather than
asserted in prose. The live smoke run (REQ-INK-10) is manual and documented in
the README; nothing here needs a Modal token.
"""
import json
import os
import subprocess
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from beatforge import config, inkling as inkling_mod, ledger, pricing
from beatforge.inkling import (InklingAudioIgnored, InklingClient, InklingError,
                               audio_duration_s, differential_audio_probe,
                               require_audio, resolve_credentials, scrub)
from beatforge.llm import OpenAICompatError, make_llm_client


def audio_parts(content):
    """Every audio block in a message, whichever content-part shape carries it.

    The two dialects are not interchangeable on the wire, but for "did the whole
    track go out once?" they are the same question — so the tests ask it once
    rather than pinning a shape they do not care about."""
    return [p for p in content if p.get("type") in ("input_audio", "audio_url")]


def audio_payload(part):
    """(base64, container) out of either shape."""
    if part["type"] == "audio_url":
        url = part["audio_url"]["url"]
        head, _, b64 = url.partition(",")
        assert head.startswith("data:audio/"), f"not an inline data URI: {head!r}"
        return b64, head[len("data:audio/"):].split(";")[0]
    return part["input_audio"]["data"], part["input_audio"]["format"]

_LAST = {}


class _Handler(BaseHTTPRequestHandler):
    """Mock Managed Endpoint. Enforces proxy-token auth the way Modal does —
    including rejecting a Bearer-only request — so the auth test is meaningful."""

    served_model = "thinkingmachines/Inkling-NVFP4"
    # Set per-test: answers keyed by call index, so the probe can be steered.
    replies: "list[str] | None" = None

    def log_message(self, *a):
        pass

    def _authed(self) -> bool:
        _LAST["headers"] = {k.lower(): v for k, v in self.headers.items()}
        return (self.headers.get("Modal-Key") == "wk-test"
                and self.headers.get("Modal-Secret") == "ws-test")

    def do_GET(self):
        if not self._authed():
            return self._json({"error": "proxy auth required"}, 401)
        if self.path.endswith("/models"):
            self._json({"data": [{"id": self.served_model}]})
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if not self._authed():
            return self._json({"error": "proxy auth required"}, 401)
        _LAST["body"] = body
        _LAST.setdefault("bodies", []).append(body)
        replies = type(self).replies
        if replies:
            text = replies[min(len(_LAST["bodies"]) - 1, len(replies) - 1)]
        else:
            text = json.dumps({"ok": 1})
        self._json({"choices": [{"message": {"content": text}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 100,
                              "prompt_tokens_details": {"cached_tokens": 400,
                                                        "audio_tokens": 600},
                              "completion_tokens_details": {"reasoning_tokens": 60}}})

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def endpoint():
    _LAST.clear()
    _Handler.replies = None
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    yield f"http://{host}:{port}"
    srv.shutdown()


@pytest.fixture
def client(endpoint):
    return InklingClient(endpoint_url=endpoint, token_id="wk-test",
                         token_secret="ws-test")


# --------------------------------------------------------------------------- #
# REQ-INK-02 — two-header auth
# --------------------------------------------------------------------------- #
def test_the_proxy_token_never_goes_out_as_a_bearer_token(client):
    """Modal's contract is the `Modal-Key`/`Modal-Secret` pair and nothing else,
    and the empty `Authorization` slot is deliberate: the proxy takes its own
    headers so the SERVED app keeps that one for its own credential.

    Putting the proxy token there breaks exactly that case — a vLLM behind proxy
    auth started with `--api-key` sees a bearer token it never issued and 401s
    after Modal has already waved the request through, which reads as a Modal
    auth failure when Modal was fine.
    """
    client.generate_json("hello")
    sent = _LAST["headers"]
    assert sent["modal-key"] == "wk-test"
    assert sent["modal-secret"] == "ws-test"
    assert "authorization" not in sent, (
        f"proxy token leaked into Authorization: {sent.get('authorization')!r}")


def test_an_app_api_key_authenticates_the_served_app_not_the_proxy(endpoint):
    """A self-deployed server behind proxy auth authenticates TWICE. The app's
    key is its own setting, so the two credentials stay separable when a 401
    arrives and nobody has to guess which layer refused."""
    c = InklingClient(endpoint_url=endpoint, token_id="wk-test",
                      token_secret="ws-test", app_api_key="sk-app-key")
    c.generate_json("hello")
    sent = _LAST["headers"]
    assert sent["modal-key"] == "wk-test"          # proxy layer, unchanged
    assert sent["authorization"] == "Bearer sk-app-key"   # app layer, separate
    assert "wk-test" not in sent["authorization"]


def test_the_modal_headers_go_out_in_modals_documented_spelling(client):
    """`Modal-Key` / `Modal-Secret` as Modal writes them. urllib title-cases every
    header name on the way out, which happens to match here — see
    test_header_names_are_sent_in_http_title_case in test_llm.py for why that is
    a documented behaviour rather than something to "fix"."""
    import http.client
    seen = {}
    real = http.client.HTTPConnection.putheader

    def spy(self, header, *values):
        seen[header] = values
        return real(self, header, *values)

    http.client.HTTPConnection.putheader = spy
    try:
        client.generate_json("hello")
    finally:
        http.client.HTTPConnection.putheader = real
    assert "Modal-Key" in seen, f"wrong spelling; sent {list(seen)}"
    assert "Modal-Secret" in seen


def test_a_class_endpoint_url_404_is_explained_as_a_url_problem(client):
    """The live failure mode after auth succeeds: a Modal *class* endpoint needs
    the method name in the hostname. Reporting a bare 404 there would look like
    the model is missing when the credential is actually fine."""
    err = client._http_error(
        404, "modal-http: Class web endpoint requires a method name", False)
    msg = str(err)
    assert "Auth SUCCEEDED" in msg
    assert "<workspace>--<app>-<class>-<method>.modal.run" in msg


def test_the_two_modal_hostnames_are_flagged_as_separate_auth_realms(client):
    """Verified live: the same proxy token clears `*.modal.run` and is refused by
    `*.modal.direct`. Someone holding a working token and the wrong URL needs to
    be told that, or they will go mint tokens that also do not work."""
    err = client._http_error(401, '{"error": "invalid token"}', False)
    assert "DIFFERENT auth realms" in str(err)


def test_a_401_names_both_headers_and_the_token_page(endpoint):
    """Never a generic 'invalid API key' — the fix is a specific pair of headers
    and a token page where the secret is shown exactly once."""
    c = InklingClient(endpoint_url=endpoint, token_id="wk-wrong",
                      token_secret="ws-wrong", model="m")
    with pytest.raises(InklingError) as e:
        c.generate("hi")
    msg = str(e.value)
    assert "Modal-Key" in msg and "Modal-Secret" in msg
    assert "proxy-auth-tokens" in msg
    assert "shown only once" in msg


def test_credentials_never_leak_into_an_error(endpoint):
    """Errors get logged and pasted into tickets; a proxy token in one is a
    credential leak. Scrubbing is by value AND by shape, so a token this process
    did not resolve itself still cannot ride along."""
    c = InklingClient(endpoint_url=endpoint, token_id="wk-supersecret",
                      token_secret="ws-supersecret", model="m")
    with pytest.raises(InklingError) as e:
        c.generate("hi")
    assert "supersecret" not in str(e.value)
    assert scrub("token wk-abc123 and ws-def456 leaked") == \
        "token wk-<redacted> and ws-<redacted> leaked"


# --------------------------------------------------------------------------- #
# REQ-INK-03 — credential resolution order
# --------------------------------------------------------------------------- #
def test_env_beats_keychain_and_reports_its_source(monkeypatch):
    """"Which credential did it actually use?" is the first question when a 401
    appears on a machine with more than one, so the source is returned, not
    inferred."""
    monkeypatch.setenv("MODAL_TOKEN_ID", "wk-env")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "ws-env")
    monkeypatch.setattr("beatforge.inkling._keychain_get", lambda: ("wk-kc", "ws-kc"))
    assert resolve_credentials()[:3] == ("wk-env", "ws-env", "env")
    # An explicit pair still wins over both.
    assert resolve_credentials("wk-x", "ws-x") == ("wk-x", "ws-x", "explicit")


def test_keychain_is_used_when_env_is_absent(monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setattr(config, "MODAL_TOKEN_ID", "")
    monkeypatch.setattr(config, "MODAL_TOKEN_SECRET", "")
    monkeypatch.setattr("beatforge.inkling._keychain_get", lambda: ("wk-kc", "ws-kc"))
    assert resolve_credentials() == ("wk-kc", "ws-kc", "keychain")


def test_missing_credentials_point_at_the_token_page(monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setattr(config, "MODAL_TOKEN_ID", "")
    monkeypatch.setattr(config, "MODAL_TOKEN_SECRET", "")
    monkeypatch.setattr("beatforge.inkling._keychain_get", lambda: None)
    with pytest.raises(InklingError, match="proxy-auth-tokens"):
        resolve_credentials()


# --------------------------------------------------------------------------- #
# Model resolution — never hardcoded
# --------------------------------------------------------------------------- #
def test_model_is_resolved_from_the_endpoint(client):
    """vLLM deployments sometimes serve under an alias; a hardcoded id 404s with
    a message that never mentions the model."""
    assert client.resolve_model() == "thinkingmachines/Inkling-NVFP4"
    client.generate_json("hi")
    assert _LAST["body"]["model"] == "thinkingmachines/Inkling-NVFP4"


def test_an_alias_still_resolves_and_still_bills(endpoint, monkeypatch):
    """A shortened alias must not fall through to the unknown-model price, which
    would report $0 and hide the spend."""
    monkeypatch.setattr(_Handler, "served_model", "inkling")
    c = InklingClient(endpoint_url=endpoint, token_id="wk-test", token_secret="ws-test")
    assert c.resolve_model() == "inkling"
    assert pricing.price_for("inkling").out == 5.00
    assert pricing.price_for("inkling").model == "thinkingmachines/Inkling-NVFP4"


# --------------------------------------------------------------------------- #
# REQ-INK-04 — full-track audio, one call, no chunking
# --------------------------------------------------------------------------- #
def test_the_whole_track_goes_in_one_call_with_one_audio_part(client, click_wav):
    """Inkling takes ~20 minutes of audio, so the chunk-and-stitch machinery other
    audio backends need is not applied here: one call, one audio part, whole
    track. Splitting it would cost extra calls AND fracture the musical context
    the designer is being paid to hear."""
    wav = click_wav(bpm=128.0, dur_s=30.0)
    client.generate_json("design this chart", audio_path=wav)
    bodies = _LAST["bodies"]
    assert len(bodies) == 1, "one designer call, not a chunk sequence"
    audio = audio_parts(bodies[0]["messages"][-1]["content"])
    assert len(audio) == 1
    assert audio_payload(audio[0])[1] == "wav"


def test_audio_is_sent_as_16k_mono_wav(client, click_wav, tmp_path):
    """Inkling's documented input format. The source here is 22.05 kHz stereo-
    capable, so this proves conversion happens rather than the source sample rate
    riding through."""
    import base64
    wav = click_wav(bpm=120.0, dur_s=3.0, sr=22050)
    client.generate_json("x", audio_path=wav)
    part = audio_parts(_LAST["body"]["messages"][-1]["content"])[0]
    out = tmp_path / "sent.wav"
    out.write_bytes(base64.b64decode(audio_payload(part)[0]))
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1


def test_an_overlong_track_fails_before_upload_not_as_a_413(client, monkeypatch):
    """Rejecting locally beats uploading a body the gateway will refuse: the
    message names the limit and says plainly that chunking is not the answer on
    this backend."""
    monkeypatch.setattr("beatforge.inkling.audio_duration_s", lambda p: 1800.0)
    with pytest.raises(InklingError) as e:
        client.generate("design", audio_path="/tmp/whatever.wav")
    msg = str(e.value)
    assert "30.0 min" in msg and "20 min" in msg
    assert "NOT sent in chunks" in msg
    assert "bodies" not in _LAST, "nothing should have been uploaded"


def test_duration_is_read_from_the_file(click_wav):
    assert audio_duration_s(click_wav(bpm=120.0, dur_s=4.0)) == pytest.approx(4.0, abs=0.1)


# --------------------------------------------------------------------------- #
# REQ-INK-05 — the differential audio probe
# --------------------------------------------------------------------------- #
#: What a model that actually hears the two synthesised tones must answer:
#: one forced-choice word each, 110 Hz then 3 kHz.
CORRECT = ["LOW", "HIGH"]
#: Right format, wrong facts — a deaf endpoint answering from the prompt alone.
DEAF = ["HIGH", "LOW"]


def test_probe_passes_when_both_clips_are_described_correctly(client):
    _Handler.replies = list(CORRECT)
    verdict = differential_audio_probe(client)
    assert verdict["passed"] is True
    assert verdict["observed"] == verdict["expected"]


def test_probe_tolerates_prose_around_the_answer(client):
    """The verdict is parsed, not string-matched. A model that obeys the format
    but cannot resist a sentence of preamble is still a model that heard the
    clip, and failing it would stop the batch for nothing."""
    _Handler.replies = ["Sure! LOW — a deep, sustained bass tone.",
                        "That one is HIGH: bright and piercing."]
    assert differential_audio_probe(client)["passed"] is True


def test_probe_fails_when_the_tones_are_described_wrongly(client):
    """The dangerous failure mode: a deaf endpoint answers confidently from the
    prompt alone. Ground truth catches it whatever words it chooses — here it
    gets the format right and the facts backwards."""
    _Handler.replies = list(DEAF)
    verdict = differential_audio_probe(client)
    assert verdict["passed"] is False
    assert "judged the test tones incorrectly" in verdict["reason"]


def test_probe_does_not_fail_a_listening_model_for_repeating_itself(client):
    """The regression this replaced. The old probe failed any endpoint whose two
    answers were >85% similar, which punishes a boilerplate-prone model for
    something that is not the failure being hunted — and a failed probe stops the
    whole run. Correct answers pass no matter how alike they look."""
    stock = "The audio is a single sustained tone pulsing at a regular rate. "
    _Handler.replies = [stock + CORRECT[0], stock + CORRECT[1]]
    verdict = differential_audio_probe(client)
    assert verdict["similarity"] > 0.85, "these two answers are deliberately alike"
    assert verdict["passed"] is True


def test_probe_fails_an_unreadable_answer_rather_than_guessing_it(client):
    """A reply naming neither word — or both — is a miss, never a coin flip in
    the model's favour."""
    _Handler.replies = ["it is somewhere in the middle", "HIGH"]
    assert differential_audio_probe(client)["passed"] is False


def test_probe_fails_when_both_clips_get_the_same_answer(client):
    """The second check, and the one that catches a deaf endpoint even when it
    guesses the right word. At temperature 0 a model answering from the prompt
    alone is deterministic, so two tones three octaves apart come back
    identical — impossible for anything that actually listened."""
    _Handler.replies = ["LOW", "LOW"]
    verdict = differential_audio_probe(client)
    assert verdict["passed"] is False
    assert "same answer" in verdict["reason"]


def test_probe_honours_the_no_audio_sentinel(client):
    _Handler.replies = ["NO_AUDIO_RECEIVED", "NO_AUDIO_RECEIVED"]
    verdict = differential_audio_probe(client)
    assert verdict["passed"] is False
    assert "received no audio" in verdict["reason"]


def test_probe_sends_two_different_clips_at_temperature_zero(client):
    """Determinism is what makes a wrong answer the model's rather than the
    sampler's; without it the probe proves nothing."""
    _Handler.replies = list(CORRECT)
    differential_audio_probe(client)
    bodies = _LAST["bodies"]
    assert len(bodies) == 2
    assert all(b["temperature"] == 0 for b in bodies)
    clips = [audio_payload(audio_parts(b["messages"][-1]["content"])[0])[0]
             for b in bodies]
    assert clips[0] != clips[1], "the two probe clips must actually differ"
    assert bodies[0]["messages"][-1]["content"][0]["text"] == \
        bodies[1]["messages"][-1]["content"][0]["text"], "same prompt both times"


def test_require_audio_blocks_the_batch_on_a_deaf_endpoint(client):
    """REQ-INK-05 gates the batch: work must not proceed past an endpoint that
    ignores audio, because every chart it produced would be designed blind."""
    _Handler.replies = list(DEAF)
    with pytest.raises(InklingAudioIgnored) as e:
        require_audio(client)
    assert "audio probe FAILED" in str(e.value)


def test_the_gate_can_be_downgraded_to_a_warning(client, monkeypatch, capsys):
    """An operator who has verified the endpoint by hand can take the probe out
    of the critical path — but never silently: a run designed without hearing the
    music has to leave a mark."""
    monkeypatch.setattr(config, "INKLING_REQUIRE_AUDIO", False)
    _Handler.replies = ["PITCH=HIGH PULSE=FAST", "PITCH=LOW PULSE=SLOW"]
    verdict = require_audio(client)
    assert verdict["passed"] is False
    assert "WARNING" in capsys.readouterr().err


def test_the_probe_is_ledgered(client):
    """Its cost and latency are real spend and have to show up like any other
    call."""
    _Handler.replies = list(CORRECT)
    with ledger.capture() as entries:
        differential_audio_probe(client)
    probe_rows = [e for e in entries if e.get("stage") == "audio_probe"]
    assert len(probe_rows) == 2
    assert all(r["provider"] == "inkling" for r in probe_rows)
    assert all(r["latency_s"] >= 0 for r in probe_rows)


# --------------------------------------------------------------------------- #
# REQ-INK-06 / 07 — ledger parity and the reasoning dial
# --------------------------------------------------------------------------- #
def test_all_four_token_classes_are_billed(client):
    """prompt / cached prompt / completion / reasoning are priced separately, and
    a report has to be able to say which dominated."""
    with ledger.capture() as entries:
        client.generate_json("hi")
    row = [e for e in entries if e.get("kind") == "model"][0]
    assert row["provider"] == "inkling"
    assert row["cached_tokens"] == 400
    assert row["audio_tokens"] == 600
    assert row["thinking_tokens"] == 60
    cost = row["cost_usd"]
    assert cost["cached_in"] == pytest.approx(400 / 1e6 * 0.27)
    assert cost["thinking"] == pytest.approx(60 / 1e6 * 5.00)
    assert cost["out"] == pytest.approx(100 / 1e6 * 5.00)


def test_reasoning_effort_follows_the_pipeline_stage(client):
    """The designer gets the thinking budget; the critic does not. Reasoning
    bills at the full output rate, so this is a cost lever — and it must work
    without any call site passing an effort level (REQ-INK-01)."""
    with ledger.stage("designer", song="s", difficulty="hard"):
        client.generate_json("design")
    assert _LAST["body"]["reasoning_effort"] == "high"
    with ledger.stage("critic", song="s", difficulty="hard"):
        client.generate_json("judge")
    assert _LAST["body"]["reasoning_effort"] == "low"
    with ledger.stage("reprompt", song="s", difficulty="hard"):
        client.generate_json("again")
    assert _LAST["body"]["reasoning_effort"] == "medium"


def test_the_effort_used_is_recorded_on_the_ledger_line(client):
    """Otherwise a cheaper re-run cannot be attributed to the dial."""
    with ledger.capture() as entries:
        with ledger.stage("critic", song="s", difficulty="easy"):
            client.generate_json("judge")
    row = [e for e in entries if e.get("kind") == "model"][0]
    assert row["thinking_level"] == "low"


def test_rates_are_stamped_as_unverified_until_an_invoice_confirms_them(client):
    """The 3.6x-low output-rate incident is why this flag exists: a confidently
    wrong dollar figure is worse than an admittedly uncertain one."""
    p = pricing.price_for("thinkingmachines/Inkling-NVFP4")
    assert (p.text_in, p.cached_in, p.out, p.thinking_out) == (1.20, 0.27, 5.00, 5.00)
    assert p.verified is False
    assert "UNVERIFIED" in p.note


# --------------------------------------------------------------------------- #
# REQ-INK-09 — the GUI's two buttons, via the CLI they shell out to
# --------------------------------------------------------------------------- #
def _run_cli(sub: str, endpoint: str, token_id: str = "wk-test") -> tuple[int, dict]:
    """Invoke the engine exactly as the desktop app does: a subprocess with the
    credentials in its environment, one json object on stdout."""
    repo = Path(__file__).resolve().parents[3]
    env = {**os.environ,
           "PYTHONPATH": str(repo / "tools"),
           "BEATFORGE_INKLING_ENDPOINT_URL": endpoint,
           "MODAL_TOKEN_ID": token_id,
           "MODAL_TOKEN_SECRET": "ws-test"}
    r = subprocess.run([sys.executable, "-m", "beatforge", sub],
                       capture_output=True, text=True, env=env, cwd=str(repo))
    line = [l for l in r.stdout.splitlines() if l.strip().startswith("{")][-1]
    return r.returncode, json.loads(line)


def test_cli_check_resolves_auth_and_model_as_json(endpoint):
    """"Test connection" in the GUI is this command; it must answer in json the
    app can render, not prose."""
    code, out = _run_cli("inkling-check", endpoint)
    assert code == 0 and out["ok"] is True
    assert out["model"] == "thinkingmachines/Inkling-NVFP4"
    assert out["credentialSource"] == "env"


def test_cli_check_reports_a_bad_token_without_crashing(endpoint):
    """A failed check is a verdict, not a stack trace — and the message has to
    name the two headers, since that is the actual fix."""
    code, out = _run_cli("inkling-check", endpoint, token_id="wk-wrong")
    assert code == 1 and out["ok"] is False
    assert "Modal-Key" in out["error"] and "Modal-Secret" in out["error"]
    assert "wk-wrong" not in out["error"], "credentials must not survive into output"


def test_cli_probe_returns_a_renderable_verdict(endpoint):
    """The probe button needs pass/fail plus both answers, so a user can see for
    themselves why it was judged deaf or listening."""
    _Handler.replies = list(CORRECT)
    code, out = _run_cli("inkling-probe", endpoint)
    assert code == 0 and out["passed"] is True
    assert len(out["responses"]) == 2
    # What the user is shown is the comparison itself: what a listening endpoint
    # had to say versus what this one did.
    assert out["expected"] == out["observed"] == list(CORRECT)


def test_cli_probe_exits_nonzero_when_the_endpoint_is_deaf(endpoint):
    """Non-zero exit so a batch script or CI can gate on it too, not just the GUI."""
    _Handler.replies = list(DEAF)
    code, out = _run_cli("inkling-probe", endpoint)
    assert code == 1 and out["passed"] is False
    assert "judged the test tones incorrectly" in out["reason"]


# --------------------------------------------------------------------------- #
# REQ-INK-08 — prompt ordering for the cached-prompt discount
# --------------------------------------------------------------------------- #
def _analysis_fixture():
    onsets = [{"id": f"p{i:03d}", "time": i * 0.5, "nearest_beat": float(i),
               "snap_error_ms": 2.0, "strength": 0.6,
               "bands": {"low": 0.6, "mid": 0.3, "high": 0.1}, "source": "mix",
               "sustain": False, "sustain_beats": 0.0} for i in range(1, 120)]
    return {"bpm": 120.0, "offset": 0.1, "duration_s": 120.0, "onsets": onsets,
            "sections": [{"name": "S0", "start_bar": 0, "end_bar": 8,
                          "energy_pct": 0.5, "role_guess": "intro"}],
            "energy_curve": [0.5] * 16, "track_file": "t.ogg",
            "beats": [i * 0.5 for i in range(240)]}


def test_variable_prompt_content_comes_after_the_stable_content():
    """Cached prompt tokens cost 4.4x less ($0.27 vs $1.20), and the discount is
    prefix-based: it only applies up to the first byte that differs. So the parts
    that change per difficulty and per attempt must come LAST.

    This pins the ordering rather than the wording — an edit that moves the
    per-attempt violations or the per-difficulty hold list earlier would silently
    shorten the cacheable prefix and cost money with nothing failing.
    """
    from beatforge import design
    a = _analysis_fixture()
    diff = list(config.BUDGETS)[0]
    prompt = design._designer_prompt(diff, a, violations="YOU BROKE GATE X")

    brief_end = prompt.index("--- MACHINE ANALYSIS")
    holds = max(prompt.find("--- HOLD REQUIREMENT"), prompt.find("--- HOLDS ---"))
    violations = prompt.index("--- YOUR PREVIOUS ATTEMPT FAILED")
    assert brief_end < holds < violations, \
        "per-difficulty then per-attempt content must be the tail of the prompt"
    assert violations > len(prompt) * 0.5, "the retry delta belongs at the very end"


def test_the_cacheable_prefix_is_measured_not_assumed():
    """The shared prefix across difficulties is what the cache can actually bill
    at the discounted rate. Today it ends at the `YOUR JOB: design a **{difficulty}**`
    line — about 6% of a ~10k-char prompt — because the difficulty is substituted
    into the brief near the top.

    This test does not assert that 6% is good (it is not; see the README note on
    REQ-INK-08). It asserts the prefix is non-trivial and that the divergence
    point is where we think it is, so a change that helps or hurts it shows up
    here as a number instead of a surprise on the invoice.
    """
    from beatforge import design
    a = _analysis_fixture()
    ds = list(config.BUDGETS)[:2]
    x = design._designer_prompt(ds[0], a, None)
    y = design._designer_prompt(ds[1], a, None)
    shared = 0
    for ca, cb in zip(x, y):
        if ca != cb:
            break
        shared += 1
    assert shared > 400, "the standing brief should be shared across difficulties"
    assert "YOUR JOB" in x[max(0, shared - 200):shared + 40], \
        "divergence is expected at the difficulty substitution"


def test_the_cost_report_surfaces_the_cache_hit_share():
    """A near-zero cache share is money left on the table; nobody notices what is
    not reported."""
    from beatforge import costreport
    entries = [{"kind": "model", "provider": "inkling", "model": "x",
                "text_tokens": 600, "audio_tokens": 0, "cached_tokens": 400,
                "output_tokens": 50, "thinking_tokens": 10,
                "cost_usd": {"total": 0.01}, "song": "s", "difficulty": "casual"}]
    agg = costreport.aggregate(entries)
    assert agg["tokens"]["cache_hit_share_of_input"] == pytest.approx(0.4)
    assert "cache hit share of input" in costreport.render_markdown(agg)


# --------------------------------------------------------------------------- #
# REQ-INK-01 — provider isolation
# --------------------------------------------------------------------------- #
def _configure_backend(monkeypatch, endpoint):
    monkeypatch.setattr(config, "LLM_BACKEND", "inkling")
    monkeypatch.setattr(config, "INKLING_ENDPOINT_URL", endpoint)
    monkeypatch.setenv("MODAL_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "ws-test")
    inkling_mod._PROBED.clear()


def test_the_factory_selects_inkling(monkeypatch, endpoint):
    _configure_backend(monkeypatch, endpoint)
    _Handler.replies = list(CORRECT)
    assert isinstance(make_llm_client(), InklingClient)


def test_the_factory_blocks_a_batch_on_a_deaf_endpoint(monkeypatch, endpoint):
    """The gate has to sit at the choke point every run passes through. If it
    only existed as a function nobody called, a deaf endpoint would chart a whole
    pack before anyone noticed the charts were designed from the prompt alone."""
    _configure_backend(monkeypatch, endpoint)
    _Handler.replies = list(DEAF)
    with pytest.raises(InklingAudioIgnored):
        make_llm_client()


def test_the_probe_runs_once_per_endpoint_per_process(monkeypatch, endpoint):
    """Two extra calls per run is a fair price for the guarantee; two per song
    would not be."""
    _configure_backend(monkeypatch, endpoint)
    _Handler.replies = list(CORRECT)
    make_llm_client()
    after_first = len(_LAST["bodies"])
    make_llm_client()
    assert len(_LAST["bodies"]) == after_first, "second client must not re-probe"


def test_the_probe_can_be_skipped_once_an_endpoint_is_proven(monkeypatch, endpoint):
    """An escape hatch for someone who has already verified the endpoint and
    wants the two calls back — explicit, not the default."""
    _configure_backend(monkeypatch, endpoint)
    monkeypatch.setenv("BEATFORGE_INKLING_SKIP_PROBE", "1")
    _Handler.replies = list(DEAF)
    assert isinstance(make_llm_client(), InklingClient)  # deaf, but not gated
    assert "bodies" not in _LAST, "skipping the probe means making no probe calls"


def test_a_url_without_the_v1_suffix_still_works(endpoint):
    """The Modal dashboard hands you the bare endpoint URL; requiring the user to
    remember /v1 is a support ticket waiting to happen."""
    c = InklingClient(endpoint_url=endpoint, token_id="wk-test", token_secret="ws-test")
    assert c.base_url.endswith("/v1")
    assert c.resolve_model()


def test_a_missing_endpoint_says_which_setting_is_empty(monkeypatch):
    monkeypatch.setattr(config, "INKLING_ENDPOINT_URL", "")
    with pytest.raises(InklingError, match="BEATFORGE_INKLING_ENDPOINT_URL"):
        InklingClient(token_id="wk-a", token_secret="ws-b")


# --------------------------------------------------------------------------- #
# The output cap and the effort dial have to agree, or the budget is spent
# thinking and the answer never arrives
# --------------------------------------------------------------------------- #
def test_inkling_does_not_inherit_the_local_servers_output_cap(client):
    """The openai backend's 4096 was tuned to stop a slow local 12B from looping
    into its own VRAM. Inkling is a reasoning model with a 1M context, and
    reasoning tokens count against the SAME cap — so inheriting that number means
    the designer spends the whole budget thinking at effort=high and returns
    empty content on an endpoint that is working fine."""
    client.generate_json("design a chart")
    sent = _LAST["body"]
    assert sent["max_tokens"] == config.INKLING_MAX_TOKENS
    assert sent["max_tokens"] > config.OPENAI_MAX_TOKENS


def test_the_retry_effort_dial_is_actually_reachable(client, monkeypatch):
    """A retry is NOT its own ledger stage — design.py opens
    `ledger.stage("designer", attempt=n)` for every draft. Keying the retry level
    on a "reprompt" stage meant BEATFORGE_INKLING_EFFORT_RETRY was wired from the
    GUI through Rust into config and then looked up under a key nothing ever
    sets, leaving every retry at the designer's `high` — the most expensive
    setting, on the calls that need it least."""
    monkeypatch.setattr(config, "INKLING_EFFORT_RETRY", "medium")
    with ledger.stage("designer", song="s", target="core", attempt=1):
        assert client._call_effort() == config.INKLING_EFFORT["designer"]
    with ledger.stage("designer", song="s", target="core", attempt=2):
        assert client._call_effort() == "medium"


def test_effort_rides_the_stage_onto_the_wire(client):
    """The dial is only a cost lever if the endpoint receives it."""
    with ledger.stage("critic", song="s", target="core"):
        client.generate_json("check this")
    assert _LAST["body"]["reasoning_effort"] == config.INKLING_EFFORT["critic"]


def test_the_probe_does_not_pay_for_reasoning(client):
    """It runs before every batch and only has to say whether a 4-second tone is
    high or low. Reasoning bills at the full output rate."""
    _Handler.replies = list(CORRECT)
    differential_audio_probe(client)
    assert all(b["reasoning_effort"] == config.INKLING_EFFORT["audio_probe"]
               for b in _LAST["bodies"])


def test_audio_goes_out_in_the_format_the_part_claims(client, click_wav, monkeypatch):
    """`format` names the container the bytes are actually in — the server picks
    its decoder off that string, so a mismatch surfaces as a decode failure deep
    in librosa rather than as anything the caller can read. The knob exists
    because 20 min of 16 kHz PCM is ~51MB base64 and flac halves that."""
    monkeypatch.setattr(config, "INKLING_AUDIO_FORMAT", "flac")
    c = InklingClient(endpoint_url=client.base_url.removesuffix("/v1"),
                      token_id="wk-test", token_secret="ws-test", model="m")
    c.generate_json("analyze", audio_path=click_wav(bpm=120.0, dur_s=4.0))
    b64, container = audio_payload(
        audio_parts(_LAST["body"]["messages"][-1]["content"])[0])
    assert container == "flac"
    # FLAC starts with the magic "fLaC"; proof the bytes match the claim.
    import base64 as _b64
    assert _b64.b64decode(b64)[:4] == b"fLaC"


# --------------------------------------------------------------------------- #
# Runaway reasoning: the budget goes to the scratchpad and no answer arrives
# --------------------------------------------------------------------------- #
class _TruncatingHandler(BaseHTTPRequestHandler):
    """Burns the whole output cap thinking on the first call, answers on the
    second. What a live endpoint did to the `hard` designer call at effort=high."""

    answer_at_effort = "medium"
    seen: "list[str]" = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        effort = body.get("reasoning_effort")
        type(self).seen.append(effort)
        if effort == self.answer_at_effort:
            msg, fin, usage = ({"content": '{"ok": 1}'}, "stop",
                               {"completion_tokens": 300, "reasoning_tokens": 120})
        else:
            msg, fin, usage = ({"content": "", "reasoning_content": "let me think..."},
                               "length",
                               {"completion_tokens": 65536, "reasoning_tokens": 65536})
        data = json.dumps({"choices": [{"message": msg, "finish_reason": fin}],
                           "usage": usage}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def truncating_endpoint():
    _TruncatingHandler.seen = []
    _TruncatingHandler.answer_at_effort = "medium"
    srv = HTTPServer(("127.0.0.1", 0), _TruncatingHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    yield f"http://{host}:{port}"
    srv.shutdown()


def _client(endpoint, **kw):
    return InklingClient(endpoint_url=endpoint, token_id="wk-test",
                         token_secret="ws-test", model="m", **kw)


def test_a_runaway_reasoning_call_retries_one_step_cheaper(truncating_endpoint):
    """A truncated call is a TOTAL loss: every reasoning token is billed and no
    chart comes back. Retrying at the same effort would burn the cap again, so
    the dial steps down once — a less-deliberated chart beats none."""
    c = _client(truncating_endpoint, effort="high")
    with ledger.stage("designer", song="s", target="core", attempt=1):
        assert c.generate_json("design") == {"ok": 1}
    assert _TruncatingHandler.seen == ["high", "medium"]


def test_the_quality_downgrade_is_announced(truncating_endpoint, capsys):
    """Silently charting at a lower effort than the settings ask for would make
    the run irreproducible and the quality drop invisible."""
    c = _client(truncating_endpoint, effort="high")
    with ledger.stage("designer", song="s", target="core", attempt=1):
        c.generate_json("design")
    err = capsys.readouterr().err
    assert "effort=high" in err and "effort=medium" in err
    assert "less deliberated" in err


def test_it_steps_down_once_not_all_the_way(truncating_endpoint):
    """One rung, not a negotiation. Bargaining the quality down until something
    comes back would spend more than the original call and return a chart nobody
    chose the settings for."""
    _TruncatingHandler.answer_at_effort = "never"      # truncate every time
    c = _client(truncating_endpoint, effort="high")
    with pytest.raises(OpenAICompatError) as e:
        with ledger.stage("designer", song="s", target="core", attempt=1):
            c.generate_json("design")
    assert _TruncatingHandler.seen == ["high", "medium"], "exactly one retry"
    assert "reasoning" in str(e.value).lower()


def test_the_retry_is_billed_at_the_effort_it_actually_used(truncating_endpoint):
    """Both calls are real spend. The failed one especially — a full cap of
    reasoning tokens that produced nothing still lands on the invoice, and a
    ledger that hid it would understate the cost of running at high."""
    c = _client(truncating_endpoint, effort="high")
    with ledger.capture() as rows:
        with ledger.stage("designer", song="s", target="core", attempt=1):
            c.generate_json("design")
    designer = [r for r in rows if r.get("stage") == "designer"]
    assert [r["thinking_level"] for r in designer] == ["high", "medium"]
    assert designer[0]["thinking_tokens"] == 65536, "the wasted thinking is billed"


def test_the_downgrade_can_be_switched_off(truncating_endpoint, monkeypatch):
    """Someone benchmarking effort levels needs the failure, not a rescue."""
    monkeypatch.setattr(config, "INKLING_DEGRADE_ON_TRUNCATION", False)
    c = _client(truncating_endpoint, effort="high")
    with pytest.raises(OpenAICompatError):
        with ledger.stage("designer", song="s", target="core", attempt=1):
            c.generate_json("design")
    assert _TruncatingHandler.seen == ["high"], "no retry when disabled"


def test_the_output_cap_leaves_room_for_the_answer_after_reasoning():
    """Measured, not guessed: one `hard` designer call against a live endpoint
    spent 1.9k output tokens at low, 12.1k at medium and 25.3k at high — and a
    second high run blew past 32.7k. The cap has to cover that tail, because
    exhausting it throws the whole (already billed) call away."""
    assert config.INKLING_MAX_TOKENS >= 65536
