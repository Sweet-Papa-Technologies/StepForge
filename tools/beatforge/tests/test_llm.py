"""OpenAI-compatible backend tests. The real Gemma server is on another subnet
and unreachable from CI, so we validate the client end-to-end against a local
mock that records what it receives — proving the request shape, the audio
attachment, and JSON parsing are correct before it ever talks to the real box."""
import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from beatforge import config
from beatforge.llm import OpenAICompatClient, OpenAICompatError, make_llm_client

_LAST = {}  # records the last request body the mock received


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        _LAST["headers"] = {k.lower(): v for k, v in self.headers.items()}
        if self.path.endswith("/models"):
            self._json({"data": [{"id": "gemma-4-12b"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _LAST["body"] = body
        _LAST["auth"] = self.headers.get("Authorization")
        _LAST["headers"] = {k.lower(): v for k, v in self.headers.items()}
        # echo a canned "analysis" as an OpenAI chat completion
        content = json.dumps({"tempo_bpm": 120, "has_kick": True,
                              "section_count": 3, "one_line": "driving synthwave"})
        self._json({"choices": [{"message": {"role": "assistant", "content": content}}]})

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def mock_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    host, port = srv.server_address
    yield f"http://{host}:{port}/v1"
    srv.shutdown()


def test_factory_selects_openai(monkeypatch):
    monkeypatch.setattr("beatforge.config.LLM_BACKEND", "openai")
    assert isinstance(make_llm_client(), OpenAICompatClient)


def test_list_models(mock_server):
    c = OpenAICompatClient(base_url=mock_server, model="gemma-4-12b")
    assert "gemma-4-12b" in c.list_models()


def test_text_generate_json(mock_server):
    c = OpenAICompatClient(base_url=mock_server, model="gemma-4-12b", api_key="k")
    out = c.generate_json("describe", audio_path=None)
    assert out["tempo_bpm"] == 120 and out["has_kick"] is True
    # request carried a bearer token and a text-only user message
    assert _LAST["auth"] == "Bearer k"
    msg = _LAST["body"]["messages"][-1]
    assert msg["role"] == "user"
    # A bare string, not [{"type":"text",...}]: strict servers reject the list
    # form with "content: Input should be a valid string", and with no audio to
    # carry the list buys nothing.
    assert msg["content"] == "describe"
    assert _LAST["body"]["response_format"] == {"type": "json_object"}


def test_audio_attached_as_input_audio(mock_server, click_wav):
    """The audio part must be an OpenAI `input_audio` block with the configured
    format and non-empty base64 — the whole point of an audio-in model."""
    wav = click_wav(bpm=120.0, dur_s=4.0)
    c = OpenAICompatClient(base_url=mock_server, model="gemma-4-12b", audio_format="wav")
    c.generate_json("analyze this audio", audio_path=wav)
    parts = _LAST["body"]["messages"][-1]["content"]
    audio = [p for p in parts if p["type"] == "input_audio"]
    assert len(audio) == 1
    assert audio[0]["input_audio"]["format"] == "wav"
    assert len(audio[0]["input_audio"]["data"]) > 100  # real base64 payload


def test_custom_headers_are_sent(mock_server):
    """Arbitrary headers are the escape hatch for gateways that want more than a
    bearer token (an API-key header, a tenant id, a version pin). They must reach
    the wire on every call — chat completions AND the /models probe."""
    c = OpenAICompatClient(base_url=mock_server, model="gemma-4-12b", api_key="k",
                           headers={"X-Api-Key": "abc", "X-Tenant": "acme"})
    c.generate_json("describe")
    assert _LAST["headers"]["x-api-key"] == "abc"
    assert _LAST["headers"]["x-tenant"] == "acme"
    assert _LAST["auth"] == "Bearer k"  # defaults survive alongside them
    c.list_models()
    assert _LAST["headers"]["x-api-key"] == "abc"


def test_custom_header_overrides_default_case_insensitively(mock_server):
    """A gateway that authenticates some other way needs the bearer token GONE,
    not duplicated — so a custom `authorization` (any casing) replaces it."""
    c = OpenAICompatClient(base_url=mock_server, model="gemma-4-12b", api_key="k",
                           headers={"authorization": "Basic dXNlcjpwdw=="})
    c.generate_json("describe")
    assert _LAST["auth"] == "Basic dXNlcjpwdw=="


def test_blank_custom_header_drops_the_default(mock_server):
    """Empty value = "don't send this header at all"; some gateways 400 on an
    unexpected Authorization."""
    c = OpenAICompatClient(base_url=mock_server, model="gemma-4-12b", api_key="k",
                           headers={"Authorization": ""})
    c.generate_json("describe")
    assert _LAST["auth"] is None


def test_headers_default_to_config(mock_server, monkeypatch):
    """With no explicit argument the client picks up BEATFORGE_OPENAI_HEADERS —
    that is how the GUI's custom-header table reaches the engine."""
    monkeypatch.setattr("beatforge.config.OPENAI_HEADERS", {"X-Env": "from-config"})
    c = OpenAICompatClient(base_url=mock_server, model="gemma-4-12b")
    c.generate_json("describe")
    assert _LAST["headers"]["x-env"] == "from-config"


@pytest.fixture
def wire_headers(monkeypatch):
    """Record the header lines http.client actually writes.

    `_LAST["headers"]` is what the *server* parsed, which is case-insensitive and
    silently merges duplicates — so it cannot see the two failures that matter
    most here: capitalisation mangled in transit, and an override that leaves two
    conflicting headers on the request. This spies one layer lower, on the bytes
    the client emits.
    """
    import http.client
    sent: list[tuple[str, str]] = []
    real = http.client.HTTPConnection.putheader

    def spy(self, header, *values):
        for v in values:
            sent.append((header if isinstance(header, str) else header.decode(),
                         v if isinstance(v, str) else v.decode()))
        return real(self, header, *values)

    monkeypatch.setattr(http.client.HTTPConnection, "putheader", spy)
    return sent


def test_custom_headers_reach_the_wire_with_exact_values(mock_server, wire_headers):
    """Each configured header goes out once, with its value byte-exact."""
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k",
                           headers={"X-Api-Key": "abc", "anthropic-version": "2023-06-01"})
    c.generate_json("hi")
    sent = {n.lower(): v for n, v in wire_headers}
    assert sent["x-api-key"] == "abc"
    assert sent["anthropic-version"] == "2023-06-01"


def test_header_names_are_sent_in_http_title_case(mock_server, wire_headers):
    """Documents what urllib actually does, so nobody "fixes" it twice.

    `AbstractHTTPHandler.do_open` runs every name through `str.title()` on the
    way out, so `anthropic-version` becomes `Anthropic-Version` no matter what
    `Request.headers` holds. Field names are case-insensitive (RFC 7230), so this
    is correct — but an earlier attempt here rewrote `Request.headers` to
    "preserve" the user's capitalisation and was a pure no-op, because title() runs
    afterwards. The limitation it leaves is real though: a server demanding a
    literal `X-API-Key` receives `X-Api-Key`, and only bypassing urllib fixes that.
    """
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k",
                           headers={"anthropic-version": "2023-06-01",
                                    "X-API-Key": "abc"})
    c.generate_json("hi")
    names = [n for n, _ in wire_headers]
    assert "Anthropic-Version" in names
    assert "X-Api-Key" in names, "title() folds X-API-Key to X-Api-Key"


def test_overriding_a_default_sends_exactly_one_copy(mock_server, wire_headers):
    """Two Authorization headers is worse than the wrong one: gateways differ on
    which they honour, so the bug would reproduce only on some deployments."""
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k",
                           headers={"authorization": "Basic dXNlcjpwdw=="})
    c.generate_json("hi")
    auth = [(n, v) for n, v in wire_headers if n.lower() == "authorization"]
    assert len(auth) == 1, f"expected one Authorization, got {auth}"
    assert auth[0][1] == "Basic dXNlcjpwdw=="


def test_a_dropped_default_never_reaches_the_wire(mock_server, wire_headers):
    """Blank value means "do not send this header" — it must be absent from the
    request, not sent empty."""
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k",
                           headers={"Authorization": ""})
    c.generate_json("hi")
    assert not [n for n, _ in wire_headers if n.lower() == "authorization"]


def test_custom_headers_ride_the_models_probe_too(mock_server, wire_headers):
    """The probe is the first request a new endpoint gets; a gateway that needs a
    header needs it there as well, or "test connection" fails on a config that
    would actually work."""
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k",
                           headers={"X-Api-Key": "abc"})
    c.list_models()
    assert ("X-Api-Key", "abc") in wire_headers


def test_a_later_row_wins_when_two_names_differ_only_in_case(mock_server, wire_headers):
    """The GUI warns about duplicates, but it cannot stop them. Whatever the
    resolution is, exactly one header must go out."""
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k",
                           headers={"X-Api-Key": "first", "x-api-key": "second"})
    c.generate_json("hi")
    keys = [(n, v) for n, v in wire_headers if n.lower() == "x-api-key"]
    assert len(keys) == 1 and keys[0][1] == "second"


# A settings box takes arbitrary typing, so these are things real users do. Each
# one otherwise surfaces from inside http.client as a bare ValueError or a
# latin-1 UnicodeEncodeError that names neither the header nor the fix.
@pytest.mark.parametrize("name,value,expect", [
    ("X-Api-Key", "abc\r\nX-Injected: evil", "line break"),   # header injection
    ("X-Api-Key", "line one\nline two", "line break"),        # pasted multi-line
    ("X-Api-Key", "sk-café–smart", "non-ASCII"),              # smart paste
    ("X-Bad:Name", "abc", "not valid"),                       # colon in the name
    ("X Bad Name", "abc", "not valid"),                       # space in the name
])
def test_an_unsendable_header_is_refused_by_name(mock_server, name, value, expect):
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k",
                           headers={name: value})
    with pytest.raises(OpenAICompatError) as e:
        c.generate_json("hi")
    msg = str(e.value)
    assert expect in msg
    assert repr(name) in msg, "the message must say WHICH header is at fault"
    assert "Custom headers" in msg, "and where to fix it"


def test_a_crlf_value_is_refused_rather_than_quietly_stripped(mock_server, wire_headers):
    """Sanitising an injection attempt hides it. Nothing should be sent at all."""
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k",
                           headers={"X-Api-Key": "ok\r\nX-Injected: evil"})
    with pytest.raises(OpenAICompatError):
        c.generate_json("hi")
    assert not [n for n, _ in wire_headers if n.lower() == "x-injected"]


def test_the_json_the_desktop_app_writes_is_what_the_engine_parses(mock_server, wire_headers):
    """Contract across the Rust/Python boundary: the desktop app exports the
    header table as a serde_json object in BEATFORGE_OPENAI_HEADERS. This is that
    exact shape — including serde's \\u escaping — parsed and put on the wire."""
    exported = '{"X-Api-Key":"abc","X-Tenant":"acme corp","X-Quote":"a\\"b\\\\c"}'
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k",
                           headers=config._parse_headers(exported))
    c.generate_json("hi")
    assert ("X-Api-Key", "abc") in wire_headers
    assert ("X-Tenant", "acme corp") in wire_headers
    assert ("X-Quote", 'a"b\\c') in wire_headers


@pytest.mark.parametrize("raw,expected", [
    ('{"X-Api-Key": "abc", "X-Tenant": "acme"}', {"X-Api-Key": "abc", "X-Tenant": "acme"}),
    ("X-Api-Key: abc\nX-Tenant: acme", {"X-Api-Key": "abc", "X-Tenant": "acme"}),
    ("X-Api-Key: abc; X-Tenant: acme", {"X-Api-Key": "abc", "X-Tenant": "acme"}),
    ("", {}),
    ("   ", {}),
    ("{not json", {}),          # a typo must not take down an otherwise-fine run
    ("no-colon-here", {}),
    ('{"X-Num": 7}', {"X-Num": "7"}),  # values are coerced to str for urllib
])
def test_header_env_parsing(raw, expected):
    assert config._parse_headers(raw) == expected


class _RejectUploadHandler(BaseHTTPRequestHandler):
    """A gateway that refuses a request BEFORE reading its body — how Modal's
    proxy (and any auth-checking reverse proxy) answers an unauthorized call.
    The client is still uploading megabytes when the socket dies, so it never
    receives the status: it just sees a broken pipe."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._reject()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self._reject()
        if n > 100_000:
            # The big upload: hang up mid-stream without draining the body.
            try:
                self.connection.close()
            except OSError:
                pass

    def _reject(self):
        payload = b'{"error":"proxy auth required"}'
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True


@pytest.fixture
def rejecting_server():
    srv = HTTPServer(("127.0.0.1", 0), _RejectUploadHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    yield f"http://{host}:{port}/v1"
    srv.shutdown()


def test_rejected_upload_reports_the_real_status_not_broken_pipe(rejecting_server):
    """The regression that sent an operator chasing a phantom network fault: a
    gateway that 401s before reading the body leaves urllib holding a dead
    socket, so the run died with "[Errno 32] Broken pipe" and no mention of
    auth. The client must re-ask with a small body and surface what the server
    actually said."""
    c = OpenAICompatClient(base_url=rejecting_server, model="gemma-4-12b", api_key="k")
    with pytest.raises(OpenAICompatError) as e:
        c.generate("x" * 4_000_000)
    msg = str(e.value)
    assert "401" in msg and "proxy auth required" in msg
    assert "Custom headers" in msg, "must point at the fix, not just the symptom"
    assert "still uploading" in msg


def test_oversized_payload_is_distinguished_from_a_rejected_one(mock_server):
    """Same dead socket, opposite cause: when a small request to the same
    endpoint SUCCEEDS, the credentials are fine and the payload size is what got
    refused. Saying "fix your auth" there would be a wild goose chase."""
    c = OpenAICompatClient(base_url=mock_server, model="gemma-4-12b", api_key="k")
    err = c._upload_cut_off(6_400_000, BrokenPipeError(32, "Broken pipe"))
    msg = str(err)
    assert "6.4MB" in msg
    assert "credentials are accepted" in msg
    assert "BEATFORGE_OPENAI_AUDIO_SR" in msg


@pytest.mark.parametrize("exc", [
    BrokenPipeError(32, "Broken pipe"),
    ConnectionResetError(54, "Connection reset by peer"),
    # macOS surfaces the same mid-upload hangup through the TLS layer instead.
    ssl.SSLError(5, "[SYS] unknown error (_ssl.c:2417)"),
])
def test_mid_upload_hangups_are_recognized(exc):
    from beatforge.llm import _is_cut_off
    assert _is_cut_off(exc)


@pytest.mark.parametrize("exc", [
    ConnectionRefusedError(61, "Connection refused"),
    OSError(65, "No route to host"),
    # A rejected certificate is an SSLError, but the connection was refused on
    # its merits — re-probing just repeats it, and calling it a payload-size
    # problem would be actively misleading.
    ssl.SSLCertVerificationError(1, "certificate verify failed"),
])
def test_plain_connection_failures_are_not_mistaken_for_hangups(exc):
    """A server that was never reachable must keep its own "cannot reach"
    message — re-probing it would only repeat the failure."""
    from beatforge.llm import _is_cut_off
    assert not _is_cut_off(exc)


def test_an_inconclusive_probe_does_not_invent_a_cause():
    """When the follow-up request dies the same way we know nothing new, so the
    error must not claim the credentials are fine (the size branch) — that sent
    the diagnosis in exactly the wrong direction."""
    c = OpenAICompatClient(base_url="http://127.0.0.1:1/v1", model="x")
    msg = str(c._upload_cut_off(6_000_000, BrokenPipeError(32, "Broken pipe")))
    assert "failed the same way" in msg
    assert "credentials are accepted" not in msg
    assert "host/port" in msg, "should point at the transport, the one thing left"


def test_list_models_surfaces_an_http_error_readably(rejecting_server):
    """`compare --probe-only` is the first thing a user runs against a new
    endpoint; a 401 there has to name the problem, not raise a raw urllib
    traceback out of the probe."""
    c = OpenAICompatClient(base_url=rejecting_server, model="x", api_key="k")
    with pytest.raises(OpenAICompatError, match="401"):
        c.list_models()


def test_audio_can_be_withheld_for_a_text_only_endpoint(mock_server, click_wav):
    """The degraded escape hatch: against a text-only endpoint the call still goes
    out, as a plain string — the one content form every server accepts."""
    wav = click_wav(bpm=120.0, dur_s=4.0)
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k", send_audio=False)
    c.generate_json("design this chart", audio_path=wav)
    assert _LAST["body"]["messages"][-1]["content"] == "design this chart"


def test_a_deaf_run_says_so_loudly(mock_server, click_wav, capsys):
    """Listening IS the pipeline, so a run that stops doing it must not read as a
    normal one. Silence here would let a deaf run pass for the real thing and the
    charts would just quietly get worse."""
    wav = click_wav(bpm=120.0, dur_s=4.0)
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k", send_audio=False)
    c.generate_json("design", audio_path=wav)
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "NOT hearing the music" in err
    assert "worse" in err


def test_audio_is_attached_by_default(mock_server, click_wav):
    """The flag is opt-out: an audio-capable endpoint keeps hearing the track."""
    wav = click_wav(bpm=120.0, dur_s=4.0)
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k")
    c.generate_json("analyze", audio_path=wav)
    parts = _LAST["body"]["messages"][-1]["content"]
    # Text first, then the audio: the order in Inkling's documented chat
    # template, and the one that leaves a cacheable prefix in front of the single
    # most variable part of the request.
    assert [p["type"] for p in parts] == ["text", "input_audio"]


def test_withheld_audio_is_not_ledgered_as_heard(mock_server, click_wav, monkeypatch):
    """`audio_attached` answers "did the model actually hear this?" — recording
    the caller's intent instead of the request's contents would make every
    text-only run look like an audio one in the cost ledger."""
    seen = {}
    monkeypatch.setattr("beatforge.ledger.record_model_call",
                        lambda **kw: seen.update(kw))
    wav = click_wav(bpm=120.0, dur_s=4.0)
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k", send_audio=False)
    c.generate_json("design", audio_path=wav)
    assert seen["audio_attached"] is False
    assert seen["audio_bytes"] == 0


class _RejectAudioHandler(BaseHTTPRequestHandler):
    """A text-only server: it validates the message shape and 400s on a content
    parts list, echoing the offending request (base64 audio and all) back."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        content = body["messages"][-1]["content"]
        if isinstance(content, str):
            payload = json.dumps({"choices": [{"message": {"content": "{\"ok\": 1}"}}]})
            code = 200
        else:
            # Found by type, not by position — the client is free to order the
            # parts however the target model's chat template wants them. Both
            # audio dialects are rejected, because a genuinely text-only server
            # has no more use for one than the other; recognising only
            # `input_audio` would make this mock accidentally "support"
            # `audio_url` and swallow the client's fallback retry.
            part = next(p for p in content
                        if p["type"] in ("input_audio", "audio_url"))
            blob = (part["input_audio"]["data"] if part["type"] == "input_audio"
                    else part["audio_url"]["url"])
            payload = json.dumps({"object": "error", "message":
                                  "1 validation error:\n  {'loc': ('body','messages',0,"
                                  "'content','str'), 'msg': 'Input should be a valid "
                                  f"string', 'input': [{{'input_audio': '{blob}'}}]}}"})
            code = 400
        data = payload.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def text_only_server():
    srv = HTTPServer(("127.0.0.1", 0), _RejectAudioHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    yield f"http://{host}:{port}/v1"
    srv.shutdown()


def test_text_only_endpoint_400_names_the_fix(text_only_server, click_wav):
    """The operator's second wall: past auth, the endpoint 400s because it can't
    take audio. The raw pydantic dump quotes megabytes of base64 back, so the
    real complaint has to survive and the way out has to be named."""
    wav = click_wav(bpm=120.0, dur_s=4.0)
    c = OpenAICompatClient(base_url=text_only_server, model="m", api_key="k")
    with pytest.raises(OpenAICompatError) as e:
        c.generate("design", audio_path=wav)
    msg = str(e.value)
    assert "Input should be a valid string" in msg, "the server's own words must survive"
    assert "base64 elided" in msg, "the echoed payload must not bury the message"
    assert "UklGR" not in msg


def test_a_rejected_audio_part_is_not_reported_as_a_deaf_model(
        text_only_server, click_wav):
    """A 400 on the message shape does NOT prove the model cannot hear.

    A validation dump lists one error per message variant, so the visible errors
    are usually about variants that never applied and the real cause has scrolled
    off — an audio-capable model behind a strict chat schema fails identically.
    Asserting "this model is deaf" from that evidence sends someone to replace a
    model that was fine, which is exactly what happened with Inkling.
    """
    wav = click_wav(bpm=120.0, dur_s=4.0)
    c = OpenAICompatClient(base_url=text_only_server, model="m", api_key="k")
    with pytest.raises(OpenAICompatError) as e:
        c.generate("design", audio_path=wav)
    msg = str(e.value)
    # The schema explanation must come before the capability one: it is both the
    # likelier cause and the cheaper thing to check.
    assert msg.index("WRONG AUDIO PART") < msg.index("TEXT-ONLY MODEL")
    assert "BEATFORGE_LLM_BACKEND=inkling" in msg, "name the backend that speaks it"
    assert "cannot hear audio" not in msg, "must not assert a capability it cannot know"
    # The disable-audio flag stays available but labelled for what it is.
    assert "BEATFORGE_OPENAI_SEND_AUDIO=0" in msg
    assert "materially worse" in msg


def test_the_same_endpoint_works_once_audio_is_off(text_only_server, click_wav):
    """The fix the error recommends has to actually work against that server."""
    wav = click_wav(bpm=120.0, dur_s=4.0)
    c = OpenAICompatClient(base_url=text_only_server, model="m", api_key="k",
                           send_audio=False)
    assert c.generate_json("design", audio_path=wav) == {"ok": 1}


def test_a_400_without_audio_is_not_blamed_on_audio(text_only_server):
    """An unrelated 400 (bad model id, bad params) must not be met with "turn off
    audio" — that would send someone chasing the wrong knob."""
    c = OpenAICompatClient(base_url=text_only_server, model="m", api_key="k")
    err = c._http_error(400, '{"message": "model `nope` does not exist"}', sent_audio=False)
    assert "SEND_AUDIO" not in str(err)
    assert "does not exist" in str(err)


@pytest.mark.parametrize("content,expected", [
    ([{"type": "input_audio", "input_audio": {"data": "x"}}, {"type": "text"}], True),
    ([{"type": "text", "text": "hi"}], False),
    ("plain string", False),
])
def test_audio_detection_reads_the_request_itself(content, expected):
    """The 400 hint keys off this, so it must describe what was actually sent —
    not what the caller asked for."""
    from beatforge.llm import _has_audio
    assert _has_audio({"messages": [{"role": "user", "content": content}]}) is expected


def test_audio_probe_is_skipped_rather_than_faked(click_wav):
    """`compare` scores the probe's tempo answer against the DSP truth. With no
    audio sent the model can only guess from the prompt, and scoring that guess
    would report a hallucination as a measurement."""
    from beatforge import compare
    c = OpenAICompatClient(base_url="http://127.0.0.1:1/v1", model="m", send_audio=False)
    out = compare.audio_probe(c, click_wav(bpm=120.0, dur_s=2.0))
    assert "skipped" in out and "tempo_bpm" not in out


def test_redaction_keeps_short_tokens_intact():
    """Redaction must not eat model ids, hashes or the prose around them — only
    the runaway payload."""
    from beatforge.llm import _redact_blobs
    assert _redact_blobs("model gemma-4-12b rejected abc123") == "model gemma-4-12b rejected abc123"
    out = _redact_blobs("data: " + "A" * 500 + " end")
    assert "500 chars of base64 elided" in out and out.endswith(" end")


def test_unreachable_server_fails_loudly():
    c = OpenAICompatClient(base_url="http://127.0.0.1:1/v1", model="x")
    with pytest.raises(OpenAICompatError, match="cannot reach"):
        c.generate("hi")


class _SlowHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        import time
        time.sleep(3)  # exceed the client timeout -> read timeout


@pytest.fixture
def slow_server():
    srv = HTTPServer(("127.0.0.1", 0), _SlowHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    yield f"http://{host}:{port}/v1"
    srv.shutdown()


def test_timeout_becomes_graceful_error(slow_server):
    """A slow/looping model (read timeout after the request is accepted) must
    surface as an OpenAICompatError, not a raw TimeoutError that crashes the run
    — this is exactly what took down the operator's `compare` on Gemma."""
    c = OpenAICompatClient(base_url=slow_server, model="gemma-4-12b", timeout=1)
    with pytest.raises(OpenAICompatError, match="timed out"):
        c.generate("analyze")


# --------------------------------------------------------------------------- #
# The output cap: which field carries it, and what an exhausted one looks like
# --------------------------------------------------------------------------- #
def test_a_self_hosted_server_gets_max_tokens(mock_server):
    """`max_tokens` is the only spelling most OpenAI-compatible servers ever
    implemented. Sending `max_completion_tokens` to one of those is not a
    graceful degradation — the field is ignored and the server generates to its
    own default, which is the silent version of the bug."""
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k")
    c.generate_json("hi")
    assert _LAST["body"]["max_tokens"] == config.OPENAI_MAX_TOKENS
    assert "max_completion_tokens" not in _LAST["body"]


def test_openai_proper_gets_max_completion_tokens(monkeypatch):
    """api.openai.com REJECTS `max_tokens` on its reasoning models with an
    explicit "use 'max_completion_tokens' instead". The two spellings are
    mutually exclusive and OpenAI 400s on the pair, so this is a choice per
    endpoint rather than an upgrade everyone gets."""
    c = OpenAICompatClient(base_url="https://api.openai.com/v1", model="gpt-5",
                           api_key="sk-x")
    assert c._max_tokens_field() == "max_completion_tokens"


def test_the_max_tokens_field_can_be_pinned(mock_server, monkeypatch):
    """A gateway that proxies to OpenAI under its own hostname defeats the
    hostname heuristic, so the guess has to be overridable."""
    monkeypatch.setattr(config, "OPENAI_MAX_TOKENS_FIELD", "max_completion_tokens")
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k")
    c.generate_json("hi")
    assert "max_completion_tokens" in _LAST["body"]
    assert "max_tokens" not in _LAST["body"]


class _EmptyReasoningHandler(BaseHTTPRequestHandler):
    """A reasoning endpoint that spent its whole budget thinking. HTTP 200, a
    well-formed choice, a usage block — and no answer."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        data = json.dumps({
            "choices": [{"message": {"content": "", "reasoning_content":
                                     "Let me work through the onset list..."},
                         "finish_reason": "length"}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 4096,
                      "completion_tokens_details": {"reasoning_tokens": 4096}},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def thinking_server():
    srv = HTTPServer(("127.0.0.1", 0), _EmptyReasoningHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    yield f"http://{host}:{port}/v1"
    srv.shutdown()


def test_a_budget_spent_on_reasoning_is_not_reported_as_a_dead_endpoint(thinking_server):
    """The most misleading success in the pipeline: reasoning tokens are
    generated tokens and count against the output cap, so a reasoning model can
    burn the entire budget in its scratchpad and return an empty `content` with
    HTTP 200. "empty response from <model>" sends people to replace an endpoint
    that was working perfectly; the fix is a bigger cap or a lower effort, and
    the message has to say so."""
    c = OpenAICompatClient(base_url=thinking_server, model="m", api_key="k")
    with pytest.raises(OpenAICompatError) as e:
        c.generate("design a chart")
    msg = str(e.value)
    assert "finish_reason=length" in msg
    assert "4096" in msg and "reasoning" in msg.lower()
    assert "Raise the cap" in msg


def test_an_oversized_body_is_refused_before_it_is_uploaded(mock_server, monkeypatch):
    """A gateway that rejects an oversized request answers and closes without
    reading the rest of the upload, so the client never sees the status — just a
    broken pipe, minutes in, after pushing tens of MB. Checking locally is what
    turns that into a sentence naming the three levers."""
    monkeypatch.setattr(config, "OPENAI_MAX_REQUEST_MB", 0.001)
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k")
    with pytest.raises(OpenAICompatError) as e:
        c.generate("x" * 5000)
    msg = str(e.value)
    assert "not uploading it" in msg
    assert "BEATFORGE_OPENAI_AUDIO_SR" in msg and "flac" in msg


def test_the_size_guard_can_be_disabled(mock_server, monkeypatch):
    """0 means "the endpoint really does take a body this big" — the guard is a
    default, not a ceiling on what the transport can do."""
    monkeypatch.setattr(config, "OPENAI_MAX_REQUEST_MB", 0)
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k")
    c.generate_json("x" * 5000)          # no raise


def test_cover_art_does_not_break_the_audio_encode(tmp_path, mock_server):
    """An imported .mp3 with embedded album art is a TWO-stream input. Without
    an explicit audio-stream selection ffmpeg tries to carry the picture into a
    container that cannot hold it and the encode fails — on a file that plays
    fine everywhere else."""
    import shutil as _shutil
    import subprocess as _sub
    if not _shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    art = tmp_path / "cover.png"
    _sub.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
              "-i", "color=c=red:s=64x64:d=1", "-frames:v", "1", str(art)],
             check=True, capture_output=True)
    src = tmp_path / "song.mp3"
    _sub.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
              "-i", "sine=frequency=440:duration=2", "-i", str(art),
              "-map", "0:a", "-map", "1:v", "-c:v", "copy", "-id3v2_version", "3",
              "-metadata:s:v", "comment=Cover (front)", str(src)],
             check=True, capture_output=True)
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k")
    part = c._audio_part(str(src))       # must not raise
    assert part["input_audio"]["data"]


# --------------------------------------------------------------------------- #
# Audio transport: two incompatible dialects, one backend
# --------------------------------------------------------------------------- #
class _AudioUrlOnlyHandler(BaseHTTPRequestHandler):
    """An SGLang-shaped server: it takes `audio_url` with a data: URI and 400s on
    OpenAI's `input_audio` part, which is what a live Modal Auto Endpoint does."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _LAST.setdefault("bodies", []).append(body)
        content = body["messages"][-1]["content"]
        parts = content if isinstance(content, list) else []
        if any(p.get("type") == "input_audio" for p in parts):
            payload, code = json.dumps({"object": "error", "message":
                "25 validation errors:\n  {'type': 'string_type', 'loc': ('body',"
                "'messages',0,'content','str'), 'msg': 'Input should be a valid "
                "string'}"}), 400
        else:
            payload, code = json.dumps(
                {"choices": [{"message": {"content": '{"ok": 1}'}}]}), 200
        data = payload.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def audio_url_server():
    _LAST.pop("bodies", None)
    srv = HTTPServer(("127.0.0.1", 0), _AudioUrlOnlyHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    yield f"http://{host}:{port}/v1"
    srv.shutdown()


def _kinds(body):
    c = body["messages"][-1]["content"]
    return [p["type"] for p in c] if isinstance(c, list) else ["str"]


def test_a_rejected_audio_shape_is_retried_in_the_other_one(audio_url_server, click_wav):
    """The failure this exists for is a wall, not a degradation: the endpoint
    takes audio perfectly well and refuses the message anyway, with a pydantic
    union dump that never says which part was wrong. Retrying the same bytes in
    the other dialect turns a dead end into a working call, and is what lets one
    backend serve both api.openai.com/vLLM and SGLang/Modal."""
    c = OpenAICompatClient(base_url=audio_url_server, model="m", api_key="k")
    assert c.generate_json("design", audio_path=click_wav(bpm=120.0, dur_s=3.0))
    kinds = [_kinds(b) for b in _LAST["bodies"]]
    assert kinds == [["text", "input_audio"], ["text", "audio_url"]]


def test_the_working_shape_is_remembered_for_the_rest_of_the_run(
        audio_url_server, click_wav):
    """The fallback costs a wasted upload of the whole track — up to twenty
    minutes of audio on this backend. Paying that once per process is the price
    of compatibility; paying it on every call would not be."""
    wav = click_wav(bpm=120.0, dur_s=3.0)
    c = OpenAICompatClient(base_url=audio_url_server, model="m", api_key="k")
    c.generate_json("first", audio_path=wav)
    c.generate_json("second", audio_path=wav)
    assert [_kinds(b) for b in _LAST["bodies"]][1:] == [
        ["text", "audio_url"], ["text", "audio_url"]], "no second rediscovery"


def test_a_pinned_transport_is_never_second_guessed(audio_url_server, click_wav, monkeypatch):
    """Explicit beats clever: someone who has named the shape is debugging, and a
    silent retry would hide the very answer they are trying to see."""
    monkeypatch.setattr(config, "OPENAI_AUDIO_TRANSPORT", "input_audio")
    c = OpenAICompatClient(base_url=audio_url_server, model="m", api_key="k")
    with pytest.raises(OpenAICompatError):
        c.generate("design", audio_path=click_wav(bpm=120.0, dur_s=3.0))
    assert len(_LAST["bodies"]) == 1, "pinned means one attempt, not two"


def test_the_audio_url_shape_carries_the_bytes_inline(mock_server, click_wav, monkeypatch):
    """A `data:` URI, never an http one. The `audio_url` name invites the
    assumption that the server fetches it — but the track is a file on the user's
    machine, which no endpoint can reach."""
    monkeypatch.setattr(config, "OPENAI_AUDIO_TRANSPORT", "audio_url")
    c = OpenAICompatClient(base_url=mock_server, model="m", api_key="k",
                           audio_format="wav")
    c.generate_json("x", audio_path=click_wav(bpm=120.0, dur_s=3.0))
    part = [p for p in _LAST["body"]["messages"][-1]["content"]
            if p["type"] == "audio_url"][0]
    url = part["audio_url"]["url"]
    assert url.startswith("data:audio/wav;base64,")
    assert len(url) > 500, "the audio itself must be in the URI"


def test_reasoning_tokens_are_counted_in_either_spelling():
    """OpenAI nests `reasoning_tokens` under `completion_tokens_details`; SGLang
    — what a Modal Auto Endpoint serves Inkling on — puts it at the top level of
    `usage`. Reading only the nested one books every reasoning token as zero, on
    the backend where reasoning is the single largest cost driver."""
    from beatforge import ledger
    nested = ledger.usage_from_openai(
        {"usage": {"completion_tokens": 90,
                   "completion_tokens_details": {"reasoning_tokens": 60}}})
    flat = ledger.usage_from_openai(
        {"usage": {"completion_tokens": 90, "reasoning_tokens": 60}})
    assert nested["thinking_tokens"] == flat["thinking_tokens"] == 60
