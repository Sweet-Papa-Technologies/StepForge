"""llm.py — swappable LLM backend for the designer / critic / A&R calls.

The pipeline's audio-understanding model is behind one small interface so an
alternative (e.g. a self-hosted Gemma 4 12B on an OpenAI-compatible server) can
be dropped in and benchmarked against Gemini 3.5 Flash with zero caller changes.

  * LLMClient        — the protocol every backend satisfies (VertexClient already
                       does; see vertex.py).
  * OpenAICompatClient — talks to any OpenAI /v1/chat/completions server, sending
                       audio as an `input_audio` content part (the format vLLM &
                       friends expect for audio-in models).
  * HybridClient     — the DEFAULT. Routes each call to a lite or heavy Gemini by
                       the ROLE it is playing (config.ROLE_TIERS). See below.
  * tier_for         — the pure routing policy, so a caller can ask "would this
                       role run on the cheap model?" without making a call.
  * make_llm_client  — factory picking the backend from config.LLM_BACKEND.

Lyria (music generation) is Vertex-only and does NOT go through here.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from . import config, ledger
from .vertex import VertexClient, VertexError, _parse_json, json_retry_note


class LLMClient(Protocol):
    def generate(self, prompt: str, *, audio_path=None, system=None, model=None,
                 thinking_level=None, json_out=False, timeout=600) -> str: ...

    def generate_json(self, prompt: str, *, audio_path=None, system=None,
                      model=None, thinking_level=None, timeout=600) -> dict: ...


class OpenAICompatError(RuntimeError):
    #: Set when the endpoint refused the message SHAPE of an audio request, so
    #: `generate` can re-wrap the same bytes in the other content-part form and
    #: try once more. Declared here rather than stuck on ad hoc, so the retry
    #: path has a real attribute to read and type checkers can see it.
    audio_schema_rejected: bool = False


# RFC 7230 token: what a header field-name may legally contain.
_LEGAL_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def _validate_header(name: str, value: str) -> None:
    """Reject a custom header that cannot go on the wire, naming the offender.

    These values are typed into a settings box, so all three of these are things
    a real person does — and every one of them otherwise surfaces from deep
    inside http.client as a bare `ValueError: Invalid header value b'...'` or a
    latin-1 `UnicodeEncodeError`, with no clue which header caused it or where to
    fix it.

    The newline check is also the injection guard: a value carrying CRLF would
    otherwise be an attempt to smuggle extra headers into the request. We refuse
    it here rather than trusting the transport to catch it, and we refuse instead
    of stripping, because silently sanitising hides the attempt.
    """
    if not _LEGAL_HEADER_NAME.match(name):
        raise OpenAICompatError(
            f"custom header name {name!r} is not valid. A header name may only "
            f"contain letters, digits and !#$%&'*+-.^_`|~ — no spaces, colons or "
            f"quotes. Check Settings -> Custom headers.")
    if any(c in value for c in "\r\n\x00"):
        raise OpenAICompatError(
            f"custom header {name!r} has a line break in its value. Headers are "
            f"single-line; a pasted multi-line secret or an accidental newline "
            f"cannot be sent (and a CRLF in a value would inject extra headers). "
            f"Re-paste it as one line under Settings -> Custom headers.")
    try:
        value.encode("latin-1")
    except UnicodeEncodeError:
        raise OpenAICompatError(
            f"custom header {name!r} has a non-ASCII value. HTTP headers are "
            f"latin-1 only, so this cannot be sent as-is — this usually means a "
            f"smart quote or an em dash came along with a copy-paste. Retype the "
            f"value under Settings -> Custom headers.")


def _has_audio(body: dict) -> bool:
    """Did this request body actually carry audio, in EITHER content-part shape?

    Read from the body rather than tracked alongside it, so the error handler can
    only ever describe what was really sent. Both dialects have to count: after
    the transport fallback re-wraps a rejected request as `audio_url`, a check
    that knew only `input_audio` would decide the retry carried no audio and drop
    the audio-specific guidance from the very error that needed it most.
    """
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") in ("input_audio", "audio_url")
                for p in content):
            return True
    return False


def _redact_blobs(text: str) -> str:
    """Replace long base64 runs with a placeholder. A server that echoes the
    offending request back quotes the whole audio payload, so the actual
    complaint scrolls off the end of the message and the user is left reading
    megabytes of `UklGRkbsGABXQVZ…`."""
    return re.sub(r"[A-Za-z0-9+/]{120,}={0,2}",
                  lambda m: f"<{len(m.group(0))} chars of base64 elided>", text)


def _is_cut_off(exc) -> bool:
    """True when the peer tore the connection down mid-request rather than
    refusing to connect. Broken pipe / connection reset are the plain cases; a
    TLS link reports the same event as an opaque SSLError instead (macOS raises
    `SSLError(5, '[SYS] unknown error')` where Linux gives EPIPE), which is why
    matching on errno alone misses the HTTPS endpoints this matters most for.

    A certificate failure is an SSLError too but is NOT this: the connection was
    refused on its merits and re-probing would only repeat it, so it keeps its
    own message."""
    if isinstance(exc, ssl.SSLCertVerificationError):
        return False
    return isinstance(exc, (BrokenPipeError, ConnectionResetError, ssl.SSLError))


def _encode_audio(path: str | Path, fmt: str, sr: int | None = None,
                  mono: bool | None = None) -> str:
    """Return base64 of the audio in `fmt`, transcoding from the source (usually
    .ogg) with ffmpeg when needed. wav is the safe default — universally decoded
    by audio-in serving stacks.

    `sr`/`mono` default to the OpenAI-backend settings but are explicit
    parameters because some endpoints specify the input format rather than
    accepting whatever arrives: Inkling requires 16 kHz mono WAV, and it must get
    that whether or not the OpenAI knobs happen to be set that way."""
    path = str(path)
    src_ext = os.path.splitext(path)[1].lower().lstrip(".")
    sr = config.OPENAI_AUDIO_SR if sr is None else sr
    mono = config.OPENAI_AUDIO_MONO if mono is None else mono
    # When no resample/downmix is requested and the source already matches, pass
    # the bytes straight through. Otherwise transcode (and shrink) via ffmpeg.
    if src_ext == fmt and sr <= 0 and not mono:
        data = Path(path).read_bytes()
    else:
        if not shutil.which("ffmpeg"):
            raise OpenAICompatError(
                f"ffmpeg needed to transcode {src_ext}->{fmt} for the audio part")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / f"a.{fmt}"
            codec = {"wav": "pcm_s16le", "mp3": "libmp3lame", "flac": "flac",
                     "ogg": "libvorbis"}.get(fmt, "pcm_s16le")
            # `-map 0:a:0 -vn` takes the first audio stream and nothing else.
            # Without it an imported .mp3/.m4a carrying embedded cover art is a
            # two-stream input, and ffmpeg's default stream selection tries to
            # carry the picture into a container that cannot hold it — the whole
            # encode fails, on a file that plays fine everywhere else.
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                   "-map", "0:a:0", "-vn", "-c:a", codec]
            if sr > 0:                       # downsample: fewer audio tokens -> less VRAM
                cmd += ["-ar", str(sr)]
            if mono:                         # downmix to 1 channel: half the payload
                cmd += ["-ac", "1"]
            cmd += ["-f", fmt if fmt != "ogg" else "ogg", str(out)]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                # ffmpeg's complaint is on stderr and is the only thing that says
                # WHY; without it this surfaces as a bare "returned non-zero".
                err = (e.stderr or b"").decode("utf-8", "replace").strip()
                raise OpenAICompatError(
                    f"ffmpeg could not convert {path} ({src_ext} -> {fmt} @ "
                    f"{sr or 'source'}Hz {'mono' if mono else 'stereo'}):\n"
                    f"{err[-800:]}")
            data = out.read_bytes()
    if not data:
        raise OpenAICompatError(
            f"audio encode produced 0 bytes for {path} — the track is empty or "
            f"unreadable, and sending it would look to the model like silence.")
    return base64.b64encode(data).decode()


class OpenAICompatClient:
    """OpenAI /v1/chat/completions client with audio-in support. Matches the
    VertexClient.generate/generate_json interface so it's a drop-in swap."""

    name = "openai"

    #: Audio shape to try FIRST when the transport setting is "auto". OpenAI's
    #: own documented content part, which api.openai.com and vLLM both take. A
    #: subclass that knows better about its endpoint overrides it (inkling.py).
    preferred_transport = "input_audio"

    def __init__(self, base_url=None, model=None, api_key=None, audio_format=None,
                 timeout=None, temperature=None, headers=None, send_audio=None,
                 max_tokens=None, audio_sr=None, audio_mono=None,
                 audio_transport=None):
        self.base_url = (base_url or config.OPENAI_BASE_URL).rstrip("/")
        self.model = model or config.OPENAI_MODEL
        self.api_key = api_key or config.OPENAI_API_KEY
        self.audio_format = audio_format or config.OPENAI_AUDIO_FORMAT
        # Per-client so a backend with its own documented input spec (Inkling:
        # 16 kHz mono) is not at the mercy of the knobs tuned for a local server.
        self.audio_sr = config.OPENAI_AUDIO_SR if audio_sr is None else audio_sr
        self.audio_mono = config.OPENAI_AUDIO_MONO if audio_mono is None else audio_mono
        self.timeout = timeout if timeout is not None else config.OPENAI_TIMEOUT
        self.temperature = temperature if temperature is not None else config.OPENAI_TEMPERATURE
        self.max_tokens = config.OPENAI_MAX_TOKENS if max_tokens is None else max_tokens
        # Arbitrary extra headers for gateways that need more than a bearer token
        # (BEATFORGE_OPENAI_HEADERS / the GUI's custom-headers table).
        self.headers = dict(config.OPENAI_HEADERS if headers is None else headers)
        # Off => the model never hears the track. A crippled mode, not a variant;
        # see the warning below and BEATFORGE_OPENAI_SEND_AUDIO in config.py.
        self.send_audio = config.OPENAI_SEND_AUDIO if send_audio is None else send_audio
        # Which content-part shape carries the audio, and which one to try first
        # when the setting is "auto". `preferred_transport` is the subclass's
        # knowledge about its own endpoint; `_proven_transport` is what a live
        # call has since established, and outranks the guess.
        self.audio_transport = (audio_transport or config.OPENAI_AUDIO_TRANSPORT)
        self._proven_transport: str | None = None
        self._warned_audio = False

    def _warn_audio_dropped(self) -> None:
        """Say it once per client, and say it as a warning.

        Listening to the track is what this pipeline IS — the DSP finds where the
        beats are, the model is what decides which of them are worth playing, and
        it can only do that by ear. Designing from the numbers alone still emits
        a chart, but it is a downgrade of the product, not a configuration
        choice, so a run must never do it quietly."""
        if self._warned_audio:
            return
        self._warned_audio = True
        print("[llm] WARNING: BEATFORGE_OPENAI_SEND_AUDIO is off — this endpoint is "
              "NOT hearing the music. Charts are being designed blind from the DSP "
              "numbers and will be materially worse. Point the backend at an "
              "audio-capable model to get the real thing.", file=sys.stderr)

    def _auth_headers(self) -> dict:
        """How this endpoint is authenticated. A hook, not a constant: Modal
        Managed Endpoints authenticate with two custom headers and reject a
        Bearer token outright, so a subclass has to be able to replace this
        wholesale rather than add to it (see inkling.InklingClient)."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def _extra_body(self) -> dict:
        """Provider-specific chat-completions fields merged into every request
        (e.g. Inkling's `reasoning_effort`). Empty for a plain OpenAI server."""
        return {}

    #: MIME subtype for a container whose name is not its subtype.
    _AUDIO_MIME = {"mp3": "mpeg", "ogg": "ogg", "wav": "wav", "flac": "flac"}

    def _wrap_audio(self, b64: str, transport: str) -> dict:
        """One encoded track, in whichever content-part shape `transport` names.

        Both carry the bytes inline — the `audio_url` form uses a `data:` URI, so
        nothing has to be reachable by the server. They are NOT interchangeable:
        an endpoint that wants one rejects the other with a pydantic union dump
        listing an error per message variant, none of which says "wrong audio
        part". See config.OPENAI_AUDIO_TRANSPORT.
        """
        if transport == "audio_url":
            mime = self._AUDIO_MIME.get(self.audio_format, self.audio_format)
            return {"type": "audio_url",
                    "audio_url": {"url": f"data:audio/{mime};base64,{b64}"}}
        return {"type": "input_audio",
                "input_audio": {"data": b64, "format": self.audio_format}}

    def _encode_for_request(self, audio_path) -> str:
        """Base64 of the track, ready to send — and the ONE place a backend gets
        to refuse it first.

        Separate from `_wrap_audio` because the two vary independently: the
        encoding is per-backend policy (Inkling pins 16 kHz mono WAV and a
        duration ceiling), while the wrapper is per-endpoint dialect and may be
        retried in the other shape with the same bytes. `generate` calls this
        exactly once, so a subclass guard here really does run before any upload
        — putting the check in `_audio_part` instead would let the retry path
        skip it entirely.
        """
        return _encode_audio(audio_path, self.audio_format,
                             sr=self.audio_sr, mono=self.audio_mono)

    def _audio_part(self, audio_path, transport: str | None = None) -> dict:
        """The complete audio content block: encode, then wrap in the shape this
        endpoint speaks."""
        return self._wrap_audio(self._encode_for_request(audio_path),
                                transport or self._transport())

    def _transport(self) -> str:
        """The audio shape to try FIRST. Once a call has proved which one this
        endpoint accepts, that answer sticks for the rest of the process — the
        fallback costs a wasted upload of the whole track, and paying it once per
        run is acceptable where paying it per call is not."""
        if self._proven_transport:
            return self._proven_transport
        pref = self.audio_transport
        return pref if pref in ("input_audio", "audio_url") else self.preferred_transport

    def _other_transport(self, used: str) -> str:
        return "input_audio" if used == "audio_url" else "audio_url"

    def _max_tokens_field(self) -> str:
        """Which request field carries the output cap.

        Not a constant, because the two spellings are mutually exclusive in
        opposite directions and there is no version that takes both: OpenAI's
        reasoning models 400 on `max_tokens` and demand `max_completion_tokens`,
        while plenty of self-hosted OpenAI-compatible servers have only ever
        implemented `max_tokens` and ignore the other one — silently generating
        to their own default, which is the worse failure of the two.
        """
        field = config.OPENAI_MAX_TOKENS_FIELD
        if field in ("max_tokens", "max_completion_tokens"):
            return field
        return ("max_completion_tokens"
                if "api.openai.com" in self.base_url else "max_tokens")

    def _call_effort(self) -> str | None:
        """Reasoning-effort level for the call being made, recorded on the ledger
        line. None for backends without the dial."""
        return None

    def _next_effort_down(self, effort: str | None) -> str | None:
        """The effort to retry at after a reply was cut off mid-reasoning, or
        None to fail instead. None here — a backend without the dial has nothing
        to trade away (see inkling.InklingClient)."""
        return None

    def _apply_effort(self, body: dict, effort: str) -> None:
        """Put `effort` on an outgoing body. A hook so the base class never has
        to know the field name a provider spells it with."""

    def _was_truncated(self, choice: dict, resp: dict) -> bool:
        """Did the model run out of output budget rather than finish?

        Both signals are needed. `finish_reason` is the direct answer but is
        absent or non-standard on some servers; a nonzero reasoning count with an
        empty answer means the budget went to thinking whatever the server called
        it.
        """
        return (choice.get("finish_reason") == "length"
                or ledger.usage_from_openai(resp).get("thinking_tokens", 0) > 0)

    def _headers(self, base: dict) -> dict:
        """Merge the configured custom headers over `base`. Matching is
        case-insensitive so a custom `authorization` really does replace the
        default bearer token instead of sending two conflicting headers, and a
        blank value drops the header entirely (some gateways 400 on an
        unexpected `Authorization`).

        On what actually reaches the wire: VALUES are sent byte-exact, but header
        NAMES are normalised to HTTP title-case by urllib
        (`AbstractHTTPHandler.do_open` does `name.title()`), so `anthropic-version`
        leaves as `Anthropic-Version`. That is RFC 7230-compliant — field names
        are case-insensitive — and it is not worth bypassing urllib to change,
        but it does mean a server demanding a literal `X-API-Key` would receive
        `X-Api-Key`. Don't add a "fix" for this without checking do_open first:
        anything that only rewrites `Request.headers` is a no-op.
        """
        merged = dict(base)
        for name, value in self.headers.items():
            name = name.strip()
            if not name:
                continue
            # Deleting the existing spelling before inserting is what keeps an
            # override to ONE header on the wire rather than two conflicting ones.
            for existing in list(merged):
                if existing.lower() == name.lower():
                    del merged[existing]
            if value != "":
                _validate_header(name, value)
                merged[name] = value
        return merged

    def _probe_rejection(self, timeout=15):
        """Re-ask the same endpoint with a one-line body to find out WHY a big
        upload was cut off. Three outcomes, and they need different advice:
          str   — the server answered with an HTTP error: that IS the reason.
          True  — the small request went through: the payload size was the problem.
          None  — the probe died the same way: inconclusive, don't guess.
        """
        tiny = json.dumps({"model": self.model, self._max_tokens_field(): 1,
                           "messages": [{"role": "user", "content": "ping"}]}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=tiny, method="POST",
            headers=self._headers({**self._auth_headers(),
                                   "Content-Type": "application/json"}))
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace").strip()
            return f"HTTP {e.code}: {detail[:600]}"
        except Exception:
            return None  # diagnosis is best-effort; never mask the original error

    def _upload_cut_off(self, sent_bytes: int, reason) -> OpenAICompatError:
        """The connection died while the request body was still going out.

        The socket error is a symptom, not the cause: a gateway that rejects a
        request (bad/missing auth, body too large) answers and closes *before*
        reading the upload, so urllib only ever sees the dead socket — the real
        HTTP status is never delivered. Ask again with a tiny body to recover it,
        because "Broken pipe" on its own sends people hunting a network fault
        that isn't there.
        """
        mb = sent_bytes / 1e6
        head = (f"{self.base_url} closed the connection while the {mb:.1f}MB request "
                f"was still uploading ({reason}).")
        verdict = self._probe_rejection()
        if isinstance(verdict, str):
            return OpenAICompatError(
                f"{head}\nIt rejects a small request to the same endpoint too:\n"
                f"  {verdict}\nSo this is the endpoint refusing the request, not a "
                f"network fault. Fix the credentials it asks for — set the API key, "
                f"or add what it needs under Settings -> Custom headers "
                f"(BEATFORGE_OPENAI_HEADERS).")
        if verdict is True:
            return OpenAICompatError(
                f"{head}\nA small request to the same endpoint succeeds, so the "
                f"endpoint is up and the credentials are accepted — it is the size of "
                f"the audio payload it refused. Shrink it with "
                f"BEATFORGE_OPENAI_AUDIO_SR (currently {config.OPENAI_AUDIO_SR}), or "
                f"raise the request-body limit on the server/proxy in front of it.")
        return OpenAICompatError(
            f"{head}\nA follow-up request failed the same way, so the cause is below "
            f"the HTTP layer: check the host/port and scheme (an https:// URL to a "
            f"plain-http server, or a TLS tunnel that needs its own port, fails "
            f"exactly like this).")

    def _rejected_the_audio(self, code: int, detail: str, sent_audio: bool) -> bool:
        """Did a 400 land on a request that carried an `input_audio` part?

        This says the endpoint would not ACCEPT the message as sent. It does NOT
        establish that the model cannot hear — a pydantic union dump lists one
        failure per message variant, so the visible errors are usually about the
        variants that never applied, and the real cause sits in the ones that
        scrolled off. An audio-capable model behind a strict or non-standard chat
        schema fails exactly the same way. The message below has to leave both
        possibilities open, because asserting the wrong one sends people to
        replace a model that was fine.
        """
        if code != 400 or not sent_audio:
            return False
        low = detail.lower()
        return any(k in low for k in (
            "input_audio", "audio", "content", "messages", "valid string"))

    def _http_error(self, code: int, detail: str, sent_audio: bool,
                    where: str = "") -> OpenAICompatError:
        msg = f"HTTP {code} from {self.base_url}{where}:\n{_redact_blobs(detail)[:1500]}"
        err = OpenAICompatError(msg)
        if self._rejected_the_audio(code, detail, sent_audio):
            used = self._transport()
            alt = self._other_transport(used)
            msg += (
                f"\n\nThe request carried the track as a `{used}` part and this "
                f"endpoint would not accept the message. Two different causes look "
                f"identical here, so check them in this order:\n\n"
                f"1. WRONG AUDIO PART — the model may hear perfectly well while its "
                f"server wants the OTHER content-part shape. The two are not "
                f"interchangeable: OpenAI and vLLM take `input_audio` with base64, "
                f"while SGLang and Modal Auto Endpoints take `audio_url` with a "
                f"`data:` URI. A validation dump lists one error per message "
                f"variant, so most of what is printed above is about variants that "
                f"never applied and none of it names the audio part. Try "
                f"BEATFORGE_OPENAI_AUDIO_TRANSPORT={alt} (or leave it at `auto`, "
                f"which retries the other shape by itself). If this is a Modal "
                f"endpoint serving Inkling, BEATFORGE_LLM_BACKEND=inkling picks "
                f"the right shape, the two-header proxy auth and the output cap "
                f"its reasoning needs, all at once.\n"
                f"2. TEXT-ONLY MODEL — if the model genuinely has no audio input, "
                f"serve one that does (Inkling, Qwen2-Audio, Gemma 3n, Ultravox …) "
                f"or use a Gemini backend, which listens natively.\n\n"
                f"BEATFORGE_OPENAI_SEND_AUDIO=0 will get a request through either "
                f"way, but the model then never hears the track and the charts are "
                f"materially worse — that is a plumbing smoke test, not a fix.")
            err = OpenAICompatError(msg)
            # Read by generate() to decide whether re-wrapping the same bytes in
            # the other shape is worth one retry.
            err.audio_schema_rejected = True
        return err

    def _oversized(self, nbytes: int) -> OpenAICompatError | None:
        """Refuse a body too big to be worth uploading, BEFORE uploading it.

        A gateway that rejects an oversized request answers and closes the socket
        without reading the rest of the upload, so the client never receives the
        status — it sees a broken pipe, minutes in, after pushing tens of MB. The
        local check is what turns that into a sentence the user can act on.
        """
        limit_mb = config.OPENAI_MAX_REQUEST_MB
        mb = nbytes / 1e6
        if not limit_mb or mb <= limit_mb:
            return None
        return OpenAICompatError(
            f"request body is {mb:.1f}MB, over the {limit_mb:.0f}MB local limit — "
            f"not uploading it.\nBase64 audio is ~1.33x the encoded track, so the "
            f"levers are, in order: lower the sample rate "
            f"(BEATFORGE_OPENAI_AUDIO_SR, now {self.audio_sr}), send a compressed "
            f"format (BEATFORGE_OPENAI_AUDIO_FORMAT=flac, now "
            f"{self.audio_format!r}), or chart a shorter cut. Raise "
            f"BEATFORGE_OPENAI_MAX_REQUEST_MB (0 disables) if the endpoint really "
            f"does accept a body this size.")

    def _post(self, body: dict, timeout: int) -> dict:
        data = json.dumps(body).encode()
        too_big = self._oversized(len(data))
        if too_big:
            raise too_big
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data, method="POST",
            headers=self._headers({**self._auth_headers(),
                                   "Content-Type": "application/json"}))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise self._http_error(e.code, detail, _has_audio(body))
        except (TimeoutError, socket.timeout) as e:
            # A read timeout AFTER the request was accepted: the model is either
            # too slow for this task on the local hardware or looping (KV cache /
            # VRAM fills as it generates toward max_tokens). NOT caught by URLError.
            raise OpenAICompatError(
                f"request to {self.base_url} timed out after {timeout}s ({e}). The "
                f"model likely ran slow or looped generating up to "
                f"{self._max_tokens_field()}={self.max_tokens}. Try "
                f"`compare --probe-only` first, lower "
                f"BEATFORGE_OPENAI_MAX_TOKENS, or raise BEATFORGE_OPENAI_TIMEOUT / the "
                f"server's --max-model-len.")
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise OpenAICompatError(
                    f"request to {self.base_url} timed out after {timeout}s. See "
                    f"`--probe-only` / BEATFORGE_OPENAI_TIMEOUT / max_tokens guidance.")
            if _is_cut_off(reason):
                raise self._upload_cut_off(len(data), reason)
            raise OpenAICompatError(
                f"cannot reach OpenAI-compatible server at {self.base_url}: {reason}")
        except OSError as e:
            # http.client can also let the socket error through unwrapped.
            if _is_cut_off(e):
                raise self._upload_cut_off(len(data), e)
            raise OpenAICompatError(f"network error talking to {self.base_url}: {e}")

    def list_models(self, timeout=15) -> list[str]:
        req = urllib.request.Request(
            f"{self.base_url}/models", headers=self._headers(self._auth_headers()))
        # This is the connectivity probe (`compare --probe-only`, the backend
        # reachability check), so it has to fail with a readable reason rather
        # than a raw urllib traceback — a 401 here names the header that's
        # missing, which is the whole point of running it first.
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace").strip()
            # Through the same hook as a chat error: this is the FIRST call a new
            # endpoint gets ("test connection", model resolution), so it is where
            # a provider's auth guidance matters most — raising a plain error here
            # would hand the user a bare 401 at exactly the wrong moment.
            raise self._http_error(e.code, detail, sent_audio=False, where="/models")
        except OSError as e:
            raise OpenAICompatError(
                f"cannot reach OpenAI-compatible server at {self.base_url}: "
                f"{getattr(e, 'reason', e)}")
        return [m.get("id") for m in data.get("data", [])]

    def generate(self, prompt: str, *, audio_path=None, system=None, model=None,
                 thinking_level=None, json_out=False, timeout=None) -> str:
        # thinking_level is Gemini-specific; ignored here (Gemma has no equivalent).
        audio_part = None
        audio_bytes = 0
        audio_b64 = None
        transport = self._transport()
        if audio_path is not None and self.send_audio:
            # Encoded ONCE and kept, so the transport fallback below re-wraps the
            # same bytes instead of paying for a second ffmpeg pass on a track
            # that can be twenty minutes long. Through the hook, so a backend's
            # pre-upload guards (Inkling's duration ceiling) still gate this.
            audio_b64 = self._encode_for_request(audio_path)
            audio_part = self._wrap_audio(audio_b64, transport)
            audio_bytes = os.path.getsize(audio_path)
        elif audio_path is not None:
            self._warn_audio_dropped()
        sent_audio = audio_part is not None

        content: str | list[dict]
        if audio_part is not None:
            # TEXT FIRST, then the audio. Two independent reasons, and they agree:
            #   * It is the order in Inkling's own documented chat template
            #     ([{"type":"text"...}, {"type":"audio"...}]), and the template is
            #     what the model was trained against.
            #   * Prefix caching works on a token prefix. The audio is the single
            #     most variable thing in the request — it changes every song — so
            #     putting it first means nothing after it can ever be a cache hit.
            content = [{"type": "text", "text": prompt}, audio_part]
        else:
            # Plain string, NOT a one-element parts list: a bare string is the
            # only content form every OpenAI-compatible server accepts, and the
            # strict ones reject `[{"type":"text",...}]` with "content: Input
            # should be a valid string". With no audio to carry there is nothing
            # the list form buys us, so don't spend compatibility on it.
            content = prompt
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        sent_model = model or self.model
        body: dict = {"model": sent_model, "messages": messages,
                      self._max_tokens_field(): self.max_tokens}
        # Omit temperature entirely rather than pin a default: OpenAI's reasoning
        # models accept ONLY the default and 400 on anything else, so an explicit
        # value is the difference between working and not on that provider.
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if json_out:
            body["response_format"] = {"type": "json_object"}
        body.update(self._extra_body())
        effort = self._call_effort()
        # REQ-R2-COST-01: the alternate backend is ledgered too. A self-hosted
        # model costs GPU-hours rather than Vertex dollars, but "which backend
        # actually served this call" is exactly what hypothesis #4 asks, and a
        # backend that leaves no trace can't answer it.
        t0 = time.monotonic()
        # `audio_attached` records what the request actually carried, not what the
        # caller offered — with audio disabled the ledger would otherwise claim
        # every call was heard, and "did the model hear it?" is the one question
        # this row exists to answer.
        try:
            resp = self._post(body, timeout if timeout is not None else self.timeout)
        except OpenAICompatError as e:
            # The endpoint refused the MESSAGE SHAPE, not the audio itself. The
            # two content-part forms are not interchangeable and the rejection
            # names neither of them — a pydantic union dump lists an error per
            # message variant, so the visible complaints are all about variants
            # that never applied. Re-wrap the SAME encoded bytes in the other
            # shape and try once. This is what makes one backend work against
            # both api.openai.com/vLLM (`input_audio`) and SGLang/Modal Auto
            # Endpoints (`audio_url`) without the user diagnosing a 400.
            if (getattr(e, "audio_schema_rejected", False) and audio_b64
                    and self.audio_transport not in ("input_audio", "audio_url")
                    and self._proven_transport is None):
                alt = self._other_transport(transport)
                print(f"[llm] {self.base_url} rejected the '{transport}' audio "
                      f"part; retrying as '{alt}'.", file=sys.stderr)
                messages[-1]["content"] = [{"type": "text", "text": prompt},
                                           self._wrap_audio(audio_b64, alt)]
                try:
                    resp = self._post(
                        body, timeout if timeout is not None else self.timeout)
                except Exception as e2:
                    ledger.record_model_call(
                        provider=self.name, model=sent_model,
                        usage=ledger.usage_from_openai({}),
                        latency_s=time.monotonic() - t0,
                        prompt_bytes=len(prompt.encode()), audio_attached=sent_audio,
                        audio_path=str(audio_path or "") or None,
                        thinking_level=effort, error=str(e2)[:300])
                    raise
                # It worked: pin it, so the rest of the run pays one upload per
                # call instead of two.
                self._proven_transport = alt
                transport = alt
            else:
                ledger.record_model_call(
                    provider=self.name, model=sent_model,
                    usage=ledger.usage_from_openai({}),
                    latency_s=time.monotonic() - t0,
                    prompt_bytes=len(prompt.encode()), audio_attached=sent_audio,
                    audio_path=str(audio_path or "") or None, thinking_level=effort,
                    error=str(e)[:300])
                raise
        except Exception as e:
            ledger.record_model_call(
                provider=self.name, model=sent_model,
                usage=ledger.usage_from_openai({}), latency_s=time.monotonic() - t0,
                prompt_bytes=len(prompt.encode()), audio_attached=sent_audio,
                audio_path=str(audio_path or "") or None, thinking_level=effort,
                error=str(e)[:300])
            raise
        if sent_audio:
            # A shape that came back 200 is the shape this endpoint takes. Pin it
            # so no later call spends an upload rediscovering that.
            self._proven_transport = transport
        ledger.record_model_call(
            provider=self.name, model=sent_model, usage=ledger.usage_from_openai(resp),
            latency_s=time.monotonic() - t0, prompt_bytes=len(prompt.encode()),
            audio_attached=sent_audio,
            audio_path=str(audio_path or "") or None,
            audio_bytes=audio_bytes, thinking_level=effort)
        try:
            choice = resp["choices"][0]
            message = choice["message"]
            text = message.get("content")
        except (KeyError, IndexError, TypeError):
            raise OpenAICompatError(f"unexpected response shape: {json.dumps(resp)[:400]}")
        if not text or not str(text).strip():
            # The budget went to the scratchpad and the answer never started.
            # Retrying at the SAME effort would most likely do it again, so step
            # the dial down once. This is not a nicety: the failed attempt has
            # already been billed for a full cap of reasoning, so failing here
            # means paying for the thinking and throwing away the chart.
            down = (self._next_effort_down(effort)
                    if self._was_truncated(choice, resp) else None)
            if down is None:
                raise self._empty_response(choice, message, resp, effort)
            print(f"[llm] {sent_model} spent the whole {self._max_tokens_field()}"
                  f"={self.max_tokens} budget reasoning at effort={effort} and "
                  f"returned no answer; retrying once at effort={down}. The chart "
                  f"will be less deliberated than the settings asked for.",
                  file=sys.stderr)
            self._apply_effort(body, down)
            t1 = time.monotonic()
            resp = self._post(body, timeout if timeout is not None else self.timeout)
            ledger.record_model_call(
                provider=self.name, model=sent_model,
                usage=ledger.usage_from_openai(resp), latency_s=time.monotonic() - t1,
                prompt_bytes=len(prompt.encode()), audio_attached=sent_audio,
                audio_path=str(audio_path or "") or None, audio_bytes=audio_bytes,
                thinking_level=down)
            choice = (resp.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = message.get("content")
            if not text or not str(text).strip():
                raise self._empty_response(choice, message, resp, down)
        return str(text).strip()

    def _empty_response(self, choice: dict, message: dict, resp: dict,
                        effort: str | None) -> OpenAICompatError:
        """The call succeeded and came back with no answer in it.

        On a REASONING endpoint this is usually not a broken endpoint at all —
        thinking tokens are generated tokens and count against the output cap, so
        the model can spend the entire budget in its scratchpad and stop before
        writing a single character of `content`. The response still looks healthy:
        HTTP 200, a choice, a usage block. `finish_reason: "length"` and a nonzero
        `reasoning_tokens` are what tell the two apart, so both go in the message
        — otherwise this reads as "the model returned nothing" and sends people to
        replace an endpoint that was working.
        """
        field = self._max_tokens_field()
        finish = choice.get("finish_reason") or "?"
        thought = ledger.usage_from_openai(resp).get("thinking_tokens", 0)
        # Some servers park the scratchpad here when the answer never arrived.
        stray = message.get("reasoning_content") or message.get("reasoning") or ""
        msg = (f"{self.model} returned an empty message "
               f"(finish_reason={finish}, {field}={self.max_tokens}"
               + (f", reasoning_effort={effort}" if effort else "") + ").")
        if finish == "length" or thought or stray:
            msg += (
                f"\n\nThe reply was cut off by the output cap, and "
                f"{thought or 'some'} of those tokens went to reasoning rather "
                f"than to the answer — reasoning bills and counts against "
                f"{field} just like output does. Raise the cap "
                f"(BEATFORGE_INKLING_MAX_TOKENS / BEATFORGE_OPENAI_MAX_TOKENS) or "
                f"lower the effort for this stage; the endpoint itself is fine.")
            if stray:
                msg += f"\nIt got as far as: {str(stray)[:300]}"
        else:
            msg += ("\n\nNo reasoning tokens were reported and the reply was not "
                    "truncated, so this is the model genuinely declining to "
                    "answer — check the prompt and any content filter in front of "
                    "the endpoint.")
        return OpenAICompatError(msg)

    def generate_json(self, prompt: str, *, audio_path=None, system=None,
                      model=None, thinking_level=None, timeout=600) -> dict:
        text = self.generate(prompt, audio_path=audio_path, system=system,
                             model=model, json_out=True, timeout=timeout)
        try:
            return _parse_json(text)
        except ValueError as e:
            retry = f"{prompt}\n\n---\n{json_retry_note(text, e)}"
            text2 = self.generate(retry, audio_path=audio_path, system=system,
                                  model=model, json_out=True, timeout=timeout)
            return _parse_json(text2)


class ChartForgeClient(OpenAICompatClient):
    """OpenAI-compatible client for the CHARTFORGE-G4 fine-tune.

    The fine-tune was trained through `chartforge_runtime.build_messages`, which
    wraps the designer prompt in a fixed system instruction and an AUDIO preamble,
    with the audio presented as a labelled window. A bare designer prompt (the
    Gemini-era path this codebase was built for) is out-of-distribution for the
    fine-tune and it degenerates. This subclass re-applies that exact wrapper on
    every call, so `design.py`/`chunked.py` don't have to change — flip it on with
    BEATFORGE_OPENAI_CHARTFORGE=1.

    Serve the model with llama.cpp and `--reasoning-budget 0`: it was trained
    thinking-off, and without that flag its output vanishes into the (stripped)
    thought channel and comes back empty.
    """

    # Copied verbatim from the model's shipped chartforge_runtime.py so BeatForge
    # stays self-contained (no dependency on a file under the weights dir).
    SYSTEM_TEXT = (
        "You generate StepMania chart designs. You select note placements "
        "exclusively from the provided ONSET TABLE by id, or as grid:<beat> "
        "references. You never invent timestamps and never place panels.")
    AUDIO_PREAMBLE = (
        "\n\n--- AUDIO ---\nYou are hearing the track as consecutive windows, in "
        "order. Each window is labelled with its time range and its bar range, "
        "and consecutive windows overlap by 2s so no onset falls in a seam. "
        "Listen to the whole arc before choosing densities.\n")

    def generate(self, prompt: str, *, audio_path=None, system=None, model=None,
                 thinking_level=None, json_out=False, timeout=None) -> str:
        # Re-apply the training-time wrapper on audio calls (the design/critic
        # passes that carry a window). generate() appends the audio part AFTER
        # this text, so the window label lands immediately before the audio —
        # matching how the fine-tune saw its examples.
        if audio_path is not None and self.send_audio:
            prompt = f"{prompt}{self.AUDIO_PREAMBLE}\nWINDOW 1 — this section's audio:\n"
            system = system or self.SYSTEM_TEXT
        return super().generate(
            prompt, audio_path=audio_path, system=system, model=model,
            thinking_level=thinking_level, json_out=json_out, timeout=timeout)


class GoogleAIStudioClient(VertexClient):
    """Gemini via Google AI Studio (generativelanguage.googleapis.com), authed with
    a plain API key — no GCP project or service account required. Reuses ALL of
    VertexClient's request/retry/thinking/JSON logic; only the endpoint and the auth
    header differ. This is the zero-setup Gemini path for a key-only user.

    Lyria music generation is NOT available here (AI Studio has no Lyria) — music
    gen stays on Vertex. Charting (designer/critic) works fully on this backend.
    """

    provider = "google-genai"

    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        if not self.api_key:
            raise VertexError(
                "No Google AI Studio API key. Get one at "
                "https://aistudio.google.com/apikey and `export GEMINI_API_KEY=...`, "
                "or use BEATFORGE_LLM_BACKEND=openai for a non-Google backend.")
        self._creds = None  # unused: AI Studio authenticates per-request by key

    def _url(self, model: str, method: str) -> str:
        # generativelanguage has no project/location and no global/regional split.
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}"

    def _auth_headers(self) -> dict:
        return {"x-goog-api-key": self.api_key}

    def _get_token(self, force: bool = False) -> str:
        # No OAuth here — the API key in _auth_headers() is the whole auth. This
        # override keeps the inherited 401-retry path a harmless no-op.
        return ""

    def lyria(self, *args, **kwargs):
        raise VertexError(
            "Lyria music generation is not available via Google AI Studio; it "
            "requires Vertex. Set BEATFORGE_LLM_BACKEND=gemini with a VERTEX_PROJECT "
            "to generate music.")


# --------------------------------------------------------------------------- #
# Hybrid routing — the opt-in mode (config.LLM_MODE == "hybrid"). The default is
# "single", where none of this engages and GEMINI_MODEL serves every call.
# --------------------------------------------------------------------------- #
#: The backends make_llm_client can wrap in a HybridClient.
_GEMINI_BACKENDS = ("gemini", "google-genai", "genai", "aistudio")


def tier_for(role: str | None, attempt: int | None = None) -> str | None:
    """Which tier ("lite"/"heavy") a call in `role` routes to, or None when hybrid
    routing is off (single-model mode, or a non-Gemini backend).

    Pure and side-effect free ON PURPOSE: the adapters need to ask "was that first
    draft written by the cheap model?" *before* deciding whether a gate failure
    deserves an escalation, and they must get the same answer the router gave.
    """
    if config.LLM_MODE != "hybrid":
        return None
    # Only the Gemini paths are ever wrapped in a HybridClient (make_llm_client).
    # On any other backend there is no second tier, and answering "lite" anyway
    # would make the adapters "escalate" a first draft to the very same model —
    # a second full designer call for nothing. The desktop app exports whatever
    # routing mode was last selected, so a switch to Argon or Inkling with
    # `hybrid` still set is an ordinary state, not a misconfiguration.
    if config.LLM_BACKEND not in _GEMINI_BACKENDS:
        return None
    tier = config.ROLE_TIERS.get(role or "", config.HYBRID_DEFAULT_TIER)
    if role in config.HYBRID_CASCADE_ROLES:
        if not config.HYBRID_CASCADE:
            return "heavy"          # cascade disabled: never draft on the cheap model
        if attempt is not None and attempt > 1:
            return "heavy"          # a retry of a lite draft escalates
    return tier


def thinking_for(role: str | None, attempt: int | None = None) -> str | None:
    """Which thinking level a call in `role` should use, or None when the
    per-role map is not in play (disabled, or a global level was pinned).

    Same shape and the same contract as `tier_for`: pure, side-effect free, and
    keyed on the ledger STAGE, so the reasoning budget rides the same ambient
    context as model routing and cost attribution. A call billed to "critic" was
    routed as "critic" AND thought as "critic" — the three can never disagree.

    `attempt` is accepted for symmetry with `tier_for` and is deliberately unused:
    a retry already carries a machine-readable violation report, so it has LESS to
    derive than the first draft, not more. If a future measurement says otherwise
    the hook is here.
    """
    if not config.ROLE_THINKING_ENABLED:
        return None
    if config.THINKING_LEVEL_PINNED:
        return None                 # an explicit global level wins (A/B arm)
    return config.ROLE_THINKING.get(role or "")


def _ambient_role() -> tuple[str | None, int | None]:
    """The role/attempt of the call being made, taken from the ledger's stage
    context. Routing rides the SAME ambient context as cost attribution so the
    two can never disagree — a call billed to "critic" was routed as "critic"."""
    ctx = ledger.current()
    stage = ctx.get("stage")
    attempt = ctx.get("attempt")
    return (str(stage) if stage else None,
            int(attempt) if isinstance(attempt, int) else None)


class HybridClient:
    """Two Gemini models behind one LLMClient, chosen per call by role.

    Delegates every request to a single underlying Gemini client (Vertex or AI
    Studio) with an explicit `model=` / `thinking_level=` override, so all of the
    auth, retry, pacing, JSON-repair and ledger behaviour is shared and there is
    exactly one transport to reason about.

    An explicit `model=` from the caller always wins — `compare.py` pins models
    deliberately and a router must not second-guess a benchmark.
    """

    name = "hybrid"

    def __init__(self, base: LLMClient | None = None,
                 heavy: str | None = None, lite: str | None = None):
        self.base = base if base is not None else _gemini_base()
        self.heavy = heavy or config.HYBRID_HEAVY_MODEL
        self.lite = lite or config.HYBRID_LITE_MODEL

    # ---- routing ---------------------------------------------------------- #
    def route(self, role: str | None = None, attempt: int | None = None
              ) -> tuple[str, str, str]:
        """(model, thinking_level, tier) for a call in this role."""
        if role is None and attempt is None:
            role, attempt = _ambient_role()
        tier = tier_for(role, attempt) or "heavy"
        if tier == "lite":
            return self.lite, config.HYBRID_LITE_THINKING, "lite"
        return self.heavy, config.HYBRID_HEAVY_THINKING, "heavy"

    def _resolved(self, model, thinking_level, role, attempt):
        """Resolve (model, thinking_level, tier) for one call.

        THINKING PRECEDENCE, decided once and stated here so the two mechanisms
        cannot silently fight (cost-reduction plan, Workstream C step 4):

            explicit `thinking_level=` argument
              > tier thinking, HYBRID MODE ONLY (HYBRID_{HEAVY,LITE}_THINKING)
              > per-role thinking (config.ROLE_THINKING, resolved in vertex.py)
              > config.DEFAULT_THINKING

        In hybrid mode the tier map always wins, because a tier's reasoning level
        is a property of the MODEL that tier names and the role map was measured
        against a single model. The practical consequence — worth knowing before
        you set it — is that BEATFORGE_ROLE_THINKING has no effect while
        BEATFORGE_LLM_MODE=hybrid. Hybrid is opt-in and off by default, so the
        role map governs the default configuration.
        """
        routed_model, routed_thinking, tier = self.route(role, attempt)
        # A caller that named a model opted out of routing; it is then billed at
        # whatever tier that model belongs to, and "pinned" if it is neither.
        if model is not None:
            tier = {self.lite: "lite", self.heavy: "heavy"}.get(model, "pinned")
        return (model or routed_model,
                thinking_level or routed_thinking,
                tier)

    # ---- LLMClient surface ------------------------------------------------ #
    def generate(self, prompt: str, *, audio_path=None, system=None, model=None,
                 thinking_level=None, json_out=False, timeout=600,
                 role=None, attempt=None) -> str:
        m, t, tier = self._resolved(model, thinking_level, role, attempt)
        with ledger.annotate(tier=tier):
            return self.base.generate(
                prompt, audio_path=audio_path, system=system, model=m,
                thinking_level=t, json_out=json_out, timeout=timeout)

    def generate_json(self, prompt: str, *, audio_path=None, system=None,
                      model=None, thinking_level=None, timeout=600,
                      role=None, attempt=None) -> dict:
        m, t, tier = self._resolved(model, thinking_level, role, attempt)
        with ledger.annotate(tier=tier):
            return self.base.generate_json(
                prompt, audio_path=audio_path, system=system, model=m,
                thinking_level=t, timeout=timeout)

    def lyria(self, *args, **kwargs):
        """Music generation is not a routed role — it is one model, on Vertex."""
        lyria = getattr(self.base, "lyria", None)
        if lyria is None:
            raise VertexError(
                f"the {type(self.base).__name__} backend has no Lyria music "
                f"generation; set BEATFORGE_LLM_BACKEND=gemini with a VERTEX_PROJECT.")
        return lyria(*args, **kwargs)


def _gemini_base() -> LLMClient:
    """The transport the hybrid router delegates to: Vertex, or AI Studio when
    that is the configured Gemini path."""
    if config.LLM_BACKEND in ("google-genai", "genai", "aistudio"):
        return GoogleAIStudioClient()
    return VertexClient()


def make_llm_client(backend: str | None = None) -> LLMClient:
    """Pick the designer/critic/A&R model backend (config.LLM_BACKEND by default).

    Returns the plain client by default (BEATFORGE_LLM_MODE=single: GEMINI_MODEL
    serves every call). On either Gemini path, BEATFORGE_LLM_MODE=hybrid wraps it
    in a HybridClient instead. The OpenAI-compatible backend is one server with one
    model loaded, so there is no second tier to route to and it is returned as-is.
    """
    backend = backend or config.LLM_BACKEND
    if backend == "openai":
        # The CHARTFORGE fine-tune needs its training-time prompt wrapper; a bare
        # designer prompt makes it degenerate. On by default; set
        # BEATFORGE_OPENAI_CHARTFORGE=0 for a non-CHARTFORGE OpenAI model.
        return ChartForgeClient() if config.OPENAI_CHARTFORGE else OpenAICompatClient()
    if backend in ("inkling", "inkling-modal", "modal"):
        # Imported lazily: inkling.py resolves credentials at construction, and a
        # user on another backend should not need Modal tokens on their machine.
        # `make_client` gates on the differential audio probe (REQ-INK-05) once
        # per endpoint per process — an endpoint that ignores audio must not get
        # as far as designing a chart.
        from .inkling import make_client
        return make_client()
    if backend == "argon":
        # Single-model by construction: Argon designs and (unless
        # BEATFORGE_ARGON_CRITIC_MODEL says otherwise) judges, with the thinking
        # level chosen per stage. A cheap-draft cascade with Argon as the quality
        # tier is the `gemini` backend in hybrid mode with
        # BEATFORGE_HYBRID_HEAVY_MODEL=gemini-4-argon — the transport handles it.
        if config.LLM_MODE == "hybrid":
            print("[argon] BEATFORGE_LLM_MODE=hybrid has no effect on the argon "
                  "backend (one model, per-stage thinking). For cheap drafts with "
                  "Argon as the quality tier, use BEATFORGE_LLM_BACKEND=gemini with "
                  "BEATFORGE_HYBRID_HEAVY_MODEL=gemini-4-argon.", file=sys.stderr)
        from .argon import make_client as make_argon
        return make_argon()
    if backend == "gemini":
        return HybridClient(VertexClient()) if config.LLM_MODE == "hybrid" else VertexClient()
    if backend in ("google-genai", "genai", "aistudio"):
        return (HybridClient(GoogleAIStudioClient()) if config.LLM_MODE == "hybrid"
                else GoogleAIStudioClient())
    raise ValueError(
        f"unknown LLM backend '{backend}' (use gemini|google-genai|openai|inkling|argon)")


__all__ = ["LLMClient", "OpenAICompatClient", "ChartForgeClient", "VertexClient",
           "GoogleAIStudioClient",
           "HybridClient", "tier_for", "VertexError", "OpenAICompatError",
           "make_llm_client"]
