"""inkling.py — Thinking Machines' Inkling served on a Modal Managed Endpoint.

Inkling is an OpenAI-compatible chat-completions endpoint, so it inherits the
transport, retry and JSON-repair machinery from `llm.OpenAICompatClient` and the
designer/critic call sites do not change. Everything Modal- or Inkling-specific
lives HERE (REQ-INK-01), because four things differ from a plain OpenAI server
and each one breaks silently if it leaks into the shared path:

  1. AUTH IS TWO HEADERS, NOT A BEARER TOKEN (REQ-INK-02). Modal Managed
     Endpoints take `Modal-Key` + `Modal-Secret` proxy tokens, and that is the
     whole contract — the proxy takes its own headers precisely so the SERVED app
     keeps `Authorization` for its own credential. Putting the proxy token in a
     bearer header 401s in a way that reads like a bad API key, which is exactly
     the wrong thing to go debugging.
  2. THE MODEL STRING IS RESOLVED, NOT HARDCODED. vLLM-backed endpoints sometimes
     serve under an alias, and a hardcoded id 404s.
  3. AUDIO RUNS TO ~20 MINUTES (REQ-INK-04). Every other audio backend we have
     evaluated capped near 30 seconds and forced downbeat-aligned chunking.
     Inkling takes the whole track in ONE call. Chunking is a per-provider
     workaround and must not be applied on this path.
  4. REASONING EFFORT IS A DIAL, AND REASONING BILLS AT THE FULL OUTPUT RATE
     (REQ-INK-07). The designer gets `high`; the critic and retries do not.
     Reasoning tokens are also GENERATED tokens, so they eat the same output cap
     the answer needs — which is why this backend carries its own, much larger
     `INKLING_MAX_TOKENS` rather than the one tuned for a local text-only model.

The one thing this module refuses to assume is that the endpoint actually
listens to the audio it is sent — see `differential_audio_probe` (REQ-INK-05).
A model that silently ignores an audio part still answers confidently, so a
single call cannot tell the difference; two different clips can.
"""
from __future__ import annotations

import copy
import difflib
import math
import os
import re
import struct
import sys
import tempfile
import wave
from pathlib import Path

from . import config, ledger
from .llm import OpenAICompatClient, OpenAICompatError, _encode_audio, _redact_blobs

TOKEN_PAGE = "https://modal.com/settings/proxy-auth-tokens"

# The probe asks for a description and gives the model an explicit way to say it
# got nothing — some servers strip an unsupported content part and answer the
# text anyway, and an honest refusal is easier to detect than a hallucination.
# No "reply NO_AUDIO_RECEIVED if you got nothing" clause, though the obvious
# thing to do is offer one. Measured against a live endpoint, adding it made the
# model answer NO_AUDIO_RECEIVED on a clip it had plainly received — offering the
# escape hatch invites it. The deaf case is caught by the two verdict checks
# instead, which need no cooperation from the model. The sentinel is still
# RECOGNISED below in case a model volunteers it unprompted.
PROBE_PROMPT = (
    "Listen to the audio, a steady tone. Reply with exactly one word: HIGH if it "
    "is a bright treble tone, LOW if it is a deep bass tone."
)
NO_AUDIO_SENTINEL = "NO_AUDIO_RECEIVED"
# Two steady tones, three octaves apart, and the one word a listening model must
# answer for each. Ground truth is the point: it replaces "do the two answers
# differ?" — which a model can fail by writing boilerplate — with "are the
# answers CORRECT?", which nothing but the audio can supply.
#
# ONE question, on the simplest possible stimulus. Both of those were paid for
# against a live endpoint. Asking pitch AND pulse rate in one compound reply was
# measurably less reliable than asking either alone; and gating the tone into a
# click train, which is what makes a pulse rate legible, puts broadband
# transients on every onset and drags a 110 Hz tone's perceived pitch UP to
# "HIGH". Steady tones at 110 Hz and 3 kHz classify correctly and repeatably.
# The probe exists to catch an endpoint that is not listening at all, not to
# measure how finely one that is can hear — so it asks the easiest question that
# still cannot be answered without the audio.
PROBE_CLIPS = (
    {"name": "a", "freq": 110.0, "pitch": "LOW"},
    {"name": "b", "freq": 3000.0, "pitch": "HIGH"},
)
# Four independent binary facts across the two clips, so a deaf model guessing
# every one of them passes at 1/16. The old test — "are the two descriptions less
# than 85% similar?" — had the error the other way round: a listening model that
# opened both answers with the same stock phrasing failed it, and a failure here
# stops the entire run.
PROBE_SIMILARITY_LIMIT = 0.85


class InklingError(OpenAICompatError):
    """Inkling/Modal-specific failure. Subclasses OpenAICompatError so existing
    `except OpenAICompatError` handlers in the pipeline keep working."""


class InklingAudioIgnored(InklingError):
    """The endpoint answered, but the differential probe says it is not listening.
    Separate from a transport failure because the remedy is different: this one
    means every chart from this endpoint would be designed by a deaf model."""


# --------------------------------------------------------------------------- #
# Credentials (REQ-INK-03)
# --------------------------------------------------------------------------- #
def resolve_credentials(token_id: str | None = None,
                        token_secret: str | None = None) -> tuple[str, str, str]:
    """Return `(token_id, token_secret, source)`.

    Order is explicit → env → keychain, so a CI/one-off override never has to
    fight stored state, and the source is returned rather than inferred because
    "which credential did it actually use?" is the first question when a 401
    shows up on a machine that has more than one.

    The secret is shown ONCE at creation on Modal's side and is unrecoverable
    afterwards, so a missing-secret error has to name that page.
    """
    if token_id and token_secret:
        return token_id, token_secret, "explicit"
    env_id = os.environ.get("MODAL_TOKEN_ID", "") or config.MODAL_TOKEN_ID
    env_secret = os.environ.get("MODAL_TOKEN_SECRET", "") or config.MODAL_TOKEN_SECRET
    if env_id and env_secret:
        return env_id, env_secret, "env"
    stored = _keychain_get()
    if stored:
        return stored[0], stored[1], "keychain"
    raise InklingError(
        "no Modal proxy token configured. Create one at "
        f"{TOKEN_PAGE} (the secret is shown only once), then either export "
        "MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or save them in Settings.")


def _keychain_get() -> tuple[str, str] | None:
    """Read the token pair from the OS keychain, or None.

    `keyring` is an optional import on purpose: it is not in requirements.txt,
    and an engine that refuses to start on a machine without it would be worse
    than one that falls back to the env vars.
    """
    try:
        import keyring
    except Exception:
        return None
    try:
        tid = keyring.get_password(config.INKLING_KEYCHAIN_SERVICE, "token_id")
        sec = keyring.get_password(config.INKLING_KEYCHAIN_SERVICE, "token_secret")
    except Exception:
        return None
    return (tid, sec) if tid and sec else None


def keychain_store(token_id: str, token_secret: str) -> bool:
    """Persist the pair in the OS keychain. False if unavailable, so a caller can
    tell the user their credentials did NOT get stored rather than silently
    losing them."""
    try:
        import keyring
        keyring.set_password(config.INKLING_KEYCHAIN_SERVICE, "token_id", token_id)
        keyring.set_password(config.INKLING_KEYCHAIN_SERVICE, "token_secret", token_secret)
        return True
    except Exception:
        return False


def scrub(text: str, *secrets: str) -> str:
    """Remove credential material from anything about to be logged or ledgered.

    Modal proxy tokens are recognisable (`wk-`/`ws-` prefixes) and endpoints echo
    request headers back in some error bodies, so redaction is by value AND by
    shape — a token that arrived from a source this process does not know about
    still must not reach a log file.
    """
    out = text
    for s in secrets:
        if s and len(s) >= 4:
            out = out.replace(s, "<redacted>")
    return re.sub(r"\b(wk|ws)-[A-Za-z0-9._-]{4,}", r"\1-<redacted>", out)


# --------------------------------------------------------------------------- #
# Audio preparation (REQ-INK-04)
# --------------------------------------------------------------------------- #
def audio_duration_s(path: str | Path) -> float:
    """Duration in seconds. soundfile is a core dependency and reads the header
    only; the wave fallback keeps this working for a plain WAV if libsndfile is
    ever unavailable."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate or 1)
    except Exception:
        try:
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / float(w.getframerate() or 1)
        except Exception:
            return 0.0


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #
class InklingClient(OpenAICompatClient):
    """Inkling on a Modal Managed Endpoint. Same interface as every other
    backend; all the Modal-specific handling is confined to this class."""

    name = "inkling"

    # Verified against a live Modal Auto Endpoint 2026-07-25: it takes
    # `audio_url` with a data: URI and rejects OpenAI's `input_audio` part with a
    # 25-error pydantic dump. Stated as a preference rather than left to the
    # "auto" fallback so the FIRST call is the right one — the fallback costs a
    # wasted upload of the whole track.
    preferred_transport = "audio_url"

    def __init__(self, endpoint_url: str | None = None, model: str | None = None,
                 token_id: str | None = None, token_secret: str | None = None,
                 timeout: int | None = None, temperature: float | None = None,
                 max_audio_seconds: int | None = None, effort: str | None = None,
                 **kw):
        url = (endpoint_url or config.INKLING_ENDPOINT_URL or "").strip().rstrip("/")
        if not url:
            raise InklingError(
                "no Inkling endpoint configured. Set BEATFORGE_INKLING_ENDPOINT_URL "
                "to your Modal Managed Endpoint URL (or fill it in under Settings).")
        # Modal endpoints serve the OpenAI API under /v1; accept either form so a
        # URL pasted straight from the dashboard works.
        base_url = url if url.endswith("/v1") else f"{url}/v1"
        self.token_id, self.token_secret, self.credential_source = resolve_credentials(
            token_id, token_secret)
        # Optional and separate from the proxy token: the credential the SERVED
        # app wants, if it wants one at all (a Managed Endpoint does not).
        self.app_api_key = (kw.pop("app_api_key", None) or config.INKLING_API_KEY or "").strip()
        super().__init__(
            base_url=base_url,
            model=model or config.INKLING_MODEL or "",
            api_key="unused",          # unused: _auth_headers() replaces this wholesale
            # Inkling documents 16 kHz mono WAV. Pinned per-client rather than
            # read from the BEATFORGE_OPENAI_AUDIO_* knobs, which are tuned for
            # whatever local server the `openai` backend happens to point at and
            # must not be able to silently violate this endpoint's input spec.
            audio_format=config.INKLING_AUDIO_FORMAT,
            audio_sr=config.INKLING_AUDIO_SR,
            audio_mono=True,
            timeout=timeout if timeout is not None else config.INKLING_TIMEOUT,
            # Its own cap: reasoning tokens count against this, and the openai
            # backend's 4096 is spent thinking before the answer starts.
            max_tokens=kw.pop("max_tokens", None) or config.INKLING_MAX_TOKENS,
            temperature=(config.INKLING_TEMPERATURE if temperature is None
                         else temperature),
            send_audio=kw.pop("send_audio", True),
            **kw)
        # MUST come after super().__init__: the base falls back to
        # config.OPENAI_MODEL for an empty model, which would silently bill and
        # log this endpoint under the *other* backend's model id — and send that
        # id to Modal, which has never heard of it.
        self.model = (model or config.INKLING_MODEL or "").strip()
        self.max_audio_seconds = (config.INKLING_MAX_AUDIO_SECONDS
                                  if max_audio_seconds is None else max_audio_seconds)
        self._forced_effort = effort
        self._resolved_model: str | None = None

    # -- auth (REQ-INK-02) --------------------------------------------------- #
    def _auth_headers(self) -> dict:
        """Modal's documented proxy-auth pair, and nothing else by default.

        "In requests to the Web Function, clients supply the Token ID and Token
        Secret in the `Modal-Key` and `Modal-Secret` HTTP headers." That is the
        whole contract, and the two custom headers are a deliberate design
        choice on Modal's side: the gateway takes its own headers precisely so
        `Authorization` stays free for whatever the SERVED application wants.

        Which is why this does NOT also send `Authorization: Bearer <id>.<secret>`
        the way it used to. That form is not in Modal's docs, and stuffing the
        proxy token into `Authorization` actively breaks the case the header was
        being kept free for — a self-deployed vLLM started with `--api-key` sees
        a bearer token it has never issued and 401s AFTER the proxy has already
        let the request through, producing an auth error that points at Modal
        when Modal was fine. It also copies the credential into a second header
        on every request for no gain. A 401 from the header pair alone is a wrong
        URL, a wrong workspace, or a wrong realm (`.modal.run` vs `.modal.direct`
        are separate auth realms) — see `_http_error`, which says so.

        `BEATFORGE_INKLING_API_KEY` is the supported way to authenticate to the
        served app on top of the proxy. It is a SEPARATE credential; the Modal
        proxy token must never be put there.
        """
        headers = {"Modal-Key": self.token_id, "Modal-Secret": self.token_secret}
        if self.app_api_key:
            headers["Authorization"] = f"Bearer {self.app_api_key}"
        return headers

    # -- reasoning effort (REQ-INK-07) --------------------------------------- #
    def _call_effort(self) -> str:
        """Effort for the call in flight, chosen by the pipeline stage already on
        the ledger context — so the designer gets `high` and the critic gets
        `low` without a single call site passing anything.

        A retry is keyed on `attempt`, NOT on a stage of its own. design.py opens
        `ledger.stage("designer", attempt=n)` for every draft, so there is no
        "reprompt" stage to match and the retry level used to be looked up under
        a key that could never be hit — BEATFORGE_INKLING_EFFORT_RETRY was wired
        from the GUI all the way to config and then silently ignored, leaving
        every retry at the designer's `high`.
        """
        if self._forced_effort:
            return self._forced_effort
        stage, attempt = "", None
        try:
            ctx = ledger.current() or {}
            stage = ctx.get("stage") or ""
            attempt = ctx.get("attempt")
        except Exception:
            pass
        if isinstance(attempt, int) and attempt > 1:
            return config.INKLING_EFFORT_RETRY
        return config.INKLING_EFFORT.get(stage, config.INKLING_EFFORT_DEFAULT)

    def _extra_body(self) -> dict:
        return {"reasoning_effort": self._call_effort()}

    def _apply_effort(self, body: dict, effort: str) -> None:
        body["reasoning_effort"] = effort

    def _next_effort_down(self, effort: str | None) -> str | None:
        """One rung down the ladder, or None at the bottom.

        Only ever ONE step, and only after a truncation: the point is to rescue a
        call whose reasoning ran away, not to keep bargaining the quality down
        until something comes back. A designer call that needed `high` and got
        `medium` produces a real chart; three retries produce an expensive one
        nobody asked for.
        """
        if not config.INKLING_DEGRADE_ON_TRUNCATION or not effort:
            return None
        ladder = config.INKLING_EFFORT_LADDER
        if effort not in ladder:
            return None
        i = ladder.index(effort)
        return ladder[i + 1] if i + 1 < len(ladder) else None

    # -- model resolution ---------------------------------------------------- #
    def resolve_model(self, timeout: int = 30) -> str:
        """The model string the endpoint actually serves, cached per client.

        Asked rather than assumed: `thinkingmachines/Inkling-NVFP4` is expected,
        but a vLLM deployment may serve an alias and a hardcoded id 400s/404s
        with a message that does not mention the model at all.
        """
        if self._resolved_model:
            return self._resolved_model
        if self.model:
            self._resolved_model = self.model
            return self.model
        try:
            served = self.list_models(timeout=timeout)
        except OpenAICompatError as e:
            # `/v1/models` is NOT a reliable capability on these endpoints. A live
            # Modal Auto Endpoint (verified 2026-07-25) answers `401 missing or
            # invalid Authorization header` on GET /v1/models with the very same
            # proxy headers that POST /v1/chat/completions accepts — the two paths
            # sit behind different auth. Letting that 401 escape would mean a
            # working endpoint refuses to chart, and reading it as "bad
            # credentials" is what sends people chasing an auth bug that is not
            # there. So: name the model instead of discovering it.
            raise InklingError(
                f"could not list models at {self.base_url}/models, so the model "
                f"string cannot be resolved automatically:\n  {str(e)[:400]}\n\n"
                f"This is expected on some Modal endpoints — GET /v1/models can be "
                f"refused while chat completions work fine with the same proxy "
                f"token. Set the model explicitly instead of resolving it: "
                f"BEATFORGE_INKLING_MODEL=thinkingmachines/Inkling-NVFP4 (or "
                f"whatever the Modal dashboard lists for this endpoint), or fill in "
                f"the Model field under Settings.") from e
        if not served:
            raise InklingError(
                f"{self.base_url}/models returned no models — the endpoint is up "
                f"but is serving nothing. Check the Modal deployment.")
        # Prefer an Inkling-looking id when several are served.
        pick = next((m for m in served if m and "inkling" in m.lower()), served[0])
        self._resolved_model = pick
        self.model = pick
        return pick

    # -- audio (REQ-INK-04) -------------------------------------------------- #
    def _encode_for_request(self, audio_path) -> str:
        """The whole track, in one part, at Inkling's documented 16 kHz mono WAV.

        No chunking: this endpoint takes ~20 minutes of audio and 1M tokens of
        context, so the downbeat-aligned chunk-and-stitch machinery other audio
        backends need is not just unnecessary here, it would cost extra calls and
        split the musical context the designer is being paid to hear.
        """
        seconds = audio_duration_s(audio_path)
        if seconds > self.max_audio_seconds:
            raise InklingError(
                f"track is {seconds/60:.1f} min, over the {self.max_audio_seconds/60:.0f} "
                f"min limit for this endpoint. Raise "
                f"BEATFORGE_INKLING_MAX_AUDIO_SECONDS if the endpoint really accepts "
                f"more, or chart a shorter cut — the track is NOT sent in chunks on "
                f"this backend by design.")
        # Whichever shape wraps these bytes, the container name has to match
        # them: the server picks its decoder off that string, so a mismatch is a
        # decode error deep in librosa rather than a schema error the caller can
        # read. `_wrap_audio` handles both shapes; see `preferred_transport`.
        return _encode_audio(audio_path, self.audio_format,
                             sr=self.audio_sr, mono=self.audio_mono)

    def generate(self, prompt: str, *, audio_path=None, system=None, model=None,
                 thinking_level=None, json_out=False, timeout=None) -> str:
        # Resolve the served model once, so every call bills and logs under the
        # string the endpoint actually knows.
        if model is None and not self.model:
            self.resolve_model()
        return super().generate(
            prompt, audio_path=audio_path, system=system, model=model,
            thinking_level=thinking_level, json_out=json_out, timeout=timeout)

    # -- errors (REQ-INK-02 / REQ-INK-04 guardrails) ------------------------- #
    def _http_error(self, code: int, detail: str, sent_audio: bool,
                    where: str = "") -> OpenAICompatError:
        detail = scrub(_redact_blobs(detail), self.token_id, self.token_secret)
        low = detail.lower()
        if code == 404 and "requires a method name" in low:
            # Verified against a live endpoint: the bare class URL 404s like this.
            return InklingError(
                f"HTTP 404 from {self.base_url}{where}: {detail[:200]}\n\n"
                f"Auth SUCCEEDED — this is a URL problem. The endpoint is a Modal "
                f"*class* web endpoint, so the method name belongs in the HOSTNAME, "
                f"not the path:\n"
                f"    https://<workspace>--<app>-<class>-<method>.modal.run\n"
                f"Copy the exact URL from the Modal dashboard (the endpoint's page "
                f"shows it) into BEATFORGE_INKLING_ENDPOINT_URL.")
        if code == 404 and "invalid function call" in low:
            return InklingError(
                f"HTTP 404 from {self.base_url}{where}: {detail[:200]}\n\n"
                f"No Modal endpoint by that name. The method segment of the hostname "
                f"is wrong — copy the exact URL from the Modal dashboard rather than "
                f"guessing it.")
        if code in (401, 403):
            hint = ""
            if "missing credentials for proxy authorization" in low:
                hint = ("Modal's proxy did not see a credential it recognises, so "
                        "the headers are not arriving as sent — check for a proxy "
                        "or corporate gateway between here and Modal that strips "
                        "unknown request headers.")
            elif "invalid token" in low:
                hint = ("The credential reached the auth check and was refused, so "
                        "the headers ARE arriving. Most likely the token belongs to "
                        "a different workspace, or — with RBAC on — is not "
                        "associated with the environment holding this endpoint. "
                        "Note also that a `*.modal.direct` host and a `*.modal.run` "
                        "host are DIFFERENT auth realms: a token valid at one can be "
                        "refused by the other. Confirm which URL the Modal dashboard "
                        "lists for this endpoint before minting a new pair.")
            elif self.app_api_key:
                hint = ("Two credentials are in play on this request and the message "
                        "does not say which failed: the Modal proxy token, and the "
                        "BEATFORGE_INKLING_API_KEY sent as `Authorization: Bearer` "
                        "for the served app. Clear the API key to test the proxy "
                        "token alone — a Managed Endpoint needs no app key.")
            return InklingError(
                f"HTTP {code} from {self.base_url}{where}: {detail[:400]}\n\n"
                f"{hint}\nModal proxy auth is the two headers `Modal-Key: wk-...` "
                f"and `Modal-Secret: ws-...`, which is what this client sends "
                f"(credentials from: {self.credential_source}"
                + (", plus an app Authorization key" if self.app_api_key else "")
                + f").\nThe proxy token is NOT a bearer credential — do not put it "
                f"in an Authorization header or in BEATFORGE_INKLING_API_KEY.\n"
                f"Mint a fresh pair at {TOKEN_PAGE} if needed — the secret is shown "
                f"only once and cannot be retrieved later.")
        if code == 413:
            return InklingError(
                f"HTTP 413 from {self.base_url}{where}: the request body was too large.\n"
                f"The whole track goes in one call on this backend (no chunking); "
                f"the limit here is {self.max_audio_seconds/60:.0f} min of 16 kHz "
                f"mono audio. Chart a shorter cut or raise the endpoint's body limit.")
        return super()._http_error(code, detail, sent_audio, where)


# --------------------------------------------------------------------------- #
# The differential audio probe (REQ-INK-05)
# --------------------------------------------------------------------------- #
def _tone_clip(path: Path, freq: float, seconds: float = 4.0,
               sr: int = 16000) -> Path:
    """Write a small 16 kHz mono WAV holding one steady tone, no dependencies.

    Synthesised rather than sliced from the corpus so the probe works on a fresh
    install with no tracks imported.

    STEADY, deliberately — no tremolo and no gating. Both were tried against a
    live endpoint and both corrupt the one thing the probe asks about: gating a
    tone into clicks puts a broadband transient on every onset, which drags a
    110 Hz tone's perceived pitch up to "HIGH" and fails an endpoint that is
    hearing it perfectly well. The 5 ms raised-cosine fades exist for the same
    reason — a waveform that starts and ends mid-cycle clicks, and the click is
    exactly the artefact that misleads the classification.

    QUIET, also deliberately, and this one cost the most to find. At -18 dBFS the
    live endpoint classified both tones correctly on every repeat; at the ~-3.5
    dBFS this used to emit, a spectrally IDENTICAL 3 kHz tone (verified by FFT —
    peak bin exactly 3000 Hz either way) came back "LOW". Loudness, not content,
    was deciding the answer. So the level is part of the stimulus, not a detail:
    do not turn it back up.
    """
    frames = bytearray()
    n = int(seconds * sr)
    fade = max(1, int(0.005 * sr))
    peak = 4000                       # ≈ -18 dBFS of int16 full scale
    for i in range(n):
        t = i / sr
        env = 1.0
        if i < fade:
            env = 0.5 * (1 - math.cos(math.pi * i / fade))
        elif i > n - fade:
            env = 0.5 * (1 - math.cos(math.pi * (n - i) / fade))
        sample = int(peak * env * math.sin(2 * math.pi * freq * t))
        frames += struct.pack("<h", max(-32768, min(32767, sample)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    return path


def _read_verdict(answer: str) -> str:
    """Pull HIGH or LOW out of a reply, tolerantly. Returns "" when the model
    said neither or said both — read as a miss, never guessed at."""
    up = answer.upper()
    found = set(re.findall(r"\b(HIGH|LOW)\b", up))
    return found.pop() if len(found) == 1 else ""


def differential_audio_probe(client, *, timeout: int = 180) -> dict:
    """Prove the endpoint is actually listening. Returns a verdict dict.

    The failure this exists for is not a crash — it is an endpoint that quietly
    drops the audio part and answers from the text prompt, producing
    plausible-looking charts designed by a model that never heard the song. One
    open-ended call cannot detect that: a deaf model still confidently describes
    "an upbeat electronic track".

    So the probe asks a question with a KNOWN ANSWER: two steady tones three
    octaves apart, one word each (HIGH / LOW), same prompt, temperature 0. Two
    checks have to pass together:

      * BOTH answers are correct. Nothing but the audio supplies that.
      * The two answers DIFFER. At temperature 0 a model answering from the
        prompt alone gets the same deterministic reply twice — which is exactly
        what a deaf endpoint was observed doing. Guessing blind therefore has to
        beat both checks, which is one chance in four rather than one in two.

    This replaced a test that compared two free-text DESCRIPTIONS with difflib and
    failed the endpoint when they were >85% similar. That measured the wrong
    thing in the more dangerous direction — a model that heard both clips but
    opened each answer with the same stock sentence failed it, and a failed probe
    stops the entire batch. The difference-check survives here because a
    forced-choice token has no boilerplate to be similar about.

    Verdict: {"passed": bool, "similarity": float, "reason": str,
              "responses": [a, b], "expected": [...], "observed": [...],
              "model": str}
    """
    # Temperature 0 so a wrong answer is the model's, not the sampler's — and so
    # that a deaf endpoint's two replies are identical rather than accidentally
    # different. A copy, so the caller's client keeps its own sampling settings.
    probe_client = copy.copy(client)
    probe_client.temperature = 0

    with tempfile.TemporaryDirectory() as td:
        answers = []
        with ledger.stage("audio_probe", song="_probe", target="core"):
            for spec in PROBE_CLIPS:
                clip = _tone_clip(Path(td) / f"{spec['name']}.wav", freq=spec["freq"])
                answers.append(probe_client.generate(
                    PROBE_PROMPT, audio_path=str(clip), timeout=timeout).strip())

    model = getattr(client, "model", "") or ""
    similarity = round(difflib.SequenceMatcher(None, answers[0], answers[1]).ratio(), 3)
    expected = [s["pitch"] for s in PROBE_CLIPS]
    observed = [_read_verdict(a) or "?" for a in answers]
    base = {"similarity": similarity, "model": model, "responses": answers,
            "expected": expected, "observed": observed}

    if any(NO_AUDIO_SENTINEL in a.upper() for a in answers):
        return {**base, "passed": False,
                "reason": f"the endpoint replied {NO_AUDIO_SENTINEL}: it received no "
                          f"audio. It is not usable for charting."}
    # Checked BEFORE the per-clip comparison, and the order is the whole point.
    # Identical answers always fail the ground-truth check too — the expected
    # answers differ, so one of them must be wrong — but "judged clip b
    # incorrectly" describes a model with poor hearing, while "same answer for
    # two tones three octaves apart" describes one that never heard either. The
    # second is the failure this probe exists for, so it gets to speak first.
    if observed[0] == observed[1]:
        return {**base, "passed": False,
                "reason": f"the endpoint gave the same answer ({observed[0]}) for two "
                          f"tones three octaves apart, at temperature 0 — so the "
                          f"answer did not come from the audio. It is not usable for "
                          f"charting."}
    wrong = [i for i in range(len(PROBE_CLIPS)) if observed[i] != expected[i]]
    if wrong:
        detail = "; ".join(f"clip {PROBE_CLIPS[i]['name']} is a "
                           f"{PROBE_CLIPS[i]['freq']:.0f} Hz tone, so the answer is "
                           f"{expected[i]}, but it said {observed[i]}" for i in wrong)
        return {**base, "passed": False,
                "reason": f"the endpoint judged the test tones incorrectly ({detail}). "
                          f"Either it is not receiving the audio part or it cannot "
                          f"hear it — charts from it would be designed by a model that "
                          f"never heard the song."}
    return {**base, "passed": True,
            "reason": f"both test tones were judged correctly and differently "
                      f"({observed[0]} then {observed[1]}) — the endpoint is listening."}


# Endpoints already proven to listen in THIS process. The probe costs two short
# calls, so it runs once per endpoint per run — not once per song, and not once
# per client — while still gating every batch that starts fresh.
_PROBED: set[str] = set()


def make_client(*, probe: bool | None = None, **kw) -> "InklingClient":
    """Build a probe-gated client. This is what the factory returns.

    The gate lives here rather than at the designer/critic call sites (REQ-INK-01)
    and fires before any charting work, because the failure it catches is silent:
    an endpoint that ignores audio produces confident, plausible, worthless
    charts. Set BEATFORGE_INKLING_SKIP_PROBE=1 to bypass it once you have proven
    an endpoint and want the two calls back.
    """
    client = InklingClient(**kw)
    if probe is None:
        probe = os.environ.get("BEATFORGE_INKLING_SKIP_PROBE", "") in ("", "0", "false")
    if probe and client.base_url not in _PROBED:
        require_audio(client)
        _PROBED.add(client.base_url)
    return client


def require_audio(client, *, timeout: int = 180, fatal: bool | None = None) -> dict:
    """Gate a batch on the probe (REQ-INK-05).

    Fatal by default, because the failure it catches is silent and expensive: a
    deaf endpoint charts a whole pack confidently and wrongly, and nothing
    downstream can tell. `BEATFORGE_INKLING_REQUIRE_AUDIO=0` downgrades it to a
    warning for the case where the operator has verified the endpoint by hand and
    would rather not have a probe stand between them and a run.
    """
    fatal = config.INKLING_REQUIRE_AUDIO if fatal is None else fatal
    verdict = differential_audio_probe(client, timeout=timeout)
    if verdict["passed"]:
        return verdict
    report = (f"audio probe FAILED for {getattr(client, 'base_url', '?')}: "
              f"{verdict['reason']}\n"
              + "\n".join(f"  clip {s['name']} -> {r[:200]}"
                          for s, r in zip(PROBE_CLIPS, verdict["responses"])))
    if fatal:
        raise InklingAudioIgnored(
            report + "\n\nSet BEATFORGE_INKLING_REQUIRE_AUDIO=0 to proceed anyway "
                     "(the charts will be designed without hearing the track).")
    print(f"[inkling] WARNING: {report}\nProceeding because "
          f"BEATFORGE_INKLING_REQUIRE_AUDIO=0 — charts from this run may have been "
          f"designed by a model that never heard the music.", file=sys.stderr)
    return verdict


__all__ = ["InklingClient", "InklingError", "InklingAudioIgnored", "make_client",
           "differential_audio_probe", "require_audio", "resolve_credentials",
           "keychain_store", "scrub", "audio_duration_s", "TOKEN_PAGE"]
