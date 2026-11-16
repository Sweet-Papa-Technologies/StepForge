"""vertex.py — shared Vertex AI client for beatforge.

Mirrors the auth/endpoint pattern of ~/.assetforge/assetforge.py and
tools/generate_overdrive_assets.py: a service-account (or ADC) bearer token,
Gemini 3.x routed to the v1beta1 `global` endpoint, everything else to the
regional endpoint. Adds what beatforge needs beyond assetforge's CLI:

  * audio-part `generateContent` (Gemini *hears* the track — the whole point),
  * family-aware "full thinking power" config (thinking_level vs thinkingBudget),
  * strict-JSON responses with defensive fence-stripping + one parse retry,
  * Lyria music prediction for Workstream A.

All model calls in beatforge go through here so they sit behind one seam that
tests stub out (see tests/fakes.py / offline mode).
"""
from __future__ import annotations

import base64
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import config, ledger

_MIME = {
    "wav": "audio/wav", "mp3": "audio/mp3", "ogg": "audio/ogg", "flac": "audio/flac",
    "m4a": "audio/mp4", "aac": "audio/aac", "png": "image/png", "jpg": "image/jpeg",
}


class VertexError(RuntimeError):
    #: Set when a long decode failed part-way: the merged response of the hops
    #: that DID complete, so the ledger can bill them instead of claiming zero.
    partial: dict | None = None


# --------------------------------------------------------------------------- #
# Global rate-limit pacing gate — enforces a minimum wall-clock interval between
# consecutive model calls across ALL clients in the process. One audio call is
# ~5s (so sequential is already ~12 RPM); BEATFORGE_LLM_MIN_INTERVAL adds extra
# headroom for batch runs. Default 0 = off (single interactive calls unaffected).
# --------------------------------------------------------------------------- #
import threading
import time as _time

_PACE_LOCK = threading.Lock()
_LAST_CALL = [0.0]
_MIN_INTERVAL = float(os.environ.get("BEATFORGE_LLM_MIN_INTERVAL", "0") or 0)


def _pace() -> None:
    if _MIN_INTERVAL <= 0:
        return
    with _PACE_LOCK:
        wait = _MIN_INTERVAL - (_time.monotonic() - _LAST_CALL[0])
        if wait > 0:
            _time.sleep(wait)
        _LAST_CALL[0] = _time.monotonic()


def _guess_mime(path: str | Path) -> str:
    ext = os.path.splitext(str(path))[1].lower().lstrip(".")
    return _MIME.get(ext, "application/octet-stream")


def _b64file(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# --------------------------------------------------------------------------- #
# Gemini 4 request shape. Three things differ from 3.x, and every one of them is
# a 400 or a silently truncated chart if the 3.x shape is sent unchanged:
#   * thinkingLevel is LOW | MEDIUM | HIGH only. `minimal` — which the 3.x path
#     and the desktop's global level both offer — is rejected, and so is any
#     `thinkingBudget` (the 2.5-era field the non-3.x branch below emits).
#   * A non-streaming response carries at most 32,768 output tokens, thinking
#     included. A whole-song designer call on 3.6 Flash already measured 23-28k
#     output+thinking, and Argon at high thinks ~3x as hard, so a single response
#     is not enough room. Asking for more turns on LONG DECODE: the server stops
#     at the per-response cap with finishReason=CONTINUATION and a token, and the
#     same request is re-sent with that token until it says STOP.
#   * Sampling parameters are ignored (this client never sent any).
# These are properties of the MODEL FAMILY, not of the Argon backend, so they
# live in the transport: BEATFORGE_GEMINI_MODEL=gemini-4-argon on the plain
# `gemini` backend, or Argon as the hybrid heavy tier, gets them too.
# --------------------------------------------------------------------------- #
GEMINI4_LEVELS = ("low", "medium", "high")
#: Per-response output ceiling of a non-streaming generateContent on Gemini 4.
GEMINI4_HOP_TOKENS = 32768
#: The model's TOTAL output limit across all hops — 1,048,576, and unified with
#: the input (prompt + output <= 1M). A budget above it is a 400, so it is
#: clamped rather than sent.
GEMINI4_MAX_OUTPUT_TOKENS = 1_048_576


def gemini4_max_hops(budget: int) -> int:
    """Hops one call may take before the client stops following continuations.

    The SERVER enforces the token budget (it answers MAX_TOKENS once it is
    spent), so this is only a guard against a sequence that never ends — and it
    must scale with the budget. A fixed cap silently truncated any budget above
    cap x 32,768 tokens: at 12 hops a 1M budget stopped near 393k. Twice the
    full-hop count allows for a server that returns short hops.
    """
    return max(4, 2 * -(-int(budget) // GEMINI4_HOP_TOKENS))


def is_gemini4(model: str) -> bool:
    return (model or "").startswith("gemini-4")


def gemini4_level(level: str | None) -> str:
    """The thinking level Gemini 4 will accept for `level`. `minimal` becomes
    `low` (the nearest legal rung) rather than a 400; anything unrecognised gets
    `medium`, the model's own documented default."""
    lv = (level or "").strip().lower()
    if lv in GEMINI4_LEVELS:
        return lv
    return "low" if lv == "minimal" else "medium"


def _continuation_token(resp: dict) -> str | None:
    """The token to resume with, or None when the response is complete.

    Only CONTINUATION means "more to come". MAX_TOKENS also carries a token, but
    it means the whole maxOutputTokens budget is spent — resuming with the same
    budget is a 400, and the reply is as finished as it is going to get.
    """
    cands = resp.get("candidates") or []
    if not cands or cands[0].get("finishReason") != "CONTINUATION":
        return None
    return cands[0].get("continuationToken") or resp.get("continuationToken")


def _merge_hops(hops: list[dict]) -> dict:
    """Fold a long-decode sequence into ONE response, shaped like a normal one.

    The answer text of every hop is joined into ONE part with no separator. A
    hop ends wherever the token cap falls — mid-number, mid-id, inside a string —
    so the pieces are fragments of one reply, not paragraphs: `_all_text` joins
    separate parts with a newline, which split across `"p01` / `7"` would turn
    the chart JSON invalid. Thought parts are kept as-is (`_all_text` drops them).

    Usage is SUMMED across hops, because each hop is a separately billed request
    that re-reads the whole prompt — mostly from the implicit cache, which its own
    cachedContentTokenCount reports. Billing only the first hop's prompt would
    under-count every call that needed a second one.
    """
    if len(hops) == 1:
        return hops[0]
    thoughts: list[dict] = []
    text: list[str] = []
    for h in hops:
        for cand in (h.get("candidates") or [])[:1]:
            for p in (cand.get("content") or {}).get("parts") or []:
                if p.get("thought"):
                    thoughts.append(p)
                elif "text" in p:
                    text.append(p["text"])
    parts = thoughts + ([{"text": "".join(text)}] if text else [])
    last = hops[-1]
    cand0 = dict((last.get("candidates") or [{}])[0])
    cand0["content"] = {"role": "model", "parts": parts}
    usage: dict[str, Any] = {}
    by_mod: dict[str, dict[str, int]] = {"promptTokensDetails": {},
                                         "cacheTokensDetails": {}}
    for h in hops:
        um = h.get("usageMetadata") or {}
        for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount",
                  "cachedContentTokenCount", "totalTokenCount"):
            usage[k] = usage.get(k, 0) + int(um.get(k, 0) or 0)
        for field in by_mod:
            for d in um.get(field) or []:
                mod = str(d.get("modality", "UNKNOWN")).upper()
                by_mod[field][mod] = by_mod[field].get(mod, 0) + int(d.get("tokenCount", 0) or 0)
        if um.get("trafficType") and "trafficType" not in usage:
            usage["trafficType"] = um["trafficType"]
    for field, counts in by_mod.items():
        if counts:
            usage[field] = [{"modality": m, "tokenCount": n} for m, n in counts.items()]
    return {**last, "candidates": [cand0], "usageMetadata": usage}


def _role_thinking() -> str | None:
    """The ambient role's thinking level, or None when the per-role map is off.

    Imported lazily because `llm.py` imports THIS module — the router is layered
    above the transport, and that direction must not reverse. `llm.thinking_for`
    is pure and cheap, so calling it per request costs nothing worth optimizing.
    """
    from .llm import _ambient_role, thinking_for
    role, attempt = _ambient_role()
    return thinking_for(role, attempt)


class VertexClient:
    """Thin generateContent / predict wrapper. One token cached per client."""

    # Ledger provider label; subclasses (e.g. Google AI Studio) override it so the
    # cost ledger records which transport actually served the call.
    provider = "vertex"
    # Long decode on Gemini 4: None = untried, False = this endpoint refused an
    # output budget above one response, so stay at GEMINI4_HOP_TOKENS for the
    # rest of the process. A class attribute because GoogleAIStudioClient does
    # not run this __init__.
    _long_decode: bool | None = None

    def __init__(self):
        self.project = config.VERTEX_PROJECT
        self.location = config.VERTEX_LOCATION
        self.key_file = config.VERTEX_KEY
        self._creds = None

    # ---- auth / endpoint (mirrors assetforge) ---------------------------- #
    def _get_token(self, force: bool = False) -> str:
        """Return a valid bearer token, refreshing when it's expired (OAuth
        access tokens live ~1h; a multi-hour batch MUST re-mint or every call
        past the first hour 401s). `force=True` re-mints regardless (used after
        a 401)."""
        try:
            from google.auth.transport.requests import Request
        except ImportError as e:
            # google-auth no longer installs `requests` itself, so an environment
            # built from a bare `google-auth` requirement fails here, on the first
            # Vertex call, with a traceback that names neither the fix nor the
            # interpreter that needs it.
            raise VertexError(
                f"Vertex authentication needs the `requests` package, which is not "
                f"installed for {sys.executable}. Install it with:\n"
                f"  \"{sys.executable}\" -m pip install 'google-auth[requests]'\n"
                f"(or re-run the desktop app's dependency installer). A Gemini API "
                f"key — the AI Studio transport — does not need it.") from e
        if self._creds is None:
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            if self.key_file and os.path.isfile(self.key_file):
                from google.oauth2 import service_account
                self._creds = service_account.Credentials.from_service_account_file(
                    self.key_file, scopes=scopes)
            else:
                import google.auth
                self._creds, _ = google.auth.default(scopes=scopes)
        if force or not self._creds.valid:      # .valid = has token AND not expired
            self._creds.refresh(Request())
        return self._creds.token

    def _auth_headers(self) -> dict:
        """Auth header(s) for a request. Vertex uses an OAuth bearer token;
        subclasses (Google AI Studio) override to send an API key instead."""
        return {"Authorization": f"Bearer {self._get_token()}"}

    def _endpoint(self, model: str) -> tuple[str, str]:
        api, loc = "v1", self.location
        # Gemini 4 is not served from us-central1 at all (global, us, eu and two
        # single regions only), global is also the cheaper endpoint, and long
        # decode exists on v1beta1 only — so it takes the 3.x route exactly.
        if model.startswith(("gemini-3", "gemini-4")):
            api, loc = "v1beta1", "global"
        elif model.startswith("lyria-3"):
            api = "v1beta1"
        return api, loc

    def _url(self, model: str, method: str) -> str:
        if not self.project:
            raise VertexError(
                "No Vertex project configured. Set VERTEX_PROJECT to your GCP "
                "project id, e.g. `export VERTEX_PROJECT=my-gcp-project` (and "
                "authenticate with `gcloud auth application-default login`). "
                "Charting can also run with no Google account via "
                "BEATFORGE_LLM_BACKEND=openai.")
        api, loc = self._endpoint(model)
        host = ("aiplatform.googleapis.com" if loc == "global"
                else f"{loc}-aiplatform.googleapis.com")
        return (f"https://{host}/{api}/projects/{self.project}"
                f"/locations/{loc}/publishers/google/models/{model}:{method}")

    def _post(self, url: str, body: dict, timeout: int = 600) -> dict:
        data = json.dumps(body).encode()
        # Retry on rate-limit (429) and transient server errors (500/503) with
        # exponential backoff, so a momentary quota spike doesn't kill a chart.
        backoffs = [5, 15, 40, 90]
        for attempt in range(len(backoffs) + 1):
            _pace()                                # rate-limit pacing (batch runs)
            req = urllib.request.Request(url, data=data, method="POST", headers={
                **self._auth_headers(),
                "Content-Type": "application/json",
            })
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                # 401: the cached token expired mid-batch — force a re-mint and
                # retry immediately (don't burn a backoff slot on a stale token).
                if e.code == 401 and attempt < len(backoffs):
                    print(f"[vertex] HTTP 401 (token expired); refreshing credentials "
                          f"(retry {attempt + 1}/{len(backoffs)})")
                    self._get_token(force=True)
                    continue
                # 403 "dunning" is a BILLING state, not a real permission error —
                # after a billing fix it clears but propagates unevenly across
                # Google's systems, so some calls still deny for a few minutes.
                # Retry those (unlike a genuine 403) so a batch rides out the lag.
                if (e.code == 403 and "dunning" in detail.lower()
                        and attempt < len(backoffs)):
                    wait = backoffs[attempt]
                    print(f"[vertex] HTTP 403 billing-dunning (propagating); backing off "
                          f"{wait}s (retry {attempt + 1}/{len(backoffs)})")
                    _time.sleep(wait)
                    continue
                if e.code in (429, 500, 502, 503, 504) and attempt < len(backoffs):
                    wait = backoffs[attempt]
                    print(f"[vertex] HTTP {e.code} (rate/transient); backing off {wait}s "
                          f"(retry {attempt + 1}/{len(backoffs)})")
                    _time.sleep(wait)
                    continue
                raise VertexError(f"HTTP {e.code} from Vertex:\n{detail[:1200]}")
            except (TimeoutError, socket.timeout, urllib.error.URLError, OSError) as e:
                # A read timeout or dropped connection is exactly as transient as a
                # 503, but it raises a different exception type and so used to fall
                # straight through — one timed-out socket killed a whole song
                # 100 minutes into a batch. A thinking-heavy designer call can sit
                # quiet for minutes, which makes this failure mode routine, not rare.
                reason = getattr(e, "reason", e)
                if isinstance(e, urllib.error.URLError) and not isinstance(
                        reason, (TimeoutError, socket.timeout, ConnectionError, OSError)):
                    raise VertexError(f"cannot reach Vertex: {reason}")
                if attempt < len(backoffs):
                    wait = backoffs[attempt]
                    print(f"[vertex] network/timeout ({type(e).__name__}: {reason}); "
                          f"backing off {wait}s (retry {attempt + 1}/{len(backoffs)})")
                    _time.sleep(wait)
                    continue
                raise VertexError(f"network failure after retries: {type(e).__name__}: {reason}")
        raise VertexError("exhausted retries")

    # ---- thinking config (spec §3: full thinking power) ------------------ #
    def _thinking_config(self, model: str, level: str) -> dict:
        """Return the generationConfig fragment that maxes reasoning for the
        model family. Gemini 3.x speaks `thinkingLevel`; Gemini 2.5 speaks
        `thinkingConfig.thinkingBudget` (-1 = dynamic/max). Gemini 4 speaks
        `thinkingLevel` too, but over a shorter ladder — see `gemini4_level`."""
        if is_gemini4(model):
            return {"thinkingConfig": {"thinkingLevel": gemini4_level(level)}}
        if model.startswith("gemini-3"):
            return {"thinkingConfig": {"thinkingLevel": level}}
        # Gemini 2.5 family: -1 lets the model think as much as it needs.
        budget = {"high": -1, "medium": 8192, "low": 2048, "minimal": 512}.get(level, -1)
        return {"thinkingConfig": {"thinkingBudget": budget}}

    def _all_text(self, resp: dict) -> str:
        out = []
        for cand in resp.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if part.get("thought"):
                    continue
                if "text" in part:
                    out.append(part["text"])
        return "\n".join(out).strip()

    # ---- public: multimodal generate ------------------------------------ #
    def generate(
        self,
        prompt: str,
        *,
        audio_path: str | Path | None = None,
        system: str | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
        json_out: bool = False,
        timeout: int = 600,
    ) -> str:
        """One Gemini call. Optionally attaches an audio part so the model
        *hears* the track. Returns the concatenated non-thought text."""
        model = model or config.GEMINI_MODEL
        level = thinking_level or _role_thinking() or config.DEFAULT_THINKING
        if is_gemini4(model):
            # Normalised HERE rather than only inside _thinking_config so the
            # ledger records the level that actually ran, not the one requested.
            level = gemini4_level(level)
        parts: list[dict] = []
        audio_bytes = 0
        if audio_path is not None:
            audio_bytes = os.path.getsize(audio_path)
            parts.append({"inlineData": {
                "mimeType": _guess_mime(audio_path), "data": _b64file(audio_path)}})
        parts.append({"text": prompt})
        gen_cfg: dict[str, Any] = self._thinking_config(model, level)
        cap = self._output_cap(model)
        if cap:
            gen_cfg["maxOutputTokens"] = cap
        if json_out:
            gen_cfg["responseMimeType"] = "application/json"
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": gen_cfg,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        body.update(self._request_fields(model))

        # REQ-R2-COST-01: bill the call from the response's own usageMetadata and
        # the model string we ACTUALLY sent — never from config, which has been
        # wrong before. Recorded on the failure path too, because a call that
        # errored after the tokens were consumed still costs money.
        t0 = _time.monotonic()
        try:
            resp, hops = self._generate_content(model, body, timeout)
        except Exception as e:
            # A long decode that died on hop N had already been billed for hops
            # 1..N-1; `partial` carries them so the ledger does not claim zero.
            partial = getattr(e, "partial", None) or {}
            ledger.record_model_call(
                provider=self.provider, model=model,
                usage=ledger.usage_from_vertex(partial),
                latency_s=_time.monotonic() - t0, prompt_bytes=len(prompt.encode()),
                audio_attached=audio_path is not None, audio_path=str(audio_path or "") or None,
                audio_bytes=audio_bytes, thinking_level=level, error=str(e)[:300],
                service_tier=self._served_tier(partial) if partial else None)
            raise
        ledger.record_model_call(
            provider=self.provider, model=model, usage=ledger.usage_from_vertex(resp),
            latency_s=_time.monotonic() - t0, prompt_bytes=len(prompt.encode()),
            audio_attached=audio_path is not None, audio_path=str(audio_path or "") or None,
            audio_bytes=audio_bytes, thinking_level=level,
            service_tier=self._served_tier(resp),
            detail={"long_decode_hops": hops} if hops > 1 else None)

        text = self._all_text(resp)
        if not text:
            raise VertexError(f"empty response from {model}: {json.dumps(resp)[:400]}")
        return text

    # ---- request-shape hooks (overridden by argon.py) -------------------- #
    def _output_cap(self, model: str) -> int | None:
        """`maxOutputTokens` to send, or None to leave the server default.

        None for 3.x: its default has always been room enough, and pinning a
        number there would be a new variable in every benchmark already run.
        Gemini 4 gets the long-decode budget unless this endpoint has refused it.
        """
        if not is_gemini4(model):
            return None
        if self._long_decode is False:
            return GEMINI4_HOP_TOKENS
        return min(GEMINI4_MAX_OUTPUT_TOKENS,
                   max(GEMINI4_HOP_TOKENS, config.ARGON_MAX_OUTPUT_TOKENS))

    def _request_fields(self, model: str) -> dict:
        """Extra top-level request fields (a service tier, say). None by default."""
        return {}

    def _served_tier(self, resp: dict) -> str | None:
        """The service tier the server reports having used, for billing. None
        means Standard — the plain client never asks for anything else."""
        return None

    def _generate_content(self, model: str, body: dict, timeout: int
                          ) -> tuple[dict, int]:
        """POST generateContent, following long-decode continuations to the end.

        Returns (one merged response, number of hops). Every hop resends the SAME
        body plus the token — never the partial output appended to `contents`,
        which the API rejects — and that identical prefix is what lets each
        re-read of a 60k-token prompt bill at the 95% cached rate.
        """
        url = self._url(model, "generateContent")
        try:
            first = self._post(url, body, timeout)
        except VertexError as e:
            if not self._long_decode_refused(model, body, e):
                raise
            # One surface or API version may not offer long decode (the docs
            # disagree with each other on non-streaming REST). Fall back to one
            # response's worth rather than failing the chart, and say so once:
            # a whole-song call can now be cut off where it would not have been.
            self._long_decode = False
            body["generationConfig"]["maxOutputTokens"] = GEMINI4_HOP_TOKENS
            print(f"[vertex] {model} refused maxOutputTokens above "
                  f"{GEMINI4_HOP_TOKENS} on this endpoint (no long decode); "
                  f"continuing with one response per call. A long whole-song reply "
                  f"may be cut off — chunked mode avoids that.", file=sys.stderr)
            first = self._post(url, body, timeout)
        hops = [first]
        max_hops = gemini4_max_hops(
            (body.get("generationConfig") or {}).get("maxOutputTokens") or GEMINI4_HOP_TOKENS)
        while (token := _continuation_token(hops[-1])) and len(hops) < max_hops:
            try:
                hops.append(self._post(url, {**body, "continuationToken": token}, timeout))
            except VertexError as e:
                e.partial = _merge_hops(hops)
                raise
        if _continuation_token(hops[-1]):
            print(f"[vertex] {model} was still generating after {max_hops} "
                  f"long-decode hops; keeping what it wrote.", file=sys.stderr)
        finish = ((hops[-1].get("candidates") or [{}])[0]).get("finishReason")
        if finish == "MAX_TOKENS" and is_gemini4(model):
            print(f"[vertex] {model} spent its whole maxOutputTokens="
                  f"{body['generationConfig'].get('maxOutputTokens')} budget (thinking "
                  f"included) before finishing. Raise BEATFORGE_ARGON_MAX_OUTPUT_TOKENS "
                  f"or lower the thinking level for this stage.", file=sys.stderr)
        return _merge_hops(hops), len(hops)

    def _long_decode_refused(self, model: str, body: dict, err: Exception) -> bool:
        """Was a 400 about the output budget we asked for, and nothing else?"""
        cap = (body.get("generationConfig") or {}).get("maxOutputTokens") or 0
        if not is_gemini4(model) or cap <= GEMINI4_HOP_TOKENS:
            return False
        msg = str(err)
        low = msg.lower()
        return "HTTP 400" in msg and any(
            k in low for k in ("maxoutputtokens", "max_output_tokens",
                               "continuation", "long decode"))

    def generate_json(
        self,
        prompt: str,
        *,
        audio_path: str | Path | None = None,
        system: str | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
        timeout: int = 600,
    ) -> dict:
        """generate() + strict JSON parse with defensive fence-strip and one
        retry that feeds the parse error back (spec §3 structured output)."""
        text = self.generate(prompt, audio_path=audio_path, system=system,
                             model=model, thinking_level=thinking_level,
                             json_out=True, timeout=timeout)
        try:
            return _parse_json(text)
        except ValueError as e:
            retry_prompt = f"{prompt}\n\n---\n{json_retry_note(text, e)}"
            text2 = self.generate(retry_prompt, audio_path=audio_path, system=system,
                                  model=model, thinking_level=thinking_level,
                                  json_out=True, timeout=timeout)
            return _parse_json(text2)  # raises on second failure -> stage fails

    # ---- public: Lyria music (Workstream A) ----------------------------- #
    def lyria(self, prompt: str, *, seed: int | None = None,
              negative: str | None = None, model: str | None = None) -> bytes:
        model = model or config.LYRIA_MODEL
        inst: dict[str, Any] = {"prompt": prompt}
        if negative:
            inst["negative_prompt"] = negative
        if seed is not None:
            inst["seed"] = seed
        t0 = _time.monotonic()
        resp = self._post(self._url(model, "predict"),
                          {"instances": [inst], "parameters": {}})
        # Lyria bills per generated clip, not per token, so the token counts are
        # legitimately zero here; the clip charge is added by the cost report from
        # pricing.CLIP_PRICES keyed on this entry's model + stage.
        ledger.record_model_call(
            provider=self.provider, model=model, usage=ledger.usage_from_vertex(resp),
            latency_s=_time.monotonic() - t0, prompt_bytes=len(prompt.encode()),
            audio_attached=False)
        preds = resp.get("predictions", [])
        if not preds or "bytesBase64Encoded" not in preds[0]:
            raise VertexError(f"no audio returned: {json.dumps(resp)[:400]}")
        return base64.b64decode(preds[0]["bytesBase64Encoded"])


def looks_truncated(text: str) -> bool:
    """Did this reply stop before the JSON object was finished?

    Unbalanced brackets outside of strings is the tell, and it is worth
    distinguishing from ordinary malformed JSON because the two need opposite
    advice: a model that emitted a syntax error should be told to fix its
    syntax, and a model that stopped early should be told to emit LESS.
    """
    depth, in_str, esc = 0, False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return in_str or depth > 0


def salvage_truncated_json(s: str) -> dict | None:
    """Recover the complete prefix of a JSON object that stops part-way through.

    **Why this exists.** A small local model asked for a long array does not
    usually emit *broken* JSON — it emits *valid* JSON and then stops. Measured
    against a self-hosted Gemma on a chunk prompt: `finish_reason: "stop"` after
    1,664 completion tokens, mid-way through the `notes` array, ending on a bare
    `{"ref": "`. Nothing about the output cap was reached; the model simply
    emitted its end token early, so no amount of `max_tokens` fixes it and a
    plain re-prompt tends to stop in the same place.

    What it had already written, though, is real: those are onset ids taken from
    the inventory it was shown. Throwing them away costs the whole call.

    **The one rule that keeps this safe: it only ever DELETES.** The scan finds
    the last point at which a complete nested value had just closed, cuts there,
    and closes the containers that were still open. It never inserts a value,
    never repairs a delimiter, and never completes a half-written element — a
    note object cut off after its `ref` is dropped entirely rather than
    resurrected without its `kind`. So a salvaged design can be short, and cannot
    be wrong.

    Returns the parsed object, or None if there is no complete element to keep
    (a reply that stopped inside the very first value has nothing to salvage).
    """
    stack: list[str] = []
    in_str = esc = False
    cut: tuple[int, tuple[str, ...]] | None = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None                 # more closers than openers: not ours
            stack.pop()
            # A nested value just ended, so everything up to here is complete.
            # Only nested ones count: closing the LAST container means the object
            # was never truncated, and this function would have nothing to do.
            if stack:
                cut = (i + 1, tuple(stack))
    if cut is None:
        return None
    end, open_at_cut = cut
    closers = "".join("}" if c == "{" else "]" for c in reversed(open_at_cut))
    try:
        obj = json.loads(s[:end] + closers)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_json(text: str) -> dict:
    """Extract a JSON object from model text, stripping ``` fences and any
    leading/trailing prose.

    A reply that was cut off mid-object is salvaged down to its last complete
    element rather than discarded — see `salvage_truncated_json` for why that is
    safe and why it matters on a local model.
    """
    s = text.strip()
    # strip ```json ... ``` fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    # if still surrounded by prose, grab the outermost {...}. A TRUNCATED reply
    # has no closing brace, so falling back to the text from the first `{` is
    # what gives the salvage path something to work with.
    if not s.startswith("{"):
        start, end = s.find("{"), s.rfind("}")
        if start == -1:
            raise ValueError("no JSON object found in response")
        s = s[start:end + 1] if end > start else s[start:]
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        salvaged = salvage_truncated_json(s) if looks_truncated(s) else None
        if salvaged is None:
            raise ValueError(f"JSON decode failed: {e}")
        print(f"[llm] the model stopped mid-reply ({len(s)} chars, unclosed JSON); "
              f"kept the {_element_count(salvaged)} complete element(s) it did "
              f"write and discarded the partial one.", file=sys.stderr)
        salvaged["_truncated"] = True
        return salvaged
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def _element_count(obj: dict) -> str:
    """A short human summary of what survived a salvage, for the warning line."""
    parts = [f"{len(v)} {k}" for k, v in obj.items() if isinstance(v, list)]
    return ", ".join(parts) if parts else f"{len(obj)} field"


def json_retry_note(text: str, err: Exception) -> str:
    """What to tell a model whose reply would not parse.

    The two causes need OPPOSITE instructions, and telling a model the wrong one
    makes the retry fail the same way:

      * Stopped early (unbalanced brackets). The syntax was fine as far as it
        got — the model just emitted its end token in the middle of a long
        array. "Fix your JSON" invites it to write the same too-long reply
        again. What helps is asking for LESS.
      * Genuinely malformed. Here the length was not the problem and asking for
        fewer notes would needlessly thin the chart, so say what broke.
    """
    if looks_truncated(text):
        return (
            "Your previous reply STOPPED PART-WAY THROUGH and the JSON object was "
            "never closed — it ended in the middle of a list. The syntax was fine "
            "up to that point; the reply was simply too long to finish.\n"
            "Answer again, SHORTER: select only the strongest, most musically "
            "important onsets rather than every one you can justify, keep the "
            "reply under roughly 150 notes, and make absolutely sure the final "
            "`}` is written. A complete shorter chart is worth far more than a "
            "longer one that gets cut off.")
    return (f"Your previous reply could not be parsed as JSON: {err}. Respond with "
            f"ONLY the valid JSON object, no prose, no code fences.")
