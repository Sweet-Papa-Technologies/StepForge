"""argon.py — Gemini 4 Argon, Google's large general model, as a charting backend.

`BEATFORGE_LLM_BACKEND=argon`. Argon is the same audio-native Gemini stack the
pipeline was built on, so unlike Inkling it needs no new wire protocol: the
request goes through `VertexClient` (or `GoogleAIStudioClient`) unchanged, and
the transport already knows the Gemini 4 request shape (vertex.py). What lives
HERE is the POLICY that makes a model 2.7x the price of 3.8 Flash affordable to
chart with, plus the one thing the transport cannot know — which service tier a
call should ride:

  1. THINKING PER STAGE. Thinking bills at the $10/1M output rate and is most of
     what a designer call emits. The designer gets `high`; retries, which arrive
     carrying a machine-readable violation report, get `medium`; the critic, which
     emits a ~200-token score, gets `low`. Chosen from the ledger stage already on
     the context, so no call site passes anything — the same mechanism as
     inkling.InklingClient._call_effort and llm.thinking_for.
  2. WHOLE SONG BY DEFAULT. Not implemented here but because of here:
     config.auto_chunk_threshold() makes `auto` chunk mode send a song up to 15
     minutes in one call on this backend. Argon's only cache is implicit and
     prefix-based, so the saving depends on the prompt PREFIX repeating — and in
     whole-song mode the audio + brief + onset inventory are byte-identical across
     all five difficulties, every retry, and (the audio) every critic call.
  3. FLEX, OPT-IN. 50% off for variable latency. Gemini API only: Vertex Flex
     PayGo "is not supported in newer Gemini models such as Gemini 4 Argon". A
     capacity refusal on Flex falls back to Standard for that one call rather
     than losing the chart, and a call is billed as Flex only when the response
     says it was served as Flex.

There is deliberately NO audio-probe gate before a run, unlike Inkling. That gate
exists because a third-party endpoint can silently drop an audio part; Gemini's
audio input is first-party, and an auth or model-access failure surfaces on the
first call anyway. `beatforge argon-check` runs the same tone probe on demand.
"""
from __future__ import annotations

import sys

from . import config, ledger
from .llm import GoogleAIStudioClient
from .vertex import VertexClient, VertexError, is_gemini4

_TIERS = ("standard", "flex")


class ArgonError(VertexError):
    """Argon-specific configuration failure. A VertexError so the pipeline's
    existing handlers treat it like any other Gemini failure."""


def _warn(msg: str) -> None:
    print(f"[argon] {msg}", file=sys.stderr)


class ArgonPolicy:
    """The Argon policy, layered over either Gemini transport (mixin).

    Overrides `generate` only: `generate_json` and the JSON-repair retry inherit
    from VertexClient and route through it, so they get the same per-stage model,
    thinking level and tier as the first attempt.
    """

    name = "argon"
    #: "vertex" | "aistudio", set by the concrete class.
    transport = "?"

    def _argon_init(self, *, model: str | None = None,
                    service_tier: str | None = None,
                    critic_model: str | None = None,
                    thinking: str | None = None) -> None:
        self.model = (model or config.ARGON_MODEL).strip()
        tier = (service_tier or config.ARGON_SERVICE_TIER or "standard").strip().lower()
        if tier not in _TIERS:
            _warn(f"unknown service tier {tier!r}; using 'standard' "
                  f"(choose one of {', '.join(_TIERS)}).")
            tier = "standard"
        if tier == "flex" and self.transport == "vertex":
            # Refused here rather than sent: Vertex documents Flex PayGo as
            # unsupported on Gemini 4, so the header could only ever be ignored
            # or 400 — and a run that thinks it is paying half is worse than one
            # that knows it is not.
            _warn("Flex is not available for Gemini 4 on Vertex AI (Flex PayGo "
                  "excludes it). Charting at Standard. Use the AI Studio transport "
                  "(BEATFORGE_ARGON_TRANSPORT=aistudio + GEMINI_API_KEY) for Flex.")
            tier = "standard"
        self.service_tier = tier
        self.critic_model = (config.ARGON_CRITIC_MODEL if critic_model is None
                             else critic_model).strip()
        self._forced_thinking = thinking
        # Set once the endpoint has refused the tier field; from then on the
        # process stops asking rather than paying a 400 per call.
        self._flex_refused = False
        # True for the duration of one Standard retry after a Flex capacity miss.
        self._standard_once = False

    # ---- per-stage policy ------------------------------------------------- #
    @staticmethod
    def _stage() -> tuple[str, int | None]:
        ctx = ledger.current() or {}
        attempt = ctx.get("attempt")
        return str(ctx.get("stage") or ""), attempt if isinstance(attempt, int) else None

    def stage_thinking(self) -> str:
        """Thinking level for the call in flight, from the ledger stage.

        A retry is keyed on `attempt` as well as on the reprompt stage names:
        chunked mode re-prompts a chunk as `designer` with attempt=2, and that is
        a retry whatever the stage is called (see inkling._call_effort, where
        missing this left every retry at the designer's level)."""
        if self._forced_thinking:
            return self._forced_thinking
        stage, attempt = self._stage()
        if stage in config.ARGON_RETRY_STAGES or (attempt is not None and attempt > 1):
            return config.ARGON_THINKING_RETRY
        return config.ARGON_THINKING.get(stage, config.ARGON_THINKING_DEFAULT)

    def stage_model(self) -> str:
        """The designer model, or ARGON_CRITIC_MODEL on a judging stage."""
        stage, _ = self._stage()
        if self.critic_model and stage in ("critic", "scorecard"):
            return self.critic_model
        return self.model

    def describe(self) -> dict:
        """What this client will do, for `argon-check` and run logs. Contains no
        credential — the endpoint URL carries none on either transport."""
        return {
            "transport": self.transport,
            "provider": getattr(self, "provider", None),
            "model": self.model,
            "criticModel": self.critic_model or self.model,
            "serviceTier": self.service_tier,
            "thinking": {"designer": config.ARGON_THINKING.get("designer"),
                         "retry": config.ARGON_THINKING_RETRY,
                         "critic": config.ARGON_THINKING.get("critic")},
            "maxOutputTokens": config.ARGON_MAX_OUTPUT_TOKENS,
            "wholeSongMaxSeconds": config.ARGON_WHOLE_SONG_MAX_SECONDS,
            "endpoint": self._url(self.model, "generateContent"),  # type: ignore[attr-defined]
        }

    # ---- service tier ----------------------------------------------------- #
    def _flex_wanted(self) -> bool:
        return (self.service_tier == "flex" and self.transport == "aistudio"
                and not self._flex_refused and not self._standard_once)

    def _request_fields(self, model: str) -> dict:
        return {"serviceTier": "flex"} if self._flex_wanted() else {}

    def _served_tier(self, resp: dict) -> str | None:
        """The tier the RESPONSE says served the call — never just the one asked
        for. Vertex reports `usageMetadata.trafficType` (ON_DEMAND_FLEX for Flex);
        the Gemini API's report is not documented for generateContent, so either
        spelling is accepted and anything else bills at Standard. That can only
        over-state the bill, which is the safe direction."""
        um = (resp or {}).get("usageMetadata") or {}
        if "FLEX" in str(um.get("trafficType") or "").upper():
            return "flex"
        for v in ((resp or {}).get("serviceTier"), um.get("serviceTier")):
            if v and "flex" in str(v).lower():
                return "flex"
        return None

    @staticmethod
    def _tier_field_rejected(err: Exception) -> bool:
        low = str(err).lower()
        return "http 400" in low and ("servicetier" in low or "service_tier" in low
                                      or "service tier" in low)

    @staticmethod
    def _at_capacity(err: Exception) -> bool:
        msg = str(err)
        return "HTTP 429" in msg or "HTTP 503" in msg

    # ---- LLMClient surface ------------------------------------------------ #
    def generate(self, prompt: str, *, audio_path=None, system=None, model=None,
                 thinking_level=None, json_out=False, timeout=600) -> str:
        """An explicit `model=` / `thinking_level=` from the caller wins — compare.py
        pins both on purpose and a policy must not second-guess a benchmark."""
        kw = dict(audio_path=audio_path, system=system,
                  model=model or self.stage_model(),
                  thinking_level=thinking_level or self.stage_thinking(),
                  json_out=json_out,
                  # A high-thinking whole-song hop can be silent for minutes, and
                  # Flex queues on top of that; the call sites' 600 s default was
                  # sized for Flash.
                  timeout=max(int(timeout or 0), config.ARGON_TIMEOUT))
        flex = self._flex_wanted()
        try:
            return super().generate(prompt, **kw)            # type: ignore[misc]
        except VertexError as e:
            if flex and self._tier_field_rejected(e):
                self._flex_refused = True
                _warn("the Gemini API rejected the Flex service-tier field on "
                      "generateContent; continuing at Standard price for the rest "
                      "of this run.")
                return super().generate(prompt, **kw)        # type: ignore[misc]
            if flex and self._at_capacity(e) and config.ARGON_FLEX_FALLBACK:
                # Flex is best-effort and is never upgraded server-side. One
                # Standard call costs double that call; losing it costs the chart.
                _warn("Flex had no capacity after the transport's backoff; retrying "
                      "this one call at Standard (BEATFORGE_ARGON_FLEX_FALLBACK=0 to "
                      "fail instead).")
                self._standard_once = True
                try:
                    return super().generate(prompt, **kw)    # type: ignore[misc]
                finally:
                    self._standard_once = False
            raise


class ArgonVertexClient(ArgonPolicy, VertexClient):
    """Argon on Vertex AI, global endpoint. Needs VERTEX_PROJECT + ADC/VERTEX_KEY."""

    transport = "vertex"

    def __init__(self, **kw):
        VertexClient.__init__(self)
        self._argon_init(**kw)


class ArgonAIStudioClient(ArgonPolicy, GoogleAIStudioClient):
    """Argon on the Gemini API (AI Studio). Needs GEMINI_API_KEY. The only path
    with a Flex tier for Gemini 4."""

    transport = "aistudio"

    def __init__(self, **kw):
        GoogleAIStudioClient.__init__(self)
        self._argon_init(**kw)


def resolve_transport(service_tier: str | None = None) -> str:
    """Which Gemini path Argon rides. `auto` prefers Vertex — the project's
    default Gemini path — except that asking for Flex picks AI Studio when a key
    is available, because only AI Studio offers Flex on Gemini 4."""
    t = (config.ARGON_TRANSPORT or "auto").strip().lower()
    if t in ("vertex", "aistudio"):
        return t
    if t not in ("auto", ""):
        _warn(f"unknown BEATFORGE_ARGON_TRANSPORT {t!r}; choosing automatically.")
    tier = (service_tier or config.ARGON_SERVICE_TIER or "standard").strip().lower()
    has_vertex, has_key = bool(config.VERTEX_PROJECT), bool(config.GEMINI_API_KEY)
    if tier == "flex" and has_key:
        return "aistudio"
    if has_vertex:
        return "vertex"
    if has_key:
        return "aistudio"
    raise ArgonError(
        "Gemini 4 Argon needs Google credentials and none are configured. Either "
        "set VERTEX_PROJECT (and run `gcloud auth application-default login` or "
        "set VERTEX_KEY) for Vertex AI, or set GEMINI_API_KEY "
        "(https://aistudio.google.com/apikey) for Google AI Studio — the only path "
        "with a Flex tier. Both are fields under Settings.")


def make_client(*, transport: str | None = None, **kw) -> ArgonPolicy:
    """Build the Argon client the factory returns (llm.make_llm_client)."""
    transport = transport or resolve_transport(kw.get("service_tier"))
    cls = ArgonVertexClient if transport == "vertex" else ArgonAIStudioClient
    client = cls(**kw)
    if not is_gemini4(client.model):
        _warn(f"BEATFORGE_ARGON_MODEL={client.model!r} is not a Gemini 4 model; it "
              f"will be sent with that model's own request shape.")
    return client


__all__ = ["ArgonPolicy", "ArgonVertexClient", "ArgonAIStudioClient", "ArgonError",
           "make_client", "resolve_transport"]
