"""config.py — ALL beatforge tunables in one place (spec §2, §3, §6).

Models, budgets, thresholds, paths, difficulty table. Nothing tunable should
live outside this file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# repo root = three levels up from this file: tools/beatforge/config.py -> repo
REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKS_SRC = REPO_ROOT / "assets" / "tracks"
TRACKS_PUB = REPO_ROOT / "public" / "assets" / "tracks"
MAPS_PUB = REPO_ROOT / "public" / "maps"
BUILD_DIR = REPO_ROOT / "build" / "analysis"
STEPMANIA_DIR = REPO_ROOT / "build" / "stepmania"
SABERFORGE_DIR = REPO_ROOT / "build" / "saberforge"     # Beat Saber song folders + drafts
# Cost telemetry (REQ-R2-COST-01): one `<song>/cost_ledger.jsonl` per track.
COST_DIR = REPO_ROOT / "build" / "cost"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"
JOBS_DIR = Path(__file__).resolve().parent / "jobs"

# --------------------------------------------------------------------------- #
# Models & API (spec §3 — verified against the Vertex model list in us-central1
# on 2026-07-05).
# --------------------------------------------------------------------------- #
#
# Gemini 3.5 Flash — the audio-analysis brain of this whole pipeline. VERIFIED
# LIVE against the Vertex project on 2026-07-05 (spec §3 mandates verify-before-
# hardcode; do NOT trust the assetforge cheat-sheet, whose static text predates
# the 3.5 release and omits this id):
#   * `gemini-3.5-flash` :generateContent -> 200 OK, project HAS access.
#   * Reachable ONLY from location=`global` on the `v1beta1` endpoint. This is
#     why the assetforge `models` list (which probes regional us-central1) misses
#     it. vertex.py routes any `gemini-3*` id to global/v1beta1 automatically.
#   * Accepts AUDIO input (inlineData audio part -> usage modality=AUDIO). It
#     actually hears the track: on overdrive_pulse it returned ~115 BPM + "clear
#     kick" from audio alone.
#   * Supports reasoning via `generationConfig.thinkingConfig.thinkingLevel`
#     (minimal|low|medium|high); high produced thoughtsTokenCount>0.
# DEFAULT: gemini-3.8-flash, the successor to gemini-3.6-flash. 3.6 was chosen on
# measured evidence, not vibes — a same-song, same-pipeline bench
# (docs/model-comparison.md, Lucky Lucky, 5 charts each) put it ahead of 3.5
# Flash on every axis that matters AND cheaper:
#   cost/chart  $0.2534 vs $0.4095   (-38%)
#   flow_cost_max     9.35 vs 14.32  (best foot flow of the three)
#   density rho      0.796 vs 0.781
#   wall clock   6.4 min vs 12.6 min
# 3.8 has NOT been through that bench yet — it is the default on the strength of
# being the same family at half 3.6's launch price (pricing.py), and those numbers
# are 3.6's. Re-run docs/model-comparison.md before quoting them for 3.8.
# gemini-3.5-flash-lite is cheaper again (-86%) and scores fine on rho, but its
# jump_share came out 0.267 against 3.6's 0.046 — nearly 6x the two-foot
# commitments, which is the exact pad-comfort problem R3 existed to fix. Cheap is
# not free if the chart stops being danceable.
GEMINI_MODEL = os.environ.get("BEATFORGE_GEMINI_MODEL", "gemini-3.8-flash")

# thinking_level (spec §3): Gemini 3.x uses `thinkingLevel`; the pipeline requests
# "high" (full thinking power, user directive). vertex.py emits the correct
# request shape for the model family GEMINI_MODEL belongs to.
THINKING_LEVEL = os.environ.get("BEATFORGE_THINKING_LEVEL", "high")
# Whether that level was named EXPLICITLY. A named level is a global pin — it
# beats the per-role map below, which is how the "uniform high" arm of the
# Workstream C A/B is requested without also having to disable the map.
THINKING_LEVEL_PINNED = "BEATFORGE_THINKING_LEVEL" in os.environ

# --------------------------------------------------------------------------- #
# Model routing. DEFAULT IS "single": GEMINI_MODEL (gemini-3.8-flash) serves every
# call — designer, retries and critic alike. "hybrid" is an OPT-IN alternative.
#
# Hybrid was built to be the default and demoted on its own evidence. The one
# full-length measurement we have (docs/song-quality-review.md §4) is that it cost
# ~17% MORE than single-model on a 178s track — 0 of 3 cheap drafts cleared the
# gates — and produced no better chart, because the quality floor is enforced by
# referees that are identical in both modes. A mode that is more expensive and no
# better does not get to be the default, however good the idea is.
#
# It stays available, and the rest of this comment is why it is still worth having:
# the mechanism is sound and the measurement points at a fixable cause (the section
# map, review finding B), so it is likely to pay off later on shorter or
# better-segmented material. Turn it on with BEATFORGE_LLM_MODE=hybrid and check
# `beatforge cost-report --stdout`, which breaks the bill down by tier.
#
# --- why the cascade is shaped the way it is ---
#
# The money is not spread evenly across this pipeline. The instrumented run in
# docs/cost-autopsy.md says where it actually goes:
#
#     designer            5 calls   $1.8478   43.2%
#     re-prompt (gates)   4 calls   $1.4698   34.4%
#     re-prompt (density) 1 call    $0.6141   14.4%
#     critic              8 calls   $0.3453    8.1%
#
# So "put the cheap model on the critic" — the obvious hybrid — buys 8%. The
# lever that matters is the FIRST DRAFT plus its retries, and that is exactly the
# work docs/model-comparison.md says Flash Lite is bad at: same-song, same-
# pipeline, Lite scored density rho 0.766 (vs 3.6 Flash's 0.796 — fine) and the
# same 80% flow-gate pass rate, but jump_share 0.267 against 0.046. Nearly 6x the
# two-foot commitments. Cheap, and not danceable.
#
# The read: Lite is competent at WHICH onsets to hit and HOW MANY, and reckless
# about TEXTURE. Every one of those is a property something deterministic already
# measures. So hybrid mode is a CASCADE refereed by those checks, not a guess:
#
#   1. `designer` — the first draft — runs on Flash Lite (~5x cheaper input,
#      ~3x cheaper output).
#   2. The draft faces the full deterministic battery: onset alignment, panel
#      balance, hold share, foot-flow ceiling, density-vs-energy Spearman
#      (adapters/stepmania/qa.py::_gates). A draft that clears all of them is a
#      chart Flash would also have had to produce. Note the jump problem above
#      needs no gate of its own: validate.py::cap_jump_share already DEMOTES
#      surplus jumps against realize.py's authored-DDR ceiling before any metric
#      is taken, so 0.267 cannot reach a simfile from any model (pinned by
#      tests/test_cascade.py).
#   3. Anything that fails escalates to Flash — as a `reprompt`, carrying the
#      machine-readable violation report. Retries were already the most expensive
#      stage; they are now also the ONLY place the expensive model is asked to
#      author from a blank page.
#   4. The `critic` stays on Flash. It is 8% of spend and it is the only judge of
#      the musicality no gate can measure — Author != Judge, and the judge must
#      not be weaker than the author.
#
# WHAT THIS SAVES IS TRACK-DEPENDENT, AND CAN BE NEGATIVE. A draft that clears the
# gates saves ~78% on that chart; a draft that fails costs a wasted lite call on top
# of the heavy one (~+17-22%). The sign is set by the lite pass RATE, which is not a
# constant, and the two measurements we have disagree:
#   * docs/model-comparison.md (lucky_lucky, 33s): Lite matched 3.6 Flash on the
#     flow gate (80%) and the density gate (100%) — there, the cascade would win.
#   * docs/song-quality-review.md (a 178s track, 12 heuristic sections): Lite failed
#     on all three drafts (`density_energy` on two) and the run came out MORE
#     expensive. The cause looks fixable and is not the model — the density target
#     it missed is derived from a section map that is wrong on long tracks
#     (review finding B) — but until that is fixed this is the honest expectation.
# This asymmetry is exactly why the default is "single".
#
# What does NOT vary is the quality floor. Nothing ships that the deterministic
# gates and the heavy critic did not both clear, and an escalation is kept only if
# it fails strictly fewer gates — so a wasted escalation costs money, never quality.
# Hybrid is a cost bet, never a quality one, in either direction.
#
#   BEATFORGE_LLM_MODE = "single" (default) | "hybrid"
# --------------------------------------------------------------------------- #
LLM_MODE = os.environ.get("BEATFORGE_LLM_MODE", "single")

# The two tiers. `heavy` must be an audio-capable reasoning model; `lite` must be
# audio-capable too (it authors from the track) — both verified in pricing.py.
#
# `heavy` falls back to GEMINI_MODEL rather than to a second hardcoded default, so
# `BEATFORGE_GEMINI_MODEL=…` keeps choosing the model that writes and judges the
# chart in BOTH modes. Switching to hybrid then changes only what drafts, which is
# the one variable the mode is supposed to introduce.
HYBRID_HEAVY_MODEL = os.environ.get("BEATFORGE_HYBRID_HEAVY_MODEL", GEMINI_MODEL)
HYBRID_LITE_MODEL = os.environ.get("BEATFORGE_HYBRID_LITE_MODEL", "gemini-3.5-flash-lite")
# Thinking level per tier. Both default to THINKING_LEVEL ("high") ON PURPOSE:
# the model-comparison numbers above were measured with Lite thinking at high, so
# that is the configuration whose quality we can actually vouch for. Turning the
# lite tier's reasoning down is a second variable and belongs in a second
# benchmark — the knob is here, the default does not pull it.
HYBRID_HEAVY_THINKING = os.environ.get("BEATFORGE_HYBRID_HEAVY_THINKING", THINKING_LEVEL)
HYBRID_LITE_THINKING = os.environ.get("BEATFORGE_HYBRID_LITE_THINKING", THINKING_LEVEL)

# role -> tier. The role is the ledger STAGE the call is made under, so routing
# and cost attribution can never disagree about what a call was for.
ROLE_TIERS = {
    "designer": "lite",             # first draft; escalates on any gate failure
    "reprompt": "heavy",            # gate/intent repair — already the pricey path
    "reprompt_density": "heavy",    # phrase-scoped density repair
    "critic": "heavy",              # Author != Judge; the judge is never cheaper
    "scorecard": "heavy",           # Workstream A A&R ear (gen.py)
}
# An unrecognized role bills to the expensive model. A new call site that nobody
# routed should cost too much, not silently design charts on the cheap model.
HYBRID_DEFAULT_TIER = "heavy"

# --------------------------------------------------------------------------- #
# Per-stage thinking budget (cost-reduction plan, Workstream C).
#
# MEASURED: thinking is 74.3% of all output-rate tokens, and output-rate tokens
# bill at 5x the input rate — the single largest line item in the bill. Today one
# global THINKING_LEVEL is applied to every call by vertex.py::_thinking_config:
# the designer authoring from a blank page and the critic emitting a 195-token
# score both think at "high". The critic spends ~3,373 thinking tokens per call
# to emit that score — a 17:1 think-to-output ratio on a task that is handed its
# rho value pre-computed and is not deriving anything.
#
# DEFAULT OFF, and that is deliberate. The reduction ratios this is projected on
# are ASSUMPTIONS about model behaviour, not measurements, and a weaker critic is
# a quality regression no deterministic gate can catch (the gates and the critic
# measure deliberately different things — Author != Judge, REQ-SM-13). The
# mechanism ships so the A/B can be run; the A/B decides whether the default
# flips. Ship per-role, measuring after each, critic first.
#
#   BEATFORGE_ROLE_THINKING = 0 (default, off) | 1
#   BEATFORGE_THINKING_LEVEL = <level>   still pins every call, for the "uniform
#                                        high" arm of the A/B
# --------------------------------------------------------------------------- #
ROLE_THINKING = {
    "designer": "high",             # authoring from a blank page; earns it
    "reprompt": "medium",           # carries a machine-readable violation report
    "reprompt_density": "medium",   # phrase-scoped repair, narrower still
    "critic": "low",                # scoring task, 17:1 think-to-output
    "scorecard": "medium",          # A&R rubric scoring (gen.py)
}
# An unrouted role keeps today's behaviour. Mirrors HYBRID_DEFAULT_TIER: a new
# call site that nobody routed should cost too much, not silently think too little.
DEFAULT_THINKING = THINKING_LEVEL
# The clean off switch the A/B needs. Off => every call resolves to
# DEFAULT_THINKING exactly as before this existed.
ROLE_THINKING_ENABLED = os.environ.get("BEATFORGE_ROLE_THINKING", "0") not in (
    "0", "", "false")

# Roles whose lite draft escalates to `heavy` on retry. Without this a chart that
# failed the gates would be re-drafted by the same model that just failed them.
HYBRID_CASCADE_ROLES = ("designer",)
# Set BEATFORGE_HYBRID_CASCADE=0 to pin the cascade roles to `heavy` — i.e. keep
# hybrid routing for everything else but never let Lite author a first draft.
HYBRID_CASCADE = os.environ.get("BEATFORGE_HYBRID_CASCADE", "1") not in ("0", "", "false")

# --------------------------------------------------------------------------- #
# LLM backend selection — the designer/critic/A&R model is swappable so an
# alternative audio-capable model (e.g. a self-hosted Gemma 4 12B on an
# OpenAI-compatible server) can be dropped in and benchmarked against Gemini 3.5
# Flash. Lyria (music generation) always stays on Vertex.
#   BEATFORGE_LLM_BACKEND =
#       "gemini"      (Gemini via Vertex AI; default) — needs a GCP project + creds
#     | "google-genai"(Gemini via Google AI Studio) — needs only a GEMINI_API_KEY,
#                     no GCP project. The zero-setup path for a key-only user.
#     | "openai"      (any OpenAI-compatible /v1 server) — no Google account at all
#     | "inkling"     (Inkling on a Modal Managed Endpoint) — see inkling.py
#     | "argon"       (Gemini 4 Argon, over Vertex or AI Studio) — see argon.py
# --------------------------------------------------------------------------- #
LLM_BACKEND = os.environ.get("BEATFORGE_LLM_BACKEND", "gemini")
# OpenAI-compatible server (vLLM/llama.cpp/etc). Defaults to a server on this
# machine; point BEATFORGE_OPENAI_BASE_URL at a remote host if it lives elsewhere:
#   export BEATFORGE_LLM_BACKEND=openai
#   export BEATFORGE_OPENAI_BASE_URL=http://<host>:8000/v1
#   export BEATFORGE_OPENAI_MODEL=<model id from GET /v1/models>
OPENAI_BASE_URL = os.environ.get("BEATFORGE_OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_MODEL = os.environ.get("BEATFORGE_OPENAI_MODEL", "gemma-4-12b")
# The OpenAI backend routes through ChartForgeClient by DEFAULT: the local
# Gemma 4 fine-tune needs its training-time prompt wrapper (system instruction +
# audio preamble + window label), and a bare designer prompt makes it degenerate.
# Set BEATFORGE_OPENAI_CHARTFORGE=0 only if you point the openai backend at a
# different (non-CHARTFORGE) model. Serve with llama.cpp and `--reasoning-budget 0`.
OPENAI_CHARTFORGE = os.environ.get(
    "BEATFORGE_OPENAI_CHARTFORGE", "1").strip().lower() in ("1", "true", "yes", "on")
OPENAI_API_KEY = os.environ.get("BEATFORGE_OPENAI_API_KEY", "not-needed")


def _parse_headers(raw: str) -> dict:
    """Parse BEATFORGE_OPENAI_HEADERS into a {name: value} dict.

    Two accepted forms, so this is usable both from a shell and from the GUI
    (which writes JSON):
      * JSON object  — {"X-Api-Key": "abc", "anthropic-version": "2023-06-01"}
      * plain lines  — `Name: value` per line (also accepts `;` separators)
    Malformed input is ignored rather than fatal: a typo in an optional header
    must not take down a run that would otherwise work.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
        except ValueError:
            return {}
        return {str(k).strip(): str(v) for k, v in obj.items() if str(k).strip()}
    out = {}
    for line in raw.replace(";", "\n").splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip():
            out[name.strip()] = value.strip()
    return out


# Arbitrary extra HTTP headers sent with every OpenAI-compatible request. This is
# the escape hatch for gateways that need more than a bearer token — an API-key
# header, a tenant/project id, a proxy auth token, a required version pin:
#   export BEATFORGE_OPENAI_HEADERS='{"X-Api-Key":"abc","X-Tenant":"acme"}'
# A header set here WINS over the ones the client sends by default, so
# `{"Authorization": "Basic ..."}` replaces the bearer token entirely, and an
# empty value ("") drops that default header from the request.
OPENAI_HEADERS = _parse_headers(os.environ.get("BEATFORGE_OPENAI_HEADERS", ""))
# Attach the track itself to designer/critic calls as an `input_audio` part.
# ON, and it should stay on: the model listening to the music is the pipeline.
# The DSP locates the beats; the model decides which of them are worth playing,
# which is a musical judgement it can only make by ear. An `openai` backend is
# expected to serve an AUDIO-CAPABLE model for that reason.
#
# Setting this to 0 is an escape hatch for smoke-testing a text-only endpoint
# (most OpenAI-compatible servers), which otherwise 400s on the content parts:
#   export BEATFORGE_OPENAI_SEND_AUDIO=0
# It still emits a chart — tempo/offset/onsets/sections are measured locally by
# analyze.py and passed as numbers — but designed by a model that never heard the
# track, so the charts are materially worse. llm.py warns on every run that does
# it, and the cost ledger records audio_attached=false so such runs stay
# identifiable after the fact.
OPENAI_SEND_AUDIO = os.environ.get("BEATFORGE_OPENAI_SEND_AUDIO", "1") not in ("0", "", "false")
# Audio is sent as an OpenAI `input_audio` content part; base64 in this format.
# The tracks are .ogg — transcoded to this on the fly (wav is universally decoded;
# set to "ogg" to send as-is if your server accepts it).
OPENAI_AUDIO_FORMAT = os.environ.get("BEATFORGE_OPENAI_AUDIO_FORMAT", "wav")
# HOW the audio rides in the message. The ecosystem is genuinely split and the
# two shapes are not interchangeable — an endpoint that takes one 400s on the
# other, with a pydantic union dump that never names the real problem:
#
#   "input_audio"  {"type":"input_audio","input_audio":{"data":<b64>,"format":"wav"}}
#                  OpenAI's documented audio part. api.openai.com and vLLM.
#   "audio_url"    {"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,…"}}
#                  SGLang's shape, and what a Modal Auto Endpoint serving Inkling
#                  requires — verified against a live endpoint 2026-07-25, where
#                  `input_audio` is rejected outright. The data: URI keeps the
#                  bytes in the request; nothing has to be reachable by the server.
#
# "auto" sends the client's preferred shape and, if the endpoint rejects the
# message, retries ONCE with the other and remembers which one worked for the
# rest of the process. That is the difference between a backend that works
# everywhere and one the user has to diagnose a validation dump for.
OPENAI_AUDIO_TRANSPORT = os.environ.get(
    "BEATFORGE_OPENAI_AUDIO_TRANSPORT", "auto").strip().lower()
# Downsample the audio sent to a local model to keep the audio-token count (and
# thus KV-cache / VRAM) small. Full 44.1kHz stereo of a 2.5min song is ~25MB and
# balloons the context; 16kHz mono (~5MB) is plenty for tempo/onset perception
# and cut the RAM spill that makes a VRAM-starved 12B model crawl. 0 = leave as-is.
OPENAI_AUDIO_SR = int(os.environ.get("BEATFORGE_OPENAI_AUDIO_SR", "16000"))
OPENAI_AUDIO_MONO = os.environ.get("BEATFORGE_OPENAI_AUDIO_MONO", "1") not in ("0", "", "false")
# A full chart JSON is ~2.5-4k tokens, so 4096 is enough for the chart and
# nothing else. It is a deliberately tight cap: a runaway local model otherwise
# generates until it fills VRAM.
#
# Know what that trades away, because this default was 16384 for exactly these
# reasons and was lowered back on purpose:
#   * Overrunning the cap loses the ENTIRE call, and you are billed for every
#     token generated getting there. A slow local model is merely slow, and
#     BEATFORGE_OPENAI_TIMEOUT already bounds it.
#   * Any reasoning endpoint pointed at this backend counts its thinking against
#     this same cap, so 4096 is spent before the answer begins. Raise it (16384
#     is the tested value) if you point this backend at a reasoning model.
#
# The Inkling backend has its own, far larger cap — see INKLING_MAX_TOKENS.
OPENAI_MAX_TOKENS = int(os.environ.get("BEATFORGE_OPENAI_MAX_TOKENS", "20384"))
# Which request field carries the output cap. "auto" picks per endpoint, and it
# has to be a choice rather than an upgrade because the two are mutually
# exclusive in opposite directions:
#   * api.openai.com REJECTS `max_tokens` on its reasoning models ("Unsupported
#     parameter: 'max_tokens' is not supported with this model. Use
#     'max_completion_tokens' instead").
#   * Plenty of self-hosted OpenAI-compatible servers have never heard of
#     `max_completion_tokens` and will silently generate to their own default.
# Sending both is not a workaround — OpenAI 400s on the pair.
OPENAI_MAX_TOKENS_FIELD = os.environ.get(
    "BEATFORGE_OPENAI_MAX_TOKENS_FIELD", "auto").strip().lower()
# Refuse a request body over this many MB before uploading it. A base64 audio
# part is ~1.33x the encoded audio, and a gateway that rejects an oversized body
# does it by closing the socket mid-upload — which surfaces as a bare broken
# pipe minutes later instead of a size error. 0 disables the check.
OPENAI_MAX_REQUEST_MB = float(os.environ.get("BEATFORGE_OPENAI_MAX_REQUEST_MB", "48"))
# Per-request read timeout (s). Local 12B models are much slower than Gemini on
# the heavy designer call; raise for slow hardware, lower to fail fast.
OPENAI_TIMEOUT = int(os.environ.get("BEATFORGE_OPENAI_TIMEOUT", "300"))
# A little sampling temperature helps local models avoid the degenerate
# repetition loops that greedy (temp=0) decoding can fall into on long JSON.
OPENAI_TEMPERATURE = float(os.environ.get("BEATFORGE_OPENAI_TEMPERATURE", "0.3"))

# --------------------------------------------------------------------------- #
# Inkling on a Modal Managed Endpoint (BEATFORGE_LLM_BACKEND=inkling)
#
# Thinking Machines' Inkling is natively multimodal (text+image+audio in) with a
# 1M context and a controllable reasoning dial, billed per token on Modal rather
# than per GPU-hour. Two things make it unlike the plain `openai` backend:
#
#   1. Auth is TWO custom headers (Modal-Key / Modal-Secret), not a Bearer token.
#      Modal documents exactly this pair, and it is deliberate: the proxy takes
#      its own headers so `Authorization` stays free for whatever the SERVED app
#      wants (a vLLM `--api-key`, say). See inkling.py.
#   2. Audio input runs to ~20 minutes, not ~30 seconds, so a full track goes in
#      ONE call. No chunking, no stitching.
#
# The model string is resolved from GET /v1/models at runtime — vLLM-backed
# endpoints sometimes serve under an alias, and a hardcoded id 404s.
# --------------------------------------------------------------------------- #
INKLING_ENDPOINT_URL = os.environ.get("BEATFORGE_INKLING_ENDPOINT_URL", "")
# "" => resolve from the endpoint. Set only to pin a specific served model.
INKLING_MODEL = os.environ.get("BEATFORGE_INKLING_MODEL", "")
# Proxy tokens from modal.com/settings/proxy-auth-tokens. Env wins over the
# keychain (see inkling.resolve_credentials); neither is ever logged or ledgered.
MODAL_TOKEN_ID = os.environ.get("MODAL_TOKEN_ID", "")
MODAL_TOKEN_SECRET = os.environ.get("MODAL_TOKEN_SECRET", "")
# An API key for the SERVED application, sent as `Authorization: Bearer ...`
# alongside the Modal proxy headers. Empty by default and empty for a Managed
# Endpoint — this exists for a self-deployed vLLM behind proxy auth, which
# authenticates twice: once at Modal's gateway, once at the model server.
# NEVER put the Modal proxy token here; it is not a bearer credential.
INKLING_API_KEY = os.environ.get("BEATFORGE_INKLING_API_KEY", "")
# Inkling's documented audio input: 16 kHz mono WAV, ideally under 20 minutes.
# Enforced rather than assumed — an oversized body should fail here with a clear
# message, not as a 413 from the gateway after uploading it.
INKLING_AUDIO_SR = 16000
# WAV is what Inkling documents and what its serving stack decodes without
# question. The knob exists because 20 min of 16 kHz mono PCM is ~38MB before
# base64 (~51MB on the wire) — "flac" is lossless at roughly half that and is
# decoded by the same librosa/soundfile path, so it is the first thing to try
# against a gateway body limit.
INKLING_AUDIO_FORMAT = os.environ.get("BEATFORGE_INKLING_AUDIO_FORMAT", "wav")
# Verified against a live Modal Auto Endpoint (SGLang-backed) on 2026-07-25: this
# server takes `audio_url` with a data: URI and REJECTS OpenAI's `input_audio`
# part with a 25-error pydantic union dump that never mentions audio. The
# preference is stated rather than discovered so the first call is the right one;
# "auto" still lets it fall back if a differently-served endpoint disagrees.
INKLING_AUDIO_TRANSPORT = os.environ.get(
    "BEATFORGE_INKLING_AUDIO_TRANSPORT", "audio_url").strip().lower()
INKLING_MAX_AUDIO_SECONDS = int(os.environ.get("BEATFORGE_INKLING_MAX_AUDIO_SECONDS", "1200"))
# Output cap. Deliberately NOT OPENAI_MAX_TOKENS: reasoning tokens are generated
# tokens and count against this, so at effort=high the 4096 tuned for a local
# text-only 12B is spent thinking and the call returns empty content — which
# reads like "the endpoint is broken" and is really "the budget ran out".
#
# DELIBERATELY VERY LIBERAL. This number is not a cost control and must not be
# tuned like one — an unused budget is never billed, while an exhausted one
# throws away a call you were billed for in full. The two outcomes are not
# symmetric, so there is no reason to sail close to the wind.
#
# Every problem this setting has caused came from being too small. Measured on
# one `hard` designer call against a live endpoint: low spent 1.9k output tokens,
# medium 12.1k, high 25.3k — and a second `high` run on the same stage blew past
# 32.7k and returned nothing. At 4096 (the local-model default it used to
# inherit) even `low` fails. Effort is the cost lever; this is just headroom, and
# it needs to cover the tail of a reasoning distribution that has a long one.
#
# 128k against a 1M context, on an endpoint that accepted 200k in testing.
INKLING_MAX_TOKENS = int(os.environ.get("BEATFORGE_INKLING_MAX_TOKENS", "131072"))
# LAST RESORT, and it should never fire. With a 128k cap, a reply cut off
# mid-reasoning means the model is looping, not that it needed a little more
# room — so the retry steps the effort down once rather than losing the call
# outright after paying for a full cap of thinking.
#
# It is emphatically NOT the answer to a tight budget: the fix for "high effort
# does not complete" is headroom, because a run that silently charts at `medium`
# when the settings said `high` is not the run that was asked for. Hence the
# stderr warning, and this switch to turn it off entirely.
INKLING_DEGRADE_ON_TRUNCATION = os.environ.get(
    "BEATFORGE_INKLING_DEGRADE_ON_TRUNCATION", "1") not in ("0", "", "false")
#: Effort ladder, most to least expensive — the order the retry walks down.
INKLING_EFFORT_LADDER = ("high", "medium", "low", "minimal")
# The nonzero default on the openai backend is an anti-repetition hack for small
# local models. Inkling does not need it, and chart design wants the run to be
# close to reproducible.
INKLING_TEMPERATURE = float(os.environ.get("BEATFORGE_INKLING_TEMPERATURE", "0.2"))
# Reasoning effort per pipeline stage. Reasoning bills at the FULL output rate
# ($5/1M, the same as completion), and unbounded thinking was measured as the
# pipeline's single largest cost driver — so the designer gets the budget and the
# cheaper roles do not.
#
# The KEYS ARE LEDGER STAGE NAMES and must stay that way — `ledger.stage(...)` is
# what this is looked up against. A retry is not its own stage; it is the same
# `designer` stage with attempt>1, so the retry level is keyed separately below.
INKLING_EFFORT = {
    "designer": os.environ.get("BEATFORGE_INKLING_EFFORT_DESIGNER", "high"),
    "critic": os.environ.get("BEATFORGE_INKLING_EFFORT_CRITIC", "low"),
    # The probe only has to say whether a 4-second tone is high or low. Paying
    # for reasoning on that is pure waste, and it runs before every batch.
    "audio_probe": os.environ.get("BEATFORGE_INKLING_EFFORT_PROBE", "low"),
}
# Any attempt after the first. A retry arrives with a machine-readable violation
# report, so it has less to derive than the first draft, not more.
INKLING_EFFORT_RETRY = os.environ.get("BEATFORGE_INKLING_EFFORT_RETRY", "medium")
INKLING_EFFORT_DEFAULT = os.environ.get("BEATFORGE_INKLING_EFFORT_DEFAULT", "medium")
# Whether a failed audio probe STOPS the run (1) or only warns loudly (0).
# Fatal by default: an endpoint that ignores audio still emits confident,
# plausible, worthless charts, and that is the failure the probe exists for.
INKLING_REQUIRE_AUDIO = os.environ.get(
    "BEATFORGE_INKLING_REQUIRE_AUDIO", "1") not in ("0", "", "false")
# A Managed Endpoint scales to zero, so the first call after an idle period pays
# a cold start. That is latency, not failure — hence a generous default.
INKLING_TIMEOUT = int(os.environ.get("BEATFORGE_INKLING_TIMEOUT", "600"))
# Keychain service name for the proxy-token pair.
INKLING_KEYCHAIN_SERVICE = "beatforge-modal"

# --------------------------------------------------------------------------- #
# Gemini 4 Argon (BEATFORGE_LLM_BACKEND=argon) — see argon.py
#
# Argon is the LARGE general model: same audio-native Gemini stack the pipeline
# was built on, 1M-token context, up to ~8.4 hours of audio in one prompt. The
# question this backend exists to answer is whether a much bigger general model
# designs better charts than a tuned Flash, so it is set up to give Argon its
# best shot AND to keep the bill sane. Measured rates (pricing.py) put it at
# 2.7x 3.8 Flash on input and output, with thinking billed as output — so the
# cost levers, in the order they matter:
#
#   1. THINKING PER STAGE. The designer authoring from a blank page gets `high`;
#      retries (which arrive with a machine-readable violation report) get
#      `medium`; the critic, which emits a ~200-token score, gets `low`. Argon has
#      no `minimal` — the transport maps it to `low` instead of taking a 400.
#   2. WHOLE SONG, NOT CHUNKS. `auto` chunk mode on this backend charts a track up
#      to ARGON_WHOLE_SONG_MAX_SECONDS in one call. That is one designer call per
#      difficulty instead of ~9, and it is what makes caching pay: the audio and
#      the onset inventory lead the prompt and are byte-identical across
#      difficulties, retries and critic calls, and Argon's implicit cache bills a
#      repeated prefix at 5% of the input rate. Chunked prompts are each too small
#      (30 s of audio is ~1k tokens) to reach the cache minimum at all.
#   3. FLEX TIER, opt-in. Half price for variable latency — but only on the
#      Gemini API (AI Studio); Vertex Flex PayGo explicitly excludes Gemini 4.
#
# Transport: the same two Gemini paths the `gemini` / `google-genai` backends use,
# chosen by ARGON_TRANSPORT. No new credential — VERTEX_PROJECT (+ ADC or
# VERTEX_KEY), or GEMINI_API_KEY.
# --------------------------------------------------------------------------- #
ARGON_MODEL = os.environ.get("BEATFORGE_ARGON_MODEL", "gemini-4-argon")
#   auto     — Vertex when VERTEX_PROJECT is set, else AI Studio; but Flex forces
#              AI Studio when a GEMINI_API_KEY is available, since Vertex has none
#   vertex   — Vertex AI, global endpoint (needs VERTEX_PROJECT)
#   aistudio — Google AI Studio (needs GEMINI_API_KEY)
ARGON_TRANSPORT = os.environ.get("BEATFORGE_ARGON_TRANSPORT", "auto").strip().lower()
# "standard" (default) | "flex". Flex is 50% off with minutes of latency and no
# availability guarantee; a capacity refusal falls back to Standard for that one
# call rather than failing the chart (ARGON_FLEX_FALLBACK). Flex on
# generateContent is documented for the Interactions API and not yet for this
# endpoint, so the field is sent and, if the server rejects it, dropped with a
# warning — and a call is only BILLED as Flex when the response confirms it.
ARGON_SERVICE_TIER = os.environ.get("BEATFORGE_ARGON_SERVICE_TIER", "standard").strip().lower()
ARGON_FLEX_FALLBACK = os.environ.get("BEATFORGE_ARGON_FLEX_FALLBACK", "1") not in (
    "0", "", "false")
# Thinking per ledger stage. KEYS ARE LEDGER STAGE NAMES (see llm.thinking_for):
# a call billed to "critic" also thinks as "critic". Anything with attempt > 1,
# and both reprompt stages, take ARGON_THINKING_RETRY.
ARGON_THINKING = {
    "designer": os.environ.get("BEATFORGE_ARGON_THINKING_DESIGNER", "high"),
    "critic": os.environ.get("BEATFORGE_ARGON_THINKING_CRITIC", "low"),
    # "is this tone high or low?" — reasoning on it is pure waste.
    "audio_probe": "low",
    "scorecard": os.environ.get("BEATFORGE_ARGON_THINKING_CRITIC", "low"),
}
ARGON_THINKING_RETRY = os.environ.get("BEATFORGE_ARGON_THINKING_RETRY", "medium")
ARGON_RETRY_STAGES = ("reprompt", "reprompt_density")
# Argon's own documented default, for a stage nobody listed.
ARGON_THINKING_DEFAULT = "medium"
# Optional: judge with a different model than the one that designs. Blank = the
# designer model. Set it to e.g. gemini-3.8-flash for a designer bake-off, so
# every designer is scored by the SAME critic and the scores are comparable.
ARGON_CRITIC_MODEL = os.environ.get("BEATFORGE_ARGON_CRITIC_MODEL", "").strip()
# Total output budget per call, thinking included, across long-decode hops.
# Applies to every gemini-4 model whichever backend sends it.
#
# The MODEL's limit is 1,048,576 (vertex.GEMINI4_MAX_OUTPUT_TOKENS), unified with
# the input — a 65k-token whole-song prompt leaves ~983k — and any value up to it
# works: the client follows as many hops as the budget needs. Higher values are
# clamped to the limit rather than sent as a 400.
#
# 128k is a DEFAULT, not a ceiling, and it is liberal on purpose (the reasoning
# of INKLING_MAX_TOKENS: an unused budget is never billed, an exhausted one
# throws the chart away). A chart JSON is 3-20k tokens and the heaviest designer
# call measured so far spent ~28k on output plus thinking, so this is ~4-5x the
# observed tail. What it does bound is a model that loops instead of answering:
# at 128k that costs at most $1.28 and ~4 hops; at the full 1M it is $10.49 per
# call ($20.97 from 2026-12-01) and ~32 hops of up to 10 minutes each.
ARGON_MAX_OUTPUT_TOKENS = int(os.environ.get("BEATFORGE_ARGON_MAX_OUTPUT_TOKENS", "131072"))
# Per-request read timeout. A high-thinking whole-song call can sit silent for
# minutes per hop, and Flex adds queueing on top; the API closes a non-streaming
# hop at 10 min, so this only has to outlast that.
ARGON_TIMEOUT = int(os.environ.get("BEATFORGE_ARGON_TIMEOUT", "900"))
# `auto` chunk mode charts a track up to this long whole-song on this backend.
# Past it, chunk anyway: the output for one call grows with the song, and a
# 15-minute challenge chart is a lot of JSON to stake on one request.
ARGON_WHOLE_SONG_MAX_SECONDS = float(
    os.environ.get("BEATFORGE_ARGON_WHOLE_SONG_MAX_SECONDS", "900"))

# Music generation (Workstream A) — unchanged from tools/generate_overdrive_assets.py.
LYRIA_MODEL = os.environ.get("BEATFORGE_LYRIA_MODEL", "lyria-002")

# Vertex project / location / creds.
#
# There is NO default project: a hardcoded one would bill the wrong account and
# 403 for everyone else. A user who wants any Vertex-backed
# feature (Gemini designer/critic on the default backend, or Lyria music gen) MUST
# set VERTEX_PROJECT; vertex.py raises a clear "set VERTEX_PROJECT" error otherwise.
# Charting on the `openai` backend needs neither this nor a Google account.
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT")  # required for Vertex; None = unset
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
# Vertex credentials. Prefer Application Default Credentials — run
# `gcloud auth application-default login` once and leave VERTEX_KEY unset. Point
# VERTEX_KEY (or the standard GOOGLE_APPLICATION_CREDENTIALS) at a service-account
# JSON only if you don't use ADC. Unset (None) => vertex.py falls back to ADC.
_vertex_key = os.environ.get("VERTEX_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
VERTEX_KEY = os.path.expanduser(_vertex_key) if _vertex_key else None

# Google AI Studio (API-key Gemini) — the no-GCP-project path. Get a key at
# https://aistudio.google.com/apikey and `export GEMINI_API_KEY=...`, then
# `export BEATFORGE_LLM_BACKEND=google-genai`. Reaches the same Gemini models
# (GEMINI_MODEL) through generativelanguage.googleapis.com instead of Vertex.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# --------------------------------------------------------------------------- #
# DSP analysis parameters (Workstream B). These are baked into the cache key, so
# changing any of them invalidates cached analysis (REQ-COMPUTE-04).
# --------------------------------------------------------------------------- #
ANALYSIS_SR = 22050          # mono analysis sample rate (matches diagnosis §0)
HOP_LENGTH = 256             # STFT hop (samples) -> ~11.6ms frames at 22050
N_FFT = 2048
# Tempo octave/metre candidates to disambiguate (REQ-DSP-02).
TEMPO_MULTIPLIERS = (0.5, 2.0 / 3.0, 0.75, 1.0, 4.0 / 3.0, 1.5, 2.0)
TEMPO_PRIOR_BAND = (84.0, 150.0)    # synthwave genre prior; tie-break target
# Log-Gaussian perceptual tempo prior (octaves): weights the alignment score so
# octave/metre errors (e.g. 258 vs 129 BPM) are suppressed without discarding a
# genuinely fast or slow track. Center ~ middle of the synthwave band.
TEMPO_PRIOR_CENTER = 120.0
TEMPO_PRIOR_SIGMA_OCT = 0.9
OFFSET_SWEEP_MS = 60.0       # ±60ms offset refinement (REQ-DSP-03)
OFFSET_STEP_MS = 1.0
ONSET_SNAP_MAX_MS = 35.0     # grid alignment tolerance (REQ-DSP-03/REQ-VAL-02)
SUSTAIN_MIN_BEATS = 1.0      # harmonic plateau >= 1 beat -> hold candidate
# REQ-R2-OUT-01 second-pass offset fit. `fit_offset` samples the onset envelope by
# frame index (~11.6ms at HOP_LENGTH=256/SR=22050), so it cannot resolve sub-frame
# beat phase; the parabolically-refined onsets can. When their median snap error
# exceeds this threshold the grid is re-phased by that median. Below it the median
# is noise and re-phasing would jitter tracks that were already correct.
# Set to a large number to disable the correction entirely.
OFFSET_SNAP_CORRECTION_MIN_MS = 5.0
OFFSET_SNAP_CORRECTION_MIN_ONSETS = 30
ONSET_COUNT_SANITY = (50, 400)  # sanity gate for a ~33s track (REQ-DSP-04)
# v2 adds the additive `density_plan` block (REQ-R2-DYN-01). Per REQ-R2-SACRED-04
# the cache key is bumped BECAUSE the output schema was extended; no existing
# field changed meaning.
# v3: DENSITY_TARGET_UTILIZATION rescales `density_plan`'s tier ceilings, so a
# cached v2 analysis carries Round 2's (too dense) plan. The utilization is an
# analysis INPUT and therefore has to be part of the cache identity — without this
# bump, changing it would silently do nothing to any track already analyzed.
BEAT_ANALYSIS_VERSION = "local_dsp_v3"  # bump to invalidate all caches

# --------------------------------------------------------------------------- #
# Density plan (REQ-R2-DYN-01) — DSP-side ground truth for dynamics.
#
# Round 1 lost dynamics because "density follows energy" was a sentence in a
# prompt. This makes it a number derived from the energy curve, with the same
# citizenship as the onset inventory: computed in core/, carried in analysis.json,
# handed to the designer as a budget, and enforced by a deterministic repair pass.
# --------------------------------------------------------------------------- #
# The quietest bar still gets this share of the tier ceiling. A floor of 0 would
# produce empty bars, which violates the "never a dead pause" rule the designer
# prompt has enforced since Round 1 — dynamics must not be bought with silence.
DENSITY_PLAN_FLOOR = 0.18
# Shaping exponent on normalized bar energy. 1.0 = linear. MUST stay > 0: the
# density-vs-energy gate is Spearman (rank) correlation, so the plan is only
# useful if it is strictly MONOTONE in energy. Any exponent > 0 preserves that;
# changing it re-shapes magnitudes without breaking the ranking.
DENSITY_PLAN_GAMMA = 1.0
# Half-width of the acceptable band around each bar's target, as a fraction of
# the target. Wide enough that the designer has musical freedom (a fill, a held
# chord), tight enough that the shape survives.
DENSITY_BAND_TOLERANCE = 0.40
# Sustained-NPS ceiling per tier. MIRRORS adapters/stepmania/grammar.py BUDGETS
# max_nps_4s — core/ cannot import an adapter, so the values are duplicated here
# and test_density.py::test_density_tier_nps_matches_stepmania_budgets fails if
# the two ever drift apart.
DENSITY_TIER_MAX_NPS = {
    "beginner": 2.0, "easy": 3.0, "medium": 5.0, "hard": 8.0, "challenge": 10.0,
}
# Share of budgeted spans that must land inside their band for the chart to ship.
# Not 100%: a span whose bars genuinely offer no usable onsets cannot be filled
# without inventing transients, and inventing them is the failure mode the Round 1
# brief warned about ("do not chase rho by globally increasing note density").
# R3 — how much of a tier's sustained-NPS ceiling a PEAK-energy bar should
# actually target. Round 2 shipped this at 1.0, i.e. it treated `max_nps_4s` — a
# hard ceiling for the busiest four seconds — as the goal for every loud bar. The
# result played exactly as that sounds: charts came out 26-79% denser than Round 1
# (hard +79%, challenge +62%) and the pack was judged too hard on pad.
#
# Because the density gate is a RANK correlation, scaling every bar's target by a
# constant leaves the shape — and therefore rho — essentially untouched while
# cutting absolute difficulty. That is the whole trick: dynamics and difficulty
# are separable here, and only difficulty needs to come down.
#
# 0.45 lands the deterministic path at ~211 notes/chart against Round 1's ~231,
# and the Gemini designer runs denser than deterministic, so the shipped charts
# land near Round 1's difficulty — the level that was not complained about — while
# keeping Round 2's dynamics.
# How the per-bar note budget is anchored.
#   "redistribute" — keep the note TOTAL the tier naturally produces and move
#       those notes toward high-energy bars. Conservative by construction: it
#       changes the distribution, never the budget, so difficulty stays put and
#       only dynamics moves. This is what R3 ships.
#   "ceiling"      — R2 behaviour: anchor each bar to a fraction of the tier NPS
#       ceiling. Inflated dense tracks and starved sparse ones. Kept for A/B only.
DENSITY_MODE = os.environ.get("BEATFORGE_DENSITY_MODE", "redistribute")
# In "redistribute" mode this is only a SAFETY CLAMP: no bar may be pushed past
# this fraction of the tier's sustained-NPS ceiling however loud it is. It is no
# longer the anchor for the target (that is the conserved note total), so it can
# sit high without re-creating R2's inflation.
DENSITY_TARGET_UTILIZATION = 0.75
DENSITY_PLAN_MIN_IN_BAND = 0.6

# --------------------------------------------------------------------------- #
# CHUNKED PROCESSING MODE (docs/chunked-mode-brief.md, REQ-CHUNK-08)
#
# A chunk is a SCOPED DESIGNER CALL: one slice of the song with all four of its
# inputs scoped down together — the audio for that slice, the onset inventory
# rows for that slice, the density budget for that slice, and a short summary of
# what previous chunks decided. Trimming only the audio is not a chunked call.
#
# Why it is the default: whole-song calls put the entire onset inventory and the
# entire track in one context, which on consumer hardware (a 24GB 4090, or the
# ~3GB of KV cache left on a local 12B) spills out of VRAM into system RAM and
# crawls. Chunking is also the prerequisite for evaluating any local/open model.
# Gemini does not need it — but it costs Gemini nothing measurable either (the
# same onsets are sent, in slices), and one default is one code path to trust.
#
# The DSP layer is untouched: it still runs ONCE over the whole track, before any
# model call, and tempo/offset/beat grid/sections/density plan/onset IDs are
# global values handed to each chunk as numbers (REQ-CHUNK-03).
#
#   BEATFORGE_CHUNK_MODE = "auto" (default) | "always" | "never"
#     auto   — chunk when the track is longer than MAX_CHUNK_SECONDS
#     always — chunk every track, however short
#     never  — whole-song mode, byte-for-byte the pre-chunking behaviour
# --------------------------------------------------------------------------- #
CHUNK_MODE = os.environ.get("BEATFORGE_CHUNK_MODE", "auto").strip().lower()
# Longest a single chunk may be. Also the `auto` threshold. 30s of 16kHz mono is
# ~1MB of audio and a few dozen inventory rows — comfortably inside the KV cache
# a 12B model has left over on a 24GB card.
MAX_CHUNK_SECONDS = float(os.environ.get("BEATFORGE_MAX_CHUNK_SECONDS", "30"))


def auto_chunk_threshold() -> float:
    """How long a track `auto` mode sends in ONE call, for the active backend.

    MAX_CHUNK_SECONDS everywhere except Argon. Chunking exists for KV-cache
    headroom on local hardware; on a 1M-context model that hears hours of audio
    it only multiplies calls and splits the musical arc, and it prices Argon out
    of its implicit cache (see the Argon block). A function rather than a
    constant so a test or the CLI changing LLM_BACKEND is seen at call time.
    """
    if LLM_BACKEND == "argon":
        return max(MAX_CHUNK_SECONDS, ARGON_WHOLE_SONG_MAX_SECONDS)
    return MAX_CHUNK_SECONDS
# Shorter than this and a section is merged into its neighbour instead of
# becoming its own call: a 3-second stab is not worth a round trip, and a chunk
# that small carries no musical context to design against.
MIN_CHUNK_SECONDS = float(os.environ.get("BEATFORGE_MIN_CHUNK_SECONDS", "8"))
# Bars of lead-in/lead-out included as CONTEXT ONLY (REQ-CHUNK-02). The model
# hears and sees the transition into and out of its slice; notes it places in an
# overlap bar are discarded, because that bar belongs to the adjacent chunk.
CHUNK_OVERLAP_BARS = int(os.environ.get("BEATFORGE_CHUNK_OVERLAP_BARS", "1"))
# Ceiling on the prior-decision summary (REQ-CHUNK-04), in tokens. This is the
# ONLY cross-chunk memory, and it carries decisions — never audio, never
# inventory rows. Oldest, least-relevant sections are dropped first; a section
# whose role repeats the current chunk's is kept preferentially.
CHUNK_SUMMARY_TOKEN_BUDGET = int(
    os.environ.get("BEATFORGE_CHUNK_SUMMARY_TOKEN_BUDGET", "400"))
# Per-chunk retries for a chunk whose output could not be used (unparseable ref,
# empty response). Retry budgets apply PER CHUNK (REQ-CHUNK-07) — one bad slice
# must not burn the whole song's attempt budget.
CHUNK_MAX_ATTEMPTS = int(os.environ.get("BEATFORGE_CHUNK_MAX_ATTEMPTS", "2"))
# Where sliced chunk audio is written. One wav per (track, chunk), reused across
# difficulties and across runs — slicing is deterministic, so a cached slice is
# the same slice.
CHUNK_AUDIO_DIR = REPO_ROOT / "build" / "chunks"

# --------------------------------------------------------------------------- #
# Compute backend (spec §3.5)
# --------------------------------------------------------------------------- #
DEFAULT_BACKEND = os.environ.get("BEATFORGE_BACKEND", "local")  # local | colab
COLAB_GPU = os.environ.get("BEATFORGE_COLAB_GPU", "T4")         # T4 | L4 | A100 | H100
COLAB_SESSION_PREFIX = "beatforge"

# --------------------------------------------------------------------------- #
# Generation loop (Workstream A)
# --------------------------------------------------------------------------- #
GEN_CANDIDATES_PER_ROUND = 4   # REQ-GEN-01 N
GEN_MAX_ROUNDS = 3             # REQ-GEN-01 K
GEN_SHIP_THRESHOLD = 8.0       # weighted scorecard ship threshold (REQ-GEN-03)
# Scorecard rubric weights (REQ-GEN-02). Keys must match schemas/scorecard.json.
GEN_RUBRIC_WEIGHTS = {
    "tempo_stability": 1.0,
    "transient_clarity": 1.5,
    "structural_contrast": 1.5,
    "intro_cleanliness": 1.0,
    "mix_punch": 1.0,
    "genre_fit": 1.0,
}
TARGET_LUFS = -14.0            # REQ-GEN-04
TARGET_TRUE_PEAK_DBTP = -1.0
MAX_LEADING_SILENCE_MS = 120.0
ENABLE_LONGER_TRACKS = False   # REQ-GEN-06 stretch, default off

# --------------------------------------------------------------------------- #
# Difficulty budgets (spec §6 — hard constraints enforced in Workstream D)
# --------------------------------------------------------------------------- #
DIFFICULTIES = ("casual", "standard", "overdrive")


@dataclass(frozen=True)
class DifficultyBudget:
    name: str
    min_gap_beats: float
    min_gap_ms: float
    finest_subdiv: float          # smallest legal grid step, in beats
    max_nps_4s: float             # max sustained notes-per-second over any 4s window
    jack_limit: int               # same-lane consecutive limit
    taps_during_hold: int         # allowed taps in OTHER lanes during a hold
    hold_len_beats: tuple         # (min, max) hold length in beats
    hold_share: tuple             # (min, max) fraction of events that are holds
    onset_align_min: float        # REQ-QA-01 alignment floor


BUDGETS = {
    "casual": DifficultyBudget(
        name="casual",
        min_gap_beats=1.0, min_gap_ms=350.0,
        finest_subdiv=1.0,
        max_nps_4s=2.0,
        jack_limit=2,
        taps_during_hold=0,
        hold_len_beats=(2.0, 8.0),
        hold_share=(0.10, 0.20),
        onset_align_min=0.80,   # sparse charts skip onsets, not miss them
    ),
    "standard": DifficultyBudget(
        name="standard",
        min_gap_beats=0.5, min_gap_ms=200.0,
        finest_subdiv=0.5,
        max_nps_4s=4.0,
        jack_limit=3,
        taps_during_hold=1,
        hold_len_beats=(1.0, 8.0),
        hold_share=(0.05, 0.15),
        onset_align_min=0.90,
    ),
    "overdrive": DifficultyBudget(
        name="overdrive",
        min_gap_beats=0.25, min_gap_ms=120.0,
        finest_subdiv=0.25,
        max_nps_4s=7.0,
        jack_limit=4,
        taps_during_hold=2,
        hold_len_beats=(1.0, 8.0),
        hold_share=(0.05, 0.12),
        onset_align_min=0.90,
    ),
}

# QA gate thresholds (spec §7)
LANE_BALANCE_RANGE = (0.22, 0.45)   # each lane 22-45% (REQ-QA-01)
MAX_REPAIR_FRACTION = 0.15          # >15% repaired -> chart fails (REQ-VAL-03)
# REQ-QA-02. MEASURED, like the hold_share bands — the 0.55 this replaces was a
# drafted number and described nothing. Scored over the 471 charts in
# tools/chartbench/out/ it accepted 29%, and it failed the MEDIAN chart of every
# generator in the corpus: DDC (trained on thousands of human charts) medians
# 0.530, AutoStepper 0.155, and our own Founders Mix #2 medians 0.011.
#
# Unlike the hold floor this gate is NOT being dropped — "density follows the
# song's energy" is the property that separates a chart from a metronome, and
# AutoStepper at 0.155 should fail. It is being put where it discriminates. At
# 0.30 the corpus passes 54%, DDC mostly passes, AutoStepper and Mix #2 mostly
# fail — the ordering a gate is for. The CHARTFORGE-LoRA run that prompted this
# medians 0.739 and passes 89%, and the charts it still fails measure -0.037,
# 0.177 and 0.284, which are genuinely not following anything.
#
# Deliberately NOT scaled per difficulty. Low tiers do score lower (the corpus
# medians run 0.185 at beginner to 0.472 at hard) because a rank correlation over
# few, mostly-tied per-bar counts is noisy — but the corpus curve is not monotonic
# (challenge dips to 0.383), so a per-tier ladder would be fitting noise. Re-derive
# with tools/chartbench if the corpus grows.
DENSITY_ENERGY_SPEARMAN_MIN = 0.30
# REQ-QA-02 density-SHAPE gates (spearman/peak-in-drop/breathes) assume the track
# HAS dynamic contrast (build→drop). Uniform loops (the existing keep-the-hits
# tracks measure CV 0.04-0.08) cannot follow an energy curve that doesn't move —
# forcing it would make the designer fabricate contrast the music lacks. Tracks
# below this energy-curve CV are recorded `flat_track_exempt` for the shape gates
# (alignment/lane/NPS/hold/repair still fully enforced). Workstream A tracks,
# which are generated WITH contrast, clear this bar and get the full gate.
DENSITY_GATE_MIN_CV = 0.12
# The section-level shape gates (peak-in-drop, breathes) require the track to have
# ≥2 sections whose energy actually differs by at least this much (0..1 scale). A
# steady groove with bar-level oscillation but one uniform section does not, so
# those two gates exempt (the bar-level Spearman gate still applies).
SECTION_STRUCTURE_MIN_SPREAD = 0.15
CRITIC_SHIP_THRESHOLD = 7.0         # REQ-QA-04
DESIGN_MAX_ATTEMPTS = 3             # REQ-QA-03 re-prompt loop cap

# Playable window (mirrors make_overdrive_maps.py): events before beat 4 or after
# (duration - PLAYABLE_TAIL_S) are dropped (REQ-VAL-02).
FIRST_PLAYABLE_BEAT = 4.0
PLAYABLE_TAIL_S = 1.4

# Judgment windows — the contract from src/core/timing.ts DEFAULT_WINDOWS.
PERFECT_WINDOW_S = 0.05
GOOD_WINDOW_S = 0.12

# Catalogue (id -> source ogg basename). Mirrors src/game/tracks.ts.
TRACK_CATALOGUE = {
    "overdrive": "overdrive_pulse",
    "midnight": "midnight_run",
    "neon": "neon_nights",
    "groove": "sweetpapa_groove",
    "hope": "worth_the_hope",
}

# Song metadata for StepMania export (title/artist), mirrors src/game/tracks.ts.
TRACK_META = {
    "overdrive": ("Overdrive Pulse", "Sweet Papa & the Tones"),
    "midnight": ("Midnight Run", "Sweet Papa & the Tones"),
    "neon": ("Neon Nights", "Sweet Papa & the Tones"),
    "groove": ("Sweet Papa Groove", "Sweet Papa & the Tones"),
    "hope": ("Worth the Hope", "Sweet Papa Technologies"),
}


@dataclass
class RunOptions:
    """Runtime knobs threaded from the CLI into every stage."""
    backend: str = DEFAULT_BACKEND
    force: bool = False
    skip_gen: bool = True          # keep-the-hits default (REQ-GEN-05)
    allow_local_fallback: bool = False
    tracks: tuple = field(default_factory=lambda: tuple(TRACK_CATALOGUE))
    difficulties: tuple = DIFFICULTIES
    gpu: str = COLAB_GPU
    offline: bool = False          # skip all Vertex calls (fixture/test mode)

# SweetPapa's Founders Mix #2 (2026-07-21). Registered so the simfile carries
# the real title/artist — without this every export is titled `fm2_<base>`.
TRACK_META['fm2_banana_banana'] = ('Banana Banana', 'Sweet Papa & the Tones')
TRACK_META['fm2_do_it_for_me_now'] = ('Do It For Me Now', 'Sweet Papa & the Tones')
TRACK_META['fm2_lucky_lucky'] = ('Lucky Lucky', 'Sweet Papa & the Tones')
TRACK_META['fm2_smile_and_dance'] = ('Smile and Dance', 'Sweet Papa & the Tones')
TRACK_META['fm2_some_will_say'] = ('Some Will Say', 'Sweet Papa & the Tones')
TRACK_META['fm2_stay_awake_for_me'] = ('Stay Awake For Me', 'Sweet Papa & the Tones')
TRACK_META['fm2_streaming_is_the_life_for_me'] = ('Streaming Is The Life For Me', 'Sweet Papa & the Tones')
TRACK_META['fm2_token_economy'] = ('Token Economy', 'Sweet Papa & the Tones')
TRACK_META['fm2_kieteyuku'] = ('消えてゆく', 'Sweet Papa & the Tones')
