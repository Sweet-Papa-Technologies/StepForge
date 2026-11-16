"""pricing.py — the checked-in price table the cost ledger bills against
(REQ-R2-COST-01).

Why a file and not a constant: the $80-for-30-songs number in the Round 2 brief
was unattributable because nothing in the pipeline ever converted tokens to
dollars. Cost has to be computed from a table that is *in the repo*, *dated*, and
*editable*, so that a number in a report can be re-derived months later and a
price change is a diff instead of a mystery.

Rates are USD per 1,000,000 tokens, per modality, because Gemini bills audio
input at a different rate from text input and that distinction is the entire
point of hypothesis #1 in the autopsy.

`verified` marks whether the rate was checked against the live Vertex pricing
page/SKU by a human. UNVERIFIED rates still produce numbers, but every report
that touches one carries a warning — a confidently-wrong dollar figure is worse
than an admittedly-uncertain one.

Note the spec asked for `config/pricing.py`. `beatforge/config` cannot be a
package because `beatforge/config.py` already exists (Python would not resolve
both), so the table lives here, next to it, and `config` re-exports nothing.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

# Date the rates below were last reviewed. Bump it when you touch a rate.
PRICING_AS_OF = "2026-10-23"

# IMPORTANT — endpoint affects the rate. Vertex charges ~10% more on non-global
# (regional) endpoints as of 2026-07-01. `vertex.py::_endpoint` routes every
# `gemini-3*` model to location=`global`, so the GLOBAL rates below are the ones
# beatforge actually pays. If that routing ever changes, these rates get ~10%
# worse and this comment is the reason why.

# Source of record for the Vertex rates (paste the page you checked, don't guess).
PRICING_SOURCE = "https://cloud.google.com/vertex-ai/generative-ai/pricing"
# Inkling's rates come from Modal's endpoint-creation screen, not a public page —
# re-check them there (they are shown per 1M tokens beside the endpoint config).
PRICING_SOURCE_MODAL = "https://modal.com/  (Managed Endpoint creation screen)"


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1e6 tokens, split by the modalities Vertex meters separately."""
    model: str
    text_in: float
    audio_in: float
    out: float
    # Cached input tokens (Vertex context caching) bill at a discount. If a model
    # has no caching SKU, leave this equal to text_in.
    cached_in: float = 0.0
    # Thinking/reasoning tokens. Gemini bills them as OUTPUT tokens; keeping the
    # field explicit means the ledger can attribute them separately even though
    # the rate matches `out`.
    thinking_out: float = 0.0
    verified: bool = False
    note: str = ""

    def __post_init__(self):
        # dataclass is frozen; use object.__setattr__ for the derived defaults.
        if self.cached_in == 0.0:
            object.__setattr__(self, "cached_in", round(self.text_in * 0.25, 6))
        if self.thinking_out == 0.0:
            object.__setattr__(self, "thinking_out", self.out)


# --------------------------------------------------------------------------- #
# The table.
# --------------------------------------------------------------------------- #
# gemini-3.5-flash is what the pipeline is CONFIGURED to use. Whether it is what
# the pipeline actually used is a question for the ledger, not for this file
# (autopsy hypothesis #4 exists because config lied once already).
TABLE: dict[str, ModelPrice] = {
    # VERIFIED 2026-07-21 against cloud.google.com/vertex-ai/generative-ai/pricing.
    # Two things here are counter-intuitive and both matter to the autopsy:
    #   1. Audio input is NOT priced at a premium — 3.5 Flash bills audio at the
    #      same $1.50 as text ("Text/Image/Video/Audio input" is one line item).
    #      An earlier guess had audio at 3.3x text, which overstated its share.
    #   2. Reasoning tokens bill at the OUTPUT rate ("Text output (response and
    #      reasoning)"), i.e. $9.00 — six times the input rate. Thinking is the
    #      most expensive token this pipeline can emit.
    # Cached input at $0.15 is a 10x discount on $1.50, which is what makes
    # context caching the highest-leverage structural fix available.
    # Verified 2026-07-21 against the Vertex pricing page AND probed live against
    # the project (both return 200 and both accept an audio part — checked, because
    # a designer that cannot hear the track is useless to us however cheap it is).
    # Kept after 3.8 replaced it as the default: every ledger line written before
    # the switch names this model, and dropping its rate would re-bill all of them
    # at the unknown-model fallback.
    "gemini-3.6-flash": ModelPrice(
        "gemini-3.6-flash", text_in=1.50, audio_in=1.50, out=7.50,
        cached_in=0.15, thinking_out=7.50, verified=True,
        note="Verified 2026-07-21, global endpoint. Same input rate as 3.5 Flash, "
             "output 17% cheaper ($7.50 vs $9.00). Audio billed at the input rate."),
    # The current default. Read 2026-10-23 off ai.google.dev/gemini-api/docs/pricing
    # (Standard tier). This is a LAUNCH rate with an end date — half the 3.6 price
    # until 2027-01-01, when it doubles to exactly 3.6's $1.50 / $7.50. The doubled
    # rate is in PRICE_CHANGES below, so a call made after that date bills right
    # without anyone remembering to edit this file. Audio is not priced separately.
    "gemini-3.8-flash": ModelPrice(
        "gemini-3.8-flash", text_in=0.75, audio_in=0.75, out=3.75,
        cached_in=0.075, thinking_out=3.75, verified=True,
        note="Verified 2026-10-23 on the Gemini API pricing page (Vertex global "
             "assumed identical, as it has been for every 3.x model here). Launch "
             "rate through 2026-12-31; $1.50 / $7.50 / $0.15 from 2027-01-01."),
    # Gemini 4 Argon — the large general model (BEATFORGE_LLM_BACKEND=argon, see
    # argon.py). Same source and date as 3.8 Flash. Three things matter here:
    #   1. Output is 2.7x 3.8 Flash's and thinking bills at that output rate, so the
    #      per-stage thinking level is the cost lever on this model, by a wide margin.
    #   2. Cached input is a 20x discount ($0.10 vs $2.00). Argon has IMPLICIT
    #      caching only, so the discount is claimed by keeping the prompt prefix
    #      byte-identical across calls — which whole-song mode on this backend does.
    #   3. The launch rate ends 2026-11-30; everything doubles on 2026-12-01.
    # Non-global endpoints bill $2.20 input; vertex.py routes gemini-4* to global.
    "gemini-4-argon": ModelPrice(
        "gemini-4-argon", text_in=2.00, audio_in=2.00, out=10.00,
        cached_in=0.10, thinking_out=10.00, verified=True,
        note="Verified 2026-10-23 on the Gemini API pricing page, global endpoint. "
             "Launch rate through 2026-11-30; $4.00 / $20.00 / $0.20 from "
             "2026-12-01. Audio billed at the input rate."),
    "gemini-3.5-flash-lite": ModelPrice(
        "gemini-3.5-flash-lite", text_in=0.30, audio_in=0.30, out=2.50,
        cached_in=0.03, thinking_out=2.50, verified=True,
        note="Verified 2026-07-21, global endpoint. 5x cheaper input and 3.6x "
             "cheaper output than 3.5 Flash. Non-global would be 0.33/2.75/0.033."),
    "gemini-3.5-flash": ModelPrice(
        "gemini-3.5-flash", text_in=1.50, audio_in=1.50, out=9.00,
        cached_in=0.15, thinking_out=9.00, verified=True,
        note="Verified 2026-07-21, GLOBAL endpoint (what vertex.py routes to). "
             "Non-global would be $1.65 / $1.65 / $9.90 / $0.165."),
    "gemini-3-flash-preview": ModelPrice(
        "gemini-3-flash-preview", text_in=0.50, audio_in=1.00, out=3.00,
        cached_in=0.05, thinking_out=3.00, verified=True,
        note="Verified 2026-07-21. Unlike 3.5 Flash, this one DOES price audio "
             "at a 2x premium over text."),
    "gemini-3.1-flash-lite": ModelPrice(
        "gemini-3.1-flash-lite", text_in=0.25, audio_in=0.50, out=1.50,
        cached_in=0.025, thinking_out=1.50, verified=True,
        note="Verified 2026-07-21, global endpoint."),
    "gemini-2.5-flash": ModelPrice(
        "gemini-2.5-flash", text_in=0.30, audio_in=1.00, out=2.50,
        verified=False, note="UNVERIFIED — legacy entry, not used by the pipeline."),
    "gemini-2.5-pro": ModelPrice(
        "gemini-2.5-pro", text_in=1.25, audio_in=1.25, out=10.00,
        verified=False, note="UNVERIFIED — legacy entry, not used by the pipeline."),
    "lyria-002": ModelPrice(
        "lyria-002", text_in=0.0, audio_in=0.0, out=0.0,
        verified=False,
        note="Lyria bills PER GENERATED CLIP, not per token; see CLIP_PRICES."),

    # Inkling on a Modal Managed Endpoint — token-billed, not GPU-hour billed.
    # Rates as shown on Modal's endpoint-creation screen (see PRICING_SOURCE_MODAL).
    #   prompt $1.20 | cached prompt $0.27 | completion $5.00 | reasoning $5.00
    # Two notes that matter to a cost report:
    #   1. Cached prompt is a 4.4x discount, and this pipeline's prompts are built
    #      for it — the same brief and onset inventory repeat across five
    #      difficulties plus critic calls (see REQ-INK-08 prompt ordering).
    #   2. Reasoning bills at the FULL completion rate, so the reasoning-effort
    #      dial is a direct cost lever, not a quality-only knob.
    # audio_in is set to the prompt rate because Modal's screen prices one prompt
    # token class; whether audio is metered differently is NOT confirmed, which is
    # why `verified` stays False until a real run's usage block is compared to a
    # billed invoice. An unverified rate still produces numbers, but every report
    # touching it carries a warning — see the 3.6x-low incident this flag exists for.
    "thinkingmachines/Inkling-NVFP4": ModelPrice(
        "thinkingmachines/Inkling-NVFP4", text_in=1.20, audio_in=1.20, out=5.00,
        cached_in=0.27, thinking_out=5.00, verified=False,
        note="Rates from Modal's endpoint-creation screen 2026-07-25 (prompt 1.20 / "
             "cached 0.27 / completion 5.00 / reasoning 5.00). UNVERIFIED against a "
             "billed invoice, and audio metering is assumed to equal the prompt "
             "rate — confirm both before trusting a per-song figure."),
}

# Rates that change on a known date, keyed by TABLE model. Each entry is
# (first ISO date the new rate applies, the rate). Both models released this
# autumn launched at a discount with a published end date, so a single static
# rate is guaranteed to go wrong on a day nobody is watching — and a cost report
# that silently halves the bill is the failure this module exists to prevent.
PRICE_CHANGES: dict[str, list[tuple[str, ModelPrice]]] = {
    "gemini-3.8-flash": [
        ("2027-01-01", ModelPrice(
            "gemini-3.8-flash", text_in=1.50, audio_in=1.50, out=7.50,
            cached_in=0.15, thinking_out=7.50, verified=True,
            note="Post-launch rate, published on the Gemini API pricing page.")),
    ],
    "gemini-4-argon": [
        ("2026-12-01", ModelPrice(
            "gemini-4-argon", text_in=4.00, audio_in=4.00, out=20.00,
            cached_in=0.20, thinking_out=20.00, verified=True,
            note="Post-launch rate, published on the Gemini API pricing page.")),
    ],
}

# Service-tier multipliers on the Standard rate. Read off the same pricing page:
# Batch and Flex are exactly half of Standard for both new models, Priority is
# exactly 1.8x. Applied only to a tier the SERVER confirmed — see
# argon.ArgonPolicy._served_tier — because billing a request at the Flex
# discount when the server quietly ran it at Standard would under-report it.
TIER_MULTIPLIER: dict[str, float] = {
    "standard": 1.0,
    "flex": 0.5,
    "batch": 0.5,
    "priority": 1.8,
}

# Endpoints serve Inkling under varying aliases (vLLM sometimes shortens the id),
# so bill any of them at the same table entry rather than falling through to the
# unknown-model path and reporting $0.
MODEL_ALIASES: dict[str, str] = {
    "inkling": "thinkingmachines/Inkling-NVFP4",
    "inkling-nvfp4": "thinkingmachines/Inkling-NVFP4",
    "thinkingmachines/inkling": "thinkingmachines/Inkling-NVFP4",
}

# Non-token SKUs: flat price per unit of work.
CLIP_PRICES: dict[str, float] = {
    "lyria-002": 0.06,          # USD per generated 30s clip — UNVERIFIED
}

# Compute (non-LLM) rates, REQ-R2-COST-02. Colab bills in compute units, not
# dollars, so this is the operator's own conversion — override it to match the
# plan you actually pay for.
GPU_USD_PER_MINUTE: dict[str, float] = {
    "T4": float(os.environ.get("BEATFORGE_PRICE_GPU_T4", "0.0035")),
    "L4": float(os.environ.get("BEATFORGE_PRICE_GPU_L4", "0.0117")),
    "A100": float(os.environ.get("BEATFORGE_PRICE_GPU_A100", "0.0613")),
    "V100": float(os.environ.get("BEATFORGE_PRICE_GPU_V100", "0.0410")),
    "none": 0.0,                # local CPU backend: no metered spend
    "cpu": 0.0,
}

# A model string we have never seen must NOT silently bill at $0 — that would
# hide exactly the kind of model swap hypothesis #4 is looking for.
UNKNOWN_MODEL_FALLBACK = ModelPrice(
    "<unknown>", text_in=1.50, audio_in=1.50, out=9.00, verified=False,
    note="UNKNOWN MODEL — billed at the gemini-3.5-flash rate as a placeholder. "
         "Add it to pricing.TABLE.")


def _table_key(model: str) -> str | None:
    """The TABLE entry a model string bills against, or None if unknown."""
    if model in TABLE:
        return model
    # An endpoint may serve a known model under a shorter alias (Inkling on vLLM
    # does); billing that at the unknown-model fallback would misreport a model we
    # have rates for.
    alias = MODEL_ALIASES.get(model.strip().lower())
    if alias:
        return alias
    # tolerate versioned suffixes: "gemini-3.5-flash-002" -> "gemini-3.5-flash".
    # Longest match first, so "gemini-4-argon-cyber-permissive" can never be
    # claimed by a shorter key that happens to iterate earlier.
    for known in sorted(TABLE, key=len, reverse=True):
        if model.startswith(known):
            return known
    return None


def price_for(model: str, at: float | None = None) -> ModelPrice:
    """Look up a model's rate as of `at` (epoch seconds; default now). Unknown ids
    fall back loudly, never to zero."""
    key = _table_key(model)
    if key is None:
        return ModelPrice(
            model, text_in=UNKNOWN_MODEL_FALLBACK.text_in,
            audio_in=UNKNOWN_MODEL_FALLBACK.audio_in, out=UNKNOWN_MODEL_FALLBACK.out,
            verified=False, note=UNKNOWN_MODEL_FALLBACK.note)
    price = TABLE[key]
    changes = PRICE_CHANGES.get(key)
    if changes:
        # ISO dates compare correctly as strings. UTC, because the pricing page
        # publishes a calendar date and does not say whose midnight it means.
        today = time.strftime("%Y-%m-%d", time.gmtime(time.time() if at is None else at))
        for effective, later in sorted(changes):
            if today >= effective:
                price = later
    return price


def cost_usd(
    model: str, *, text_in: int = 0, audio_in: int = 0, cached_in: int = 0,
    out: int = 0, thinking: int = 0, tier: str | None = None,
    at: float | None = None,
) -> dict:
    """Bill one model call. Returns the per-component breakdown AND the total, so
    a report can say *which* component dominated without re-deriving it.

    `cached_in` is a SUBSET of the prompt tokens that were served from cache; the
    caller passes non-cached text separately (see ledger.usage_from_vertex).
    `thinking` is billed at the output rate but reported apart from visible output.
    `tier` is the service tier the server CONFIRMED (None/standard = no change).
    """
    p = price_for(model, at=at)
    mult = TIER_MULTIPLIER.get((tier or "standard").lower(), 1.0)
    parts = {
        "text_in": text_in / 1e6 * p.text_in * mult,
        "audio_in": audio_in / 1e6 * p.audio_in * mult,
        "cached_in": cached_in / 1e6 * p.cached_in * mult,
        "out": out / 1e6 * p.out * mult,
        "thinking": thinking / 1e6 * p.thinking_out * mult,
    }
    parts: dict = {k: round(v, 8) for k, v in parts.items()}
    parts["total"] = round(sum(parts.values()), 8)
    parts["rate_verified"] = p.verified
    if mult != 1.0:
        parts["tier"] = tier
    return parts


def gpu_cost_usd(gpu: str, minutes: float) -> float:
    rate = GPU_USD_PER_MINUTE.get(gpu, GPU_USD_PER_MINUTE.get(str(gpu).upper(), 0.0))
    return round(rate * max(0.0, minutes), 8)


def table_snapshot() -> dict:
    """The exact rates a report was computed against, for embedding in artifacts.
    A cost report without its price table is not reproducible."""
    return {
        "pricing_as_of": PRICING_AS_OF,
        "source": PRICING_SOURCE,
        "models": {
            m: {"text_in": p.text_in, "audio_in": p.audio_in, "out": p.out,
                "cached_in": p.cached_in, "thinking_out": p.thinking_out,
                "verified": p.verified, "note": p.note}
            for m, p in TABLE.items()},
        "price_changes": {
            m: [{"effective": d, "text_in": p.text_in, "audio_in": p.audio_in,
                 "out": p.out, "cached_in": p.cached_in} for d, p in changes]
            for m, changes in PRICE_CHANGES.items()},
        "tier_multiplier": dict(TIER_MULTIPLIER),
        "gpu_usd_per_minute": dict(GPU_USD_PER_MINUTE),
        "clip_prices": dict(CLIP_PRICES),
    }


def unverified_models() -> list[str]:
    return sorted(m for m, p in TABLE.items() if not p.verified)
