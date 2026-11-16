"""design.py — the StepMania intent designer (Gemini 3.5 Flash, swappable via
llm.py) and the fresh-context critic (STEPFORGE §6, REQ-SM-05/13). The model
hears the audio and emits INTENT (which onsets, kinds, phrase textures) — never
panels or times."""
from __future__ import annotations

import json
from pathlib import Path

from ... import config
from ...inventory import rows as _inventory_rows
from . import validate
from .grammar import BUDGETS

_PROMPTS = Path(__file__).resolve().parent / "prompts"


def _hold_rule(difficulty: str) -> str:
    """The tier's hold-candidate threshold, as a rule stated ONCE in the budget
    text rather than baked per-onset into the inventory.

    This used to be a `sustain` boolean computed per onset inside
    `_compact_analysis`, which meant `easy` and `medium` — identical inventories
    in every other respect — differed on nine onsets and could never share a
    cached prompt prefix. The threshold is a property of the TIER, so it belongs
    with the tier's other budget numbers.

    Equivalence note: the old test was `o["sustain"] and sustain_beats >= floor`.
    The DSP only ever sets `sustain=True` together with a plateau of at least
    SUSTAIN_MIN_BEATS, and leaves `sustain_beats` at 0.0 otherwise, so
    `sustain` <-> `sustain_beats > 0` and the rule below selects exactly the same
    onsets. `adapters/stepmania/resolve.py` remains the enforcer either way.
    """
    floor = BUDGETS[difficulty].hold_len_beats[0] - 1.0
    return f"sustain_beats >= {floor:.1f}" if floor > 0 else "sustain_beats > 0"


def _budget_text(difficulty: str) -> str:
    b = BUDGETS[difficulty]
    _, hi_h = b.hold_share
    lo_l, hi_l = b.hold_len_beats
    return (f"- finest row subdivision: {b.finest_subdiv} beat\n"
            f"- max sustained NPS (4s window): {b.max_nps_4s}\n"
            f"- crossovers: {b.crossover}\n"
            f"- footswitches: {'allowed (sparse)' if b.footswitch else 'none'}\n"
            f"- jacks: max {b.jack_limit} in a burst\n"
            f"- jumps: {b.jumps}\n"
            f"- holds/rolls share of notes: at most {hi_h:.0%} (no minimum)\n"
            f"- holds/rolls: at most {validate.MAX_CONCURRENT_HOLDS} may overlap in "
            f"time, and briefly. Two at once pins both feet\n"
            f"- holds/rolls are legal ONLY on an onset with {_hold_rule(difficulty)}; "
            f"every other onset in the inventory can only be a tap or a mine\n"
            f"- hold_beats must be {lo_l}-{hi_l} beats and within +/-1 beat of that "
            f"onset's sustain_beats\n"
            f"- density must follow the per-bar energy "
            f"(Spearman >= {config.DENSITY_ENERGY_SPEARMAN_MIN})")


def _compact_analysis(analysis: dict, difficulty: str) -> dict:
    b = BUDGETS[difficulty]
    subdiv = b.finest_subdiv
    lo = config.FIRST_PLAYABLE_BEAT
    hi = (analysis["duration_s"] - config.PLAYABLE_TAIL_S - analysis["offset"]) * analysis["bpm"] / 60.0
    onsets = []
    for o in analysis.get("onsets", []):
        beat = o["nearest_beat"]
        if beat < lo or beat > hi:
            continue
        if abs(beat - round(beat / subdiv) * subdiv) > 1e-3:
            continue
        if abs(o.get("snap_error_ms", 0)) > config.ONSET_SNAP_MAX_MS:
            continue
        # sustain_beats travels RAW. Baking this tier's hold-length floor into a
        # per-onset boolean here was the one thing making the `easy` and `medium`
        # inventories differ (9 onsets out of 517) — see _hold_rule.
        onsets.append({"id": o["id"], "beat": beat, "strength": o["strength"],
                       "bands": o["bands"],
                       "sustain_beats": o.get("sustain_beats", 0)})
    return {"bpm": analysis["bpm"], "offset": analysis["offset"], "meter": analysis.get("meter", 4),
            "duration_s": analysis["duration_s"], "finest_subdivision": subdiv,
            "sections": analysis.get("sections", []),
            "energy_curve": analysis.get("energy_curve", []), "onsets": onsets,
            # REQ-R2-DYN-02: the density budget travels WITH the inventory, as
            # numbers. Only this tier's bands are sent — the other four would be
            # noise in a prompt that is already 97% onset menu.
            "density_plan": _density_view(analysis, difficulty)}


def _density_view(analysis: dict, difficulty: str) -> dict:
    """This tier's slice of the density plan: per-section notes-per-bar bands.

    Deliberately per-SECTION and not per-bar. Per-bar bands for a 96-bar track
    would add ~4KB of numbers the designer cannot act on at that resolution (it
    declares intent per phrase, not per bar), and the cost autopsy is unambiguous
    that prompt bulk is the one thing this pipeline has too much of already."""
    plan = analysis.get("density_plan") or {}
    if not plan.get("per_bar"):
        return {}
    return {
        "units": "notes per bar",
        "tier": difficulty,
        "flat_track": plan.get("flat", False),
        "sections": [
            {"start_bar": s["start_bar"], "end_bar": s["end_bar"],
             "role": s.get("role_guess"),
             "min": s["band"][difficulty][0], "max": s["band"][difficulty][1],
             "target": s["target_notes"][difficulty]}
            for s in plan.get("per_section", [])
            if difficulty in s.get("band", {})
        ],
    }


def _density_text(analysis: dict, difficulty: str) -> str:
    """The hard-budget block. Round 1 said 'density must follow energy' in prose
    and the designer ignored it. This says the numbers."""
    view = _density_view(analysis, difficulty)
    if not view.get("sections"):
        return ("(no density plan available for this track — follow the energy "
                "curve as best you can)")
    if view.get("flat_track"):
        head = ("This track has NO energy contrast, so the budget is deliberately "
                "uniform. Do not fabricate a build/drop the music does not have.\n")
    else:
        head = ("Each phrase you declare MUST carry a `density` field: your intended "
                "notes per bar for that phrase. It is validated against the band for "
                "the bars it covers and REJECTED if it falls outside. Quiet sections "
                "get fewer notes, loud sections get more — this is the budget, not a "
                "suggestion.\n")
    rows = "\n".join(
        f"  bars {s['start_bar']:>3}-{s['end_bar']:<3} ({s['role'] or '?':<6}) "
        f"target {s['target']:.1f}  allowed {s['min']:.1f}-{s['max']:.1f} notes/bar"
        for s in view["sections"])
    return head + rows


# Fields of `_compact_analysis` that are TIER-SPECIFIC and already stated, in
# full, in the variable block at the back of the prompt: `finest_subdivision` is
# the first line of `_budget_text`, and `density_plan` is exactly what
# `_density_text` renders. Sending them here too would cost bytes, say nothing
# new, and — the reason it matters — make the analysis block differ per
# difficulty, which is precisely the byte-identity the caching depends on.
# Excluded from the PROMPT only; `_compact_analysis` still returns them.
_TIER_SPECIFIC = ("finest_subdivision", "density_plan")


def analysis_block(difficulty: str, analysis: dict) -> str:
    """The MACHINE ANALYSIS block: a small JSON header for the song-level facts,
    then the onset inventory as rows (Workstream A).

    Everything in here is song-level, not tier-level, so `easy` and `medium`
    produce byte-identical text and so do `hard` and `challenge` — five designer
    prompts carrying three distinct payloads. Exposed (not private) because the
    byte-identity test asserts on exactly this string, which is the property the
    whole caching saving rests on.
    """
    compact = _compact_analysis(analysis, difficulty)
    header = {k: v for k, v in compact.items()
              if k not in _TIER_SPECIFIC and k != "onsets"}
    return json.dumps(header) + "\n" + _inventory_rows(compact["onsets"])


def designer_prompt(difficulty: str, analysis: dict) -> str:
    tmpl = (_PROMPTS / "designer.md").read_text()
    exemplars = _PROMPTS / "exemplars.md"
    return (tmpl.replace("{difficulty}", difficulty)
                .replace("{budget}", _budget_text(difficulty))
                .replace("{density_budget}", _density_text(analysis, difficulty))
                .replace("{exemplars}", exemplars.read_text() if exemplars.exists() else "")
                .replace("{analysis}", analysis_block(difficulty, analysis)))


def design_intent(track_id: str, difficulty: str, analysis: dict, audio: str, client,
                  seed_feedback: str | None = None) -> dict:
    prompt = designer_prompt(difficulty, analysis)
    if seed_feedback:
        prompt += ("\n\n--- REVISION: your previous chart drew this critique; address "
                   "it while keeping every budget rule ---\n" + seed_feedback)
    return client.generate_json(prompt, audio_path=audio)


# --------------------------------------------------------------------------- #
# Chunked mode (docs/chunked-mode-brief.md)
# --------------------------------------------------------------------------- #
def chunk_designer_prompt(difficulty: str, analysis: dict, chunk,
                          prior_summary: str) -> str:
    """One chunk's designer prompt.

    ORDER IS LOAD-BEARING (REQ-CHUNK-09). Prompt caching works on a token PREFIX,
    and under chunking the prefix has a different shape than it does whole-song:
    the static brief, the craft rules, the exemplars and the tier budget are
    identical for every chunk of a given song and difficulty, so they go first
    and stay byte-identical; the prior-decision summary, the chunk's assignment,
    its density budget and its inventory come after, in increasing order of how
    much they vary. The audio is appended last by the LLM client, for the same
    reason — it is the single most variable part of the request, so nothing
    after it could ever be a cache hit.

    `analysis` is the SCOPED analysis for this chunk (`chunking.slice_analysis`),
    which is what makes `{density_budget}` and `{analysis}` render this chunk's
    budget and this chunk's inventory rather than the whole song's.
    """
    from ...chunking import chunk_brief

    tmpl = (_PROMPTS / "designer_chunk.md").read_text()
    exemplars = _PROMPTS / "exemplars.md"
    return (tmpl.replace("{difficulty}", difficulty)
                .replace("{budget}", _budget_text(difficulty))
                .replace("{exemplars}", exemplars.read_text() if exemplars.exists() else "")
                .replace("{prior_summary}", prior_summary)
                .replace("{chunk_brief}", chunk_brief(chunk, analysis))
                .replace("{density_budget}", _density_text(analysis, difficulty))
                .replace("{analysis}", analysis_block(difficulty, analysis)))


def cacheable_prefix(difficulty: str) -> str:
    """The part of a chunk prompt that is byte-identical across every chunk of a
    song at this difficulty.

    Exposed (not private) because that invariance is the whole basis of
    REQ-CHUNK-09's cache-hit claim, and a property that is asserted in a test is
    a property that cannot quietly regress. Mirrors `chunk_designer_prompt` by
    construction: it is the same template, cut at the first varying placeholder.
    """
    tmpl = (_PROMPTS / "designer_chunk.md").read_text()
    exemplars = _PROMPTS / "exemplars.md"
    head = tmpl.split("{prior_summary}")[0]
    return (head.replace("{difficulty}", difficulty)
                .replace("{budget}", _budget_text(difficulty))
                .replace("{exemplars}", exemplars.read_text() if exemplars.exists() else ""))


def design_chunk_intent(difficulty: str, scoped_analysis: dict, chunk,
                        audio: str, client, prior_summary: str,
                        feedback: str | None = None) -> dict:
    """One scoped designer call. Returns the raw (unscoped) model output."""
    prompt = chunk_designer_prompt(difficulty, scoped_analysis, chunk, prior_summary)
    if feedback:
        prompt += ("\n\n--- YOUR PREVIOUS ATTEMPT AT THESE BARS WAS REJECTED; FIX IT "
                   "AND RESUBMIT ---\n" + feedback)
    return client.generate_json(prompt, audio_path=audio)


def _dynamics_block(analysis: dict, metrics: dict | None, difficulty: str) -> str:
    """REQ-R2-DYN-04: hand the critic the MEASURED density-vs-energy correlation
    and the plan's per-phrase verdict, so its judgement cites a number instead of
    an impression. The critic is still Author≠Judge — it did not design this chart
    and is not being told what to conclude, only what was measured."""
    if not metrics:
        return ""
    rho = metrics.get("density_energy_spearman")
    gate = config.DENSITY_ENERGY_SPEARMAN_MIN
    lines = ["\n--- MEASURED DYNAMICS (computed, not your impression) ---"]
    if rho is None:
        lines.append("density-vs-energy Spearman: not computable for this track "
                     "(too few bars or a flat energy curve).")
    else:
        verdict = "PASSES" if rho >= gate else "FAILS"
        lines.append(
            f"density-vs-energy Spearman rho = {rho:+.3f} (gate >= {gate:.2f}: "
            f"{verdict}). This is the correlation between notes-per-bar and the "
            f"song's measured per-bar energy.")
    dp = metrics.get("density_plan") or {}
    if dp.get("plan_available"):
        lines.append(
            f"density plan: {dp.get('in_band', 0)}/{len(dp.get('spans', []))} phrases "
            f"inside their notes-per-bar budget.")
        for r in (dp.get("under_dense") or [])[:4]:
            lines.append(f"  UNDER budget: bars {r['bars'][0]}-{r['bars'][1]} have "
                         f"{r['measured']:.1f} notes/bar, budget "
                         f"{r['band'][0]:.1f}-{r['band'][1]:.1f}")
        for r in (dp.get("over_dense") or [])[:4]:
            lines.append(f"  OVER budget: bars {r['bars'][0]}-{r['bars'][1]} have "
                         f"{r['measured']:.1f} notes/bar, budget "
                         f"{r['band'][0]:.1f}-{r['band'][1]:.1f}")
    lines.append(
        "In your reply, set `measured_rho` to the rho value above (copy it exactly; "
        "if it was not computable use null) and make your `dynamics` comment "
        "consistent with these measurements — do not contradict them by ear.")
    return "\n".join(lines)


def critic_review(track_id: str, difficulty: str, analysis: dict, placements: list,
                  audio: str, client, metrics: dict | None = None) -> dict:
    """Fresh-context critic (Author≠Judge, REQ-SM-13): render the realized chart
    UNAMBIGUOUSLY (holds as explicit start→end spans so a post-hold tap is never
    misread as a concurrent conflict) and have Gemini judge musicality/playability."""
    from .grammar import PANELS
    bpm, offset = analysis["bpm"], analysis["offset"]

    def t(beat):
        return round(offset + beat / bpm * 60, 3)

    rows = []
    for p in placements[:600]:
        panels = "".join(PANELS[c] for c in p.panels)
        if p.hold_beats:
            rows.append(f"{t(p.beat):.3f}: HOLD {panels} until {t(p.beat + p.hold_beats):.3f}")
        elif len(p.panels) > 1:
            rows.append(f"{t(p.beat):.3f}: JUMP {panels}")
        else:
            rows.append(f"{t(p.beat):.3f}: tap {panels}")
    prompt = (
        f"You are a StepMania (ITG) chart critic. You are HEARING the audio and "
        f"reading a realized {difficulty} dance-single chart. Each line is "
        f"'<time_s>: EVENT panels'. A 'HOLD X until T' occupies panel X from its "
        f"time until T — any later 'tap X' AFTER that end time is a normal, legal "
        f"step, NOT a conflict; do not flag it. Note that the validator already "
        f"guarantees no note lands on a panel while it is held, and jumps are <=2 "
        f"panels. You did NOT design this — judge it fresh: do steps land on the "
        f"music, is the foot flow comfortable, does density track energy, does the "
        f"drop hit?\n"
        f"Chart:\n" + "\n".join(rows) + "\n"
        + _dynamics_block(analysis, metrics, difficulty) + "\n\n"
        f"Reply ONLY JSON: {{\"score\": <0-10>, \"verdict\": \"<one line>\", "
        f"\"measured_rho\": <the rho value above, or null>, "
        f"\"dynamics\": \"<one line on whether density tracks the song's energy>\", "
        f"\"issues\": [{{\"where\":\"..\",\"problem\":\"..\"}}]}}")
    review = client.generate_json(prompt, audio_path=audio)
    # The measurement is ours, not the model's: overwrite whatever it echoed so a
    # QA report can never carry a hallucinated correlation. Keeping the model's
    # value alongside makes a mismatch visible instead of silently corrected.
    if metrics is not None:
        echoed = review.get("measured_rho")
        truth = metrics.get("density_energy_spearman")
        review["measured_rho"] = truth
        if echoed is not None and truth is not None and abs(float(echoed) - truth) > 0.01:
            review["measured_rho_echoed_by_critic"] = echoed
    return review
