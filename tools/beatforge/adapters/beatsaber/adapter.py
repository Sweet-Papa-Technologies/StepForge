"""adapter.py — BeatSaberAdapter: the TargetAdapter #2 for Beat Saber Standard
mode (SABERFORGE spec §2, REQ-ARCH-01). Ties design (intent) -> realize (parity
DP) -> validate/repair -> simulate -> serialize (v3 via JSMap) -> QA + external
referees, touching nothing outside adapters/beatsaber/.

Positioning (spec §0, REQ-POS-01..03): every output is an AI-assisted *draft* for
ChroMapper refinement — labelled in metadata, own-music-first, and there is NO
BeatSaver upload/publish path anywhere in this package.
"""
from __future__ import annotations

import json
from pathlib import Path

from ... import chunking, config, ledger, llm
from ...analyze import analyze_track
from . import check, difficulty as diffmod, qa as bs_qa, simulate
from .grammar import BUDGETS, DIFFICULTIES, grammar_description
from .lighting import basic_lighting
from .resolve import (IntentError, deterministic_intent, resolve_intent)
from .realize import realize as realize_objects
from .serialize import AI_MARKER, write_song_folder
from .validate import validate_repair


class BeatSaberAdapter:
    name = "beatsaber"

    def grammar(self) -> dict:
        return grammar_description()

    def design_brief(self, analysis: dict, difficulty: str) -> str:
        from .design import designer_prompt
        return designer_prompt(difficulty, analysis)

    def realize(self, design: dict, analysis: dict, difficulty: str) -> list:
        resolved = resolve_intent(design, analysis, difficulty)
        return realize_objects(resolved, analysis, difficulty, BUDGETS[difficulty])

    def validate_repair(self, placements, analysis, difficulty):
        rr = validate_repair(placements, analysis, difficulty)
        return rr.objects, {"repairs": rr.repairs, "ok": rr.ok, "original": rr.original}

    def serialize(self, per_difficulty, analysis, track_id, out_dir):
        meta = _track_meta(track_id, analysis)
        return write_song_folder(meta, analysis, per_difficulty,
                                 _audio_path(track_id), Path(out_dir))

    def qa_metrics(self, placements, analysis, difficulty):
        return bs_qa.chart_metrics(placements, analysis, difficulty)


# --------------------------------------------------------------------------- #
# Orchestration (REQ-BS-13): analysis -> design -> realize -> validate ->
# simulate -> serialize -> QA + referees, one track. Deterministic by default;
# the Gemini designer + critic run only with a client.
# --------------------------------------------------------------------------- #
COPYRIGHT_CAVEAT = (
    "[saberforge] COPYRIGHT: --i-have-rights acknowledged. SABERFORGE is a "
    "mapper's drafting assistant for YOUR OWN music (paid-plan Suno / original). "
    "Mapping third-party copyrighted audio is your responsibility; these drafts "
    "are AI-assisted and are NOT for BeatSaver upload.")


def build_song(track_id: str, opts: config.RunOptions, difficulties=DIFFICULTIES,
               deterministic: bool = True, client=None, i_have_rights: bool = False) -> dict:
    # Attribute every model call under this block to this song and target
    # (REQ-R2-COST-01). SABERFORGE calls used to land in `_unattributed`, which
    # made a Beat Saber run invisible to `cost-report` — and in hybrid mode the
    # stage is also the routing key, so an unattributed call is an UNROUTED one.
    with ledger.stage("song", song=track_id, target="beatsaber"):
        return _build_song(track_id, opts, difficulties, deterministic, client,
                           i_have_rights)


def _build_song(track_id, opts, difficulties, deterministic, client,
                i_have_rights) -> dict:
    if i_have_rights:
        print(COPYRIGHT_CAVEAT)
    adapter = BeatSaberAdapter()
    analysis = analyze_track(track_id, opts)
    bpm = analysis["bpm"]
    out_dir = config.SABERFORGE_DIR / config.TRACK_CATALOGUE.get(track_id, track_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    njs_table = diffmod.njs_offset_table(difficulties, bpm)
    per_difficulty = {}
    report = {"track_id": track_id, "bpm": bpm, "offset": analysis["offset"],
              # The backend that actually designed this, not a hardcoded guess —
              # see the same fix in the stepmania adapter.
              "mode": ("deterministic" if (deterministic or client is None)
                       else getattr(client, "name", None) or "llm"),
              "credit": AI_MARKER, "ai_assisted": True,
              # `chunk_mode` is the SETTING; `chunked` per chart is what actually
              # happened (deterministic never chunks, and `auto` declines on a
              # short track). REQ-CHUNK-10 compares the two modes side by side,
              # and a report that does not say which mode made it cannot go in
              # that table.
              "chunk_mode": config.CHUNK_MODE,
              "own_music_default": True, "auto_publish": False, "charts": {}}

    for diff in difficulties:
        best = _design_once(adapter, track_id, diff, analysis, deterministic, client, None)

        if not deterministic and client is not None:
            best["critic"] = _run_critic(track_id, diff, analysis, best["objects"],
                                         client, chunk_run=best.get("chunk_run"))
            if float((best["critic"] or {}).get("score", 0)) < config.CRITIC_SHIP_THRESHOLD:
                issues = "; ".join(f"{i.get('where','')}: {i.get('problem','')}"
                                   for i in (best["critic"] or {}).get("issues", []))
                revised = _design_once(adapter, track_id, diff, analysis, False, client,
                                       f"A critic raised these issues — fix them while keeping "
                                       f"every budget rule: {issues}")
                revised["critic"] = _run_critic(track_id, diff, analysis,
                                                revised["objects"], client, pass_no=2,
                                                chunk_run=revised.get("chunk_run"))
                if float(revised["critic"].get("score", 0)) >= float(best["critic"].get("score", 0)):
                    best = revised
                if float(best["critic"].get("score", 0)) < config.CRITIC_SHIP_THRESHOLD:
                    best["critic"]["human_review"] = True
            (out_dir / f"{diff}.critic.json").write_text(json.dumps(best["critic"], indent=2))

        (out_dir / f"{diff}.design.json").write_text(json.dumps(best["design"], indent=2))

        njs, offset, rank = njs_table[diff]
        objs = best["objects"]
        lighting = basic_lighting(analysis)
        per_difficulty[diff] = {"difficulty": BUDGETS[diff].difficulty_name,
                                "rank": rank, "njs": njs, "offset": offset,
                                "objects": objs, "lighting": lighting}

        sim = simulate.simulate(objs, analysis)
        previews = bs_qa.render_previews(objs, analysis, _audio_path(track_id), out_dir / diff)
        report["charts"][diff] = {
            "notes": len([o for o in objs if o.kind in ("note", "arc", "chain")]),
            "bombs": sum(1 for o in objs if o.kind == "bomb"),
            "arcs": sum(1 for o in objs if o.kind == "arc"),
            "chains": sum(1 for o in objs if o.kind == "chain"),
            "njs": njs, "offset": offset, "difficulty_rank": rank,
            "repaired": best["vinfo"]["original"] - len([o for o in objs if o.kind in ("note", "arc", "chain")]),
            "repair_ok": best["vinfo"]["ok"], "simulator_clean": sim.clean,
            "forced_resets": len(sim.forced_resets),
            "critic": best.get("critic"), "previews": previews,
            "metrics": best["metrics"],
            "chunked": best.get("chunk_run") is not None,
            "chunks": (len(best["chunk_run"].chunks)
                       if best.get("chunk_run") is not None else 0),
            "chunk_calls": (best["chunk_run"].calls
                            if best.get("chunk_run") is not None else 0),
            # Bars the mapper could not design, charted from the DSP instead.
            # Empty on a healthy run; present so a degraded section is auditable.
            "chunk_fallback_bars": (best["chunk_run"].fallback_bars()
                                    if best.get("chunk_run") is not None else [])}

    files = adapter.serialize(per_difficulty, analysis, track_id, out_dir)
    report["files"] = files.get("files", {})
    report["jsmap_verified"] = files.get("verified", False)
    report["fallback_serializer"] = files.get("fallback", False)

    # triple-referee legality (REQ-BS-09): in-core validator + simulator + external.
    externals = check.run_external_checkers(str(out_dir))
    in_core_ok = all(c["repair_ok"] for c in report["charts"].values())
    sim_clean = all(c["simulator_clean"] for c in report["charts"].values())
    report["referees"] = check.referee_summary(in_core_ok, sim_clean, externals)

    report["njs_constant"] = all(
        report["charts"][d]["njs"] == njs_table[d][0] for d in difficulties)
    report["difficulty_monotonic"] = diffmod.is_monotonic(difficulties)
    report["out"] = str(out_dir.relative_to(config.REPO_ROOT))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _design_once(adapter, track_id, diff, analysis, deterministic, client, seed_feedback):
    use_llm = not (deterministic or client is None)
    # CHUNKED MODE (REQ-CHUNK-08), the default. Only the DESIGNER call changes:
    # the chunks are assembled into a whole-song design before anything below
    # this point runs, so realize/validate/simulate/QA are identical in both
    # modes. The deterministic path never chunks — it makes no model calls, so
    # there is no context to keep small.
    use_chunked = use_llm and chunking.should_chunk(analysis)
    attempts = config.DESIGN_MAX_ATTEMPTS if use_llm else 1
    feedback = seed_feedback
    design, objects, last_err = {}, None, None
    drafted_on = None            # hybrid tier that authored the design we kept
    chunk_run = None             # the per-chunk record, for scoped repair + critic
    for attempt in range(attempts):
        if not use_llm:
            design = deterministic_intent(analysis, diff)
        elif use_chunked:
            from .chunked import design_chunked
            # Stage names are set per chunk inside design_chunked (each chunk is
            # its own `designer`/`reprompt` call), so there is no outer stage
            # here — one would re-label every chunk call as a single designer
            # call and lose the per-chunk cost attribution the ledger exists for.
            chunk_run = design_chunked(track_id, diff, analysis,
                                       _audio_path(track_id), client,
                                       seed_feedback=feedback)
            design = chunk_run.design(analysis)
            drafted_on = llm.tier_for("designer" if attempt == 0 else "reprompt",
                                      attempt + 1)
        else:
            from .design import design_intent
            # The stage name is both the cost bucket and, in hybrid mode, the
            # routing key: "designer" is the lite first draft, "reprompt" the
            # heavy repair (config.ROLE_TIERS).
            stage_name = "designer" if (attempt == 0 and not seed_feedback) else "reprompt"
            with ledger.stage(stage_name, difficulty=diff, attempt=attempt + 1):
                design = design_intent(track_id, diff, analysis, _audio_path(track_id),
                                       client, seed_feedback=feedback)
            drafted_on = llm.tier_for(stage_name, attempt + 1)
        try:
            objects = adapter.realize(design, analysis, diff)
            break
        except IntentError as e:
            last_err = e
            feedback = (f"Your previous output was rejected: {e}. Reference onsets by "
                        f"their EXACT id or grid:<beat>, kind must be note/arc/chain/"
                        f"bomb_reset, hand must be left/right/either, and NEVER include a "
                        f"coordinate, cut direction, colour or parity. Fix and resubmit.")
            objects = None
    if objects is None:
        raise last_err or RuntimeError("design failed")
    objects, vinfo = adapter.validate_repair(objects, analysis, diff)
    metrics = adapter.qa_metrics(objects, analysis, diff)

    # HYBRID CASCADE: a cheap first draft ships only if the deterministic
    # referees accept it — here that includes the swing simulator and the
    # unforced-reset count, which is precisely the "is it playable" question a
    # weaker model is most likely to get wrong. Otherwise re-author on the heavy
    # tier with the failures named.
    if drafted_on == "lite" and not all(metrics["gates"].values()):
        esc = _escalate_to_heavy(adapter, track_id, diff, analysis, client, metrics)
        if esc is not None:
            design, objects, vinfo, metrics = esc
            drafted_on = "heavy"

    # REQ-CHUNK-07 SCOPED RE-PROMPT: a failed gate re-prompts only the chunks
    # responsible, not the whole song. This is the payoff of keeping the per-chunk
    # outputs around — the alternative is a whole-song re-design, which throws
    # away every section that was already right.
    if chunk_run is not None and not seed_feedback:
        rep = _repair_failing_chunks(adapter, track_id, diff, analysis, client,
                                     chunk_run, objects, metrics)
        if rep is not None:
            design, objects, vinfo, metrics = rep

    metrics["designed_by_tier"] = drafted_on
    return {"design": design, "objects": objects, "vinfo": vinfo,
            "metrics": metrics, "critic": None,
            # The per-chunk record, so the critic can review per chunk the same
            # way the mapper wrote per chunk (REQ-CHUNK-06). None in whole-song
            # mode, which is what the critic branches on.
            "chunk_run": chunk_run}


def _repair_failing_chunks(adapter, track_id, diff, analysis, client, chunk_run,
                           objects, metrics):
    """REQ-CHUNK-07. Re-prompt the chunks a failed gate can be attributed to.

    Returns `(design, objects, vinfo, metrics)` if the repair produced a map that
    fails STRICTLY FEWER gates, else None. Same rule as the hybrid escalation, for
    the same reason: a re-draft that is not better is not worth the swap, so a
    failed repair costs a call and never a map.
    """
    from .chunked import chunk_gate_failures, reprompt_failing_chunks

    failed = [g for g, ok in metrics["gates"].items() if not ok]
    if not failed:
        return None
    failing = chunk_gate_failures(objects, analysis, diff, chunk_run.chunks, failed)
    if not failing:
        return None
    # Snapshot what we would be replacing: a repair that does not clear more
    # gates has to be undone, and `run` is mutated in place.
    before = list(chunk_run.scoped), list(chunk_run.decisions)
    if not reprompt_failing_chunks(chunk_run, track_id, diff, analysis,
                                   _audio_path(track_id), client, failing):
        return None
    try:
        design = chunk_run.design(analysis)
        objs = adapter.realize(design, analysis, diff)
        objs, rvinfo = adapter.validate_repair(objs, analysis, diff)
        rmetrics = adapter.qa_metrics(objs, analysis, diff)
    except Exception as e:                    # a failed repair is not fatal
        print(f"    [{diff}] scoped chunk repair failed ({e}); keeping the original",
              flush=True)
        chunk_run.scoped, chunk_run.decisions = before
        return None
    still_failing = sum(1 for ok in rmetrics["gates"].values() if not ok)
    if still_failing >= len(failed):
        print(f"    [{diff}] scoped chunk repair did not clear more gates; "
              f"keeping the original", flush=True)
        chunk_run.scoped, chunk_run.decisions = before
        return None
    print(f"    [{diff}] scoped chunk repair kept — {len(failed)} -> "
          f"{still_failing} gate failure(s)", flush=True)
    return design, objs, rvinfo, rmetrics


def _escalate_to_heavy(adapter, track_id, diff, analysis, client, metrics):
    """Re-author a gate-failing lite draft on the heavy tier. Kept only if it
    fails strictly fewer gates; otherwise the original stands, so a wasted
    escalation costs money but never quality."""
    failed = [g for g, ok in metrics["gates"].items() if not ok]
    print(f"    [{diff}] lite draft failed {', '.join(failed)} — escalating to "
          f"{config.HYBRID_HEAVY_MODEL}…", flush=True)
    brief = ("Your previous map failed these objective gates — fix them while "
             "keeping every budget rule: " + ", ".join(failed) + ". "
             "Remember you declare INTENT only (which onsets, kind, hand, feel); "
             "the parity realizer owns geometry, so fix these by changing WHICH "
             "onsets you take, how dense each phrase is, and how you alternate hands.")
    try:
        from .design import design_intent
        with ledger.stage("reprompt", difficulty=diff, attempt=1,
                          escalation="lite->heavy"):
            redesign = design_intent(track_id, diff, analysis, _audio_path(track_id),
                                     client, seed_feedback=brief)
        objs = adapter.realize(redesign, analysis, diff)
        objs, rvinfo = adapter.validate_repair(objs, analysis, diff)
    except Exception as e:
        print(f"    [{diff}] escalation failed ({e}); keeping the lite draft", flush=True)
        return None
    rmetrics = adapter.qa_metrics(objs, analysis, diff)
    if sum(1 for ok in rmetrics["gates"].values() if not ok) >= len(failed):
        print(f"    [{diff}] escalation did not clear more gates; keeping the lite draft",
              flush=True)
        return None
    return redesign, objs, rvinfo, rmetrics


def _run_critic(track_id, diff, analysis, objs, client, pass_no: int = 1,
                chunk_run=None) -> dict:
    try:
        # REQ-CHUNK-06: in chunked mode the critic reviews per chunk, scoped the
        # same way the mapper was, plus one final pass over the assembled map's
        # structure. It returns the SAME dict shape either way, so the revision
        # loop, the ship threshold and the QA report are identical.
        if chunk_run is not None:
            from .chunked import critic_chunked
            with ledger.stage("critic", difficulty=diff, attempt=pass_no):
                return critic_chunked(track_id, diff, analysis, objs,
                                      _audio_path(track_id), client, chunk_run)
        from .design import critic_review
        with ledger.stage("critic", difficulty=diff, attempt=pass_no):
            return critic_review(track_id, diff, analysis, objs, _audio_path(track_id), client)
    except Exception as e:
        return {"error": str(e)[:200], "score": 0}


def _track_meta(track_id: str, analysis: dict) -> dict:
    base = config.TRACK_CATALOGUE.get(track_id, track_id)
    title, artist = config.TRACK_META.get(track_id, (base, "beatforge"))
    return {"title": title, "artist": artist, "music": f"{base}.ogg"}


def _audio_path(track_id: str) -> str:
    base = config.TRACK_CATALOGUE.get(track_id, track_id)
    for root in (config.TRACKS_PUB, config.TRACKS_SRC):
        p = root / f"{base}.ogg"
        if p.exists():
            return str(p)
    raise FileNotFoundError(track_id)
