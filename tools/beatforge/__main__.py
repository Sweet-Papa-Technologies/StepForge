"""beatforge CLI — `python -m beatforge {generate|analyze|chart|validate|qa|all}`.

Run from anywhere with `tools/` on PYTHONPATH, or `cd tools && python -m beatforge`.
Idempotent: stages skip work whose output exists unless --force.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import config
from .analyze import analyze_track, load_cached
from .compute import make_backend
from .validate import BeatmapError, parse_beatmap_py


def _opts(a) -> config.RunOptions:
    tracks = (a.track,) if getattr(a, "track", None) else tuple(config.TRACK_CATALOGUE)
    diffs = (a.difficulty,) if getattr(a, "difficulty", None) else config.DIFFICULTIES
    # CHUNKED MODE (REQ-CHUNK-08). Pinned on the module rather than threaded
    # through RunOptions because the designer surfaces read it directly and
    # RunOptions does not reach them — and because that keeps the CLI flag, the
    # GUI setting and the env var resolving to exactly one value that every
    # stage reads, instead of three that can disagree.
    chunk_mode = getattr(a, "chunk_mode", None)
    if chunk_mode:
        config.CHUNK_MODE = chunk_mode
    return config.RunOptions(
        backend=getattr(a, "backend", config.DEFAULT_BACKEND),
        force=getattr(a, "force", False),
        skip_gen=getattr(a, "skip_gen", True),
        allow_local_fallback=getattr(a, "allow_local_fallback", False),
        tracks=tracks, difficulties=diffs, gpu=getattr(a, "gpu", config.COLAB_GPU),
        offline=getattr(a, "offline", False),
    )


def cmd_analyze(a):
    opts = _opts(a)
    backend = make_backend(opts, run_id="cli")
    try:
        for tid in opts.tracks:
            an = analyze_track(tid, opts, backend=backend)
            print(f"[analyze] {tid}: bpm={an['bpm']} offset={an['offset']} "
                  f"onsets={len(an['onsets'])} holds={an.get('sustain_available')} "
                  f"backend={an['beat_backend']} stems={an['stem_source']}")
    finally:
        backend.close()


def cmd_chart(a):
    from .pipeline import chart_track
    from .llm import make_llm_client
    opts = _opts(a)
    client = make_llm_client()
    backend = make_backend(opts, run_id="cli")
    try:
        for tid in opts.tracks:
            summary = chart_track(tid, opts, client, backend=backend)
            for diff, info in summary["charts"].items():
                if "error" in info:
                    print(f"[chart] {tid}/{diff}: FAILED — {info['error'][:120]}")
                    continue
                print(f"[chart] {tid}/{diff}: {info['events']} events, "
                      f"{info['attempts']} attempt(s), critic={info['critic_score']}, "
                      f"gates_pass={info['gates_pass']} -> {info['map']}")
    finally:
        backend.close()


def cmd_validate(a):
    """Re-parse emitted maps with the schema-parity validator (REQ-VAL-01)."""
    opts = _opts(a)
    ok = True
    for tid in opts.tracks:
        base = config.TRACK_CATALOGUE.get(tid, tid)
        for diff in opts.difficulties:
            p = config.MAPS_PUB / f"{base}.{diff}.beatmap.json"
            if not p.exists():
                continue
            try:
                bm = parse_beatmap_py(json.loads(p.read_text()))
                print(f"[validate] {p.name}: OK ({len(bm['events'])} events)")
            except BeatmapError as e:
                ok = False
                print(f"[validate] {p.name}: FAIL {e}")
    sys.exit(0 if ok else 1)


def cmd_qa(a):
    opts = _opts(a)
    for tid in opts.tracks:
        base = config.TRACK_CATALOGUE.get(tid, tid)
        for diff in opts.difficulties:
            p = config.BUILD_DIR / f"{base}.{diff}.qa.json"
            if p.exists():
                r = json.loads(p.read_text())
                m = r["metrics"]
                print(f"[qa] {tid}/{diff}: align={m['onset_alignment']:.2f} "
                      f"nps={m['peak_nps_4s']:.1f} critic={(r.get('critic') or {}).get('score')} "
                      f"gates={all(m['gates'].values())}")


def cmd_generate(a):
    from .gen import generate_track
    from .vertex import VertexClient
    opts = _opts(a)
    client = VertexClient()
    for tid in opts.tracks:
        res = generate_track(tid, opts, client)
        print(f"[generate] {tid}: winner score={res['winner_score']:.1f} "
              f"rounds={res['rounds']} -> {res['audio']}")


def cmd_compare(a):
    """Benchmark the OpenAI-compatible model (e.g. local Gemma 4 12B) against
    Gemini 3.5 Flash on the audio-understanding probe + designer/critic. With
    --track it runs one track; without, it sweeps the whole catalogue."""
    from .compare import compare_track, format_report
    opts = _opts(a)
    tracks = (a.track,) if a.track else tuple(config.TRACK_CATALOGUE)
    diff = a.difficulty or "standard"
    reachable = _openai_reachable()
    if not reachable:
        print("[compare] WARNING: OpenAI-compatible server "
              f"{config.OPENAI_BASE_URL} is not reachable; the Gemma column will "
              "show errors. Gemini baseline still runs.")
    for tid in tracks:
        report = compare_track(tid, diff, opts, probe_only=a.probe_only)
        print(format_report(report))
        out = config.BUILD_DIR / f"compare.{config.TRACK_CATALOGUE.get(tid, tid)}.{diff}.json"
        out.write_text(json.dumps(report, indent=2))
    print(f"\nreports -> {config.BUILD_DIR.relative_to(config.REPO_ROOT)}/compare.*.json")


def cmd_inkling_check(a):
    """Resolve auth + the served model against a Modal Managed Endpoint.

    Emits ONE json object on stdout so the desktop app can render the result
    without scraping logs (REQ-INK-09). Never raises: a failure is a verdict,
    not a crash, and the message is the whole point of running this.
    """
    from .inkling import InklingClient, InklingError, scrub
    out = {"ok": False, "model": None, "credentialSource": None, "error": None}
    try:
        c = InklingClient()
        out["credentialSource"] = c.credential_source
        out["model"] = c.resolve_model()
        out["endpoint"] = c.base_url
        out["ok"] = True
    except (InklingError, Exception) as e:  # noqa: B014 - report anything readably
        out["error"] = scrub(str(e)[:1200], os.environ.get("MODAL_TOKEN_ID", ""),
                             os.environ.get("MODAL_TOKEN_SECRET", ""))
    print(json.dumps(out))
    return 0 if out["ok"] else 1


def cmd_inkling_probe(a):
    """Differential audio probe (REQ-INK-05), as json on stdout.

    This is the check that matters: an endpoint which silently ignores the audio
    part still answers confidently, so charts would come from a model that never
    heard the song. Two different clips, same prompt, temperature 0.
    """
    from .inkling import InklingClient, differential_audio_probe, scrub
    out = {"passed": False, "similarity": None, "reason": None,
           "responses": [], "model": None, "error": None}
    try:
        c = InklingClient()
        c.resolve_model()
        out.update(differential_audio_probe(c))
    except Exception as e:
        out["error"] = scrub(str(e)[:1200], os.environ.get("MODAL_TOKEN_ID", ""),
                             os.environ.get("MODAL_TOKEN_SECRET", ""))
        out["reason"] = out["error"]
    print(json.dumps(out))
    return 0 if out["passed"] else 1


def cmd_argon_check(a):
    """Gemini 4 Argon end to end, as ONE json object on stdout (the desktop
    Settings panel's "Test Argon" button shells out to this).

    Resolves the transport and credentials, then runs the same two-tone audio
    probe Inkling uses — two tiny calls at `low` thinking — because that one
    exchange proves all three things a first chart would otherwise discover the
    expensive way: the credentials work, this project can reach the model, and an
    audio part is accepted and heard. It also reports which tier the server
    actually served and what the check cost, from the ledger rows it wrote.
    """
    from . import ledger
    from .argon import make_client
    from .inkling import differential_audio_probe
    out = {"ok": False, "error": None, "probe": None}
    try:
        c = make_client()
        out.update(c.describe())
        with ledger.capture(write=True) as rows:
            verdict = differential_audio_probe(c)
        out["probe"] = {k: verdict.get(k) for k in
                        ("passed", "reason", "expected", "observed", "responses")}
        out["servedTier"] = next((r.get("service_tier") for r in rows
                                  if r.get("service_tier")), "standard")
        out["costUsd"] = round(sum((r.get("cost_usd") or {}).get("total", 0.0)
                                   for r in rows), 6)
        out["ok"] = bool(verdict.get("passed"))
        if not out["ok"]:
            out["error"] = verdict.get("reason")
    except Exception as e:  # noqa: BLE001 - a failed check is a verdict, not a crash
        out["error"] = str(e)[:1200]
    print(json.dumps(out))
    return 0 if out["ok"] else 1


def _openai_reachable() -> bool:
    from .llm import OpenAICompatClient
    try:
        OpenAICompatClient().list_models(timeout=6)
        return True
    except Exception:
        return False


def resolve_engine(engine: str | None, deterministic: bool, model_dir=None) -> str:
    """`--engine` -> the engine that will actually run (REQ-SFN spec §9.1).

      neural         the local ONNX model; an error if its weights are missing
      llm            the Gemini / Inkling / OpenAI designer path (unchanged)
      deterministic  the DSP-only path (unchanged)
      auto           neural if weights are installed, else llm — or
                     deterministic when --deterministic was passed

    `--deterministic` on its own keeps meaning what it always has. With no
    `--engine`, $BEATFORGE_ENGINE is consulted, then `auto`.
    """
    engine = (engine or os.environ.get("BEATFORGE_ENGINE", "").strip() or "auto").lower()
    if engine not in ("neural", "llm", "deterministic", "auto"):
        raise SystemExit(f"unknown engine '{engine}' (neural|llm|deterministic|auto)")
    if engine == "auto":
        if deterministic:
            return "deterministic"
        from .adapters.stepmania.neural import infer
        return "neural" if infer.is_installed(model_dir) else "llm"
    return engine


def _style_overrides(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--style-json is not valid JSON: {e}") from None
    if not isinstance(data, dict) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in data.values()):
        raise SystemExit("--style-json must be an object of slider name -> number")
    return data


def _stepforge_neural(a, tracks, diffs):
    from .adapters.stepmania.neural import infer, pipeline
    try:
        engine = infer.StepManiaNeuralEngine(a.model_dir, int8=a.int8)
    except infer.ModelNotInstalled as e:
        raise SystemExit(f"[stepforge] {e}") from None
    overrides = _style_overrides(a.style_json)
    opts = _opts(a)
    for tid in tracks:
        try:
            r = pipeline.build_song(tid, opts, difficulties=diffs, engine=engine,
                                    style_preset=a.style_preset, style_overrides=overrides,
                                    seed=a.seed, style_rag=a.style_rag)
        except ValueError as e:                       # bad preset / slider name
            raise SystemExit(f"[stepforge] {e}") from None
        cs = " ".join(f"{s}:{i['notes']}n/m{i['meter']}/gates{all(i['metrics']['gates'].values())}"
                      for s, i in r["charts"].items())
        print(f"[stepforge] {tid} ({r['mode']}, mono={r['meter_monotonic']}) -> {r['out']}  [{cs}]")


def cmd_stepforge(a):
    """STEPFORGE: export tracks as StepMania song folders (.ssc/.sm) via the
    TargetAdapter — DSP truth -> intent designer -> foot-flow realizer ->
    validate -> simfile. `--engine` picks who writes the chart: the local neural
    model, the LLM designer, or the DSP-only deterministic path."""
    from .adapters.stepmania.adapter import build_song
    from .adapters.stepmania.grammar import ALL_DIFFICULTIES, DIFFICULTIES
    tracks = (a.track,) if a.track else tuple(config.TRACK_CATALOGUE)
    diffs = tuple(a.difficulties.split(",")) if a.difficulties else DIFFICULTIES
    bad = [d for d in diffs if d not in ALL_DIFFICULTIES]
    if bad:
        raise SystemExit(f"unknown difficulty {', '.join(bad)} "
                         f"(choose from {', '.join(ALL_DIFFICULTIES)})")
    engine = resolve_engine(a.engine, a.deterministic, a.model_dir)
    if engine == "neural":
        return _stepforge_neural(a, tracks, diffs)
    opts = _opts(a)
    deterministic = engine == "deterministic"
    client = None
    if not deterministic:
        from .llm import make_llm_client
        client = make_llm_client()
    for tid in tracks:
        r = build_song(tid, opts, difficulties=diffs, deterministic=deterministic, client=client)
        cs = " ".join(f"{s}:{i['notes']}n/m{i['meter']}/gates{all(i['metrics']['gates'].values())}"
                      for s, i in r["charts"].items())
        print(f"[stepforge] {tid} ({r['mode']}, mono={r['meter_monotonic']}) -> {r['out']}  [{cs}]")


def cmd_stepforge_inpaint(a):
    """Re-roll a bar range of one neural chart in place (spec §6.5.4)."""
    from .adapters.stepmania.neural import infer, pipeline
    try:
        bars = pipeline.parse_bars(a.bars)
        r = pipeline.inpaint_song(a.difficulty, bars, track_id=a.track, ssc=a.ssc,
                                  seed=a.seed, model_dir=a.model_dir, int8=a.int8,
                                  style_preset=a.style_preset,
                                  style_overrides=_style_overrides(a.style_json))
    except (infer.ModelNotInstalled, FileNotFoundError, ValueError) as e:
        raise SystemExit(f"[inpaint] {e}") from None
    print(f"[inpaint] {r['track_id']}/{r['difficulty']} bars {bars[0]}:{bars[1]} -> "
          f"{r['changed']} notes changed (of {r['notes_in_section']} in the section, "
          f"seed {r['seed']}) -> {r['out']}")


def cmd_to_ogg(a):
    """Transcode an audio file to Ogg Vorbis without ffmpeg, through libsndfile
    (the `soundfile` wheels bundle it). Reads WAV, FLAC, OGG/Opus, AIFF and MP3;
    the desktop app uses this to import audio when ffmpeg is not installed.
    Prints one json object; exits 1 when the file cannot be read or written."""
    from .audio_convert import to_ogg
    try:
        info = to_ogg(a.src, a.out)
    except Exception as e:                       # noqa: BLE001 — reported, not raised
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return 1
    print(json.dumps({"ok": True, **info}))
    return 0


def cmd_neural_status(a):
    """Is the neural engine usable? One json object on stdout (the desktop
    Settings panel shells out to this). Always exits 0: a missing model is a
    verdict, not a crash."""
    from .adapters.stepmania.neural import infer
    status = infer.neural_status(a.model_dir)
    if a.json:
        print(json.dumps(status))
    else:
        print(f"installed: {status['installed']}\nmodel dir: {status['model_dir']}\n"
              f"quantization: {status['quantization']}\n"
              f"onnxruntime: {status['onnxruntime']}\n"
              f"style RAG index: {status['style_rag_available']}\n"
              f"presets: {', '.join(status['presets'])}"
              + (f"\n{status['error']}" if status["error"] else ""))
    return 0


def cmd_saberforge(a):
    """SABERFORGE: export tracks as Beat Saber Standard song folders (v3 .dat +
    Info.dat via BeatSaber-JSMap) through the TargetAdapter — DSP truth -> intent
    designer -> parity realizer -> validate -> simulate -> serialize -> QA +
    external referees. Output is an AI-assisted DRAFT for ChroMapper; there is no
    BeatSaver upload path (REQ-POS-01/02). Default uses the Gemini designer;
    --deterministic is DSP-only (offline)."""
    from .adapters.beatsaber.adapter import build_song
    from .adapters.beatsaber.grammar import DIFFICULTIES
    opts = _opts(a)
    tracks = (a.track,) if a.track else tuple(config.TRACK_CATALOGUE)
    diffs = tuple(a.difficulties.split(",")) if a.difficulties else DIFFICULTIES
    deterministic = a.deterministic
    client = None
    if not deterministic:
        from .llm import make_llm_client
        client = make_llm_client()
    for tid in tracks:
        r = build_song(tid, opts, difficulties=diffs, deterministic=deterministic,
                       client=client, i_have_rights=a.i_have_rights)
        cs = " ".join(f"{s}:{i['notes']}n/njs{i['njs']}/sim{i['simulator_clean']}/"
                      f"gates{all(i['metrics']['gates'].values())}"
                      for s, i in r["charts"].items())
        ver = "verified" if r["jsmap_verified"] else "UNVERIFIED"
        print(f"[saberforge] {tid} ({r['mode']}, mono={r['difficulty_monotonic']}, "
              f"njs_const={r['njs_constant']}, {ver}) -> {r['out']}  [{cs}]")


def cmd_cost_report(a):
    """REQ-R2-COST-03: roll build/cost/*/cost_ledger.jsonl into $/song, $/chart,
    $ by stage, the token breakdown, and the five priciest calls."""
    from .costreport import render_markdown, write_report
    rollup = write_report(song=getattr(a, "track", None))
    if rollup["totals"]["model_calls"] == 0 and rollup["totals"]["compute_events"] == 0:
        print(f"[cost-report] no ledger entries under "
              f"{config.COST_DIR.relative_to(config.REPO_ROOT)} — run a chart first.")
        return
    print(render_markdown(rollup) if a.stdout else
          f"[cost-report] {rollup['totals']['model_calls']} model call(s), "
          f"${rollup['totals']['usd']:.4f} total, "
          f"${rollup['totals']['usd_per_song']:.4f}/song, "
          f"${rollup['totals']['usd_per_chart']:.4f}/chart -> "
          f"{(config.COST_DIR / 'cost-report.md').relative_to(config.REPO_ROOT)}")


def cmd_all(a):
    from .pipeline import chart_track
    from .llm import make_llm_client
    opts = _opts(a)
    client = make_llm_client()          # charting: swappable (Gemini or Gemma)
    if not opts.skip_gen:
        # Generation (Lyria + A&R) is Vertex-only, independent of the chart backend.
        from .gen import generate_track
        from .vertex import VertexClient
        gen_client = VertexClient()
        for tid in opts.tracks:
            generate_track(tid, opts, gen_client)
    # ONE backend/session for the whole batch (REQ-COMPUTE-03)
    backend = make_backend(opts, run_id="batch")
    try:
        for tid in opts.tracks:
            summary = chart_track(tid, opts, client, backend=backend)
            print(f"[all] {tid}: " + ", ".join(
                f"{d}={i['events']}ev/critic{i['critic_score']}"
                for d, i in summary["charts"].items()))
    finally:
        backend.close()


def build_parser():
    p = argparse.ArgumentParser(prog="beatforge",
                                description="Music-aware track & chart pipeline for OVERDRIVE.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def chunk_flag(sp):
        """The one place the chunked-mode switch is declared, so `stepforge`,
        `saberforge` and `chart` cannot end up with three spellings of it."""
        sp.add_argument(
            "--chunk-mode", dest="chunk_mode", choices=["auto", "always", "never"],
            help=("design the song section by section (default: auto — chunks a "
                  "track longer than --max-chunk-seconds). `never` is whole-song "
                  "mode, for backend comparison. Env: BEATFORGE_CHUNK_MODE."))

    def common(sp, difficulty=True):
        sp.add_argument("--track", help="single track id (default: all)")
        if difficulty:
            sp.add_argument("--difficulty", choices=config.DIFFICULTIES)
        sp.add_argument("--backend", choices=["local", "colab"], default=config.DEFAULT_BACKEND)
        sp.add_argument("--gpu", default=config.COLAB_GPU)
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--allow-local-fallback", action="store_true", dest="allow_local_fallback")
        sp.add_argument("--offline", action="store_true", help="skip Vertex calls")
        chunk_flag(sp)

    common(sub.add_parser("analyze", help="Workstream B: DSP ground truth"), difficulty=False)
    common(sub.add_parser("chart", help="Workstreams C/D: design + validate + QA"))
    common(sub.add_parser("validate", help="re-parse emitted maps (schema parity)"))
    common(sub.add_parser("qa", help="print QA metrics for emitted charts"))
    g = sub.add_parser("generate", help="Workstream A: Lyria x Gemini A&R loop")
    common(g, difficulty=False)
    av = sub.add_parser("all", help="generate?(skip) -> analyze -> chart everything")
    common(av)
    av.add_argument("--skip-gen", action="store_true", default=True, dest="skip_gen",
                    help="keep existing audio (default)")
    av.add_argument("--with-gen", action="store_false", dest="skip_gen",
                    help="run the generation loop first")

    cr = sub.add_parser("cost-report",
                        help="roll cost ledgers into $/song, $/chart, $ by stage")
    cr.add_argument("--track", help="single track id (default: every ledger)")
    cr.add_argument("--stdout", action="store_true",
                    help="print the full markdown report instead of a one-line summary")

    cmp = sub.add_parser("compare", help="benchmark OpenAI-compatible model vs Gemini 3.5 Flash")
    common(cmp)
    cmp.add_argument("--probe-only", action="store_true", dest="probe_only",
                     help="only run the audio-understanding probe (skip full chart design)")

    sf = sub.add_parser("stepforge", help="export ITGmania .ssc/.sm song folders (StepMania adapter)")
    sf.add_argument("--track", help="single track id (default: all)")
    sf.add_argument("--difficulties", help="comma list e.g. easy,medium,hard (default: easy,medium,hard)")
    sf.add_argument("--deterministic", action="store_true",
                    help="DSP-only foot-flow placement, no LLM (offline)")
    sf.add_argument("--backend", choices=["local", "colab"], default=config.DEFAULT_BACKEND)
    sf.add_argument("--gpu", default=config.COLAB_GPU)
    sf.add_argument("--force", action="store_true")
    sf.add_argument("--allow-local-fallback", action="store_true", dest="allow_local_fallback")
    sf.add_argument("--offline", action="store_true")
    chunk_flag(sf)

    def neural_flags(sp):
        sp.add_argument("--style-preset", dest="style_preset",
                        help="neural: casual_flow | itg_stamina | tech_footswitch | jumpstream")
        sp.add_argument("--style-json", dest="style_json",
                        help='neural: slider overrides, e.g. \'{"jump_share":0.12,"nps":5}\'')
        sp.add_argument("--model-dir", dest="model_dir",
                        help="neural: exported model directory (default: "
                             "$BEATFORGE_NEURAL_MODEL_DIR or models/stepforge_flow)")
        sp.add_argument("--seed", type=int, help="neural: sampling seed")
        sp.add_argument("--int8", action="store_true",
                        help="neural: use the int8-quantized graphs if exported")

    sf.add_argument("--style-rag", dest="style_rag", action="store_true",
                    help="neural: fill unset sliders from the phrase style index, if "
                         "one is installed beside the model")
    sf.add_argument("--engine", choices=["neural", "llm", "deterministic", "auto"],
                    help="who writes the chart (default: auto — neural when its weights "
                         "are installed, else llm; env: BEATFORGE_ENGINE)")
    neural_flags(sf)

    ip = sub.add_parser("stepforge-inpaint",
                        help="re-roll a bar range of an existing neural chart in place")
    ip.add_argument("--track", help="track id (uses build/stepmania/<track>/)")
    ip.add_argument("--ssc", help="path to the .ssc to edit (alternative to --track)")
    ip.add_argument("--difficulty", required=True,
                    choices=["beginner", "easy", "medium", "hard", "challenge"])
    ip.add_argument("--bars", required=True, help="START:END, 0-based, END exclusive")
    neural_flags(ip)

    to = sub.add_parser("to-ogg", help="transcode audio to .ogg without ffmpeg (json)")
    to.add_argument("--in", dest="src", required=True)
    to.add_argument("--out", required=True)

    ns = sub.add_parser("neural-status", help="is the neural engine installed? (json)")
    ns.add_argument("--json", action="store_true")
    ns.add_argument("--model-dir", dest="model_dir")

    sb = sub.add_parser("saberforge", help="export Beat Saber Standard song folders (BeatSaber adapter)")
    sb.add_argument("--track", help="single track id (default: all)")
    sb.add_argument("--difficulties", help="comma list e.g. expert,hard (default: easy,normal,hard,expert,expertplus)")
    sb.add_argument("--deterministic", action="store_true",
                    help="DSP-only parity placement, no LLM (offline)")
    sb.add_argument("--i-have-rights", action="store_true", dest="i_have_rights",
                    help="acknowledge you hold the rights to map this audio (prints the copyright caveat)")
    sb.add_argument("--backend", choices=["local", "colab"], default=config.DEFAULT_BACKEND)
    sb.add_argument("--gpu", default=config.COLAB_GPU)
    sb.add_argument("--force", action="store_true")
    sb.add_argument("--allow-local-fallback", action="store_true", dest="allow_local_fallback")
    sb.add_argument("--offline", action="store_true")
    chunk_flag(sb)

    # Inkling/Modal configuration checks. Both print one json object and are what
    # the desktop Settings panel's "Test connection" / "Run audio probe" buttons
    # shell out to, so a user never has to open a terminal (REQ-INK-09).
    sub.add_parser("inkling-check",
                   help="resolve Modal proxy-token auth + the served model (json)")
    sub.add_parser("inkling-probe",
                   help="differential audio probe: does the endpoint actually listen? (json)")
    # Gemini 4 Argon: credentials + model access + audio, in one cheap check.
    sub.add_parser("argon-check",
                   help="Gemini 4 Argon: resolve transport/credentials and run the "
                        "two-tone audio probe (json)")

    for name, fn in {"analyze": cmd_analyze, "chart": cmd_chart, "validate": cmd_validate,
                     "qa": cmd_qa, "generate": cmd_generate, "all": cmd_all,
                     "compare": cmd_compare, "stepforge": cmd_stepforge,
                     "stepforge-inpaint": cmd_stepforge_inpaint,
                     "neural-status": cmd_neural_status,
                     "to-ogg": cmd_to_ogg,
                     "saberforge": cmd_saberforge,
                     "inkling-check": cmd_inkling_check,
                     "inkling-probe": cmd_inkling_probe,
                     "argon-check": cmd_argon_check,
                     "cost-report": cmd_cost_report}.items():
        sub.choices[name].set_defaults(fn=fn)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # A command that returns a code means it (the inkling checks) reports a
    # verdict rather than raising — propagate it so scripting and CI can branch
    # on the result instead of grepping stdout. Commands returning None are
    # unaffected.
    rc = args.fn(args)
    if isinstance(rc, int) and rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
