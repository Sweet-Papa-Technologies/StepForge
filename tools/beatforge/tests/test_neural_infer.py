"""REQ-SFN-01 / REQ-SFN-05 (2) / REQ-SFN-07 — the torch-free runtime.

Runs the real ONNX path against tests/fixtures/neural/tiny_model, a randomly
initialised export of a few thousand parameters. Its charts are musical
nonsense, which is the point: the legality guarantees below must hold whatever
the networks say.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from beatforge.adapters.stepmania.grammar import ALL_DIFFICULTIES, BUDGETS  # noqa: E402
from beatforge.adapters.stepmania.neural import codec as C  # noqa: E402
from beatforge.adapters.stepmania.neural import conditioning as cond  # noqa: E402
from beatforge.adapters.stepmania.neural import grid_features as GF  # noqa: E402
from beatforge.adapters.stepmania.neural import infer  # noqa: E402

TINY = Path(__file__).parent / "fixtures" / "neural" / "tiny_model"
TOOLS = Path(__file__).resolve().parents[2]
BPM, OFFSET, DUR = 150.0, 0.2, 40.0


def click_audio(bpm=BPM, dur=DUR, offset=OFFSET, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(int(dur * GF.SR), np.float32)
    t, i = offset, 0
    while t < dur - 0.1:
        k = int(t * GF.SR)
        amp = 1.0 if i % 2 == 0 else 0.5
        y[k:k + 60] += amp * np.hanning(120)[60:].astype(np.float32) * rng.standard_normal(60)
        t += 60.0 / bpm / 2
        i += 1
    return y / np.abs(y).max()


@pytest.fixture(scope="module")
def engine():
    return infer.StepManiaNeuralEngine(TINY)


@pytest.fixture(scope="module")
def grid():
    return GF.extract_grid_features(click_audio(), GF.TimingMap.constant(BPM, OFFSET))


def fake_logits(t: int, seed: int = 0, density: float = 0.5) -> dict:
    """Plausible-looking RhythmNet output: strong on quarters, weaker on 8ths
    and 16ths, noise elsewhere."""
    rng = np.random.default_rng(seed)
    k = np.arange(t)
    base = np.full(t, -5.0)
    base[k % 3 == 0] = -1.5
    base[k % 6 == 0] = 0.5
    base[k % 12 == 0] = 2.5
    onset = base + rng.normal(0, 1.0, t) + np.log(density / (1 - density))
    return {"onset": onset.astype(np.float32),
            "jump": rng.normal(-1.0, 1.5, t).astype(np.float32),
            "hold_type": rng.normal(0, 1.0, (t, 6)).astype(np.float32),
            "hold_dur": rng.normal(0, 1.0, (t, 10)).astype(np.float32),
            "mine": rng.normal(-2.0, 1.0, t).astype(np.float32)}


def tick_times(t: int, bpm: float = BPM) -> np.ndarray:
    return OFFSET + np.arange(t) * 60.0 / bpm / 12


# --------------------------------------------------------------------------- #
# REQ-SFN-01: runtime dependency isolation
# --------------------------------------------------------------------------- #
def test_inference_runs_end_to_end_with_torch_blocked(tmp_path):
    """A clean interpreter in which `import torch` fails generates all five
    difficulties from audio and writes a .ssc that simfile re-opens."""
    script = textwrap.dedent(f"""
        import sys
        import importlib.abc

        class _NoTorch(importlib.abc.MetaPathFinder):
            # `import torch` raises, without leaving a placeholder in
            # sys.modules (SciPy's array-API helpers inspect that).
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ModuleNotFoundError("torch is blocked", name=name)
                return None

        sys.meta_path.insert(0, _NoTorch())
        import json
        import numpy as np
        import simfile
        from beatforge.adapters.stepmania.grammar import ALL_DIFFICULTIES
        from beatforge.adapters.stepmania.neural import conditioning as cond
        from beatforge.adapters.stepmania.neural import grid_features as GF
        from beatforge.adapters.stepmania.neural import infer, pipeline
        from beatforge.adapters.stepmania.serialize import build_simfile
        from beatforge.adapters.stepmania.difficulty import compute_meter

        rng = np.random.default_rng(3)
        y = np.zeros(int(40 * GF.SR), np.float32)
        for b in range(190):
            k = int((0.2 + b * 0.2) * GF.SR)
            y[k:k + 60] += rng.standard_normal(60).astype(np.float32)
        g = GF.extract_grid_features(y, GF.TimingMap.constant(150.0, 0.2))
        eng = infer.StepManiaNeuralEngine({str(TINY)!r})
        analysis = {{"bpm": 150.0, "offset": 0.2, "duration_s": 40.0, "sections": []}}
        charts = {{}}
        for d in ALL_DIFFICULTIES:
            ch = eng.generate(g.X, g.tick_times_sec, d, cond.resolve_style(d), 40.0, seed=1)
            pl = infer.to_placements(ch)
            charts[d] = (compute_meter(pl, 150.0, d), pl)
        sf = build_simfile({{"title": "t", "artist": "a", "music": "t.ogg"}}, analysis, charts,
                           subdiv={{d: pipeline.TICK_SUBDIV for d in charts}})
        path = {str(tmp_path / "t.ssc")!r}
        open(path, "w").write(str(sf))
        pipeline.stabilize_simfile(path)
        text = open(path).read()
        assert str(simfile.loads(text)) == text        # byte-stable round trip
        bad = [m for m in sys.modules if m == "torch" or m.startswith("torch.")]
        assert not bad, bad
        train = [m for m in sys.modules if ".neural.train" in m]
        assert not train, train
        print(json.dumps({{d: len(p) for d, (_, p) in charts.items()}}))
    """)
    res = subprocess.run([sys.executable, "-W", "ignore", "-c", script], capture_output=True,
                         text=True, env={"PYTHONPATH": str(TOOLS), "PATH": "/usr/bin:/bin"})
    assert res.returncode == 0, res.stderr[-2000:]
    counts = json.loads(res.stdout.strip().splitlines()[-1])
    assert set(counts) == set(ALL_DIFFICULTIES)
    assert (tmp_path / "t.ssc").exists()


def test_runtime_modules_do_not_import_the_training_package():
    import ast
    neural = TOOLS / "beatforge" / "adapters" / "stepmania" / "neural"
    for path in neural.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert name.split(".")[0] != "torch", f"{path.name} imports torch"
                assert "train" not in name.split("."), f"{path.name} imports {name}"


# --------------------------------------------------------------------------- #
# status / missing weights
# --------------------------------------------------------------------------- #
def test_status_reports_an_installed_model():
    s = infer.neural_status(TINY)
    assert s["installed"] is True and s["error"] is None
    assert s["model_dir"] == str(TINY) and s["quantization"] == "fp32"
    assert s["presets"] == list(cond.PRESET_NAMES)
    assert s["codec_version"] == "1.0" and s["trained_with_stems"] is False
    assert {f["key"] for f in s["style_fields"]} == {
        "nps", "stream_share", "jump_share", "hold_share", "crossover_share",
        "jack_share", "footswitch_share", "mine_share"}
    assert isinstance(s["style_rag_available"], bool)
    json.dumps(s)                                   # the CLI prints it as JSON


def test_missing_weights_fail_with_an_actionable_error(tmp_path, monkeypatch):
    s = infer.neural_status(tmp_path)
    assert s["installed"] is False and "model_meta.json" in s["error"]
    with pytest.raises(infer.ModelNotInstalled) as e:
        infer.StepManiaNeuralEngine(tmp_path)
    assert str(tmp_path) in str(e.value) and "--engine" in str(e.value)
    monkeypatch.delenv(infer.ENV_MODEL_DIR, raising=False)
    assert infer.resolve_model_dir().parts[-2:] == ("models", "stepforge_flow")
    monkeypatch.setenv(infer.ENV_MODEL_DIR, str(tmp_path))
    assert infer.resolve_model_dir() == tmp_path
    assert infer.resolve_model_dir("/x/y") == Path("/x/y")       # the flag wins


def test_incompatible_model_is_refused(tmp_path):
    meta = json.loads((TINY / infer.META_FILE).read_text())
    for name in meta["files"].values():
        (tmp_path / name).write_bytes((TINY / name).read_bytes())
    meta["codec_version"] = "0.9"
    (tmp_path / infer.META_FILE).write_text(json.dumps(meta))
    s = infer.neural_status(tmp_path)
    assert s["installed"] is False and "codec" in s["error"]
    with pytest.raises(ValueError, match="codec"):
        infer.StepManiaNeuralEngine(tmp_path)


# --------------------------------------------------------------------------- #
# Stage A decode: masks, calibration, budgets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("difficulty", ALL_DIFFICULTIES)
def test_decode_respects_the_tier(difficulty):
    t = 48 * 60
    tt = tick_times(t)
    dur = float(tt[-1]) + 2.0
    b = BUDGETS[difficulty]
    plan = infer.decode_rhythm(fake_logits(t, seed=5), tt, difficulty,
                               cond.resolve_style(difficulty), dur)
    n = len(plan.tick)
    assert n > 0 and np.all(np.diff(plan.tick) > 0)
    # legal subdivisions only, inside the playable window
    assert set((plan.tick % 12).tolist()) <= set(infer._SUBDIVISIONS[difficulty])
    assert plan.tick.min() >= 48 and tt[plan.tick.max()] <= dur - infer.PLAYABLE_TAIL_S
    # physical gap and peak rate
    gaps = np.diff(tt[plan.tick])
    assert gaps.min() >= infer.min_gap_s(difficulty) - 1e-9
    assert cond.peak_nps(tt[plan.tick]) <= b.max_nps_4s + 1e-9
    # shares
    assert plan.is_jump.sum() <= infer._JUMP_MAX_FRAC[b.jumps] * n + 1
    holds = plan.flavor > 0
    assert holds.sum() <= b.hold_share[1] * n + 1
    fine = (plan.tick % 12 != 0).sum()
    assert plan.stats["fine_share_met"] and fine <= b.fine_frac * n + 1
    # streams no longer than the tier allows
    run = longest = 1
    for g in np.diff(plan.tick):
        run = run + 1 if g <= 6 else 1
        longest = max(longest, run)
    assert longest <= infer.RUN_LIMIT[difficulty]


def test_threshold_calibration_hits_the_nps_slider():
    t = 48 * 80
    tt = tick_times(t)
    logits = fake_logits(t, seed=2)
    got = {}
    for nps in (2.0, 3.5, 5.0):
        style = cond.resolve_style("challenge", overrides={"nps": nps})
        plan = infer.decode_rhythm(logits, tt, "challenge", style, float(tt[-1]) + 2)
        got[nps] = cond.sustained_nps(tt[plan.tick])
        assert got[nps] == pytest.approx(nps, abs=0.5), (nps, got[nps])
        assert infer.THETA_RANGE[0] <= plan.stats["theta"] <= infer.THETA_RANGE[1]
    assert got[2.0] < got[3.5] < got[5.0]            # monotone in the slider


def test_unconditioned_density_follows_the_models_expected_count():
    t = 48 * 60
    tt = tick_times(t)
    style = cond.tier_only_vector("challenge")
    sparse = infer.decode_rhythm(fake_logits(t, 1, density=0.2), tt, "challenge", style,
                                 float(tt[-1]) + 2)
    dense = infer.decode_rhythm(fake_logits(t, 1, density=0.7), tt, "challenge", style,
                                float(tt[-1]) + 2)
    assert len(sparse.tick) < len(dense.tick)
    # the count that ships is the model's own expectation wherever the tier's
    # limits and the threshold range leave room for it
    assert sparse.stats["notes"] == pytest.approx(sparse.stats["expected_count"], rel=0.35)


def test_density_follows_energy_when_given():
    t = 48 * 64
    tt = tick_times(t)
    logits = fake_logits(t, seed=4)
    energy = np.concatenate([np.full(32, 0.2), np.full(32, 0.9)])     # quiet, then loud
    style = cond.resolve_style("hard", overrides={"nps": 3.0})
    plan = infer.decode_rhythm(logits, tt, "hard", style, float(tt[-1]) + 2, energy)
    first = (plan.tick < 32 * 48).sum()
    assert (plan.tick >= 32 * 48).sum() > first


def test_freezes_never_enclose_a_jump_or_another_freeze():
    t = 48 * 60
    tt = tick_times(t)
    style = cond.resolve_style("challenge", overrides={"hold_share": 0.06,
                                                       "jump_share": 0.12, "nps": 6.0})
    for seed in range(6):
        plan = infer.decode_rhythm(fake_logits(t, seed), tt, "challenge", style,
                                   float(tt[-1]) + 2)
        for i in np.nonzero(plan.flavor > 0)[0]:
            end = plan.tick[i] + plan.dur[i]
            inside = (plan.tick > plan.tick[i]) & (plan.tick <= end)
            assert not plan.is_jump[inside].any()
            assert not (plan.flavor[inside] > 0).any()
            assert inside.sum() <= infer._MAX_NOTES_IN_HOLD["challenge"]
            assert plan.dur[i] >= BUDGETS["challenge"].hold_len_beats[0] * 12


# --------------------------------------------------------------------------- #
# REQ-SFN-05 (2): constrained Viterbi legality for ANY logits
# --------------------------------------------------------------------------- #
def random_plan(rng, difficulty: str, n_ticks: int = 48 * 40) -> infer.RhythmPlan:
    tt = tick_times(n_ticks)
    style = cond.resolve_style(difficulty, overrides={"hold_share": 0.3, "jump_share": 0.3})
    return infer.decode_rhythm(fake_logits(n_ticks, int(rng.integers(1 << 30))), tt,
                               difficulty, style, float(tt[-1]) + 2)


def solve(plan: infer.RhythmPlan, unary, difficulty, tt):
    rules = infer.TierRules.for_difficulty(difficulty)
    dt = np.concatenate([[1e9], np.diff(tt[plan.tick] * 1000.0)])
    slots, kind = infer.lock_schedule(plan.tick, plan.is_jump, plan.flavor, plan.dur)
    res = infer.enforce_run_limit(
        dict(unary=unary, dtb=C.dt_bucket(dt), is_jump=plan.is_jump, slots=slots,
             hold_kind=kind, rules=rules), plan.tick, rules)
    enc = infer._encoded_from(plan, res.token, res.f_init, np.zeros((0, 2), np.int32))
    return res, infer.NeuralChart(C.decode(enc), enc, {})


@pytest.mark.parametrize("difficulty", ALL_DIFFICULTIES)
def test_viterbi_is_legal_for_random_logits(difficulty):
    rng = np.random.default_rng(11)
    rules = infer.TierRules.for_difficulty(difficulty)
    tt = tick_times(48 * 40)
    for trial in range(5):
        plan = random_plan(rng, difficulty)
        n = len(plan.tick)
        unary = rng.normal(0, 4.0, (n, 20))            # adversarial: pure noise
        res, chart = solve(plan, unary, difficulty, tt)
        st = infer.chart_stats(chart, tt)
        assert res.relaxed == 0
        assert st["lock_violations"] == 0, "a foot holding a freeze moved"
        assert st["spins"] == 0, "a >180 degree twist"
        # tokens stay in the class Stage A assigned
        assert np.array_equal(res.token >= 8, plan.is_jump)
        if rules.no_double_step:
            assert st["double_steps"] == 0
        if rules.no_cross:
            assert st["crossed"] == 0
        if rules.no_footswitch:
            assert st["footswitches"] == 0
        # no run of identical rows beyond the tier's limit
        rows = infer.token_rows(res.token, res.f_init)
        run = longest = 1
        for a, b in zip(rows, rows[1:]):
            run = run + 1 if a == b else 1
            longest = max(longest, run)
        assert longest <= rules.panel_run_limit
        # and the result is a real chart: it re-encodes losslessly
        enc2, info = C.encode(chart.events, tt[plan.tick] * 1000.0)
        assert C.decode(enc2).equals(chart.events) and info.lock_violations == 0


def test_viterbi_never_moves_a_holding_foot_even_when_the_logits_beg():
    # left foot starts a 2-beat freeze; logits scream for ALT (the holding foot)
    ticks = np.array([48, 54, 60, 66, 72, 96], np.int64)
    plan = infer.RhythmPlan(ticks, np.zeros(6, bool),
                            np.array([C.HOLD_ALT, 0, 0, 0, 0, 0]),
                            np.array([24, 0, 0, 0, 0, 0]), np.zeros(0, np.int64))
    unary = np.full((6, 20), -20.0)
    unary[:, :4] = 0.0                                  # only ALT tokens look good
    tt = tick_times(200)
    for difficulty in ("easy", "hard"):
        res, chart = solve(plan, unary, difficulty, tt)
        before, _, _ = C.foot_path(chart.encoded)
        holder = 1 - (int(before[0]) & 1) if res.token[0] < 4 else int(before[0]) & 1
        moves = C.tables()["moves_l" if holder == C.LEFT else "moves_r"]
        held_panel = int(chart.events.p0[0])
        for i in range(1, 5):                           # notes at ticks 54..72 <= 48+24
            assert not moves[int(before[i]), int(res.token[i])]
            assert held_panel not in infer.token_rows(res.token, res.f_init)[i]
        assert infer.chart_stats(chart, tt)["lock_violations"] == 0


def test_viterbi_prefers_the_models_tokens_when_they_are_legal():
    ticks = np.arange(48, 48 + 16 * 3, 3, dtype=np.int64)
    n = len(ticks)
    plan = infer.RhythmPlan(ticks, np.zeros(n, bool), np.zeros(n, np.int64),
                            np.zeros(n, np.int64), np.zeros(0, np.int64))
    want = np.array([C.HOME, C.DOWN, C.UP, C.HOME] * 4)        # a clean stream
    unary = np.full((n, 20), -8.0)
    unary[np.arange(n), want] = 0.0
    res, _ = solve(plan, unary, "hard", tick_times(200))
    assert np.array_equal(res.token, want)


def test_initial_parity_is_chosen_or_honoured():
    plan = random_plan(np.random.default_rng(2), "hard")
    n = len(plan.tick)
    unary = np.random.default_rng(3).normal(0, 2, (n, 20))
    rules = infer.TierRules.for_difficulty("hard")
    tt = tick_times(48 * 40)
    dt = np.concatenate([[1e9], np.diff(tt[plan.tick] * 1000.0)])
    slots, kind = infer.lock_schedule(plan.tick, plan.is_jump, plan.flavor, plan.dur)
    args = dict(unary=unary, dtb=C.dt_bucket(dt), is_jump=plan.is_jump, slots=slots,
                hold_kind=kind, rules=rules)
    free = infer.constrained_viterbi(**args)
    left = infer.constrained_viterbi(f_init=C.LEFT, **args)
    right = infer.constrained_viterbi(f_init=C.RIGHT, **args)
    assert left.f_init == C.LEFT and right.f_init == C.RIGHT
    assert free.score == pytest.approx(max(left.score, right.score))


# --------------------------------------------------------------------------- #
# the engine end to end (tiny ONNX model)
# --------------------------------------------------------------------------- #
def test_generate_all_difficulties_is_legal_and_reproducible(engine, grid):
    counts = {}
    for d in ALL_DIFFICULTIES:
        style = cond.resolve_style(d)
        a = engine.generate(grid.X, grid.tick_times_sec, d, style, DUR, seed=7)
        b = engine.generate(grid.X, grid.tick_times_sec, d, style, DUR, seed=7)
        assert a.events.equals(b.events), "same seed must give the same chart"
        st = infer.chart_stats(a, grid.tick_times_sec)
        assert st["spins"] == 0 and st["lock_violations"] == 0
        if d in ("beginner", "easy", "medium"):
            assert st["double_steps"] == 0
        assert a.stats["choreo"]["relaxed"] == 0
        counts[d] = len(a.events)
        assert counts[d] > 0
        assert cond.peak_nps(grid.tick_times_sec[a.events.tick]) <= BUDGETS[d].max_nps_4s
    # a different seed re-rolls the choreography but not the rhythm
    hard = cond.resolve_style("hard")
    a = engine.generate(grid.X, grid.tick_times_sec, "hard", hard, DUR, seed=1)
    b = engine.generate(grid.X, grid.tick_times_sec, "hard", hard, DUR, seed=2)
    assert np.array_equal(a.events.tick, b.events.tick)
    assert not a.events.equals(b.events)


def test_windowed_inference_matches_shapes_on_a_long_song(engine):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((infer.WINDOW_TICKS * 2 + 480, 64)).astype(np.float32)
    out = engine.model.rhythm(x, cond.cond19(cond.tier_only_vector("hard")))
    assert out["onset"].shape == (len(x),) and out["hold_type"].shape == (len(x), 6)
    assert np.isfinite(out["onset"]).all()
    n = infer.EVENT_WINDOW + 300
    logits = engine.model.choreo(rng.standard_normal((n, 17, 64)).astype(np.float32),
                                 rng.standard_normal((n, 31)).astype(np.float32),
                                 np.full(n, C.MASK), cond.cond19(cond.tier_only_vector("hard")))
    assert logits.shape == (n, 20) and np.isfinite(logits).all()


def test_placements_serialize_with_no_orphan_holds(engine, grid, tmp_path):
    import simfile
    from simfile.notes import NoteData
    from beatforge.adapters.stepmania.neural import pipeline
    from beatforge.adapters.stepmania.serialize import build_simfile
    style = cond.resolve_style("challenge", overrides={"hold_share": 0.06, "jump_share": 0.1})
    chart = engine.generate(grid.X, grid.tick_times_sec, "challenge", style, DUR, seed=5)
    placements = infer.to_placements(chart)
    analysis = {"bpm": BPM, "offset": OFFSET, "duration_s": DUR, "sections": []}
    for ssc in (True, False):
        sf = build_simfile({"title": "t", "artist": "a", "music": "t.ogg"}, analysis,
                           {"challenge": (11, placements)}, ssc=ssc,
                           subdiv={"challenge": pipeline.TICK_SUBDIV})
        path = tmp_path / ("t.ssc" if ssc else "t.sm")
        path.write_text(str(sf))
        pipeline.stabilize_simfile(path)
        text = path.read_text()
        assert str(simfile.loads(text)) == text                  # byte-stable round trip
        open_heads, heads, tails = {}, 0, 0
        rows = {}
        for n in NoteData(simfile.loads(text).charts[0]):
            ch = n.note_type.value
            rows.setdefault(n.beat, []).append(n.column)
            if ch in "24":
                assert n.column not in open_heads
                open_heads[n.column] = n.beat
                heads += 1
            elif ch == "3":
                assert n.column in open_heads, "tail without a head"
                del open_heads[n.column]
                tails += 1
        assert heads == tails and not open_heads, "orphan hold"
        assert all(len(cols) <= 2 for cols in rows.values())
        # every row is on the 48-tick grid
        assert all((b * 12).denominator == 1 for b in rows)
    # what was written is exactly the chart the engine produced
    triples, dropped, snapped = C.simfile_chart_to_ticks(
        simfile.loads(text).charts[0], lambda beat: BPM)
    back, st = C.events_from_notes(triples)
    assert dropped == snapped == st.hands == 0 and back.equals(chart.events)


def test_meters_are_made_strictly_monotonic():
    from beatforge.adapters.stepmania.neural.pipeline import _strict_meters
    out = _strict_meters({"beginner": 2, "easy": 2, "medium": 5, "hard": 9, "challenge": 9})
    vals = [out[d] for d in ALL_DIFFICULTIES]
    assert all(a < b for a, b in zip(vals, vals[1:])), vals
    for d, v in out.items():
        lo, hi = BUDGETS[d].meter_range
        assert lo - 1 <= v <= hi


# --------------------------------------------------------------------------- #
# inpainting
# --------------------------------------------------------------------------- #
def test_inpaint_preserves_everything_outside_the_section(engine, grid):
    style = cond.resolve_style("hard", overrides={"nps": 4.0})
    base = engine.generate(grid.X, grid.tick_times_sec, "hard", style, DUR, seed=3)
    last_bar = int(base.events.tick[-1]) // 48
    start, end = 6, min(12, last_bar)
    inside = (base.events.tick >= start * 48) & (base.events.tick < end * 48)
    assert 4 < inside.sum() < len(base.events)
    seen = set()
    for seed in (1, 2, 3, 4):
        out = engine.inpaint_section(base.events, grid.X, grid.tick_times_sec, "hard",
                                     style, start, end, seed=seed)
        new, old = out.events.rows(), base.events.rows()
        assert len(new) == len(old)
        assert np.array_equal(out.events.tick, base.events.tick)     # rhythm untouched
        for keep, a, b in zip(~inside, new, old):
            if keep:
                assert a == b, "a note outside the section changed"
        assert out.stats["changed"] == sum(a != b for a, b in zip(new, old))
        st = infer.chart_stats(out, grid.tick_times_sec)
        assert st["spins"] == 0 and st["lock_violations"] == 0
        seen.add(tuple(map(str, new)))
    assert len(seen) > 1, "different seeds should re-roll the section differently"
    # with the project's flow gate as referee, the re-rolled chart still passes it
    from beatforge.adapters.stepmania import footflow as ff
    from beatforge.adapters.stepmania import qa as sm_qa
    from beatforge.adapters.stepmania.neural import pipeline
    veto = pipeline.flow_veto("hard")
    clean = engine.generate(grid.X, grid.tick_times_sec, "hard", style, DUR, seed=3,
                            veto=veto)
    out = engine.inpaint_section(clean.events, grid.X, grid.tick_times_sec, "hard", style,
                                 start, end, seed=9, veto=veto)
    worst, _ = sm_qa._flow_costs(infer.to_placements(out), BUDGETS["hard"])
    assert worst < ff.DOUBLE_STEP + ff.JACK + 1e-6


def test_inpaint_of_an_empty_range_is_a_no_op(engine, grid):
    style = cond.resolve_style("easy")
    base = engine.generate(grid.X, grid.tick_times_sec, "easy", style, DUR, seed=3)
    out = engine.inpaint_section(base.events, grid.X, grid.tick_times_sec, "easy", style,
                                 500, 510, seed=1)
    assert out.events.equals(base.events) and out.stats["changed"] == 0


def test_parse_bars():
    from beatforge.adapters.stepmania.neural.pipeline import parse_bars
    assert parse_bars("16:24") == (16, 24)
    for bad in ("24:16", "5", "a:b", "-1:4", "3:3"):
        with pytest.raises(ValueError):
            parse_bars(bad)


# --------------------------------------------------------------------------- #
# REQ-SFN-07: the CLI end to end, and chartbench compatibility
# --------------------------------------------------------------------------- #
@pytest.fixture
def cli_track(tmp_path, monkeypatch):
    """A catalogue track backed by synthetic audio, with every output directory
    redirected into tmp_path."""
    import soundfile as sf
    from beatforge import config
    from beatforge.adapters.stepmania import qa as sm_qa
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    sf.write(tracks / "nrl_click.ogg", click_audio(), GF.SR, format="OGG", subtype="VORBIS")
    monkeypatch.setattr(config, "TRACKS_PUB", tracks)
    monkeypatch.setattr(config, "TRACKS_SRC", tracks)
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path / "analysis")
    monkeypatch.setattr(config, "STEPMANIA_DIR", tmp_path / "stepmania")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(config.TRACK_CATALOGUE, "nrl_click", "nrl_click")
    monkeypatch.setitem(config.TRACK_META, "nrl_click", ("Neural Click", "Tests"))
    # previews shell out to matplotlib + ffmpeg; not what this test is about
    monkeypatch.setattr(sm_qa, "render_previews", lambda *a, **k: {})
    from beatforge.adapters.stepmania.neural import pipeline
    monkeypatch.setattr(pipeline, "render_previews", lambda *a, **k: {})
    monkeypatch.delenv("BEATFORGE_ENGINE", raising=False)
    monkeypatch.delenv(infer.ENV_MODEL_DIR, raising=False)
    return tmp_path / "stepmania" / "nrl_click"


def test_cli_neural_run_emits_five_monotonic_charts(cli_track, capsys):
    import simfile
    from simfile.notes import NoteData
    from beatforge.__main__ import main
    main(["stepforge", "--track", "nrl_click", "--engine", "neural",
          "--difficulties", "beginner,easy,medium,hard,challenge",
          "--model-dir", str(TINY), "--style-json", '{"hold_share": 0.05}'])
    out = capsys.readouterr().out
    assert "neural: rhythm" in out and "neural: choreography" in out
    assert "[stepforge] nrl_click (neural, mono=True)" in out

    report = json.loads((cli_track / "report.json").read_text())
    assert report["mode"] == "neural" and report["meter_monotonic"] is True
    assert list(report["charts"]) == list(ALL_DIFFICULTIES)
    meters = [report["charts"][d]["meter"] for d in ALL_DIFFICULTIES]
    assert all(a < b for a, b in zip(meters, meters[1:])), meters
    for d in ALL_DIFFICULTIES:
        row = report["charts"][d]
        qa = json.loads((cli_track / f"{d}.qa.json").read_text())
        assert qa["metrics"]["gates"] == row["metrics"]["gates"]
        assert set(row["metrics"]["gates"]) >= {"onset_alignment", "panel_balance",
                                                "hold_share", "flow_ceiling"}
        assert row["notes"] > 0 and row["bars"] > 0
        assert row["engine"]["name"] == "neural"
        assert row["engine"]["style"]["hold_share"] is not None
        assert row["metrics"]["neural"]["spins"] == 0
        assert row["metrics"]["neural"]["lock_violations"] == 0
        assert (cli_track / f"{d}.design.json").exists()

    for ext in ("ssc", "sm"):
        path = cli_track / f"nrl_click.{ext}"
        text = path.read_text()
        sf = simfile.loads(text)
        assert str(sf) == text, f".{ext} is not byte-stable through simfile"
        assert [c.difficulty for c in sf.charts] == ["Beginner", "Easy", "Medium", "Hard",
                                                     "Challenge"]
        assert [int(c.meter) for c in sf.charts] == meters
        for ch in sf.charts:                       # zero orphan holds, legal rows
            open_heads = set()
            for n in NoteData(ch):
                assert (n.beat * 12).denominator == 1
                if n.note_type.value in "24":
                    open_heads.add(n.column)
                elif n.note_type.value == "3":
                    assert n.column in open_heads
                    open_heads.discard(n.column)
            assert not open_heads
    assert (cli_track / "nrl_click.ogg").exists()

    # chartbench (the benchmark harness, not part of every checkout) reads it
    # without any schema change
    if not (TOOLS / "chartbench" / "sm_parse.py").is_file():
        return
    sys.path.insert(0, str(TOOLS / "chartbench"))
    try:
        import sm_parse
        from beatforge.adapters.stepmania.qa import chart_metrics
        bpm, offset, charts = sm_parse.parse(cli_track / "nrl_click.sm")
        assert len(charts) == 5 and bpm == pytest.approx(report["bpm"], abs=0.01)
        analysis = json.loads(next((cli_track.parents[1] / "analysis").glob(
            "*.analysis.json")).read_text())
        for ch in charts:
            m = chart_metrics(ch.placements, analysis, ch.difficulty.lower())
            assert m["notes"] == report["charts"][ch.difficulty.lower()]["notes"]
    finally:
        sys.path.remove(str(TOOLS / "chartbench"))


def test_cli_inpaint_rerolls_one_section_and_nothing_else(cli_track, capsys):
    import simfile
    from beatforge.__main__ import main
    main(["stepforge", "--track", "nrl_click", "--engine", "neural",
          "--difficulties", "easy,hard", "--model-dir", str(TINY),
          "--style-json", '{"nps": 4.0}'])
    ssc = cli_track / "nrl_click.ssc"
    before = simfile.loads(ssc.read_text())
    before_report = json.loads((cli_track / "report.json").read_text())
    capsys.readouterr()
    main(["stepforge-inpaint", "--track", "nrl_click", "--difficulty", "hard",
          "--bars", "4:10", "--seed", "42", "--model-dir", str(TINY)])
    out = capsys.readouterr().out
    assert "[inpaint] nrl_click/hard bars 4:10" in out
    after = simfile.loads(ssc.read_text())
    assert str(after) == ssc.read_text()
    easy_b, hard_b = before.charts
    easy_a, hard_a = after.charts
    assert str(easy_a.notes) == str(easy_b.notes), "the other chart must not change"
    mb, ma = str(hard_b.notes).split(","), str(hard_a.notes).split(",")
    assert len(mb) == len(ma)
    changed = [i for i, (x, y) in enumerate(zip(mb, ma)) if x.strip() != y.strip()]
    assert changed and all(4 <= i < 10 for i in changed), changed
    # the .sm twin got the same edit, and the report recorded it
    sm_after = simfile.loads((cli_track / "nrl_click.sm").read_text())
    assert str(sm_after.charts[1].notes).strip() == str(hard_a.notes).strip()
    report = json.loads((cli_track / "report.json").read_text())
    assert report["charts"]["hard"]["engine"]["inpainted"][0]["bars"] == [4, 10]
    assert report["charts"]["hard"]["engine"]["inpainted"][0]["seed"] == 42
    assert report["charts"]["easy"] == before_report["charts"]["easy"]
    # re-rolled in the style it was generated with (recorded in the report)
    assert report["charts"]["hard"]["engine"]["style"]["nps_norm"] == pytest.approx(
        before_report["charts"]["hard"]["engine"]["style"]["nps_norm"])


def test_cli_engine_resolution(cli_track, monkeypatch, tmp_path):
    from beatforge.__main__ import main, resolve_engine
    empty = tmp_path / "no_model"
    assert resolve_engine("auto", False, empty) == "llm"
    assert resolve_engine("auto", True, empty) == "deterministic"
    assert resolve_engine("auto", False, TINY) == "neural"
    assert resolve_engine(None, False, TINY) == "neural"            # default is auto
    assert resolve_engine("llm", True, TINY) == "llm"
    assert resolve_engine("deterministic", False, TINY) == "deterministic"
    monkeypatch.setenv("BEATFORGE_ENGINE", "deterministic")
    assert resolve_engine(None, False, TINY) == "deterministic"
    monkeypatch.delenv("BEATFORGE_ENGINE")
    # neural without weights: a clear, actionable failure naming the directory
    with pytest.raises(SystemExit) as e:
        main(["stepforge", "--track", "nrl_click", "--engine", "neural",
              "--model-dir", str(empty)])
    assert str(empty) in str(e.value) and "model_meta.json" in str(e.value)
    # the legacy flag still runs the deterministic path, untouched
    main(["stepforge", "--track", "nrl_click", "--deterministic", "--difficulties", "easy",
          "--model-dir", str(TINY)])
    report = json.loads((cli_track / "report.json").read_text())
    assert report["mode"] == "deterministic"
    for bad in (["--style-json", "{not json"], ["--style-json", '{"jump_share": "x"}'],
                ["--style-json", '{"nope": 1}'], ["--style-preset", "nope"]):
        with pytest.raises(SystemExit):
            main(["stepforge", "--track", "nrl_click", "--engine", "neural",
                  "--model-dir", str(TINY), "--difficulties", "easy", *bad])


def test_cli_neural_status_json(capsys):
    from beatforge.__main__ import main
    assert main(["neural-status", "--json", "--model-dir", str(TINY)]) is None
    s = json.loads(capsys.readouterr().out)
    assert s["installed"] is True and s["presets"] and s["default_model_dir"]
    for key in ("installed", "model_dir", "quantization", "style_rag_available", "presets"):
        assert key in s                               # the spec §9.2 contract


def test_optional_hooks_are_none_safe(tmp_path, monkeypatch):
    from beatforge.adapters.stepmania.neural import decisions_hook as dh
    from beatforge.adapters.stepmania.neural import style_rag
    assert style_rag.load_index(tmp_path) is None
    monkeypatch.delenv(dh.ENV_URL, raising=False)
    assert dh.DecisionsClient.from_env() is None
    assert dh.score_chart(None, "x") is None and dh.choose_preset(None, "x") is None
    # a working endpoint (faked transport): typed answers, clamped and validated
    seen = {}

    def transport(body):
        seen.update(body)
        return {"decisions": [{"id": q["id"], "value": 14 if q["type"] == "score"
                               else "jumpstream"} for q in body["decisions"]]}
    client = dh.DecisionsClient("http://x/v1", transport=transport)
    assert dh.score_chart(client, "a chart") == 10.0           # clamped to 0..10
    assert seen["max_output_tokens"] == 0                      # zero output tokens
    assert dh.choose_preset(client, "a song") == "jumpstream"
    broken = dh.DecisionsClient("http://x/v1", transport=lambda b: {"nope": 1})
    assert dh.score_chart(broken, "x") is None                 # failure = no opinion

    # the phrase index: one matmul, nearest phrase first, per-difficulty rows
    emb = style_rag.GridStatEmbedder()
    rng = np.random.default_rng(0)
    cores = rng.standard_normal((6, style_rag.PHRASE_TICKS, 31)).astype(np.float32)
    vecs = np.stack([emb.embed_core(c) for c in cores])
    assert vecs.shape == (6, 256)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    styles = np.tile(np.arange(6, dtype=np.float32)[:, None] / 10, (1, 8))
    slot = np.array([3, 3, 3, 3, 1, 1], np.int8)                # hard x4, easy x2
    style_rag.StyleIndex(vecs, styles, slot, emb.name).save(style_rag.index_path(tmp_path))
    idx = style_rag.load_index(tmp_path)
    assert len(idx) == 6 and idx.vectors.dtype == np.float32 and idx.embedder == emb.name
    hit = idx.query(emb.embed_core(cores[2] + 0.01 * rng.standard_normal(cores[2].shape)),
                    "hard", k=2)
    assert int(hit["index"][0]) == 2 and hit["scores"][0] > 0.99
    assert all(slot[i] == 3 for i in hit["index"])             # never an easy row
    assert idx.query(vecs[0], "challenge") is None             # no rows for that tier
    over = idx.song_style(vecs[:2], "hard", k=1)
    assert set(over) <= set(style_rag.STYLE_KEYS) and over["nps"] == pytest.approx(0.05 * 16, abs=1e-2)
    assert infer.neural_status(tmp_path)["style_rag_available"] is True


# --------------------------------------------------------------------------- #
# fitting the project's existing QA gates, and true-audio sync
# --------------------------------------------------------------------------- #
def test_grid_is_seated_on_the_audio_not_on_the_dsp_offset():
    """BeatForge's DSP reports the beat grid early (frames stamped at their
    start). The neural pipeline must find where the beats really are."""
    from beatforge.adapters.stepmania.neural import pipeline
    true_offset, bpm = 0.50, 120.0
    y = click_audio(bpm=bpm, dur=30.0, offset=true_offset)
    env = GF.frame_features(y).onset_envelope
    for early in (0.072, 0.060, 0.085):                  # what the DSP might report
        a = pipeline.align_grid(env, bpm, true_offset - early, 30.0, "local_dsp")
        assert a["shift_s"] == pytest.approx(early, abs=0.008), (early, a)
        assert a["contrast"] > 1.5
    # a backend with no known bias is searched around zero
    a = pipeline.align_grid(env, bpm, true_offset - 0.01, 30.0, "madmom")
    assert a["expected_s"] == 0.0 and a["shift_s"] == pytest.approx(0.01, abs=0.008)


def test_the_real_dsp_is_early_and_the_pipeline_corrects_it(tmp_path):
    """End to end on the actual DSP: its offset for a click track is tens of
    ms before the clicks; offset + grid shift lands on them."""
    import soundfile as sf
    from beatforge import dsp
    from beatforge.adapters.stepmania.neural import pipeline
    true_offset, bpm = 0.50, 120.0
    path = tmp_path / "clicks.wav"
    sf.write(path, click_audio(bpm=bpm, dur=30.0, offset=true_offset), GF.SR)
    an = dsp.analyze_signal(str(path))
    period = 60.0 / an["bpm"]
    err = (an["offset"] - true_offset + period / 2) % period - period / 2
    assert -0.095 < err < -0.045, f"DSP grid error {err * 1000:.0f} ms"     # early
    g, align = pipeline._features(str(path), an["bpm"], an["offset"], an["duration_s"])
    fixed = (an["offset"] + align["shift_s"] - true_offset + period / 2) % period - period / 2
    assert abs(fixed) < 0.010, f"aligned grid error {fixed * 1000:.0f} ms"
    assert g.X.shape[1] == 64


def test_flow_veto_makes_charts_pass_the_projects_flow_gate(engine, grid):
    from beatforge.adapters.stepmania import footflow as ff
    from beatforge.adapters.stepmania import qa as sm_qa
    from beatforge.adapters.stepmania.neural import pipeline
    for d in ALL_DIFFICULTIES:
        style = cond.resolve_style(d, overrides={"nps": min(4.0, BUDGETS[d].max_nps_4s)})
        chart = engine.generate(grid.X, grid.tick_times_sec, d, style, DUR, seed=11,
                                veto=pipeline.flow_veto(d))
        worst, _ = sm_qa._flow_costs(infer.to_placements(chart), BUDGETS[d])
        assert worst < ff.DOUBLE_STEP + ff.JACK + 1e-6, (d, worst)
        assert pipeline.flow_veto(d)(chart) == []
        st = infer.chart_stats(chart, grid.tick_times_sec)
        assert st["spins"] == 0 and st["lock_violations"] == 0


def test_panel_balance_is_pulled_inside_the_gate(engine, grid):
    style = cond.resolve_style("hard", overrides={"nps": 5.0})

    def excess(chart):
        share = infer.panel_shares(chart.events)
        assert share.sum() == pytest.approx(1.0)
        return max(0.0, 0.15 - share.min(), share.max() - 0.35)

    # the fixture model is random and lopsided; balancing must not make it worse,
    # and must get a short chart to within a couple of notes of the band
    worse, better = [], []
    for seed in (1, 2, 3, 4):
        raw = engine.generate(grid.X, grid.tick_times_sec, "hard", style, DUR, seed=seed,
                              balance=False)
        bal = engine.generate(grid.X, grid.tick_times_sec, "hard", style, DUR, seed=seed)
        worse.append(excess(raw))
        better.append(excess(bal))
        assert excess(bal) <= excess(raw) + 1e-9
    assert max(better) <= 0.10 and sum(better) < sum(worse)
    # and it is the balance loop doing it: an all-HOME plan is rebalanced
    n = 80
    ticks = np.arange(48, 48 + n * 6, 6, dtype=np.int64)
    plan = infer.RhythmPlan(ticks, np.zeros(n, bool), np.zeros(n, np.int64),
                            np.zeros(n, np.int64), np.zeros(0, np.int64))
    unary = np.full((n, 20), -6.0)
    unary[:, C.HOME] = 0.0                               # wants L R L R forever
    tt = tick_times(48 + n * 6 + 48)
    res, lopsided = solve(plan, unary, "hard", tt)
    assert infer.panel_shares(lopsided.events)[[C.D, C.U]].sum() < 0.05
    rules = infer.TierRules.for_difficulty("hard")
    dt = np.concatenate([[1e9], np.diff(tt[ticks] * 1000.0)])
    res = infer.constrained_viterbi(unary, C.dt_bucket(dt), plan.is_jump,
                                    np.zeros(n, np.int64), np.zeros(n, np.int64), rules,
                                    panel_bias=np.array([-3.0, 3.0, 3.0, -3.0]))
    assert len(set(infer.token_rows(res.token, res.f_init))) >= 3


def test_notes_stay_on_transients_when_told_where_they_are():
    t = 48 * 60
    tt = tick_times(t)
    logits = fake_logits(t, seed=6)
    on = np.zeros(t, bool)
    on[::6] = True                                        # transients on 8ths only
    style = cond.resolve_style("challenge", overrides={"nps": 7.0})
    free = infer.decode_rhythm(logits, tt, "challenge", style, float(tt[-1]) + 2)
    held = infer.decode_rhythm(logits, tt, "challenge", style, float(tt[-1]) + 2,
                               on_transient=on)
    assert (~on[free.tick]).mean() > 0.15                 # unconstrained: plenty off
    assert (~on[held.tick]).mean() <= infer.MAX_OFF_TRANSIENT + 0.01
    assert len(held.tick) > 0.5 * len(free.tick)          # thinned, not erased


def test_transient_ticks_use_the_dsp_frame():
    from beatforge.adapters.stepmania.neural import pipeline
    an = {"bpm": 120.0, "offset": 0.1,
          "onsets": [{"time": 0.1 + 0.5 * k + (0.02 if k % 2 else 0.0)} for k in range(8)]}
    on = pipeline.transient_ticks(an, 96)
    assert on[0] and on[12] and on[24]                   # beats: an onset within 32 ms
    assert not on[6] and not on[3]                        # no onset near the 8th / 16th
    assert pipeline.transient_ticks({"bpm": 120.0, "offset": 0.0, "onsets": []}, 48).all()


def test_fast_previews_write_both_artifacts(engine, grid, tmp_path):
    import soundfile as sf
    from beatforge.adapters.stepmania.neural import pipeline
    audio = tmp_path / "a.ogg"
    sf.write(audio, click_audio(dur=20.0), GF.SR, format="OGG", subtype="VORBIS")
    chart = engine.generate(grid.X, grid.tick_times_sec, "hard",
                            cond.resolve_style("hard", overrides={"nps": 4.0}), 20.0, seed=1)
    out = pipeline.render_previews(infer.to_placements(chart),
                                   {"bpm": BPM, "offset": OFFSET, "sections": []},
                                   str(audio), tmp_path / "hard")
    assert out["assist"].endswith("hard.assist.ogg")
    y, sr = sf.read(out["assist"])
    assert abs(len(y) / sr - 20.0) < 0.5 and np.abs(y).max() > 0.1
    if "png" in out:                                      # matplotlib is optional
        assert Path(out["png"]).stat().st_size > 1000


def test_tempo_refinement_removes_drift():
    """The DSP's tempo is a few hundredths of a percent off, which is tens of ms
    of drift per minute. The refinement measures that drift against the onsets
    and takes the whole BPM the music was produced at when the fit supports it."""
    from beatforge.adapters.stepmania.neural import pipeline
    true_bpm, true_offset, dur = 110.0, 0.40, 60.0
    env = GF.frame_features(click_audio(bpm=true_bpm, dur=dur, offset=true_offset)
                            ).onset_envelope
    for wrong in (109.957, 110.06, 109.9):
        r = pipeline.refine_tempo(env, wrong, true_offset + 0.004, dur)
        assert r["snapped"] and r["bpm"] == true_bpm, (wrong, r)
        assert abs(r["offset"] - true_offset) < 0.010
        assert r["drift_ms_per_min"] == pytest.approx((true_bpm / wrong - 1) * 60000, abs=1)
    # a tempo that is not round is left exactly as the DSP measured it
    env = GF.frame_features(click_audio(bpm=127.37, dur=dur, offset=true_offset)
                            ).onset_envelope
    r = pipeline.refine_tempo(env, 127.40, true_offset, dur)
    assert not r["snapped"] and r["bpm"] == 127.40 and r["offset"] == true_offset
    # too few onsets to judge: untouched
    r = pipeline.refine_tempo(np.zeros(3000), 120.0, 0.1, 30.0)
    assert r["bpm"] == 120.0 and not r["snapped"]


def sixteenth_audio(bpm=120.0, dur=40.0, offset=0.30, seed=5) -> np.ndarray:
    """Clicks on every 16th, accented beat > 8th > 16th: a rhythm that fills its
    own grid, so a wrong tempo's grid visibly misses most of it."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(dur * GF.SR), np.float32)
    step, i = 15.0 / bpm, 0
    while offset + i * step < dur - 0.1:
        k = int((offset + i * step) * GF.SR)
        amp = (1.0, 0.35, 0.6, 0.35)[i % 4]
        y[k:k + 60] += amp * np.hanning(120)[60:].astype(np.float32) * rng.standard_normal(60)
        i += 1
    return y / np.abs(y).max()


def test_tempo_ratio_errors_are_caught_and_right_tempos_left_alone():
    """dsp.py sometimes locks onto 4/3 or 3/2 of the real tempo. The onsets then
    mostly miss its grid, and the grid of the right ratio catches them."""
    from beatforge.adapters.stepmania.neural import pipeline
    true_bpm, true_offset, dur = 120.0, 0.30, 40.0
    env = GF.frame_features(sixteenth_audio(true_bpm, dur, true_offset)).onset_envelope
    beat = 60.0 / true_bpm
    for wrong in (160.0, 90.0, 180.0, 80.0):                 # x4/3, x3/4, x3/2, x2/3
        r = pipeline.verify_tempo(env, wrong, 0.1, dur)
        assert r["ratio"] != 1.0, (wrong, r)
        assert r["bpm"] == pytest.approx(true_bpm, rel=0.002), (wrong, r)
        assert r["fit"] > r["dsp_fit"] + pipeline.TEMPO_SWITCH_MARGIN
        assert r["dsp_fit"] <= pipeline.TEMPO_SUSPECT_RATIO * r["fit"]
        # beat 0 lands on a real (accented) beat, not on the off-beat 8th
        phase = (r["offset"] - true_offset + beat / 2) % beat - beat / 2
        assert abs(phase) < 0.012, (wrong, r)
    # the right tempo, and its octaves, are not second-guessed
    for fine in (120.0, 60.0, 240.0):
        r = pipeline.verify_tempo(env, fine, true_offset, dur)
        assert r["ratio"] == 1.0 and r["bpm"] == fine and r["offset"] == true_offset, r
    # nothing to go on: untouched
    r = pipeline.verify_tempo(np.zeros(4000), 133.0, 0.2, 40.0)
    assert r["ratio"] == 1.0 and r["bpm"] == 133.0


def test_rebar_moves_per_bar_data_onto_the_new_grid():
    from beatforge.adapters.stepmania.neural import pipeline
    an = {"bpm": 160.0, "offset": 0.0, "duration_s": 60.0, "density_plan": {"per_bar": [1]},
          "energy_curve": [float(i) for i in range(40)],        # 1.5 s bars at 160
          "sections": [{"start_bar": 0, "end_bar": 20}, {"start_bar": 20, "end_bar": 40}]}
    out = pipeline._rebar(an, 120.0, 0.0)                         # 2.0 s bars at 120
    assert len(out["energy_curve"]) == 31
    assert out["energy_curve"][0] == 0.0 and out["energy_curve"][3] == 4.0   # t = 7 s
    assert [s["start_bar"] for s in out["sections"]] == [0, 15]
    assert out["sections"][-1]["end_bar"] == 30 and out["density_plan"] == {}
    assert an["energy_curve"][3] == 3.0                           # input untouched
