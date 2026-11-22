"""v1.6 choreography critic: its features say what they are named for, a chart
and its mirror read the same, the training negatives change arrows and nothing
else, and best-of-K is off unless a model ships a critic, keeps the notes and
stays legal."""
from __future__ import annotations

import json

import numpy as np
import pytest

from beatforge.adapters.stepmania.neural import codec as C
from beatforge.adapters.stepmania.neural import conditioning as cond
from beatforge.adapters.stepmania.neural import critic as CR
from beatforge.adapters.stepmania.neural import infer, pipeline
from beatforge.adapters.stepmania.neural.train import train_critic as TC

from test_v16_structure import TINY, _fake_rhythm, _grid, _song

G = C.G


def _chart(panels, step: int = 12):
    """Single notes `step` ticks apart on the given panels, read by the annotator."""
    n = len(panels)
    tick = np.arange(n, dtype=np.int32) * step
    ev = C.ChartEvents(tick, np.asarray(panels, np.int8), np.full(n, -1, np.int8),
                       np.zeros(n, np.int8), np.zeros(n, np.int8), np.zeros(n, np.int32),
                       np.zeros(n, np.int32), np.zeros((0, 2), np.int32))
    tms = tick * (60000.0 / 140 / C.TICKS_PER_BEAT)
    enc, _ = C.encode(ev, tms)
    return tick.astype(np.int64), tms, enc


def _features(panels, x=None, links=(), novelty=None, difficulty="hard", step: int = 12):
    tick, tms, enc = _chart(panels, step)
    x = np.zeros((int(tick[-1]) + G, 64), np.float32) if x is None else x
    prep = CR.prepare(x, tick, tms, enc.token >= 8, difficulty, links, novelty)
    return CR.chart_features(prep, enc.token, enc.f_init), prep, enc


def test_features_are_named_finite_and_survive_an_empty_chart():
    rng = np.random.default_rng(0)
    f, _, _ = _features(rng.integers(0, 4, 200))
    assert f.shape == (CR.N_FEATURES,) and len(set(CR.FEATURES)) == CR.N_FEATURES
    assert np.isfinite(f).all()
    assert f[CR.INDEX["is_hard"]] == 1.0 and f[CR.INDEX["is_easy"]] == 0.0
    prep = CR.prepare(np.zeros((96, 64), np.float32), np.zeros(0, np.int64), np.zeros(0),
                      np.zeros(0, bool), "easy")
    empty = CR.chart_features(prep, np.zeros(0, np.int64), C.LEFT)
    assert np.isfinite(empty).all() and empty[CR.INDEX["is_easy"]] == 1.0


def test_a_chart_and_its_mirror_score_the_same():
    rng = np.random.default_rng(1)
    panels = rng.integers(0, 4, 160)
    x = rng.normal(0, 1, (160 * 12 + G, 64)).astype(np.float32)
    f, _, _ = _features(panels, x)
    g, _, _ = _features(np.asarray(C.MIRROR_PANEL)[panels], x)
    assert np.allclose(f, g, atol=1e-6)


def test_a_loop_reads_as_a_loop_and_a_lopsided_chart_as_lopsided():
    rng = np.random.default_rng(2)
    varied = [0]
    while len(varied) < 192:                               # no arrow twice in a row
        q = int(rng.integers(0, 4))
        if q != varied[-1]:
            varied.append(q)
    v, _, _ = _features(varied)
    loop, _, _ = _features([0, 1, 3] * 64)
    i = CR.INDEX
    assert loop[i["tok4_entropy"]] < v[i["tok4_entropy"]] - 0.2
    assert loop[i["loop_run"]] > v[i["loop_run"]] + 0.5 and loop[i["period3"]] > 0.95
    assert loop[i["top4_share"]] > 3 * v[i["top4_share"]]
    lop = np.asarray(varied)
    lop[::2] = 0                                           # every other arrow Left
    f, _, _ = _features(lop)
    assert f[i["panel_max"]] > 0.45 and f[i["panel_max"]] > v[i["panel_max"]] + 0.1
    assert f[i["local_max_mean"]] > v[i["local_max_mean"]]
    jacks, _, _ = _features(np.repeat(varied[:96], 2), step=6)
    assert jacks[i["jack_share"]] > 0.4 > v[i["jack_share"]] + 0.3


def test_callback_features_count_arrows_that_come_back():
    rng = np.random.default_rng(3)
    bar = [0, 1, 3, 2]                                     # four quarter notes
    first = [int(q) for q in rng.integers(0, 4, 32)]
    links = [(b, b - 8, 0.9, 0) for b in range(8, 16)]     # bars 8..15 repeat bars 0..7
    echo, prep, _ = _features(first + first, links=links)
    other, _, _ = _features(first + [(q + 1) % 4 for q in first], links=links)
    i = CR.INDEX
    assert len(prep.link_later) == 32 and echo[i["has_links"]] == 1.0
    assert echo[i["cb_panel"]] == 1.0 and other[i["cb_panel"]] == 0.0
    assert echo[i["cb_token"]] > other[i["cb_token"]]
    assert echo[i["cb_coverage"]] == pytest.approx(0.5)
    none, _, _ = _features(bar * 16)
    assert none[i["has_links"]] == 0.0 and none[i["cb_token"]] == 0.0


def test_section_features_need_a_section_boundary():
    panels = [0, 3] * 32 + [1, 2] * 32                      # the footwork changes at bar 16
    nov = np.zeros(32, np.float32)
    nov[16] = 1.0
    f, prep, _ = _features(panels, novelty=nov)
    g, _, _ = _features(panels)
    i = CR.INDEX
    assert list(prep.boundaries) == [16]
    assert f[i["has_sections"]] == 1.0 and f[i["section_change"]] > 0.9
    assert f[i["section_contrast"]] > 0.5
    assert g[i["has_sections"]] == 0.0 and g[i["section_change"]] == 0.0


@pytest.mark.parametrize("kind", [k for k in TC.KINDS if k != "smooth"])
def test_a_training_negative_changes_arrows_of_free_single_notes_only(kind):
    rng = np.random.default_rng(4)
    n = 240
    panel = rng.integers(0, 4, n)
    panel[80:120] = panel[40:80]                           # a passage that comes back
    elig = np.nonzero(np.arange(n) % 7 != 0)[0]            # every 7th note may not change
    bar = np.arange(n) // 4
    pairs = (np.arange(80, 120), np.arange(40, 80))
    out = TC.damage(kind, panel, elig, bar, rng, pairs)
    assert out is not None and out.shape == panel.shape
    changed = np.nonzero(out != panel)[0]
    assert len(changed) > 0 and np.isin(changed, elig).all()
    assert set(np.unique(out)) <= {0, 1, 2, 3}
    if kind == "shuffle":                                  # the same arrows, elsewhere
        assert sorted(out[elig]) == sorted(panel[elig])
    if kind == "uncallback":
        assert np.isin(changed, pairs[0]).all()


def test_the_smooth_negative_takes_jacks_out_and_leaves_other_notes_alone():
    rng = np.random.default_rng(8)
    n = 200
    panel = np.repeat(rng.integers(0, 4, n // 2), 2)       # every arrow twice: jacks throughout
    elig = np.nonzero(np.arange(n) % 9 != 0)[0]

    def read(p):
        tick = np.arange(n, dtype=np.int32) * 6
        ev = C.ChartEvents(tick, p.astype(np.int8), np.full(n, -1, np.int8), np.zeros(n, np.int8),
                           np.zeros(n, np.int8), np.zeros(n, np.int32), np.zeros(n, np.int32),
                           np.zeros((0, 2), np.int32))
        return C.encode(ev, tick * (60000.0 / 140 / C.TICKS_PER_BEAT))[0].token

    out = TC.smooth(panel, elig, rng, read)
    assert out is not None
    changed = np.nonzero(out != panel)[0]
    assert len(changed) >= 4 and np.isin(changed, elig).all()
    jacks = lambda p: int((p[1:] == p[:-1]).sum())          # noqa: E731
    assert jacks(out) < jacks(panel)
    assert TC.smooth(panel, elig[:8], rng, read) is None   # too few notes to matter


def test_auc_is_the_chance_a_positive_outranks_a_negative():
    assert TC.auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert TC.auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])) == 0.0
    assert TC.auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 0, 1, 0])) == 0.5
    assert np.isnan(TC.auc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_resampling_needs_no_network_and_is_repeatable():
    rng = np.random.default_rng(5)
    logp = infer.log_softmax(rng.normal(0, 2, (50, C.N_TOKENS)))
    tok, lp = infer.resample(logp, 3, infer.DEFAULT_TAU, (1.0, 1.0, 0.0), seed=7)
    again, _ = infer.resample(logp, 3, infer.DEFAULT_TAU, (1.0, 1.0, 0.0), seed=7)
    assert tok.shape == (3, 50) and lp.shape == (3, 50, C.N_TOKENS)
    assert np.array_equal(tok, again)
    assert np.array_equal(tok[2], logp.argmax(1))          # temperature 0: the likeliest token
    assert not np.array_equal(tok[0], tok[1])


class _FavourLast:
    """A stand-in critic: the later the sample, the better."""

    def __init__(self):
        self.calls = 0

    def score(self, feats):
        self.calls += 1
        return np.arange(len(feats), dtype=np.float64)


def _engine(monkeypatch, critic, scorer=None):
    engine = infer.StepManiaNeuralEngine(TINY)
    monkeypatch.setattr(engine.model, "rhythm", _fake_rhythm)
    monkeypatch.setitem(engine.model.meta, "decoder", {} if critic is None else {"critic": critic})
    engine.model.sess_c = scorer
    return engine


def test_best_of_k_is_off_unless_the_model_ships_a_critic(monkeypatch):
    assert _engine(monkeypatch, None, _FavourLast()).model.critic() is None
    assert _engine(monkeypatch, {"k": 4}, None).model.critic() is None      # no stage_3
    assert _engine(monkeypatch, {"k": 4, "enabled": False}, _FavourLast()).model.critic() is None
    assert _engine(monkeypatch, {"k": 1}, _FavourLast()).model.critic() is None
    assert _engine(monkeypatch, {"k": 4, "source": "?"}, _FavourLast()).model.critic() is None
    cr = _engine(monkeypatch, {"k": 3, "source": "logits"}, _FavourLast()).model.critic()
    assert cr["k"] == 3 and cr["source"] == "logits" and cr["steps"] == infer.UNMASK_STEPS
    x = _song("ABAB")
    tt, dur, _ = _grid(x)
    chart = _engine(monkeypatch, None, _FavourLast()).generate(
        x, tt, "hard", cond.resolve_style("hard"), dur)
    assert "critic" not in chart.stats["choreo"]


@pytest.mark.parametrize("source", ["sample", "logits"])
def test_best_of_k_keeps_the_notes_picks_by_score_and_stays_legal(monkeypatch, source):
    x = _song("ABABAB", noise=0.02)
    tt, dur, bpm = _grid(x)
    style = cond.resolve_style("hard")
    veto = pipeline.flow_veto("hard", bpm)
    plain = _engine(monkeypatch, None).generate(x, tt, "hard", style, dur, veto=veto)
    scorer = _FavourLast()
    best = _engine(monkeypatch, {"k": 4, "source": source}, scorer).generate(
        x, tt, "hard", style, dur, veto=veto)
    info = best.stats["choreo"]["critic"]
    assert scorer.calls == 1 and info["k"] == 4 and len(info["scores"]) == 4
    assert info["eligible"][0] is True
    # the highest-scored sample the referees treat no worse than the first
    assert info["picked"] == max(j for j in range(4) if info["eligible"][j])
    # the rhythm is the rhythm network's; only arrows differ
    assert np.array_equal(plain.events.tick, best.events.tick)
    # (with the stand-in step network a further sample may be worse footed
    # than the first and so not eligible; then the first one stays)
    assert np.array_equal(plain.encoded.token, best.encoded.token) == (info["picked"] == 0)
    assert veto(best) == []
    assert pipeline.read_double_steps(best.events, tt[best.events.tick] * 1000.0) == []
    st = infer.chart_stats(best, tt)
    assert st.get("spins", 0) == 0 and st.get("lock_violations", 0) == 0

    class First:
        def score(self, feats):
            return -np.arange(len(feats), dtype=np.float64)
    same = _engine(monkeypatch, {"k": 4, "source": source}, First()).generate(
        x, tt, "hard", style, dur, veto=veto)
    # sample 0 is the chart the model writes without a critic
    assert same.stats["choreo"]["critic"]["picked"] == 0
    assert np.array_equal(same.encoded.token, plain.encoded.token)


def test_the_critic_file_is_small_and_scores_a_batch(tmp_path):
    rng = np.random.default_rng(6)
    data = {"x": rng.normal(0, 1, (400, CR.N_FEATURES)).astype(np.float32)}
    data["y"] = np.asarray(data["x"][:, CR.INDEX["cost_mean"]] < 0, np.int64)
    data["kind"] = np.where(data["y"] == 1, "human", "random")
    data["difficulty"] = np.asarray(["hard"] * 400)
    data["split"] = np.asarray(["train"] * 320 + ["val"] * 80)
    data["song"] = np.asarray([f"s{i // 8}" for i in range(400)])
    pytest.importorskip("torch")
    report = TC.train(data, tmp_path, epochs=60, log=lambda *a: None)
    path = tmp_path / "stage_3_critic.onnx"
    assert path.stat().st_size < 500_000 and report["auc_val"] > 0.9
    assert report["parity_max_abs_diff"] < 1e-3
    score = CR.Critic(path).score(data["x"][:5])
    assert score.shape == (5,) and np.isfinite(score).all()
    assert json.loads((tmp_path / "critic_meta.json").read_text())["features"] == list(CR.FEATURES)
