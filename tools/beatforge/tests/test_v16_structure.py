"""v1.6 structure analysis and callbacks: repeats are found where the song
repeats and nowhere else, the two structure channels are well formed, and the
decoder's callbacks are soft, legal and off unless a model declares them."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from beatforge.adapters.stepmania.neural import codec as C
from beatforge.adapters.stepmania.neural import grid_features as GF
from beatforge.adapters.stepmania.neural import structure as S

G = C.G


def _song(sections: str, bars_per: int = 8, seed: int = 0, noise: float = 0.05) -> np.ndarray:
    """A synthetic X (T, 64): one letter per section, each a `bars_per`-bar
    passage with its own rhythm, timbre and harmony; the same letter is the
    same passage again, give or take `noise`."""
    rng = np.random.default_rng(seed)
    proto = {}
    for ch in sorted(set(sections)):
        p = np.zeros((bars_per * G, 64), np.float32)
        p[:, GF.MEL] = rng.normal(0, 1, 20) + rng.normal(0, 0.3, (bars_per * G, 20))
        hits = rng.random((bars_per * G // 3, 6)) < 0.3
        p[::3, GF.FLUX] = hits * rng.uniform(0.5, 2.0, (bars_per * G // 3, 6))
        p[:, S.CHROMA] = np.repeat(rng.normal(0, 1, (bars_per, 5)), G, axis=0)
        proto[ch] = p
    parts = [proto[ch] + rng.normal(0, noise, proto[ch].shape).astype(np.float32)
             for ch in sections]
    return np.concatenate(parts).astype(np.float32)


def test_a_repeated_section_is_linked_to_its_first_statement_and_nothing_else_is():
    x = _song("ABABCB")                                  # B at bars 8, 24, 40
    st = S.analyze_structure(x)
    assert st.n_bars == 48
    link = st.link_of()
    # every bar of the second A and of the later Bs is linked ...
    for b in range(16, 24):
        assert link[b][0] == b - 16, b                   # second A -> first A
    for b in range(24, 32):
        assert link[b][0] == b - 16, b                   # second B -> first B
    for b in range(40, 48):
        assert link[b][0] == b - 32, b                   # third B -> the FIRST B, not the second
    # ... the first statements and the one-off C are not
    assert not any(b in link for b in list(range(0, 16)) + list(range(33, 39)))
    assert all(s >= S.TAU_SIM for _, s in link.values())
    assert all((b - j) % S.LINK_STEP_BARS == 0 and b - j >= S.MIN_LINK_BARS
               for b, (j, _) in link.items())


def test_the_two_channels_are_per_tick_bounded_and_mean_what_they_say():
    x = _song("AABA", bars_per=8)
    st = S.analyze_structure(x)
    ch = st.channels()
    assert ch.shape == (len(x), 2) and ch.dtype == np.float32
    assert ch.min() >= 0.0 and ch.max() <= 1.0 + 1e-6
    nov, rep = st.novelty, st.repeat_sim
    # novelty peaks at the section changes (bars 16 and 24), not inside a section
    top2 = set(np.argsort(-nov)[:2].tolist())
    assert top2 == {16, 24}, top2
    assert nov[16] > nov[4:12].max() and nov.max() == pytest.approx(1.0)
    # "heard this before": the repeats of A are, the first A and B are not
    assert rep[8:16].min() > 0.8 and rep[24:32].min() > 0.8
    assert rep[:4].max() == 0.0 and rep[16:24].max() < 0.5
    # per-tick channels are the per-bar values held for the bar
    assert np.allclose(st.section_novelty.reshape(-1, G)[:, 0], nov)
    assert np.allclose(st.prior_repeat_sim.reshape(-1, G)[:, -1], rep)


def test_unrelated_music_has_no_links_and_a_loop_links_everywhere():
    rng = np.random.default_rng(3)
    x = _song("ABCDEFGH", bars_per=8, seed=5)
    assert S.analyze_structure(x).motif_links == []
    loop = _song("A" * 8, bars_per=8, seed=6)
    links = S.analyze_structure(loop).link_of()
    assert len(links) >= 48
    # nearly all point at the first pass (a noisier copy can sit just outside the tie band)
    assert np.mean([j < 8 for j, _ in links.values()]) >= 0.8
    del rng


def test_short_and_empty_songs_do_not_break_it():
    for bars in (0, 1, 3, 7):
        x = _song("A", bars_per=max(bars, 1))[:bars * G]
        st = S.analyze_structure(x)
        assert st.n_bars == bars and st.motif_links == []
        assert st.channels().shape == (bars * G, 2)
    # a song that does not end on a bar line keeps a value for its last ticks
    x = _song("AB")[:-17]
    st = S.analyze_structure(x)
    assert len(st.section_novelty) == len(x) and np.isfinite(st.channels()).all()
    # silence: nothing to compare, nothing linked, nothing NaN
    st = S.analyze_structure(np.zeros((16 * G, 64), np.float32))
    assert st.motif_links == [] and np.isfinite(st.ssm).all()


def test_voice_channels_join_the_description_when_given():
    x = _song("AAAA", bars_per=8, noise=0.02)
    plain = S.analyze_structure(x)
    assert len(plain.motif_links) > 0
    # the same audio, but the second half has a different vocal line
    rng = np.random.default_rng(0)
    voice = np.zeros((len(x), 4), np.float32)
    voice[: len(x) // 2 : 6, 0] = 1.0
    voice[len(x) // 2 :: 3, 1:] = rng.random((len(x) // 2 // 3 + 1, 3))[: len(voice[len(x) // 2 :: 3])]
    with_voice = S.analyze_structure(x, voice)
    a = np.mean([s for _, s in plain.link_of().values()])
    later = [s for b, (_, s) in with_voice.link_of().items() if b >= 16]
    assert not later or np.mean(later) < a               # the vocal change makes them less alike


def test_it_takes_a_few_milliseconds():
    x = _song("ABABCBAB" * 2, bars_per=8)                # 128 bars: a six-minute song
    S.analyze_structure(x)
    t0 = time.perf_counter()
    for _ in range(5):
        S.analyze_structure(x)
    assert (time.perf_counter() - t0) / 5 < 0.02         # generous: CI machines are slow


# --------------------------------------------------------------------------- #
# the decoder's use of it
# --------------------------------------------------------------------------- #
ort = pytest.importorskip("onnxruntime")
from beatforge.adapters.stepmania.neural import conditioning as cond  # noqa: E402
from beatforge.adapters.stepmania.neural import infer, pipeline  # noqa: E402

TINY = Path(__file__).parent / "fixtures" / "neural" / "tiny_model"


def test_rhythm_echo_blends_only_linked_bars_and_only_by_the_similarity():
    t = 40 * G
    rng = np.random.default_rng(1)
    onset = rng.normal(0, 1, t).astype(np.float32)
    links = [(16, 0, 1.0, 0), (17, 1, 0.925, 1), (18, 2, 0.85, 2)]
    out = infer.echo_onset_logits(onset, links, 0.85, 0.5)
    assert out is not onset and np.array_equal(out[:16 * G], onset[:16 * G])
    assert np.array_equal(out[19 * G:], onset[19 * G:])
    assert np.allclose(out[16 * G:17 * G], 0.5 * onset[16 * G:17 * G] + 0.5 * onset[:G])
    assert np.allclose(out[17 * G:18 * G], 0.75 * onset[17 * G:18 * G] + 0.25 * onset[G:2 * G])
    assert np.array_equal(out[18 * G:19 * G], onset[18 * G:19 * G])       # at tau: nothing
    # nothing to do -> the same array back, untouched
    assert infer.echo_onset_logits(onset, links, 0.85, 0.0) is onset
    assert infer.echo_onset_logits(onset, [], 0.85, 0.5) is onset


def test_the_arrow_bonus_points_at_the_earlier_bars_token_on_the_same_tick():
    ticks = np.array([0, 12, 24, 36, 16 * G, 16 * G + 12, 16 * G + 30, 16 * G + 36])
    is_jump = np.array([0, 0, 1, 0, 0, 1, 0, 0], bool)
    tokens = np.array([1, 2, 9, 3, 0, 8, 0, 0])
    bonus, hit = infer.callback_bonus(ticks, is_jump, tokens, [(16, 0, 0.9, 0)], 2.0)
    assert hit == 2 and bonus.shape == (8, C.N_TOKENS)
    assert bonus[4, 1] == pytest.approx(1.8)               # same tick, both singles
    assert bonus[7, 3] == pytest.approx(1.8)
    assert not bonus[5].any()                              # a jump over a single: no bonus
    assert not bonus[6].any()                              # no note on that tick earlier
    assert not bonus[:4].any()                             # the first statement is left alone
    assert infer.callback_bonus(ticks, is_jump, tokens, [], 2.0)[1] == 0


def _fake_rhythm(x, cond19):
    t = len(x)
    k = np.arange(t)
    onset = np.full(t, -6.0, np.float32)
    onset[k % 6 == 0] = 1.0
    onset[k % 12 == 0] = 2.5
    return {"onset": onset, "jump": np.full(t, -5.0, np.float32),
            "hold_type": np.tile(np.array([9, 0, 0, 0, 0, 0], np.float32), (t, 1)),
            "hold_dur": np.zeros((t, 10), np.float32), "mine": np.full(t, -9.0, np.float32)}


def _engine(monkeypatch, callbacks):
    engine = infer.StepManiaNeuralEngine(TINY)
    monkeypatch.setattr(engine.model, "rhythm", _fake_rhythm)
    dec = {} if callbacks is None else {"callbacks": callbacks}
    monkeypatch.setitem(engine.model.meta, "decoder", dec)
    return engine


def _grid(x):
    bpm = 140.0
    tt = 0.2 + np.arange(len(x)) * (60.0 / bpm / C.TICKS_PER_BEAT)
    return tt, float(tt[-1] + 3.0), bpm


def _agreement(chart, links) -> float:
    """Share of linked notes with the same token as the earlier bar's note on
    the same tick, in the decoder's own footing."""
    ticks, tok = chart.events.tick, chart.encoded.token
    index = {int(t): i for i, t in enumerate(ticks)}
    same = total = 0
    for b, j, _, _ in links:
        lo, hi = np.searchsorted(ticks, [b * G, (b + 1) * G])
        for i in range(lo, hi):
            k = index.get(int(ticks[i]) - (b - j) * G)
            if k is not None and (tok[i] >= 8) == (tok[k] >= 8):
                total += 1
                same += int(tok[i] == tok[k])
    return same / max(1, total)


def test_callbacks_are_off_unless_the_model_declares_them(monkeypatch):
    assert _engine(monkeypatch, None).model.callbacks() is None
    assert _engine(monkeypatch, {"enabled": False, "choreo_bonus": 2.0}).model.callbacks() is None
    assert _engine(monkeypatch, {"choreo_bonus": 0, "rhythm_weight": 0}).model.callbacks() is None
    assert _engine(monkeypatch, {"tau_sim": 1.5, "choreo_bonus": 2.0}).model.callbacks() is None
    cb = _engine(monkeypatch, {"tau_sim": 0.7, "choreo_bonus": 2.0}).model.callbacks()
    assert cb == {"tau_sim": 0.7, "rhythm_weight": 0.0, "choreo_bonus": 2.0}
    # a chart from a model without them never looks at the structure
    engine = _engine(monkeypatch, None)

    def boom(*a, **k):
        raise AssertionError("structure analysis ran")

    monkeypatch.setattr(infer.ST, "analyze_structure", boom)
    x = _song("ABAB")
    tt, dur, _ = _grid(x)
    chart = engine.generate(x, tt, "hard", cond.resolve_style("hard"), dur)
    assert "callbacks" not in chart.stats["choreo"]


@pytest.mark.parametrize("difficulty", ["medium", "challenge"])
def test_a_repeated_section_gets_its_steps_back_and_the_chart_stays_legal(monkeypatch, difficulty):
    x = _song("ABABAB", noise=0.02)
    tt, dur, bpm = _grid(x)
    links = S.analyze_structure(x).motif_links
    assert len(links) >= 24
    style = cond.resolve_style(difficulty)
    veto = pipeline.flow_veto(difficulty, bpm)
    plain = _engine(monkeypatch, None).generate(x, tt, difficulty, style, dur, veto=veto)
    # (a strong bonus: the stand-in step network's choices are noise, and the
    # bonus has to beat the 2.5 nats a sampled token is worth)
    echo = _engine(monkeypatch, {"tau_sim": 0.85, "choreo_bonus": 8.0}).generate(
        x, tt, difficulty, style, dur, veto=veto)
    # the notes are the same (no rhythm echo asked for); the arrows come back
    assert np.array_equal(plain.events.tick, echo.events.tick)
    info = echo.stats["choreo"]["callbacks"]
    assert info["links"] == len(links) and info["anchored"] > 0
    assert _agreement(echo, links) > _agreement(plain, links) + 0.15
    # ...and it is still a chart anyone can stand on
    assert veto(echo) == []
    times_ms = tt[echo.events.tick] * 1000.0
    assert pipeline.read_double_steps(echo.events, times_ms) == []
    st = infer.chart_stats(echo, tt)
    assert st.get("spins", 0) == 0 and st.get("lock_violations", 0) == 0
