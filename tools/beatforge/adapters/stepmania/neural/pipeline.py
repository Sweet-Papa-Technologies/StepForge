"""pipeline.py — the neural engine's song-folder orchestration (spec §9).

`build_song` is the neural counterpart of `adapter.build_song`: it takes the
same DSP analysis and writes the same artifacts to the same place —
build/stepmania/<track>/ with .ssc, .sm, report.json, per-difficulty
.notefield.png / .assist.ogg and .qa.json — so the desktop app's previews and
chartbench read a neural run exactly as they read an LLM one.

`inpaint_song` re-rolls a bar range of one chart in an existing folder.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import zlib
from pathlib import Path

import numpy as np

from .... import config, ledger
from .. import qa as sm_qa
from ..difficulty import compute_meter
from ..grammar import ALL_DIFFICULTIES, BUDGETS
from ..serialize import write_song_folder
from . import codec as C
from . import conditioning as cond
from . import grid_features as GF
from . import infer

ENGINE_NAME = "neural"
#: Serialization grid for the neural engine: one tick of the 48-grid, so the
#: triplets Hard and Challenge may use survive the trip to the simfile.
TICK_SUBDIV = 1.0 / C.TICKS_PER_BEAT


def _default_seed(track_id: str, difficulty: str) -> int:
    return zlib.crc32(f"{track_id}:{difficulty}".encode()) & 0x7FFFFFFF


def _strict_meters(meters: dict) -> dict:
    """Make meters strictly increasing across tiers without leaving any tier's
    own range where that is possible (REQ-SFN-07)."""
    order = [d for d in ALL_DIFFICULTIES if d in meters]
    out = dict(meters)
    for prev, cur in zip(order, order[1:]):
        if out[cur] <= out[prev]:
            out[cur] = min(BUDGETS[cur].meter_range[1], out[prev] + 1)
    for cur, prev in zip(reversed(order), reversed(order[:-1])):
        if out[prev] >= out[cur]:
            out[prev] = max(1, out[cur] - 1)
    return out


def stabilize_simfile(path) -> None:
    """Rewrite a simfile through garcia/simfile once so the bytes on disk are
    exactly what the library itself emits (REQ-SFN-07: a byte-stable round
    trip). The first write can differ from its own re-serialization by blank
    lines inside #NOTES; after this, load -> str is the identity."""
    import simfile
    path = Path(path)
    text = path.read_text()
    norm = str(simfile.loads(text))
    if str(simfile.loads(norm)) != norm:
        raise AssertionError(f"{path.name} does not round-trip through simfile")
    if norm != text:
        path.write_text(norm)


#: How early BeatForge's DSP reports time, per beat backend. `dsp.py` stamps an
#: STFT frame at its first sample rather than its centre (2048 samples = 46 ms
#: of half-window at 22.05 kHz) and spectral flux fires as a transient enters
#: the window, so its onsets and its beat grid arrive early. Measured on
#: synthetic clicks at known sample positions: 73-76 ms early at 120, 128 and
#: 150 BPM. (The project's own timing probe saw the same thing from the other
#: side: DDC's charts sat a uniform +58 ms "late" against BeatForge's onsets.)
#: The neural features are computed in true audio time — verified by the click
#: tests in test_neural_grid.py — so the grid has to be moved onto it.
DSP_EARLY_S = {"local_dsp": 0.072}
GRID_SEARCH_S = 0.060
#: Human charters sync to the attack they hear, which precedes the spectral-flux
#: peak: over the training corpus the median chart sat 8 ms ahead of it. The
#: features use the flux-aligned grid (as in training); the simfile is written
#: this much earlier so a neural chart is synced the way a human one is.
HUMAN_SYNC_LEAD_S = 0.008


def align_grid(envelope: np.ndarray, bpm: float, offset: float, duration_s: float,
               beat_backend: str = "local_dsp") -> dict:
    """Seat the DSP's beat grid on the audio's transients.

    Returns {'shift_s', 'contrast', 'expected_s'}: `offset + shift_s` is the
    time of beat 0 in true audio time. The search is centred on the backend's
    known early bias and limited to +-60 ms around it, which is less than half
    a beat at any tempo the DSP reports, so it cannot lock onto a neighbour.
    """
    expected = DSP_EARLY_S.get(beat_backend, 0.0)
    period = 60.0 / bpm
    beats = np.arange(offset, duration_s, period)
    sync = GF.estimate_sync(envelope, beats + expected, max_shift_s=GRID_SEARCH_S,
                            near_s=GRID_SEARCH_S)
    return {"shift_s": expected + sync["shift_s"], "contrast": sync["contrast"],
            "expected_s": expected}


def transient_ticks(analysis: dict, n_ticks: int, tol_s: float = 0.032) -> np.ndarray:
    """Per tick: is there a DSP onset within `tol_s` of it? Evaluated in the
    DSP's own time frame (its offset, its onset times), which is the frame the
    QA onset-alignment gate measures in (35 ms)."""
    onsets = np.sort(np.array([o["time"] for o in analysis.get("onsets", [])], float))
    if len(onsets) == 0:
        return np.ones(n_ticks, bool)
    t = analysis["offset"] + np.arange(n_ticks) * 60.0 / analysis["bpm"] / C.TICKS_PER_BEAT
    j = np.clip(np.searchsorted(onsets, t), 1, len(onsets) - 1)
    return np.minimum(np.abs(t - onsets[j - 1]), np.abs(onsets[j] - t)) <= tol_s


def render_previews(placements, analysis, audio_path, out_stem) -> dict:
    """The same two artifacts as `qa.render_previews` — <stem>.notefield.png and
    <stem>.assist.ogg — with the click track mixed in numpy instead of through
    an ffmpeg `aevalsrc` expression of one sine term per note. That expression
    took 120 s per chart on a two-minute song (and then hit its timeout and
    produced nothing), which was 95% of a neural run's wall-clock."""
    import soundfile as sf
    out = {}
    bpm, offset = analysis["bpm"], analysis["offset"]
    notes = [p for p in placements if p.kind != "mine"]
    times = np.array([offset + p.beat / bpm * 60 for p in notes])
    png = Path(str(out_stem) + ".notefield.png")
    try:
        _notefield_png(png, notes, times, bpm, offset, analysis.get("sections", []))
        out["png"] = str(png)
    except Exception:
        pass
    try:
        y, sr = sf.read(audio_path, always_2d=True, dtype="float32")
        mix = 0.8 * y.mean(axis=1)
        k = int(0.015 * sr)
        tick = (0.5 * np.sin(2 * np.pi * 2000 * np.arange(k) / sr)
                * np.hanning(2 * k)[k:]).astype(np.float32)
        for t in np.unique(np.round(times, 4)):
            i = int(t * sr)
            if 0 <= i < len(mix) - k:
                mix[i:i + k] += tick
        ogg = Path(str(out_stem) + ".assist.ogg")
        mix = np.clip(mix, -1.0, 1.0)
        # in blocks: libsndfile's Vorbis encoder overflows its stack (segfault)
        # when handed a couple of minutes of audio in a single write
        with sf.SoundFile(str(ogg), "w", samplerate=sr, channels=1, format="OGG",
                          subtype="VORBIS") as fh:
            for a in range(0, len(mix), sr):
                fh.write(mix[a:a + sr])
        out["assist"] = str(ogg)
    except Exception:
        pass
    return out


def _notefield_png(path, notes, times, bpm, offset, sections) -> None:
    """A timeline of the chart: one row per panel, a dot per tap, a bar per
    hold, dashed lines at section starts. Drawn with Pillow (already a
    dependency) rather than matplotlib, which the app no longer bundles."""
    from PIL import Image, ImageDraw, ImageFont
    from ..grammar import PANELS
    W, H, L, R, T, B = 1760, 330, 56, 24, 18, 46
    end = max([float(t) + (p.hold_beats or 0) / bpm * 60 for p, t in zip(notes, times)] or [1.0]) + 1.0
    x = lambda s: L + (W - L - R) * max(0.0, s) / end
    row = lambda c: H - B - (H - T - B) * (c + 0.5) / 4
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    d.rectangle((L, T, W - R, H - B), outline="#444444")
    for sec in sections:
        sx = x(offset + sec["start_bar"] * 4 / bpm * 60)
        for y0 in range(T, H - B, 10):
            d.line((sx, y0, sx, min(y0 + 5, H - B)), fill="#b14cff", width=1)
    for p, t in zip(notes, times):
        for c in p.panels:
            if p.hold_beats:
                d.line((x(t), row(c), x(t + p.hold_beats / bpm * 60), row(c)), fill="#ff2d95", width=4)
            else:
                d.ellipse((x(t) - 3, row(c) - 3, x(t) + 3, row(c) + 3), fill="#2de2e6")
    for c, name in enumerate(PANELS):
        d.text((L - 10, row(c)), str(name), fill="#222222", font=font, anchor="rm")
    step = 10 if end <= 200 else 30
    for s in range(0, int(end) + 1, step):
        d.line((x(s), H - B, x(s), H - B + 5), fill="#444444")
        d.text((x(s), H - B + 8), str(s), fill="#222222", font=font, anchor="mt")
    d.text(((L + W - R) / 2, H - 14), "s", fill="#222222", font=font, anchor="mm")
    im.save(path)


def flow_veto(difficulty: str, bpm: float | None = None):
    """The project's flow-ceiling gate as a per-note referee for the engine.

    `qa.chart_metrics` fails a chart whose worst footflow transition reaches
    DOUBLE_STEP + JACK (SACRED-03: zero forbidden-tier transitions). footflow
    and the neural foot model agree on almost everything, but footflow scores a
    same-foot repeat as a forbidden jack, treats the right foot as having moved
    last after every jump, and breaks D/U ties by a fixed offset — so a chart
    the neural model considers comfortable can still trip it. Rather than copy
    those rules, replay the gate's own arithmetic and hand back the notes it
    objects to; the engine bans each and re-solves. `bpm` gives the gate the
    time between steps, as `qa.chart_metrics` does, so slow repeats pass.
    """
    from .. import footflow as ff
    budget = BUDGETS[difficulty]
    ceiling = ff.DOUBLE_STEP + ff.JACK + 1e-6

    def veto(chart) -> list:
        placements = infer.to_placements(chart)
        index = {int(t): i for i, t in enumerate(chart.events.tick)}
        state, bad, prev_beat = ff.REST_STATE, [], None
        for p in placements:
            dt_s = None if bpm is None or prev_beat is None \
                else (p.beat - prev_beat) * 60.0 / bpm
            prev_beat = p.beat
            if len(p.panels) > 1:
                c, state = ff.jump_cost(state, p.panels[0], p.panels[1], budget)
            else:
                c, state = ff.step(state, p.meta.get("foot", ff.LEFT), p.panels[0],
                                   budget, p.meta.get("movement", "static"), 0.0,
                                   dt_s=dt_s)
            if c >= ceiling and p.kind != "mine":
                i = index.get(int(round(p.beat * C.TICKS_PER_BEAT)))
                if i is not None:
                    bad.append(i)
        return bad
    return veto


TEMPO_MAX_CHANGE = 0.004        # never move the DSP's BPM by more than 0.4%
TEMPO_MATCH_S = 0.035           # an onset belongs to a grid line within this
TEMPO_MIN_ONSETS = 40
TEMPO_SNAP_TOL = 1.5e-4          # a round BPM within 9 ms/min of the fit is taken


def _onset_peaks(envelope: np.ndarray) -> np.ndarray:
    """Strong onset times (s, true audio time) from the feature envelope."""
    import scipy.signal
    e = np.asarray(envelope, np.float64)
    pk, _ = scipy.signal.find_peaks(e, height=e.mean() + 1.0 * e.std(), distance=8)
    pk = pk[(pk > 0) & (pk < len(e) - 1)]
    a, b, c = e[pk - 1], e[pk], e[pk + 1]
    return (pk + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)) * GF.HOP_S + GF.FLUX_LEAD_S


def refine_tempo(envelope: np.ndarray, bpm: float, offset: float, duration_s: float
                 ) -> dict:
    """Fit BPM and beat-0 time to the audio so the grid does not drift.

    `offset` must already be in true audio time (after `align_grid`). The DSP's
    tempo is good to a few hundredths of a percent, and that is not good
    enough: 109.957 for a 110 BPM song is 24 ms of drift per minute, so a
    chart that starts on the beat ends a Fantastic window late. Measured on the
    13-song benchmark, neural charts drifted up to 28 ms/min before this.

    Drift is measured directly: each strong onset is paired with its nearest
    8th-note grid line and a robust line is fitted to (grid time, onset - grid).
    The slope is the tempo error and the intercept the phase error. The fit
    starts on the first 20 s, where no plausible drift can mis-pair an onset,
    and widens as the tempo estimate improves. A whole or half BPM is taken
    when it lies within the fit's own uncertainty — produced music is almost
    always cut to one. (Maximising onset energy on the grid over a tempo sweep
    was tried first and rejected: on syncopated songs its maximum sat 0.1-0.2%
    away from tempos that measurably did not drift.)
    """
    onsets = _onset_peaks(envelope)
    onsets = onsets[(onsets > offset) & (onsets < duration_s)]
    out = {"bpm": bpm, "offset": offset, "drift_ms_per_min": 0.0, "snapped": False,
           "onsets": int(len(onsets))}
    if len(onsets) < TEMPO_MIN_ONSETS:
        return out
    b, off = float(bpm), float(offset)
    slope_se = 0.0
    horizon = 20.0
    while True:
        half = 30.0 / b
        use = onsets[onsets <= offset + horizon]
        n = np.round((use - off) / half)
        g = off + n * half
        e = use - g
        ok = np.abs(e) <= TEMPO_MATCH_S
        if ok.sum() >= TEMPO_MIN_ONSETS:
            x, y = g[ok] - off, e[ok]
            for _ in range(3):                           # trimmed least squares
                coef = np.polyfit(x, y, 1)
                res = y - np.polyval(coef, x)
                mad = np.median(np.abs(res - np.median(res))) + 1e-6
                keep = np.abs(res) <= 3.0 * 1.4826 * mad
                if keep.sum() < TEMPO_MIN_ONSETS or keep.all():
                    break
                x, y = x[keep], y[keep]
            slope, icpt = float(coef[0]), float(coef[1])
            sxx = float(((x - x.mean()) ** 2).sum()) + 1e-9
            slope_se = float(np.sqrt((res[keep] ** 2).mean() / sxx)) if keep.any() else 0.0
            b = b / (1.0 + slope)
            off = off + icpt
        if horizon >= duration_s:
            break
        horizon = min(duration_s, horizon * 2.0)
    if abs(b / bpm - 1.0) > TEMPO_MAX_CHANGE:
        return out                                       # not credible: keep the DSP's
    # Only a ROUND tempo is believed. On the 13-song benchmark the free fit
    # helped where it landed on a whole or half BPM (126.048 -> 126: jitter
    # 16.8 -> 5.2 ms) and hurt where it did not (170.120 -> 169.917: 7.9 ->
    # 13.6 ms): on swung or syncopated material, which onset belongs to which
    # grid line is itself uncertain, and the slope inherits that. A round
    # number inside the fit's uncertainty is two independent signs agreeing.
    tol = max(2.0 * slope_se, TEMPO_SNAP_TOL)
    for r in sorted({round(b), round(b * 2) / 2}, key=lambda r: abs(r - b)):
        if r > 0 and abs(b / r - 1.0) <= tol:
            half = 30.0 / r
            for _ in range(2):                           # re-seat the phase on it
                e = onsets - (off + np.round((onsets - off) / half) * half)
                ok = np.abs(e) <= TEMPO_MATCH_S
                if ok.sum() >= TEMPO_MIN_ONSETS:
                    off += float(np.median(e[ok]))
            out.update(bpm=float(r), offset=off, snapped=True,
                       drift_ms_per_min=round((r / bpm - 1.0) * 60000.0, 1))
            break
    return out


COHERENCE_WINDOW_BPM = 0.8      # search this far either side of the DSP's tempo
COHERENCE_STEP_BPM = 0.002
COHERENCE_MIN_GAIN = 1.05       # the refined tempo must be this much more coherent
COHERENCE_HARMONICS = ((1, 1.0), (2, 0.5), (4, 0.25))


def _grid_jitter(onsets: np.ndarray, bpm: float, offset: float) -> tuple:
    """(median |onset - nearest 8th-note line| over onsets within TEMPO_MATCH_S,
    offset re-seated on those onsets)."""
    half = 30.0 / bpm
    off = offset
    for _ in range(2):
        e = onsets - (off + np.round((onsets - off) / half) * half)
        ok = np.abs(e) <= TEMPO_MATCH_S
        if ok.sum() < TEMPO_MIN_ONSETS:
            return float("inf"), offset
        off += float(np.median(e[ok]))
    e = onsets - (off + np.round((onsets - off) / half) * half)
    ok = np.abs(e) <= TEMPO_MATCH_S
    return float(np.median(np.abs(e[ok]))), off


def refine_tempo_coherence(envelope: np.ndarray, bpm: float, offset: float,
                           duration_s: float) -> dict:
    """Refine a tempo that is NOT a round number, without pairing onsets to
    grid lines.

    `refine_tempo` only trusts a fit that lands on a whole or half BPM, because
    its regression has to decide which grid line each onset belongs to and on
    syncopated music that decision is the error. This asks a question with no
    such decision in it: wrap every onset time onto the beat circle at tempo B
    and see how tightly the onsets bunch,

        coherence(B) = sum over h in {1, 2, 4} of
                       w_h * | sum_k a_k * exp(2 pi i * h * t_k * B / 60) |

    (a_k the onset strengths; the harmonics 2 and 4 let 8ths and 16ths count).
    A tempo that drifts smears the bunch around the circle and the sum falls.
    B is searched within COHERENCE_WINDOW_BPM of the given tempo and accepted
    only if it is COHERENCE_MIN_GAIN more coherent AND the onsets sit measurably
    closer to its 8th-note grid; otherwise the given tempo stands. The grid is
    kept where it was at mid-song and re-seated on the onsets.

    Returns {'bpm', 'offset', 'refined', 'gain', 'jitter_ms': (before, after)}.
    """
    import scipy.signal
    env = np.asarray(envelope, np.float64)
    pk, _ = scipy.signal.find_peaks(env, height=env.mean() + 1.0 * env.std(), distance=8)
    t = pk * GF.HOP_S + GF.FLUX_LEAD_S
    keep = (t > offset) & (t < duration_s)
    t, a = t[keep], env[pk][keep]
    out = {"bpm": bpm, "offset": offset, "refined": False, "gain": 1.0, "jitter_ms": None}
    if len(t) < TEMPO_MIN_ONSETS:
        return out
    cands = bpm + np.arange(-COHERENCE_WINDOW_BPM, COHERENCE_WINDOW_BPM + 1e-9,
                            COHERENCE_STEP_BPM)
    cands = cands[np.abs(cands / bpm - 1.0) <= TEMPO_MAX_CHANGE]

    def coherence(b: np.ndarray) -> np.ndarray:
        ph = 2j * np.pi * np.outer(np.atleast_1d(b) / 60.0, t)          # (B, K)
        return sum(w * np.abs((a[None, :] * np.exp(h * ph)).sum(1))
                   for h, w in COHERENCE_HARMONICS)

    score = coherence(cands)
    base = float(coherence(bpm)[0])
    best = float(cands[int(score.argmax())])
    gain = float(score.max() / (base + 1e-12))
    out["gain"] = round(gain, 3)
    if gain < COHERENCE_MIN_GAIN or abs(best - bpm) < COHERENCE_STEP_BPM:
        return out
    # same grid line at mid-song under both tempos, then re-seat on the onsets
    mid = 0.5 * (offset + duration_s)
    n_mid = np.round((mid - offset) * bpm / 60.0)
    off_new = (offset + n_mid * 60.0 / bpm) - n_mid * 60.0 / best
    j0, _ = _grid_jitter(t, bpm, offset)
    j1, off_new = _grid_jitter(t, best, off_new)
    out["jitter_ms"] = (round(j0 * 1000, 1), round(j1 * 1000, 1))
    if not j1 < j0:
        return out
    # the refined tempo often lands a hair from a round one (172.001 for 172):
    # take the round value if the onsets sit on its grid at least as well
    for r in sorted({round(best), round(best * 2) / 2}, key=lambda r: abs(r - best)):
        if r > 0 and abs(best / r - 1.0) <= TEMPO_SNAP_TOL:
            off_r = (offset + n_mid * 60.0 / bpm) - n_mid * 60.0 / r
            jr, off_r = _grid_jitter(t, float(r), off_r)
            if jr <= j1 * 1.05:
                best, off_new = float(r), off_r
                out["jitter_ms"] = (round(j0 * 1000, 1), round(jr * 1000, 1))
            break
    out.update(bpm=round(best, 4), offset=off_new, refined=True)
    return out


#: Tempo relations the DSP's estimator confuses. Octaves are not here on purpose: half/double time is a
#: defensible reading, and the DSP's choice of octave is kept.
TEMPO_RATIOS = (4 / 3, 3 / 4, 3 / 2, 2 / 3, 5 / 4, 4 / 5, 8 / 5, 5 / 8)
TEMPO_FIT_WINDOW_S = 0.020
TEMPO_SUSPECT_RATIO = 0.45      # DSP grid fit under this share of the best ratio's
TEMPO_SWITCH_MARGIN = 0.07      # ...and the ratio must beat it by this much outright
TEMPO_ACF_RATIO = 0.60          # second opinion: DSP periodicity under this share
TEMPO_ACF_MARGIN = 0.10         # ...of the best ratio's, and worse by this outright
TEMPO_FOLD_RANGE = (95.0, 205.0)


def _grid_fit(onsets: np.ndarray, strength: np.ndarray, bpm: float) -> tuple:
    """How much of the onset energy sits on the 8th-note grid of `bpm`, above
    what a random grid of that density would catch (0 = chance, 1 = all of
    it), with the best phase. Returns (fit, phase_s in [0, half-beat))."""
    half = 30.0 / bpm
    total = float(strength.sum()) + 1e-12
    best, best_ph = -1.0, 0.0
    for ph in np.arange(0.0, half, 0.004):
        e = np.abs(((onsets - ph + half / 2) % half) - half / 2)
        hit = float(strength[e <= TEMPO_FIT_WINDOW_S].sum()) / total
        if hit > best:
            best, best_ph = hit, float(ph)
    chance = min(1.0, 2 * TEMPO_FIT_WINDOW_S / half)
    return (best - chance) / max(1e-9, 1 - chance), best_ph


def _periodicity(envelope: np.ndarray, bpms) -> list:
    """Autocorrelation of the onset envelope at 1, 2 and 4 beats of each tempo
    (mean of the three; 1.0 would be a perfectly beat-periodic envelope)."""
    e = np.asarray(envelope, np.float64)
    e = e - e.mean()
    n = len(e)
    f = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (ac[0] + 1e-12)
    lags = np.arange(n) * GF.HOP_S
    return [float(np.mean([np.interp(k * 60.0 / b, lags, ac) for k in (1, 2, 4)]))
            for b in bpms]


def verify_tempo(envelope: np.ndarray, bpm: float, offset: float, duration_s: float
                 ) -> dict:
    """Catch the DSP locking onto a 4:3 / 3:2-type multiple of the real tempo.

    On 180 held-out songs with human-charted tempos, dsp.py was right (or an
    octave off) on 164 and wrong by one of these ratios on 16 — and a chart on
    a grid that is 4/3 too fast is wrong everywhere. The wrong ones show in how
    little of the onset energy sits on the DSP's 8th-note grid compared with
    the grid of the best ratio: 10-44% as much, against 47% or more for every
    song whose DSP tempo was right. The rule: switch when the DSP's fit is
    under 45% of the best ratio's AND that ratio is better by 0.07 outright.

    Honest accounting of that 45%: the rule's form and a 50% threshold came
    from the first 120 songs (12 of 12 errors fixed, none broken); on the next
    60, 50% fixed all 4 errors but also moved one correct tempo (56 -> 59 of 60
    right), and the threshold was tightened to 45%, which gets all 180. The gap
    it sits in is narrow (44% vs 47%), and the near-misses are all x3/2 — a
    shuffled song really does put its onsets on the triplet grid. Expect an
    occasional miss either way; `tempo_ratio` in report.json says when it fired.

    A second, independent test backs the first up where onsets are soft and no
    grid fits them well: the envelope's autocorrelation at 1, 2 and 4 beats. If
    the DSP tempo's is under 60% of the best ratio's and lower by 0.10, that
    ratio is taken. Alone it is the weaker detector (5 of the 16 errors, none
    of the 164 right tempos moved); it is here for songs like one in the
    13-song benchmark that the grid test passed at 173.5 BPM while AutoStepper
    (130), librosa (129) and this autocorrelation all said 130.

    `offset` is beat 0 in true audio time. Returns {'bpm', 'offset', 'ratio',
    'fit', 'dsp_fit', 'by'}; ratio 1.0 means the DSP's tempo was kept.
    """
    import scipy.signal
    env = np.asarray(envelope, np.float64)
    pk, _ = scipy.signal.find_peaks(env, height=env.mean() + 0.5 * env.std(), distance=5)
    out = {"bpm": bpm, "offset": offset, "ratio": 1.0, "fit": None, "dsp_fit": None,
           "by": None}
    if len(pk) < TEMPO_MIN_ONSETS:
        return out
    onsets = pk * GF.HOP_S + GF.FLUX_LEAD_S
    strength = env[pk]
    dsp_fit, _ = _grid_fit(onsets, strength, bpm)
    out.update(fit=round(dsp_fit, 3), dsp_fit=round(dsp_fit, 3))
    best = (dsp_fit, 1.0, 0.0)
    for q in TEMPO_RATIOS:
        b = bpm * q
        if 55.0 <= b <= 250.0:
            fit, ph = _grid_fit(onsets, strength, b)
            if fit > best[0]:
                best = (fit, q, ph)
    fit, q, ph = best
    out["by"] = "grid"
    if (q == 1.0 or fit < dsp_fit + TEMPO_SWITCH_MARGIN
            or dsp_fit > TEMPO_SUSPECT_RATIO * fit):
        qs = [1.0] + [r for r in TEMPO_RATIOS if 55.0 <= bpm * r <= 250.0]
        per = _periodicity(env, [bpm * r for r in qs])
        i = int(np.argmax(per))
        if (i == 0 or per[i] < per[0] + TEMPO_ACF_MARGIN
                or per[0] > TEMPO_ACF_RATIO * per[i]):
            out["by"] = None
            return out
        q = qs[i]
        fit, ph = _grid_fit(onsets, strength, bpm * q)
        out["by"] = "periodicity"
    b = bpm * q
    while b >= TEMPO_FOLD_RANGE[1]:
        b /= 2.0
    while b < TEMPO_FOLD_RANGE[0]:
        b *= 2.0
    # the fitted phase is good to an 8th; put the beat on the stronger half
    beat = 60.0 / b
    half = beat / 2.0
    def on_grid(phase):
        e = np.abs(((onsets - phase + beat / 2) % beat) - beat / 2)
        return float(strength[e <= TEMPO_FIT_WINDOW_S].sum())
    ph = ph % half
    phase = ph if on_grid(ph) >= on_grid(ph + half) else ph + half
    out.update(bpm=round(b, 4), offset=phase, ratio=round(b / bpm, 4), fit=round(fit, 3))
    return out


def _rebar(analysis: dict, bpm: float, offset: float) -> dict:
    """Re-express the per-bar parts of a DSP analysis on a different grid (after
    `verify_tempo` replaced the tempo or `verify_phase` moved the bar line):
    energy per bar and section bounds, both by time. The density plan is
    dropped — it was computed for bars that no longer exist."""
    old_bar = 240.0 / analysis["bpm"]
    new_bar = 240.0 / bpm
    energy = analysis.get("energy_curve") or []
    n_new = int(max(0.0, analysis["duration_s"] - offset) / new_bar) + 1
    out = dict(analysis)
    if energy:
        t = offset + (np.arange(n_new) + 0.5) * new_bar
        idx = np.clip(((t - analysis["offset"]) / old_bar).astype(int), 0, len(energy) - 1)
        out["energy_curve"] = [float(energy[i]) for i in idx]
    def bar_at(old_index: float) -> int:                 # same instant, new bars
        t = analysis["offset"] + old_index * old_bar
        return max(0, int(round((t - offset) / new_bar)))

    out["sections"] = [dict(sec, start_bar=bar_at(sec["start_bar"]),
                            end_bar=bar_at(sec["end_bar"]))
                       for sec in analysis.get("sections", [])]
    out["density_plan"] = {}
    return out


PHASE_MARGIN = 1.5              # nats; see verify_phase
BAR_MARGIN = 0.02               # share of probability; see verify_phase


def phase_margin(engine) -> float:
    """The half-beat margin for this model. PHASE_MARGIN was set with the
    full-corpus weights; a model trained on less data is less sure of itself
    and needs a lower one (docs/v1.3/results.v1.3.md, section 5), so a model
    may carry its own as `decoder.phase_margin` in model_meta.json."""
    meta = getattr(getattr(engine, "model", None), "meta", None) or {}
    v = (meta.get("decoder") or {}).get("phase_margin")
    return float(v) if isinstance(v, (int, float)) else PHASE_MARGIN


def verify_phase(engine, core: np.ndarray, dt_ms: np.ndarray) -> dict:
    """Is the beat grid seated on the beats, and does bar 1 start on a downbeat?
    Ask the rhythm model.

    **Half a beat.** A beat tracker finds the pulse but can seat it on the
    off-beats. On 168 held-out songs with human simfiles, dsp.py's grid was
    half a beat from the charter's on 45 (27%). At Hard that hardly shows — the
    chart uses both halves of the beat — but a Beginner chart, which may only
    use quarter notes, lands every step between the beats: placement F1 against
    the human chart was exactly 0 on those songs, and 0.62 on the rest.

    Nothing simple in the audio separates the two (onset energy on beats vs
    off-beats got 3 of 12 right). RhythmNet does: it was trained on human
    charts over correctly seated grids, so asked for a Beginner chart it puts
    its probability on the quarter notes only when the music's beats are under
    them. The statistic is log(sum p on quarters / sum p on 8th off-beats),
    taken on the grid as given and on the grid moved half a beat; it is
    strongly bimodal (median +9 nats when the grid is wrong, -8 when right).
    Moving the grid when the gain exceeds PHASE_MARGIN — set on 59 songs —
    fixed 44 of the 45 across all 168 and moved no correct grid.

    **Which beat is beat 1.** dsp.py's beat 0 is the first beat it tracked, not
    a downbeat: on 166 songs its bar lines matched the charter's on 44 (27%).
    The networks are told where each tick sits in the bar, and jumps at the low
    tiers go on downbeats, so a wrong bar line degrades both. The same model
    settles it: of the four places the bar could start, the true one is where
    a Beginner chart's probability gathers most on beat 1. Taking the best of
    the four when it beats the current one by BAR_MARGIN matched the charter's
    bar lines on 145 of 166 (87%).

    Returns {'half_beat', 'gain' (nats), 'bar_shift_beats' (0-3, later),
    'bar_share'}.
    """
    cv = engine.cond_for(cond.resolve_style("beginner"))

    def onset_p(shift: int) -> np.ndarray:
        x = GF.assemble(core[shift:], dt_ms[shift:])
        return infer.sigmoid(engine.model.rhythm(x, cv)["onset"].astype(np.float64))

    def contrast(p: np.ndarray) -> float:
        k = np.arange(len(p)) % C.TICKS_PER_BEAT
        return float(np.log((p[k == 0].sum() + 1e-9) / (p[k == 6].sum() + 1e-9)))

    def on_beat_one(p: np.ndarray) -> float:
        k = np.arange(len(p))
        return float(p[k % C.G == 0].sum() / (p[k % C.TICKS_PER_BEAT == 0].sum() + 1e-9))

    out = {"half_beat": False, "gain": 0.0, "bar_shift_beats": 0, "bar_share": None}
    if len(core) < 8 * C.G:
        return out
    half = C.TICKS_PER_BEAT // 2
    p0, p_half = onset_p(0), onset_p(half)
    gain = contrast(p_half) - contrast(p0)
    out.update(half_beat=bool(gain > phase_margin(engine)), gain=round(gain, 2))
    base = half if out["half_beat"] else 0
    share = [on_beat_one(p_half if out["half_beat"] else p0)]
    share += [on_beat_one(onset_p(base + k * C.TICKS_PER_BEAT)) for k in (1, 2, 3)]
    best = int(np.argmax(share))
    if share[best] - share[0] > BAR_MARGIN:
        out["bar_shift_beats"] = best
    out["bar_share"] = [round(v, 3) for v in share]
    return out


def _features(audio_path: str, bpm: float, offset: float, duration_s: float,
              beat_backend: str = "local_dsp", engine=None):
    """Beat-warped features on the audio-aligned grid -> (GridFeatures, align).

    `align` carries the grid actually used: `bpm` and `offset` in true audio
    time, plus `shift_s`, how early the DSP's clock was. With `engine`, the
    grid's phase is also checked against the rhythm model (`verify_phase`): it
    is moved half a beat if it is seated on the off-beats, and by whole beats
    so that bar 1 starts on a downbeat."""
    y = GF.load_audio(audio_path)
    ff = GF.frame_features(y)
    align = align_grid(ff.onset_envelope, bpm, offset, duration_s, beat_backend)
    check = verify_tempo(ff.onset_envelope, bpm, offset + align["shift_s"], duration_s)
    tempo = refine_tempo(ff.onset_envelope, check["bpm"], check["offset"], duration_s)
    refined = False
    if not tempo["snapped"]:
        # not a round tempo: try the grid-free refinement
        coh = refine_tempo_coherence(ff.onset_envelope, tempo["bpm"], tempo["offset"],
                                     duration_s)
        if coh["refined"]:
            refined = True
            tempo = dict(tempo, bpm=coh["bpm"], offset=coh["offset"],
                         drift_ms_per_min=round(
                             (coh["bpm"] / check["bpm"] - 1.0) * 60000.0, 1))
    align.update(bpm=tempo["bpm"], offset=tempo["offset"], dsp_bpm=bpm,
                 dsp_drift_ms_per_min=tempo["drift_ms_per_min"],
                 bpm_snapped=tempo["snapped"], bpm_refined=refined,
                 tempo_ratio=check["ratio"],
                 tempo_switched_by=check["by"],
                 tempo_fit=check["fit"], dsp_tempo_fit=check["dsp_fit"],
                 phase_shift_beats=0.0, phase_gain=None, bar_share=None)

    def warp():
        timing = GF.TimingMap.constant(align["bpm"], align["offset"])
        return GF.warp_core(ff, timing, timing.ticks_for_duration(duration_s))

    core, t, dt_ms = warp()
    if engine is not None:
        phase = verify_phase(engine, core, dt_ms)
        align["phase_gain"], align["bar_share"] = phase["gain"], phase["bar_share"]
        shift = 0.5 * phase["half_beat"] + phase["bar_shift_beats"]
        if shift:
            align["offset"] += shift * 60.0 / align["bpm"]
            align["phase_shift_beats"] = shift
            core, t, dt_ms = warp()
    return GF.GridFeatures(GF.assemble(core, dt_ms), t, dt_ms, core, False), align


def _chart_report(placements, analysis, difficulty, meter, chart, tick_times, out_dir,
                  audio_path, style, preset, preview_analysis=None,
                  previews: bool = True) -> dict:
    """QA row for one chart. `analysis` is the DSP's (the frame its onsets live
    in); `preview_analysis`, if given, carries the audio-true offset the
    previews should be rendered against."""
    from .. import density as sm_density
    metrics = sm_qa.chart_metrics(placements, analysis, difficulty)
    dreport = sm_density.measure(placements, analysis, difficulty)
    metrics["density_plan"] = dreport
    metrics["designed_by_tier"] = None
    metrics["neural"] = infer.chart_stats(chart, tick_times)
    previews = render_previews(placements, preview_analysis or analysis,
                               audio_path, out_dir / difficulty) if previews else {}
    row = {
        "notes": len(placements), "meter": meter,
        "jumps": sum(1 for p in placements if len(p.panels) > 1),
        "holds": sum(1 for p in placements if p.hold_beats),
        "repaired": 0, "repair_ok": True, "critic": None, "previews": previews,
        "metrics": metrics, "chunked": False, "chunks": 0, "chunk_calls": 0,
        "chunk_fallback_bars": [],
        "density": {"report": dreport, "thinned": 0, "filled": 0, "spans": []},
        "density_energy_spearman": metrics.get("density_energy_spearman"),
        # bars the chart spans: `stepforge-inpaint --bars S:E` takes E <= bars
        "bars": (int(chart.events.tick[-1]) // C.G + 1) if len(chart.events) else 0,
        "engine": {"name": ENGINE_NAME, "preset": preset,
                   "style": {f: (None if v < 0 else round(float(v), 4))
                             for f, v in zip(cond.FIELDS, style)},
                   **chart.stats},
    }
    (out_dir / f"{difficulty}.qa.json").write_text(json.dumps(
        {"difficulty": difficulty, "meter": meter, "metrics": metrics,
         "previews": previews, "critic": None, "engine": row["engine"]}, indent=2))
    return row


def _design_note(difficulty: str, style, preset, stats: dict) -> dict:
    """The neural stand-in for <diff>.design.json: there is no designer
    prose, so say what the engine was asked for and what it did."""
    asked = [f"{f}={float(v):.2f}" for f, v in zip(cond.FIELDS[1:9], style[1:9]) if v >= 0]
    r = stats.get("rhythm", {})
    note = (f"StepForge Flow (neural). Preset: {preset or 'none'}. "
            f"Style sliders: {', '.join(asked) if asked else 'left to the model'}. "
            f"Threshold {r.get('theta')}, {r.get('notes')} notes "
            f"({r.get('jumps', 0)} jumps, {r.get('holds', 0)} freezes).")
    return {"engine": ENGINE_NAME, "design_notes": note, "sections": [], "stats": stats}


def build_song(track_id: str, opts: config.RunOptions, difficulties=ALL_DIFFICULTIES,
               *, model_dir=None, style_preset: str | None = None,
               style_overrides: dict | None = None, seed: int | None = None,
               tau: float = infer.DEFAULT_TAU, int8: bool = False,
               engine: infer.StepManiaNeuralEngine | None = None,
               style_rag: bool = False) -> dict:
    """Chart `track_id` at each of `difficulties` with the neural engine and
    write the song folder. Returns the same report dict the LLM path returns.

    With `style_rag`, sliders the caller left unset are filled from the phrase
    index (if one is installed beside the model): the style human charters used
    on the most acoustically similar 4-bar phrases. Explicit sliders win."""
    from ..adapter import _audio_path, _track_meta
    from ....analyze import analyze_track
    with ledger.stage("song", song=track_id, target="stepmania"):
        engine = engine or infer.StepManiaNeuralEngine(model_dir, int8=int8)
        analysis = analyze_track(track_id, opts)
        out_dir = config.STEPMANIA_DIR / config.TRACK_CATALOGUE.get(track_id, track_id)
        report = chart_audio(_audio_path(track_id), analysis, out_dir,
                             _track_meta(track_id, analysis), difficulties, engine,
                             track_id=track_id, style_preset=style_preset,
                             style_overrides=style_overrides, seed=seed, tau=tau,
                             style_rag=style_rag)
        report["out"] = str(out_dir.relative_to(config.REPO_ROOT))
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        return report


@dataclass
class SongContext:
    """Everything about one song that every difficulty is charted from."""
    g: GF.GridFeatures
    align: dict                 # the grid actually used (see `_features`)
    analysis: dict              # the DSP analysis on that grid, on the DSP's clock (QA)
    synced: dict                # the same on true audio time (what is written/played)
    on_transient: np.ndarray    # per tick: a DSP onset within 32 ms
    dsp_bpm: float
    dsp_offset: float


def prepare_song(audio, analysis: dict, log=print, engine=None) -> SongContext:
    """Features, grid alignment, tempo and beat-phase checks for one audio file
    — the part of charting that does not depend on difficulty or style. The
    beat-phase check needs the rhythm model, so it runs only with `engine`."""
    dsp_bpm, dsp_offset = analysis["bpm"], analysis["offset"]
    log("    [features] beat-warping audio onto the 48-tick grid…", flush=True)
    g, align = _features(str(audio), dsp_bpm, dsp_offset, analysis["duration_s"],
                         analysis.get("beat_backend", "local_dsp"), engine)
    if align["phase_shift_beats"]:
        log(f"    [features] grid moved {align['phase_shift_beats']:g} beats later "
            f"(off-beat evidence {align['phase_gain']:+.1f} nats; beat-1 share by "
            f"bar position {align['bar_share']})", flush=True)
    bpm = align["bpm"]
    # Two views of one chart, both on the refined tempo. What is written and
    # previewed plays against real audio: `synced`. QA compares notes with the
    # DSP's own onsets, whose clock runs `shift_s` early, so QA gets the same
    # grid expressed on that clock: `analysis`.
    if align["tempo_ratio"] != 1.0 or align["phase_shift_beats"]:
        # the DSP's tempo or bar line was replaced: its bars are not this chart's
        analysis = _rebar(analysis, bpm, align["offset"] - align["shift_s"])
    if align["tempo_ratio"] != 1.0:
        log(f"    [features] DSP tempo {dsp_bpm:.2f} fits the onsets badly "
            f"({align['dsp_tempo_fit']}); using {bpm:.2f} ({align['tempo_fit']})",
            flush=True)
    synced = dict(analysis, bpm=round(bpm, 3),
                  offset=round(align["offset"] - HUMAN_SYNC_LEAD_S, 4))
    analysis = dict(analysis, bpm=bpm, offset=align["offset"] - align["shift_s"])
    log(f"    [features] grid seated {align['shift_s'] * 1000:+.0f} ms from the DSP "
        f"offset (contrast {align['contrast']:.2f}); tempo {dsp_bpm:.3f} -> {bpm:.3f}",
        flush=True)
    return SongContext(g, align, analysis, synced, transient_ticks(analysis, len(g.X)),
                       dsp_bpm, dsp_offset)


def chart_song(ctx: SongContext, difficulties, engine: infer.StepManiaNeuralEngine, *,
               track_id: str, style_preset: str | None = None,
               style_overrides: dict | None = None, seed: int | None = None,
               tau: float = infer.DEFAULT_TAU, style_rag: bool = False,
               log=None) -> tuple:
    """The charts themselves: (charts, styles, retrieved), each keyed by
    difficulty. Writes nothing."""
    g, analysis = ctx.g, ctx.analysis
    rag_index, phrase_vecs = None, None
    if style_rag:
        from . import style_rag as rag
        rag_index = rag.load_index(engine.model.dir)
        if rag_index is not None and rag_index.embedder == rag.GridStatEmbedder.name:
            phrase_vecs = rag.GridStatEmbedder().embed_phrases(g.core)
        else:
            rag_index = None
    charts, styles, retrieved = {}, {}, {}
    for diff in difficulties:
        overrides = dict(style_overrides or {})
        if rag_index is not None:
            retrieved[diff] = rag_index.song_style(phrase_vecs, diff)
            preset_set = set() if not style_preset else set(retrieved[diff])
            for key, val in retrieved[diff].items():
                if key not in overrides and key not in preset_set:
                    overrides[key] = val
        style = cond.resolve_style(diff, style_preset, overrides or None)
        styles[diff] = style
        charts[diff] = engine.generate(
            g.X, g.tick_times_sec, diff, style, analysis["duration_s"],
            seed=_default_seed(track_id, diff) if seed is None else seed,
            tau=tau, energy_per_bar=analysis.get("energy_curve"),
            veto=flow_veto(diff, analysis["bpm"]), on_transient=ctx.on_transient,
            progress=(lambda stage, d=diff: log(f"    [{d}] neural: {stage}…", flush=True))
            if log else None)
    return charts, styles, retrieved


def chart_audio(audio, analysis: dict, out_dir, meta: dict,
                difficulties=ALL_DIFFICULTIES,
                engine: infer.StepManiaNeuralEngine | None = None, *,
                track_id: str | None = None, model_dir=None,
                style_preset: str | None = None, style_overrides: dict | None = None,
                seed: int | None = None, tau: float = infer.DEFAULT_TAU,
                style_rag: bool = False, previews: bool = True, log=print) -> dict:
    """Chart one audio file into `out_dir` given its DSP analysis.

    The catalogue-free core of `build_song`: `audio` is any decodable file,
    `analysis` the dict `dsp.analyze_signal` returns for it, and `meta`
    {"title", "artist", "music"} names the song and the audio file as it will
    sit in the folder. Used directly by the benchmark harness, which charts
    songs that are not in BeatForge's track catalogue.
    """
    audio, out_dir = str(audio), Path(out_dir)
    track_id = track_id or Path(meta["music"]).stem
    engine = engine or infer.StepManiaNeuralEngine(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = prepare_song(audio, analysis, log, engine)
    g, align, analysis, synced = ctx.g, ctx.align, ctx.analysis, ctx.synced
    dsp_bpm, dsp_offset, bpm = ctx.dsp_bpm, ctx.dsp_offset, align["bpm"]
    charts, styles, retrieved = chart_song(
        ctx, difficulties, engine, track_id=track_id, style_preset=style_preset,
        style_overrides=style_overrides, seed=seed, tau=tau, style_rag=style_rag, log=log)

    placements = {d: infer.to_placements(c) for d, c in charts.items()}
    meters = _strict_meters({d: compute_meter(p, bpm, d)
                             for d, p in placements.items()})
    report = {"track_id": track_id, "bpm": synced["bpm"], "offset": synced["offset"],
              "mode": ENGINE_NAME, "chunk_mode": "never", "charts": {},
              "engine": {"name": ENGINE_NAME, "model_dir": str(engine.model.dir),
                         "dsp_offset": dsp_offset, "dsp_bpm": dsp_bpm,
                         "bpm_snapped": align["bpm_snapped"],
                         "bpm_refined": align["bpm_refined"],
                         "dsp_drift_ms_per_min": align["dsp_drift_ms_per_min"],
                         "tempo_ratio": align["tempo_ratio"],
                         "tempo_switched_by": align["tempo_switched_by"],
                         "tempo_fit": align["tempo_fit"],
                         "dsp_tempo_fit": align["dsp_tempo_fit"],
                         "phase_shift_beats": align["phase_shift_beats"],
                         "phase_gain": align["phase_gain"],
                         "bar_share": align["bar_share"],
                         "grid_shift_ms": round(align["shift_s"] * 1000, 1),
                         "grid_contrast": round(align["contrast"], 3),
                         "sync_lead_ms": HUMAN_SYNC_LEAD_S * 1000,
                         "quantization": engine.model.quantization,
                         "codec_version": C.CODEC_VERSION, "preset": style_preset,
                         "style_rag": {d: {k: round(v, 4) for k, v in r.items()}
                                       for d, r in retrieved.items()} or None}}
    per_difficulty = {}
    for diff in difficulties:
        log(f"    [{diff}] rendering previews…", flush=True)
        (out_dir / f"{diff}.design.json").write_text(json.dumps(
            _design_note(diff, styles[diff], style_preset, charts[diff].stats),
            indent=2))
        report["charts"][diff] = _chart_report(
            placements[diff], analysis, diff, meters[diff], charts[diff],
            g.tick_times_sec, out_dir, audio, styles[diff], style_preset,
            preview_analysis=synced, previews=previews)
        per_difficulty[diff] = (meters[diff], placements[diff])
        log(f"    [{diff}] DONE — meter {meters[diff]}, "
            f"{len(placements[diff])} notes", flush=True)

    files = write_song_folder(meta, synced, per_difficulty, audio, out_dir,
                              formats=("ssc", "sm"),
                              subdiv={d: TICK_SUBDIV for d in per_difficulty})
    for fmt in ("ssc", "sm"):
        stabilize_simfile(files[fmt])
    report["files"] = files
    report["out"] = str(out_dir)
    order = [d for d in ALL_DIFFICULTIES if d in report["charts"]]
    ms = [report["charts"][d]["meter"] for d in order]
    report["meter_monotonic"] = all(a < b for a, b in zip(ms, ms[1:]))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- #
# Inpainting an existing chart
# --------------------------------------------------------------------------- #
def _resolve_ssc(track_id: str | None, ssc: str | None) -> tuple:
    if ssc:
        p = Path(ssc)
        if not p.exists():
            raise FileNotFoundError(f"no simfile at {p}")
        return p, p.parent.name
    if not track_id:
        raise ValueError("give --track or --ssc")
    base = config.TRACK_CATALOGUE.get(track_id, track_id)
    folder = config.STEPMANIA_DIR / base
    p = folder / f"{base}.ssc"
    if not p.exists():
        raise FileNotFoundError(
            f"no chart to inpaint: {p} does not exist. Generate it first with "
            f"`beatforge stepforge --track {track_id} --engine neural`.")
    return p, base


def _find_chart(sf, difficulty: str):
    want = BUDGETS[difficulty].sm_difficulty.lower()
    for ch in sf.charts:
        if (ch.stepstype or "").strip() == "dance-single" and \
                (ch.difficulty or "").strip().lower() == want:
            return ch
    raise ValueError(f"the simfile has no dance-single {difficulty} chart")


def _recorded_style(chart_row: dict) -> tuple:
    """(preset, slider overrides) a report row says its chart was made with."""
    eng = chart_row.get("engine") or {}
    style = eng.get("style") or {}
    over = {}
    for key in cond.FIELDS[2:9]:
        if style.get(key) is not None:
            over[key] = float(style[key])
    if style.get("nps_norm") is not None:
        over["nps"] = float(style["nps_norm"]) * 16.0
    if style.get("meter_norm") is not None:
        over["meter"] = float(style["meter_norm"]) * 15.0
    return None, over          # the sliders already include what the preset set


def parse_bars(spec: str) -> tuple:
    """'16:24' -> (16, 24). Bars are 0-based and the end is exclusive."""
    try:
        a, b = (int(x) for x in str(spec).split(":"))
    except ValueError:
        raise ValueError(f"--bars wants START:END, got '{spec}'") from None
    if a < 0 or b <= a:
        raise ValueError(f"--bars needs 0 <= START < END, got '{spec}'")
    return a, b


def inpaint_song(difficulty: str, bars: tuple, *, track_id: str | None = None,
                 ssc: str | None = None, seed: int | None = None, model_dir=None,
                 style_preset: str | None = None, style_overrides: dict | None = None,
                 tau: float = infer.DEFAULT_TAU, int8: bool = False,
                 engine: infer.StepManiaNeuralEngine | None = None) -> dict:
    """Re-roll bars [start, end) of one chart in place: rewrite that chart in
    the folder's .ssc and .sm, refresh its previews and its report entry, and
    leave every other chart byte-identical."""
    import simfile
    from simfile.timing import TimingData
    from ..quantize import to_simfile_notes

    ssc_path, base = _resolve_ssc(track_id, ssc)
    folder = ssc_path.parent
    sf = simfile.open(str(ssc_path), strict=False)
    chart = _find_chart(sf, difficulty)
    td = TimingData(sf, chart)
    bpms = [(float(b.beat), float(b.value)) for b in td.bpms]
    timing = GF.TimingMap(-float(td.offset), bpms,
                          [(float(s.beat), float(s.value)) for s in td.stops])
    audio = folder / (sf.music or f"{base}.ogg")
    if not audio.exists():
        raise FileNotFoundError(f"audio for the chart not found: {audio}")
    engine = engine or infer.StepManiaNeuralEngine(model_dir, int8=int8)

    y = GF.load_audio(str(audio))
    duration = len(y) / GF.SR
    triples, dropped, _ = C.simfile_chart_to_ticks(chart, timing.bpm_at)
    events, st = C.events_from_notes(triples)
    if dropped or st.hands or st.dropped:
        raise ValueError("this chart has notes the neural codec cannot represent "
                         "(off the 48-tick grid, or 3+ panels); it was not written "
                         "by the neural engine and cannot be inpainted")
    n_ticks = max(timing.ticks_for_duration(duration),
                  int(np.ceil((int(events.tick[-1]) + 1) / C.G)) * C.G)
    start_bar, end_bar = bars
    if start_bar * C.G > int(events.tick[-1]):
        raise ValueError(f"bar {start_bar} is past the end of the chart "
                         f"({int(events.tick[-1]) // C.G + 1} bars)")
    g = GF.extract_grid_features(y, timing, n_ticks=n_ticks)
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "track_id": track_id or base, "mode": ENGINE_NAME, "charts": {}}
    old = report.get("charts", {}).get(difficulty, {})
    if style_preset is None and not style_overrides:
        # re-roll in the style the chart was generated with, as recorded
        style_preset, style_overrides = _recorded_style(old)
    style = cond.resolve_style(difficulty, style_preset, style_overrides)
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31))
    result = engine.inpaint_section(events, g.X, g.tick_times_sec, difficulty, style,
                                    start_bar, end_bar, seed=seed, tau=tau,
                                    veto=flow_veto(difficulty, bpms[0][1]))
    placements = infer.to_placements(result)

    notes = str(to_simfile_notes(placements, TICK_SUBDIV))
    written = []
    for path in (ssc_path, ssc_path.with_suffix(".sm")):
        if not path.exists():
            continue
        target = simfile.open(str(path), strict=False)
        _find_chart(target, difficulty).notes = notes
        path.write_text(str(target))
        stabilize_simfile(path)                      # must re-open, byte-stable
        written.append(str(path))

    # refresh this chart's previews + report entry; other charts are untouched
    from ....analyze import load_cached
    analysis = (load_cached(track_id) if track_id else None) or load_cached(base) or {
        "bpm": bpms[0][1], "offset": -float(td.offset), "duration_s": duration,
        "onsets": [], "sections": [], "energy_curve": [], "meter": 4}
    meter = int(old.get("meter") or float(chart.meter))
    true_frame = dict(analysis, bpm=bpms[0][1], offset=-float(td.offset))
    row = _chart_report(placements, analysis, difficulty, meter, result,
                        g.tick_times_sec, folder, str(audio), style, style_preset,
                        preview_analysis=true_frame)
    row["engine"]["inpainted"] = (old.get("engine", {}).get("inpainted", [])
                                  + [{"bars": [start_bar, end_bar], "seed": seed,
                                      "changed": result.stats["changed"]}])
    report.setdefault("charts", {})[difficulty] = row
    report_path.write_text(json.dumps(report, indent=2))
    return {"track_id": track_id or base, "difficulty": difficulty,
            "bars": [start_bar, end_bar], "seed": seed,
            "changed": result.stats["changed"],
            "notes_in_section": result.stats["notes_in_section"],
            "files": written, "out": str(folder)}
