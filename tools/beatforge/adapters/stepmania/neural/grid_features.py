"""grid_features.py — beat-synchronous 48-tick feature extraction (spec §3,
REQ-SFN-02).

Audio features are computed on a 10 ms hop and then warped onto the musical
grid: tick k sits at beat k/12, and its timestamp comes from a `TimingMap`
(the simfile's #OFFSET/#BPMS/#STOPS in training, BeatForge's measured bpm +
offset at inference). The network therefore never has to learn tempo division
from millisecond frames; it is told how long each tick lasts (`dt_ms`) instead.

Channel layout of X (T, 64):

     0..19  broadband_mel    20 log-mel bands, 30 Hz - 11 kHz, per-track z-norm
    20..25  hpss_onset_flux  harmonic x {low, mid, high}, percussive x {low, mid, high}
    26..31  flux_delta       tick-to-tick first difference of 20..25
    32..47  stem_features    4 stems x [level, flux low, flux mid, flux high];
                             all zero when no stems are supplied
    48..53  chroma_contour   4 pitch-height quantiles, spectral centroid, d-centroid
    54..63  metric_timing    subdivision one-hot (6), 4-bar phase sin/cos (2),
                             log(dt/40) (1), song progress (1)

Only 31 of those channels depend on the audio and need storing (`core`: mel,
flux, and five contour values); the deltas and the metric block are rebuilt by
`assemble`, which keeps a training shard at half the size.

One STFT (512-sample window, 23 ms) feeds everything. The short window is a
deliberate choice for the flux channels: spectral flux fires when energy first
enters the analysis window, which is up to half a window BEFORE the sound, and
at 175 BPM a tick is only 28 ms wide. A 46 ms window can put a transient's
flux peak on the previous tick; a 23 ms one cannot.

Spec §3.1 also asks for BPM_k/150 as an input. It is not a separate channel:
the table in §3.2 allots metric_timing exactly ten slots, and log(dt/40) is the
same quantity (dt = 5000 / BPM) apart from stops, which only dt can express.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal

from .codec import G, TICKS_PER_BEAT

SR = 22050
N_FFT = 512
HOP = 220                         # 9.98 ms
HOP_S = HOP / SR
N_MEL = 20
FMIN, FMAX = 30.0, 11000.0
D_IN = 64
D_CORE = 31
#: Flux leads the sound by a fraction of the window (see module docstring):
#: n_fft/4 for a click, ~0 for a sustained onset. Shift flux timestamps by the
#: middle of that range.
FLUX_LEAD_S = 0.003

MEL = slice(0, 20)
FLUX = slice(20, 26)
DFLUX = slice(26, 32)
STEM = slice(32, 48)
CHROMA = slice(48, 54)
METRIC = slice(54, 64)
STEM_NAMES = ("drums", "bass", "vocals", "other")

_BAND_EDGES_HZ = (150.0, 2000.0)             # low | mid | high
_HPSS_TIME, _HPSS_FREQ = 17, 13              # median kernels (frames, bins)
_HPSS_STEP = 4                               # medians evaluated every 4th row
_FLUX_GAMMA = 100.0
_CONTOUR_LO, _CONTOUR_HI = 60.0, 5000.0
_QUANTILES = (0.25, 0.5, 0.75, 0.9)
_MAX_HALF_WINDOW_S = 0.25


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
# `TimingMap` (beat <-> seconds: an offset, BPM segments, stops) lives in
# ..timing so the modules outside the neural engine time a chart the same way.
from ..timing import TICKS_PER_BEAT as _T_TPB, TICKS_PER_MEASURE as _T_G, TimingMap  # noqa: E402,F401

assert (_T_TPB, _T_G) == (TICKS_PER_BEAT, G), "timing.py and codec.py disagree on the grid"


# --------------------------------------------------------------------------- #
# Audio I/O
# --------------------------------------------------------------------------- #
def load_audio(path: str) -> np.ndarray:
    """Decode to mono float32 at SR, peak-normalized. Deterministic."""
    import soundfile as sf
    y, sr = sf.read(str(path), always_2d=True, dtype="float32")
    y = y.mean(axis=1)
    if sr != SR:
        g = int(np.gcd(int(sr), SR))
        y = scipy.signal.resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y / peak).astype(np.float32) if peak > 0 else y.astype(np.float32)


# --------------------------------------------------------------------------- #
# Frame-rate features
# --------------------------------------------------------------------------- #
_WINDOW = scipy.signal.get_window("hann", N_FFT, fftbins=True).astype(np.float32)
_FREQS = np.fft.rfftfreq(N_FFT, 1.0 / SR)
_MAG_SCALE = 2.0 / _WINDOW.sum()             # full-scale sine -> magnitude 1


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, np.float64) / 700.0)


def _mel_filterbank() -> np.ndarray:
    pts = 700.0 * (10 ** (np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MEL + 2)
                          / 2595.0) - 1.0)
    fb = np.zeros((N_MEL, len(_FREQS)), np.float32)
    for m in range(N_MEL):
        lo, ctr, hi = pts[m], pts[m + 1], pts[m + 2]
        up = (_FREQS - lo) / max(ctr - lo, 1e-9)
        down = (hi - _FREQS) / max(hi - ctr, 1e-9)
        tri = np.clip(np.minimum(up, down), 0.0, None)
        if tri.sum() <= 0:                    # band narrower than one FFT bin
            tri[np.argmin(np.abs(_FREQS - ctr))] = 1.0
        fb[m] = tri / tri.sum()
    return fb


_MEL_FB = _mel_filterbank()
_BANDS = (_FREQS < _BAND_EDGES_HZ[0],
          (_FREQS >= _BAND_EDGES_HZ[0]) & (_FREQS < _BAND_EDGES_HZ[1]),
          _FREQS >= _BAND_EDGES_HZ[1])
_CONTOUR_BINS = (_FREQS >= _CONTOUR_LO) & (_FREQS <= _CONTOUR_HI)
_CONTOUR_POS = (np.log2(_FREQS[_CONTOUR_BINS] / _CONTOUR_LO)
                / np.log2(_CONTOUR_HI / _CONTOUR_LO)).astype(np.float32)


def stft_mag(y: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram (frames, bins), centered frames, amplitude-scaled."""
    y = np.asarray(y, np.float32)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(yp) - N_FFT) // HOP
    out = np.empty((n_frames, N_FFT // 2 + 1), np.float32)
    idx = np.arange(N_FFT)[None, :]
    for a in range(0, n_frames, 4096):
        b = min(n_frames, a + 4096)
        frames = yp[idx + (np.arange(a, b) * HOP)[:, None]] * _WINDOW
        out[a:b] = np.abs(np.fft.rfft(frames, axis=1)) * _MAG_SCALE
    return out


def _flux_bands(mag: np.ndarray) -> np.ndarray:
    """Positive spectral flux of log-compressed magnitude in low/mid/high."""
    s = np.log1p(_FLUX_GAMMA * mag)
    d = np.maximum(s[1:] - s[:-1], 0.0)
    out = np.zeros((len(mag), 3), np.float32)
    for i, band in enumerate(_BANDS):
        out[1:, i] = d[:, band].mean(1)
    return out


def _median_along0(x: np.ndarray, k: int, step: int) -> np.ndarray:
    """Running median of width `k` along axis 0, evaluated every `step` rows
    and linearly interpolated back to every row.

    A median filter's output is smooth along its own axis — that is what it is
    for — so sampling it sparsely loses almost nothing and costs 1/step as
    much. The full-resolution scipy filter took ~9 s on a three-minute song,
    three quarters of the whole feature pass. Pure numpy on purpose: identical
    results on any scipy version, which training/inference parity depends on.
    """
    n = len(x)
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)], 0)
    centers = np.arange(0, n + step, step)
    centers[-1] = n - 1
    centers = np.unique(np.minimum(centers, n - 1))
    med = np.empty((len(centers),) + x.shape[1:], x.dtype)
    offs = np.arange(k)
    for a in range(0, len(centers), 512):
        c = centers[a:a + 512]
        # window axis last + a full sort: numpy's SIMD float32 sort beats
        # np.partition here by ~3x
        win = np.ascontiguousarray(np.moveaxis(xp[c[:, None] + offs[None, :]], 1, -1))
        med[a:a + len(c)] = np.sort(win, axis=-1)[..., pad]
    if len(centers) == 1:
        return np.repeat(med, n, 0)
    rows = np.arange(n)
    j = np.clip(np.searchsorted(centers, rows, side="right") - 1, 0, len(centers) - 2)
    span = (centers[j + 1] - centers[j]).astype(np.float32)
    frac = ((rows - centers[j]) / span).astype(np.float32)
    frac = frac.reshape((-1,) + (1,) * (x.ndim - 1))
    return med[j] * (1.0 - frac) + med[j + 1] * frac


def _hpss(mag: np.ndarray):
    """Median-filter harmonic/percussive separation (soft masks)."""
    harm = _median_along0(mag, _HPSS_TIME, _HPSS_STEP)
    perc = _median_along0(np.ascontiguousarray(mag.T), _HPSS_FREQ, _HPSS_STEP).T
    h2, p2 = harm * harm, perc * perc
    mask = h2 / (h2 + p2 + 1e-12)
    return mag * mask, mag * (1.0 - mask)


@dataclass
class FrameFeatures:
    level: np.ndarray       # (F, 25) mel (20) + contour (5): averaged onto ticks
    flux: np.ndarray        # (F, 6)  H/P x low/mid/high: peak-pooled onto ticks
    silence: np.ndarray     # (25,)   `level` row standing for "no audio here"
    n_frames: int
    #: Uncompressed total flux, for sync estimation. The stored `flux` is
    #: log-compressed per channel, which is right for a network input and wrong
    #: for asking "do the notes sit on the transients" — compression flattens
    #: exactly the contrast that question measures.
    onset_envelope: np.ndarray = None


def frame_features(y: np.ndarray) -> FrameFeatures:
    mag = stft_mag(y)
    f = len(mag)
    # float32 matmul through Apple's Accelerate raises spurious FP warnings on
    # finite data (numpy >= 2); the results are checked finite by `assemble`.
    with np.errstate(all="ignore"):
        logmel = np.log((mag * mag) @ _MEL_FB.T + 1e-7)
    mu, sd = logmel.mean(0), logmel.std(0) + 1e-5
    mel = (logmel - mu) / sd
    mel_silence = (np.log(1e-7) - mu) / sd

    harm, perc = _hpss(mag)
    flux = np.concatenate([_flux_bands(harm), _flux_bands(perc)], axis=1)
    envelope = flux.sum(1)
    scale = flux.mean(0) + flux.std(0) + 1e-6
    flux = np.log1p(flux / scale).astype(np.float32)

    # pitch-height contour of the harmonic part, on a log-frequency axis
    w = harm[:, _CONTOUR_BINS] ** 2
    tot = w.sum(1, keepdims=True)
    cdf = np.cumsum(w, axis=1) / np.maximum(tot, 1e-12)
    quiet = tot[:, 0] <= 1e-10
    contour = np.empty((f, 5), np.float32)
    for i, qv in enumerate(_QUANTILES):
        contour[:, i] = _CONTOUR_POS[np.minimum((cdf < qv).sum(1), len(_CONTOUR_POS) - 1)]
    wm = mag[:, _CONTOUR_BINS]
    with np.errstate(all="ignore"):
        contour[:, 4] = (wm @ _CONTOUR_POS) / np.maximum(wm.sum(1), 1e-9)
    contour[quiet] = 0.0

    level = np.concatenate([mel, contour], axis=1).astype(np.float32)
    silence = np.concatenate([mel_silence, np.zeros(5)]).astype(np.float32)
    return FrameFeatures(level, flux, silence, f, envelope)


# --------------------------------------------------------------------------- #
# Warping onto the grid
# --------------------------------------------------------------------------- #
def _grid_windows(timing: TimingMap, n_ticks: int):
    k = np.arange(n_ticks, dtype=np.float64)
    t = timing.time_at(k / TICKS_PER_BEAT)
    left = t - timing.time_at((k - 0.5) / TICKS_PER_BEAT)
    right = timing.time_at((k + 0.5) / TICKS_PER_BEAT) - t
    left = np.clip(left, HOP_S, _MAX_HALF_WINDOW_S)
    right = np.clip(right, HOP_S, _MAX_HALF_WINDOW_S)
    t_next = timing.time_at((k + 1) / TICKS_PER_BEAT)
    dt_ms = np.clip((t_next - t) * 1000.0, 10.0, 500.0)
    return t, left, right, dt_ms


def _pool(values: np.ndarray, fill: np.ndarray, t, left, right, lead_s: float,
          mode: str) -> np.ndarray:
    """Triangular pooling of frame-rate `values` (F, C) onto tick windows.
    `mode='mean'` is the weighted average; `mode='peak'` the weighted maximum
    (used for flux, so a transient keeps its height whatever the tick width).
    Frames outside the audio read as `fill`."""
    f = len(values)
    tt = t - lead_s                              # frame-time each tick centres on
    lo = np.floor((tt - left) / HOP_S).astype(np.int64)
    hi = np.ceil((tt + right) / HOP_S).astype(np.int64)
    k = int(min(64, max(1, (hi - lo).max() + 1)))
    idx = lo[:, None] + np.arange(k)[None, :]
    ft = idx * HOP_S
    d = ft - tt[:, None]
    w = np.where(d < 0, 1.0 + d / left[:, None], 1.0 - d / right[:, None])
    w = np.clip(w, 0.0, 1.0).astype(np.float32)
    padded = np.concatenate([values, fill[None, :]], axis=0)
    safe = np.where((idx >= 0) & (idx < f), idx, f)
    g = padded[safe]                             # (T, K, C)
    if mode == "peak":
        return (g * w[:, :, None]).max(1)
    wsum = w.sum(1, keepdims=True)
    # a window narrower than the hop can miss every frame: fall back to nearest
    empty = wsum[:, 0] <= 1e-6
    out = (g * w[:, :, None]).sum(1) / np.maximum(wsum, 1e-6)
    if empty.any():
        near = np.clip(np.round(tt[empty] / HOP_S).astype(np.int64), -1, f)
        out[empty] = padded[np.where((near >= 0) & (near < f), near, f)]
    return out


def metric_features(n_ticks: int, dt_ms: np.ndarray, tick0: int = 0,
                    total: int | None = None) -> np.ndarray:
    """The 10 metric_timing channels — a function of the grid alone. `tick0`
    and `total` place a window inside a longer song (a training crop)."""
    k = np.arange(tick0, tick0 + n_ticks)
    rows = np.arange(n_ticks)
    total = n_ticks if total is None else total
    m = np.zeros((n_ticks, 10), np.float32)
    r = k % TICKS_PER_BEAT
    cls = np.full(n_ticks, 5)                    # 48th
    cls[r % 2 == 0] = 4                          # 24th
    cls[r % 3 == 0] = 3                          # 16th
    cls[r % 4 == 0] = 2                          # 12th (triplet 8th)
    cls[r % 6 == 0] = 1                          # 8th
    cls[r == 0] = 0                              # 4th
    m[rows, cls] = 1.0
    phase = 2.0 * np.pi * (k / TICKS_PER_BEAT) / 16.0
    m[:, 6], m[:, 7] = np.sin(phase), np.cos(phase)
    m[:, 8] = np.log(dt_ms / 40.0)
    m[:, 9] = k / max(1, total)
    return m


def assemble(core: np.ndarray, dt_ms: np.ndarray, stems: np.ndarray | None = None,
             start: int = 0, stop: int | None = None) -> np.ndarray:
    """core (T, 31) [+ stems (T, 16)] -> X float32 for ticks [start, stop).

    The window may run past either end of the song (start < 0, stop > T); the
    overhang is zero audio with a valid metric block, which is what a training
    crop or a note-context window near the edge needs. With the defaults this
    is the whole song, (T, 64).
    """
    t = len(core)
    stop = t if stop is None else stop
    n = stop - start
    a, b = max(start, 0), min(stop, t)
    x = np.zeros((n, D_IN), np.float32)
    dt = np.full(n, 40.0, np.float32)
    if b > a:
        c = np.asarray(core[a:b], np.float32)
        prev = np.asarray(core[a - 1], np.float32) if a > 0 else c[0]
        d = np.diff(np.concatenate([prev[None, :], c], axis=0), axis=0)
        o = a - start
        x[o:o + b - a, MEL] = c[:, 0:20]
        x[o:o + b - a, FLUX] = c[:, 20:26]
        x[o:o + b - a, DFLUX] = d[:, 20:26]
        if stems is not None:
            x[o:o + b - a, STEM] = stems[a:b]
        x[o:o + b - a, 48:53] = c[:, 26:31]
        x[o:o + b - a, 53] = d[:, 30]
        dt[o:o + b - a] = np.asarray(dt_ms[a:b], np.float32)
    x[:, METRIC] = metric_features(n, dt, tick0=start, total=t)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def warp_core(ff: FrameFeatures, timing: TimingMap, n_ticks: int):
    """Frame features -> (core (T, 31), tick_times_sec (T,), dt_ms (T,))."""
    t, left, right, dt_ms = _grid_windows(timing, n_ticks)
    level = _pool(ff.level, ff.silence, t, left, right, 0.0, "mean")
    flux = _pool(ff.flux, np.zeros(6, np.float32), t, left, right, FLUX_LEAD_S, "peak")
    core = np.concatenate([level[:, :20], flux, level[:, 20:]], axis=1)
    return core.astype(np.float32), t, dt_ms.astype(np.float32)


def stem_features(stems: dict, timing: TimingMap, n_ticks: int) -> np.ndarray:
    """4 stems x [level, flux low, flux mid, flux high] -> (T, 16)."""
    t, left, right, _ = _grid_windows(timing, n_ticks)
    out = np.zeros((n_ticks, 16), np.float32)
    for i, name in enumerate(STEM_NAMES):
        y = stems.get(name)
        if y is None:
            continue
        mag = stft_mag(np.asarray(y, np.float32))
        rms = np.log1p(30.0 * np.sqrt((mag * mag).mean(1, keepdims=True)))
        flux = _flux_bands(mag)
        flux = np.log1p(flux / (flux.mean(0) + flux.std(0) + 1e-6))
        out[:, 4 * i:4 * i + 1] = _pool(rms.astype(np.float32), np.zeros(1, np.float32),
                                        t, left, right, 0.0, "mean")
        out[:, 4 * i + 1:4 * i + 4] = _pool(flux.astype(np.float32),
                                            np.zeros(3, np.float32), t, left, right,
                                            FLUX_LEAD_S, "peak")
    return out


@dataclass
class GridFeatures:
    X: np.ndarray               # (T, 64) float32
    tick_times_sec: np.ndarray  # (T,) float64
    dt_ms: np.ndarray           # (T,) float32
    core: np.ndarray            # (T, 31) float32, the storable part of X
    has_stems: bool


def extract_grid_features(audio, timing: TimingMap, n_ticks: int | None = None,
                          stems: dict | None = None) -> GridFeatures:
    """Audio -> beat-warped tensor.

    `audio` is a path or a mono float array already at SR. `stems`, if given,
    maps 'drums'/'bass'/'vocals'/'other' to mono arrays at SR. The shapes and
    every non-stem channel are identical with or without stems.
    """
    if isinstance(audio, str) or hasattr(audio, "__fspath__"):
        y = load_audio(str(audio))
    else:
        y = np.asarray(audio, np.float32)
    if n_ticks is None:
        n_ticks = timing.ticks_for_duration(len(y) / SR)
    ff = frame_features(y)
    core, t, dt_ms = warp_core(ff, timing, n_ticks)
    st = stem_features(stems, timing, n_ticks) if stems else None
    return GridFeatures(assemble(core, dt_ms, st), t, dt_ms, core, bool(stems))


# --------------------------------------------------------------------------- #
# Sync check (corpus ingestion)
# --------------------------------------------------------------------------- #
def estimate_sync(onset_env: np.ndarray, note_times_s: np.ndarray,
                  max_shift_s: float = 0.1, step_s: float = 0.002,
                  near_s: float = 0.03, far_margin: float = 1.08) -> dict:
    """How well a chart's notes sit on the audio's transients, and the constant
    shift that would seat them best.

    Returns
      shift_s        the shift to apply (seconds; positive = notes sound later)
      score          mean onset envelope at the shifted notes / the track mean
      score_at_zero  the same with no shift
      contrast       score / the envelope averaged over +-60 ms around each
                     note. This is the quality number: ~1 means the notes are
                     no more on a transient than the music around them (wrong
                     audio, drifting tempo); a tight chart reaches 2-3.

    Two things measured on real packs shape the search. Charts commonly sit
    ~10 ms ahead of the transients (the ITG sync convention), and some files
    ship with #OFFSET 0 and are 70+ ms out — so the window is wide. But at 200
    BPM a 16th note is 75 ms, and a wide window can lock onto the neighbouring
    subdivision instead; so a shift beyond `near_s` must beat the best shift
    within it by `far_margin` to be believed.
    """
    env = np.asarray(onset_env, np.float64)
    base = env.mean() + 1e-9
    nt = np.asarray(note_times_s, np.float64) - FLUX_LEAD_S
    if len(nt) == 0 or len(env) < 2:
        return {"shift_s": 0.0, "score": 0.0, "score_at_zero": 0.0, "contrast": 0.0}
    ft = np.arange(len(env)) * HOP_S
    shifts = np.arange(-max_shift_s, max_shift_s + 1e-9, step_s)
    scores = np.array([np.interp(nt + s, ft, env, left=0.0, right=0.0).mean()
                       for s in shifts]) / base
    # light smoothing so a single lucky bin cannot win
    sm = np.convolve(scores, np.ones(5) / 5.0, mode="same")
    near = np.abs(shifts) <= near_s + 1e-9
    i_near = int(np.argmax(np.where(near, sm, -np.inf)))
    i_far = int(sm.argmax())
    i = i_far if sm[i_far] >= far_margin * sm[i_near] else i_near
    k = max(1, int(round(0.06 / HOP_S)))
    box = np.convolve(env, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    local = np.interp(nt + shifts[i], ft, box, left=0.0, right=0.0).mean() / base
    return {"shift_s": float(shifts[i]), "score": float(sm[i]),
            "score_at_zero": float(sm[len(shifts) // 2]),
            "contrast": float(sm[i] / (local + 1e-9))}
