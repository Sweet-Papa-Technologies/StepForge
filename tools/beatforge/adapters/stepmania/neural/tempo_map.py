"""tempo_map.py — one tempo where one tempo holds, a tempo map where it does not.

The engine used to assume a song has a single BPM. Most do, and for those
nothing here changes anything: `analyze` checks that the single-tempo grid the
pipeline already found stays locked to the audio from the first bar to the
last and, when it does, says so (`TempoMap.constant`) and the song is charted
exactly as before.

When it does not — a band stops and comes back in, a bar is cut short, the
tempo changes, a drummer without a click drifts — a single grid is right until
that moment and wrong for the rest of the song. `analyze` then cuts the song
into stretches that each hold one tempo and phase, fits a grid to each, and
joins them into one beat numbering:

    a pause (silence between two stretches)   -> a #STOPS entry
    anything else (cut bar, tempo change)     -> a short #BPMS segment that
                                                 carries the beat count across

The result is a `TimingMap` (..timing), which is what the feature warp, the QA
metrics, the previews and the simfile writer all read.

How a grid is judged. The song is cut into 8-bar windows, two bars apart. In
each, the onsets are folded onto the 16th-note circle of the grid and the
resulting profile is slid against the song's own typical profile until the two
match; how far it had to slide is that window's phase error (`_Windows.stats`).
A window within 18 ms is ALIGNED, one beyond 22 ms is OFF, and one whose
profile matches nowhere says nothing — an ambient intro, or a stretch at
another tempo, whose onsets smear round the circle.

What it takes to be an event. Songs that people chart at one tempo are not
metronomes: sections sit 10-25 ms apart from one instrument's attack to
another's, single-tempo fits drift a little, a syncopated bridge puts its
loud notes on the off-16ths. So a stretch gets a grid of its own only if the
single tempo is visibly wrong on it for long enough, the step to its neighbour
is clear (`_worth`), another tempo beats the old one where a wrong tempo has
nowhere to hide (8th notes, and the beat the music repeats at; `_regions`),
and the map leaves the song's windows clearly closer to the grid than the
single tempo did (`analyze`). On the 180 benchmark songs, all charted by
people at one tempo, that leaves all 180 untouched.

Which 16th-note line is THE beat is not something onsets can tell. Each
stretch gets a first answer here (`_seat_beats`); the pipeline then asks the
rhythm model, stretch by stretch.

numpy/scipy only. Nothing here touches the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal

from ..timing import TimingMap
from . import grid_features as GF

# -- windows ---------------------------------------------------------------- #
BLOCK_BEATS = 8                 # hop: 2 bars
WINDOW_BLOCKS = 4               # a window is 4 blocks: 8 bars
MIN_ONSETS = 16                 # fewer onsets than this in a window say nothing
BIN_S = 0.002
SMOOTH_S = 0.005                # profile smoothing (sigma)
CONF_MIN = 0.75                 # correlation with the song's own profile, to be believed
FIT_MIN = 0.15                  # ...and onset energy on the grid, above chance,
CONF_SURE = 0.85                # ...unless the profile matches this well
HIT_S = 0.020                   # an onset within this of a grid line is on it

# -- the single-tempo check (the fast path) --------------------------------- #
ALIGN_TOL_S = 0.018             # a window within this of the grid: aligned
BREAK_TOL_S = 0.022             # beyond this, window after window: a break
FAST_PATH_SHARE = 0.95          # share of confident windows that must be aligned

# -- what counts as an event -------------------------------------------------- #
STEP_MIN_S = 0.030              # a phase step with no pause must be at least this
PAUSE_STEP_MIN_S = 0.012        # ...and across a pause, this
STEP_WINDOWS = 4                # windows a re-seated stretch must hold for (~14 bars)
TRIPLET_WINDOWS = 8             # ...when it sits a triplet away from the grid
PAUSE_WINDOWS = 2               # ...when a pause comes before or after it
TEMPO_MIN = 1.5e-3              # tempos closer than this are the same tempo
STRETCH_GAIN = 0.08             # onset energy a new grid must add on its own stretch
PERIODICITY_GAIN = 0.03         # how much more the music must repeat at a new tempo's beat
LOST_FIT_MIN = 0.15             # 8th-note fit a stretch needs when the single tempo is lost
ACCEPT_RATIO = 0.80             # the map's mean window phase error vs the single tempo's
COST_CAP_S = 0.040              # a window further out than this, or unreadable, costs this

# -- finding other grids ----------------------------------------------------- #
SAME_TEMPO = 0.008              # +-0.8%: the same tempo, re-seated (pause, cut bar)
WIDE_RANGE = (0.71, 1.41)       # tempo ratios searched for a real tempo change
TEMPO_STEP = 5e-4
LOCAL_FIT_MIN = 0.20            # a stretch must fit its own grid at least this well
ABS_BPM = (95.0, 205.0)         # tempos looked for when the single tempo is lost
SWITCH_PENALTY = 1.5            # windows a change of grid must explain
MAX_GRIDS = 24
SNAP_TOL = 1.5e-4               # a round BPM within twice this of the fit may be taken

# -- pauses ------------------------------------------------------------------ #
PAUSE_DROP_DB = 20.0            # this far below the song's median level...
PAUSE_MIN_S = 0.25              # ...for this long: a pause
MIN_STOP_S = 0.004              # a shorter stop is not written


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    """A stretch of the song that holds one tempo and phase."""
    t_start: float              # the stretch the grid was fitted over (seconds)
    t_end: float
    period: float               # seconds per beat
    anchor: float               # the time of one beat line
    fit: float = 0.0            # onset energy on its grid, above chance
    kept: bool = False          # the pipeline's own single-tempo grid, unchanged
    windows: int = 0            # windows it was found aligned in
    # what it was found from: those windows, and the grid as first fitted
    _idx: np.ndarray | None = field(default=None, repr=False, compare=False)
    _grid: "_Grid | None" = field(default=None, repr=False, compare=False)

    @property
    def bpm(self) -> float:
        return 60.0 / self.period

    def line_at_or_before(self, t: float, step: float = 1.0) -> float:
        """The last grid line (every `step` beats from the anchor) <= t."""
        d = self.period * step
        return self.anchor + np.floor((t - self.anchor) / d + 1e-9) * d

    def line_at_or_after(self, t: float, step: float = 1.0) -> float:
        d = self.period * step
        return self.anchor + np.ceil((t - self.anchor) / d - 1e-9) * d


@dataclass
class TempoMap:
    """What `analyze` found. `constant` means the single-tempo grid holds and
    nothing is to be changed; otherwise `regions` (in song order), the `joins`
    between them and `timing()` describe the piecewise map."""
    constant: bool
    bpm: float                  # the single-tempo grid that was checked
    offset: float
    regions: list = field(default_factory=list)
    joins: list = field(default_factory=list)       # len(regions) - 1 dicts
    pauses: list = field(default_factory=list)      # [(start_s, end_s)] mid-song silences
    diag: dict = field(default_factory=dict)
    _built: tuple | None = None

    # -- assembling the beat numbering ------------------------------------- #
    def build(self) -> None:
        """Number the beats through every region and join -> BPM and stop lists
        (rounded the way the simfile will carry them) and the events."""
        self._built = _assemble(self.regions, self.joins, self.offset)

    def _need(self) -> tuple:
        if self._built is None:
            self.build()
        return self._built

    @property
    def first_beat_time(self) -> float:
        return self._need()[0]

    @property
    def bpms(self) -> list:
        return self._need()[1]

    @property
    def stops(self) -> list:
        return self._need()[2]

    @property
    def events(self) -> list:
        return self._need()[3]

    @property
    def region_beats(self) -> list:
        """[(first beat, last beat)] of each region in the built numbering."""
        return self._need()[4]

    def timing(self) -> TimingMap:
        if self.constant:
            return TimingMap.constant(self.bpm, self.offset)
        return TimingMap(self.first_beat_time, self.bpms, self.stops)

    def shift_region(self, index: int, beats: float) -> None:
        """Move one region's beat lines `beats` later (0.5 = seat it on what
        were the off-beats). The joins on either side are re-made by `build`;
        for the first region beat 0 moves with its lines."""
        r = self.regions[index]
        r.anchor += beats * r.period
        if index == 0:
            self.offset += beats * r.period
        self._built = None

    def move_origin(self, beats: float) -> None:
        """Make beat 0 the beat that is now beat `beats` (whole beats: this
        moves the bar lines, not the grid)."""
        self.offset += beats * self.regions[0].period
        self._built = None

    def to_report(self) -> dict:
        """The JSON summary report.json carries."""
        out = dict(self.diag, constant=self.constant)
        if not self.constant:
            out.update(bpms=[[b, v] for b, v in self.bpms],
                       stops=[[b, s] for b, s in self.stops], events=self.events)
        return out


def reseat_at_pause(bpm: float, offset: float, duration_s: float, pause, before_beats: float,
                    after_beats: float, pauses=(), diag: dict | None = None) -> TempoMap:
    """The map for a song that holds its one tempo but comes back in after a
    `pause` (start_s, end_s) on another 16th-note line: the same grid on both
    sides, seated `before_beats` and `after_beats` later than (`bpm`,
    `offset`), joined by a stop at the pause. The seats are the caller's — the
    rhythm model's, see pipeline.pause_seats; onsets cannot tell them apart."""
    period = 60.0 / bpm
    x = Region(0.0, float(pause[0]), period, float(offset) + before_beats * period,
               kept=before_beats == 0)
    y = Region(float(pause[1]), float(duration_s), period, float(offset) + after_beats * period)
    out = TempoMap(False, float(bpm), float(x.anchor), [x, y],
                   [{"last_x": float(pause[0]), "first_y": float(pause[1]),
                     "pause": (float(pause[0]), float(pause[1]))}],
                   list(pauses), dict(diag or {}))
    out.build()
    return out


# --------------------------------------------------------------------------- #
# Onsets, level
# --------------------------------------------------------------------------- #
def _moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average over `n` samples, shorter at the ends."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    a = np.clip(i - n // 2, 0, len(x))
    b = np.clip(i + n // 2 + 1, 0, len(x))
    return (c[b] - c[a]) / np.maximum(b - a, 1)


def onset_peaks(envelope: np.ndarray) -> tuple:
    """(times in true audio seconds, strengths) of the clear peaks of the onset
    envelope: those that stand 1.5 deviations above the envelope's level over
    the 4 s around them, so a quiet section is heard as well as a loud one.
    Strengths are capped at their 95th percentile so one loud hit cannot
    outvote a bar of ordinary ones."""
    e = np.asarray(envelope, np.float64)
    if len(e) < 8:
        return np.zeros(0), np.zeros(0)
    n = int(round(4.0 / GF.HOP_S))
    mu = _moving_mean(e, n)
    sd = np.sqrt(np.maximum(_moving_mean(e * e, n) - mu * mu, 0.0))
    floor = 0.25 * (e.mean() + 1.5 * e.std())
    pk, _ = scipy.signal.find_peaks(e, distance=6)
    pk = pk[(pk > 0) & (pk < len(e) - 1)]
    pk = pk[e[pk] >= np.maximum(mu[pk] + 1.5 * sd[pk], floor)]
    if len(pk) == 0:
        return np.zeros(0), np.zeros(0)
    a, b, c = e[pk - 1], e[pk], e[pk + 1]
    t = (pk + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)) * GF.HOP_S + GF.FLUX_LEAD_S
    return t, np.minimum(b, np.percentile(b, 95))


def frame_rms(y: np.ndarray, n_frames: int | None = None) -> np.ndarray:
    """RMS level per feature frame (GF.HOP), over a window of two hops."""
    y = np.asarray(y, np.float64)
    n = int(n_frames) if n_frames is not None else 1 + len(y) // GF.HOP
    c = np.concatenate([[0.0], np.cumsum(y * y)])
    k = np.arange(n) * GF.HOP
    a = np.clip(k - GF.HOP, 0, len(y))
    b = np.clip(k + GF.HOP, 0, len(y))
    return np.sqrt((c[b] - c[a]) / np.maximum(b - a, 1))


def find_pauses(rms: np.ndarray, t_first: float, t_last: float) -> list:
    """Mid-song silences: [(start_s, end_s)] where the level stays more than
    PAUSE_DROP_DB below the song's median for at least PAUSE_MIN_S, between the
    first and the last onset (lead-in and fade-out are not pauses)."""
    r = np.asarray(rms, np.float64)
    if len(r) < 4:
        return []
    db = 20.0 * np.log10(r + 1e-9)
    a = int(max(0, np.floor(t_first / GF.HOP_S)))
    b = int(min(len(r), np.ceil(t_last / GF.HOP_S)))
    if b - a < 4:
        return []
    quiet = db < np.median(db[a:b]) - PAUSE_DROP_DB
    quiet[:a] = False
    quiet[b:] = False
    edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]])))
    out = []
    for s, e in zip(edges[::2], edges[1::2]):
        if (e - s) * GF.HOP_S >= PAUSE_MIN_S and s > a and e < b:
            out.append((float(s * GF.HOP_S), float(e * GF.HOP_S)))
    return out


# --------------------------------------------------------------------------- #
# Windows and how a grid sits in them
# --------------------------------------------------------------------------- #
def _circ_smooth(h: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular convolution of each row of `h` with a centred `kernel`."""
    nb = h.shape[1]
    k = np.zeros(nb)
    half = len(kernel) // 2
    for i, v in enumerate(kernel):
        k[(i - half) % nb] += v
    return np.fft.irfft(np.fft.rfft(h, axis=1) * np.fft.rfft(k)[None, :], n=nb, axis=1)


def _shape(h: np.ndarray, bin_s: float) -> np.ndarray:
    """Raw histograms -> smoothed, square-root-compressed, zero-mean, unit-norm
    profiles. The compression makes WHERE the peaks are count for more than how
    tall each is: a section where the swung note is the loud one still looks
    like the section before it."""
    sig = SMOOTH_S / bin_s
    half = int(np.ceil(3 * sig))
    g = _circ_smooth(h, np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2))
    g = np.sqrt(np.maximum(g, 0.0))
    g = g - g.mean(1, keepdims=True)
    return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)


def _resample(ref: np.ndarray, nb: int) -> np.ndarray:
    """A profile on another number of bins (the same shape around the circle)."""
    if len(ref) == nb:
        return ref
    x = (np.arange(nb) + 0.5) / nb
    xp = (np.arange(-1, len(ref) + 1) + 0.5) / len(ref)
    out = np.interp(x, xp, np.concatenate([ref[-1:], ref, ref[:1]]))
    out = out - out.mean()
    return out / (np.linalg.norm(out) + 1e-12)


def _wrap(x, period: float):
    """x folded into [-period/2, period/2)."""
    return (np.asarray(x) + period / 2) % period - period / 2


@dataclass
class _Stats:
    """How one grid sits in every window."""
    lag: np.ndarray             # displacement of the window's profile from the reference (s)
    conf: np.ndarray            # correlation at that displacement
    fit: np.ndarray             # onset energy on the grid so displaced, above chance
    confident: np.ndarray       # enough onsets, a profile that matches, onsets on a grid
    aligned: np.ndarray         # confident and within ALIGN_TOL_S of the grid
    off: np.ndarray             # confident and beyond BREAK_TOL_S
    ref: np.ndarray             # the reference profile used


class _Windows:
    """The song's onsets, cut into blocks of BLOCK_BEATS of the checked tempo;
    a window is WINDOW_BLOCKS consecutive blocks."""

    def __init__(self, t: np.ndarray, w: np.ndarray, bpm: float, offset: float,
                 duration: float, env: np.ndarray | None = None):
        self.t, self.w = t, w
        self.env = np.zeros(0) if env is None else np.asarray(env, np.float64)
        self.block_s = BLOCK_BEATS * 60.0 / bpm
        self.t0 = offset - np.ceil(max(offset, 0.0) / self.block_s) * self.block_s
        self.n_blocks = max(WINDOW_BLOCKS,
                            int(np.ceil((duration - self.t0) / self.block_s)))
        self.block = np.clip(((t - self.t0) // self.block_s).astype(np.int64),
                             0, self.n_blocks - 1)
        self.n_win = self.n_blocks - WINDOW_BLOCKS + 1
        self.count = self._windowed(np.bincount(self.block, minlength=self.n_blocks)
                                    .astype(np.float64))
        self.enough = self.count >= MIN_ONSETS

    def _windowed(self, per_block: np.ndarray) -> np.ndarray:
        c = np.concatenate([np.zeros((1,) + per_block.shape[1:]),
                            np.cumsum(per_block, axis=0)])
        return c[WINDOW_BLOCKS:] - c[:-WINDOW_BLOCKS]

    def span(self, i: int) -> tuple:
        """Window i covers [start, end) seconds."""
        return (self.t0 + i * self.block_s,
                self.t0 + (i + WINDOW_BLOCKS) * self.block_s)

    def onsets_in(self, lo: int, hi: int) -> np.ndarray:
        """Mask of the onsets inside windows lo..hi (inclusive)."""
        return (self.block >= lo) & (self.block <= hi + WINDOW_BLOCKS - 1)

    def onsets_of(self, mask: np.ndarray) -> np.ndarray:
        """Mask of the onsets inside any window of `mask`."""
        b = np.zeros(self.n_blocks + 1, np.int64)
        for a, z in _runs(mask):
            b[a] += 1
            b[min(z + WINDOW_BLOCKS, self.n_blocks)] -= 1
        return (np.cumsum(b)[:-1] > 0)[self.block]

    def stats(self, p: float, anchor: float, ref: np.ndarray | None = None,
              core: np.ndarray | None = None) -> _Stats:
        """How the 16th-note grid (lines `p` seconds apart, one at `anchor`)
        sits in every window.

        Each window's onsets are folded onto the 16th-note circle and the
        profile is slid against a reference — the same thing for the whole
        song, built from the windows that agree with each other — until they
        match best. That displacement is the window's phase error. It is the
        whole profile that is matched, not its tallest peak, so a swung or
        triplet section reads as on the grid as long as the rest of the song
        has that shape too. And it is the 16th-note circle, so a syncopated
        stretch, whose loud notes sit on the off-16ths, is on the grid as well:
        only music that is on NO line of the grid counts as off it.

        A window is only believed when it has enough onsets, its profile
        matches the reference well (CONF_MIN) and its onsets really do sit on
        lines once displaced (FIT_MIN, waived for a profile that matches very
        well: some songs keep few of their onsets on any grid): a stretch at
        another tempo smears round the circle and fails. Measured on 5,861
        windows of single-tempo songs and 410 at a tempo 10% off, the rule
        keeps 93% of the first and 11% of the second, and at least half the
        windows of every one of the 180 songs.

        `ref` is a reference profile to use; without one it is built from the
        windows of `core`, or, with no `core`, from the windows that agree."""
        nb = max(16, int(round(p / BIN_S)))
        bin_s = p / nb
        col = np.minimum(((_wrap(self.t - anchor, p) + p / 2) / bin_s).astype(np.int64),
                         nb - 1)
        h = np.zeros((self.n_blocks, nb))
        np.add.at(h, (self.block, col), self.w)
        hw = self._windowed(h)
        g = _shape(hw, bin_s)
        fg = np.fft.rfft(g, axis=1)
        rows = np.arange(len(g))

        def against(r):
            xc = np.fft.irfft(fg * np.conj(np.fft.rfft(r))[None, :], n=nb, axis=1)
            k = xc.argmax(1)
            return ((k + nb // 2) % nb - nb // 2) * bin_s, xc[rows, k]

        if ref is None and core is not None and core.any():
            ref = _shape(hw[core].sum(0, keepdims=True), bin_s)[0]
        elif ref is None:
            use = self.enough.copy()
            if not use.any():
                use[:] = True
            for _ in range(3):
                ref = _shape(hw[use].sum(0, keepdims=True), bin_s)[0]
                lag, conf = against(ref)
                core = self.enough & (np.abs(lag) <= ALIGN_TOL_S)
                if core.sum() < 3 or (core == use).all():
                    break
                use = core
        else:
            ref = _resample(ref, nb)
        lag, conf = against(ref)
        hb = int(round(HIT_S / bin_s))
        box = _circ_smooth(hw, np.ones(2 * hb + 1))
        at = (np.round(lag / bin_s).astype(np.int64) + nb // 2) % nb
        chance = min(0.95, (2 * hb + 1) * bin_s / p)
        fit = (box[rows, at] / (hw.sum(1) + 1e-12) - chance) / (1 - chance)
        confident = self.enough & (conf >= CONF_MIN) & ((fit >= FIT_MIN) | (conf >= CONF_SURE))
        aligned = confident & (np.abs(lag) <= ALIGN_TOL_S)
        off = confident & (np.abs(lag) > BREAK_TOL_S)
        return _Stats(lag, conf, fit, confident, aligned, off, ref)

    def fit_over(self, mask: np.ndarray, p: float, anchor: float) -> float:
        """Onset energy within HIT_S of the 16th-note grid, above chance, over
        the onsets in `mask`."""
        if not mask.any():
            return 0.0
        e = np.abs(_wrap(self.t[mask] - anchor, p))
        share = float(self.w[mask][e <= HIT_S].sum() / (self.w[mask].sum() + 1e-12))
        chance = min(0.95, 2 * HIT_S / p)
        return (share - chance) / (1 - chance)


# --------------------------------------------------------------------------- #
# Fitting a grid to a stretch
# --------------------------------------------------------------------------- #
def _coherence_scan(t: np.ndarray, w: np.ndarray, lo: float, hi: float) -> tuple:
    """Candidate 16th-note periods in [lo, hi] and how tightly the onsets bunch
    on each one's circle (16ths, with 8ths counting as much: a stretch of plain
    8th notes has nothing on the off-16ths to bunch)."""
    n = max(2, int(np.ceil(np.log(hi / lo) / TEMPO_STEP)) + 1)
    cand = lo * np.exp(np.arange(n) * (np.log(hi / lo) / (n - 1)))
    z = np.exp(1j * np.pi * np.outer(1.0 / cand, t))         # the 8th-note circle
    score = np.abs((w * z).sum(1)) + np.abs((w * z * z).sum(1))
    return cand, score / (w.sum() + 1e-12)


def _seat(t: np.ndarray, w: np.ndarray, p: float) -> tuple:
    """(phase, fit): where the 16th-note grid of period `p` sits best on these
    onsets (a time in [0, p)), and the onset energy it then catches."""
    nb = max(16, int(round(p / BIN_S)))
    bin_s = p / nb
    col = np.minimum(((t % p) / bin_s).astype(np.int64), nb - 1)
    h = np.bincount(col, weights=w, minlength=nb)[None, :]
    hb = int(round(HIT_S / bin_s))
    sig = SMOOTH_S / bin_s
    half = int(np.ceil(3 * sig))
    g = _circ_smooth(h, np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2))[0]
    i = int(g.argmax())
    box = _circ_smooth(h, np.ones(2 * hb + 1))[0]
    chance = min(0.95, (2 * hb + 1) * bin_s / p)
    fit = (box[i] / (w.sum() + 1e-12) - chance) / (1 - chance)
    a, b, c = g[i - 1], g[i], g[(i + 1) % nb]
    d = 0.5 * (a - c) / (a - 2 * b + c - 1e-12)
    return (i + 0.5 + float(np.clip(d, -0.5, 0.5))) * bin_s, float(fit)


def _local_grid(t: np.ndarray, w: np.ndarray, center: float, lo: float, hi: float,
                every: int = 0):
    """Best 16th-note grid for the onsets `t` with a period in [lo, hi] ->
    (p, anchor near `center`, fit). (None, None, 0) if there is nothing to fit.
    With `every` > 0, the best `every` candidates instead, as a list of such
    triples, for the caller to choose among."""
    if len(t) < MIN_ONSETS:
        return [] if every else (None, None, 0.0)
    tt = t - center
    cand, score = _coherence_scan(tt, w, lo, hi)
    tops = []
    for i in (int(i) for i in np.argsort(score)[::-1]):      # the best few, well apart
        if all(abs(np.log(cand[i] / cand[j])) > 4 * TEMPO_STEP for j in tops):
            tops.append(i)
        if len(tops) == (every or 3):
            break
    best = (None, None, 0.0)
    found = []
    for i in tops:
        fine = np.linspace(cand[max(i - 1, 0)], cand[min(i + 1, len(cand) - 1)], 21)
        z = np.exp(1j * np.pi * np.outer(1.0 / fine, tt))
        sc = np.abs((w * z).sum(1)) + np.abs((w * z * z).sum(1))
        p = float(fine[int(sc.argmax())])
        phase, fit = _seat(tt, w, p)
        phase = phase - p * np.round(phase / p)              # the line nearest centre
        found.append((p, center + phase, fit))
        if fit > best[2]:
            best = (p, center + phase, fit)
    return found if every else best


def _tune(win: "_Windows", p: float, anchor: float, ref: np.ndarray,
          mask: np.ndarray, rounds: int = 3) -> tuple:
    """Correct a grid from the phase errors of its own windows -> (p, anchor,
    the stats under the corrected grid).

    A line through (window centre, phase error) over the windows of `mask`:
    its height is how far the grid sits from the music, its slope how fast the
    two drift apart, which is the tempo error. Window phase errors come from
    matching whole profiles, so they are steady to a couple of milliseconds
    where pairing single onsets with grid lines is not — at 190 BPM a 16th
    note is 79 ms and an onset a third of the way between two lines belongs
    to neither."""
    st = win.stats(p, anchor, ref)
    # first the plain displacement, however large: the windows may all sit
    # half a 16th from the grid they are being fitted from
    idx = np.flatnonzero(mask & st.confident)
    if len(idx) >= 2:
        shift = float(np.median(st.lag[idx]))
        if abs(shift) > 0.010:
            anchor += shift
            st = win.stats(p, anchor, ref)
    for _ in range(rounds):
        idx = np.flatnonzero(mask & st.confident & (np.abs(st.lag) <= 0.030))
        if len(idx) < 2:
            break
        c = win.t0 + (idx + WINDOW_BLOCKS / 2.0) * win.block_s
        lag = st.lag[idx]
        cm = float(c.mean())
        if len(idx) >= 4 and c.max() - c.min() >= 4 * win.block_s:
            slope, icpt = (float(v) for v in np.polyfit(c - cm, lag, 1))
        else:
            slope, icpt = 0.0, float(np.median(lag))
        if abs(icpt) < 2e-4 and abs(slope) * (c.max() - c.min()) < 5e-4:
            break
        # the music is lag(c) later than the grid at time c: move the lines
        anchor = anchor + icpt + slope * (anchor - cm)
        p = p * (1.0 + slope)
        st = win.stats(p, anchor, ref)
    return p, anchor, st


def _inner(mask: np.ndarray) -> np.ndarray:
    """A stretch's windows without the ones at its ends, which also hold some
    of whatever comes before and after it; unchanged if that leaves under
    three."""
    idx = np.flatnonzero(mask)
    cut = WINDOW_BLOCKS - 1
    if len(idx) < 2 * cut + 3:
        return mask
    out = np.zeros_like(mask)
    out[idx[cut:-cut]] = True
    return out


def _round_tempo(win: "_Windows", p: float, anchor: float, ref: np.ndarray,
                 mask: np.ndarray) -> tuple:
    """Take a whole or half BPM when the fitted tempo is within SNAP_TOL x 2 of
    one and the windows stay on the round grid from the stretch's first to its
    last (no more than 8 ms of drift between them)."""
    bpm = 15.0 / p
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return p, anchor
    lo, hi = win.span(int(idx[0]))[0], win.span(int(idx[-1]))[1]
    mid = 0.5 * (lo + hi)
    for r in sorted({round(bpm), round(bpm * 2) / 2}, key=lambda r: abs(r - bpm)):
        if r <= 0 or abs(bpm / r - 1.0) > 2 * SNAP_TOL:
            continue
        if abs(bpm / r - 1.0) * (hi - lo) > 0.008:
            continue
        q = 15.0 / r
        a = anchor + np.round((mid - anchor) / p) * p            # the same line mid-stretch
        st = win.stats(q, a, ref)
        ok = mask & st.confident
        if ok.sum() >= 2 and np.abs(st.lag[ok]).max() <= ALIGN_TOL_S:
            return q, float(a + np.median(st.lag[ok]))
    return p, anchor


# --------------------------------------------------------------------------- #
# The analysis
# --------------------------------------------------------------------------- #
def _runs(mask: np.ndarray) -> list:
    """[(first, last)] of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def _fast_path(st: _Stats) -> tuple:
    """Does the single-tempo grid hold? -> (yes/no, aligned share, breaks).

    Yes when at least FAST_PATH_SHARE of the windows that say anything are on
    the grid and no two such windows in a row are off it."""
    share = float(st.aligned.sum() / max(1, st.confident.sum()))
    seq = st.off[st.confident]
    two = bool(len(seq) > 1 and (seq[1:] & seq[:-1]).any())
    return (share >= FAST_PATH_SHARE and not two), share, int(st.off.sum())


def _step(st: _Stats, win: _Windows, p: float) -> tuple:
    """The largest sustained step in the windows' phase error -> (block the
    step falls at, its size in seconds); (None, 0.0) when there is none.

    A single-tempo fit made across a break is a compromise: it tilts its tempo
    until both halves are only a little wrong, and then no window is far
    enough off to notice. But the windows before the break and the windows
    after it still disagree with each other by the size of the break, and that
    survives any tilt. Windows that straddle the block are left out."""
    k = 3
    best = (None, 0.0)
    for b in range(WINDOW_BLOCKS + 1, win.n_win - 1):
        before = np.arange(max(0, b - WINDOW_BLOCKS - k + 1), b - WINDOW_BLOCKS + 1)
        after = np.arange(b, min(win.n_win, b + k))
        before, after = before[st.confident[before]], after[st.confident[after]]
        if len(before) < 2 or len(after) < 2:
            continue
        lb, la = st.lag[before], st.lag[after]
        if np.ptp(lb) > 0.012 or np.ptp(la) > 0.012:
            continue
        size = float(_wrap(np.median(la) - np.median(lb), p))
        if abs(size) > abs(best[1]):
            best = (b, size)
    return best


def analyze(envelope: np.ndarray, bpm: float, offset: float, duration_s: float,
            rms: np.ndarray | None = None) -> TempoMap:
    """Check the single-tempo grid (`bpm`, beat 0 at `offset`, true audio time)
    against the onset envelope, and build a tempo map if it does not hold.

    `rms` is the level per feature frame (`frame_rms`); without it no pause can
    be recognized and every join is made with a BPM segment."""
    out = TempoMap(True, float(bpm), float(offset))
    out.diag = {"windows": 0, "aligned_share": None, "grids": 1,
                # the single-tempo grid that was checked (what the song would
                # have been charted on before tempo maps)
                "single_bpm": float(bpm), "single_offset": round(float(offset), 4)}
    if bpm <= 0 or duration_s <= 0:
        return out
    t, w = onset_peaks(envelope)
    keep = (t > 0) & (t < duration_s)
    t, w = t[keep], w[keep]
    if len(t) < 4 * MIN_ONSETS:
        return out
    win = _Windows(t, w, bpm, offset, duration_s, envelope)
    p = 15.0 / bpm
    out.pauses = find_pauses(rms, float(t[0]), float(t[-1])) if rms is not None else []
    # A window with a pause inside it holds the music before the pause and the
    # music after it, which may sit on two different grids: it speaks for
    # neither.
    for s0, s1 in out.pauses:
        for i in range(win.n_win):
            a, b = win.span(i)
            if a < s0 and s1 < b:
                win.enough[i] = False
    base = win.stats(p, offset)
    fast, share, breaks = _fast_path(base)
    out.diag.update(windows=int(win.n_win), aligned_share=round(share, 3),
                    breaks=breaks, pauses=len(out.pauses),
                    # the single tempo holds from end to end by the plain rule
                    # (`_fast_path`); a song can fail it and still be left alone
                    locked=bool(fast))
    grid0 = _Grid(p, float(offset), base, True)
    given = base                                 # the single-tempo grid as it was given
    # a break the single-tempo fit papered over (see `_step`): refit that grid
    # on the longer side of it, so the other side shows for what it is
    at, size = _step(base, win, p)
    if at is not None and abs(size) >= PAUSE_STEP_MIN_S:
        out.diag["step_ms"] = round(size * 1000, 1)
        left = np.zeros(win.n_win, bool)
        left[:max(0, at - WINDOW_BLOCKS + 1)] = True
        right = np.zeros(win.n_win, bool)
        right[at:] = True
        side = left if (base.confident & left).sum() >= (base.confident & right).sum() else right
        q, qa, _ = _tune(win, p, float(offset), base.ref, side & base.confident)
        st = win.stats(q, qa, core=side & base.confident)
        if st.aligned.sum() >= STEP_WINDOWS:
            grid0, base = _Grid(q, qa, st, True), st
            fast = False
    # Windows the single-tempo grid does not explain: displaced ones, and ones
    # whose profile matches nowhere although they are full of onsets — which is
    # how a stretch at another tempo looks from here, its onsets smeared round
    # the circle.
    grids = [grid0]
    # When the single tempo explains next to nothing — two tempos in one song
    # can leave the global estimate on neither — there is no profile of the
    # song to match against and no tempo to stay near: each stretch is then
    # fitted on its own terms.
    lost = grid0.on.sum() < max(STEP_WINDOWS, 0.2 * win.enough.sum())
    out.diag["lost"] = bool(lost)
    if not fast or _open_runs(win, grid0.on, out.pauses):
        grids += _other_grids(win, grids, None if lost else base.ref, out.pauses)
    out.diag["grids"] = len(grids)
    if len(grids) == 1:
        return out
    segs = _segments(_label(grids))
    regions = _regions(win, grids, segs, out.pauses, p, float(offset),
                       np.asarray(envelope, np.float64), lost)
    if not regions or (len(regions) == 1 and regions[0].kept):
        return out
    # An event needs the single tempo to be visibly wrong somewhere, for long
    # enough: windows off its grid, or unreadable on it, inside the stretches
    # being given a new one. A song that single tempo keeps within tolerance
    # from end to end is left alone however much tighter two grids would fit
    # it — the tighter fit is real, and not worth a tempo event.
    changed = np.zeros(win.n_win, bool)
    for r in regions:
        if not r.kept:
            changed[r._idx] = True
    wrong = changed & win.enough & ~given.aligned
    out.diag["wrong_windows"] = int(wrong.sum())
    if wrong.sum() < (PAUSE_WINDOWS if out.pauses else STEP_WINDOWS):
        out.diag["rejected"] = "single tempo within tolerance"
        return out
    joins = [_join(win, a, b, out.pauses) for a, b in zip(regions, regions[1:])]
    cand = TempoMap(False, float(bpm), float(offset), regions, joins, out.pauses,
                    dict(out.diag))
    # The map must leave the song's windows clearly closer to the grid than
    # the single tempo did: a fit that is perfect for one half and 25 ms out
    # for the other is no better to play than one that splits the difference.
    before, after = _phase_cost(win, grids[0].st, None), _phase_cost(win, None, regions)
    cand.diag.update(cost_single=round(before * 1000, 1), cost_map=round(after * 1000, 1),
                     regions=len(regions))
    if after > ACCEPT_RATIO * before:
        out.diag.update(cand.diag, rejected="no gain")
        return out
    cand.build()
    return cand


@dataclass
class _Grid:
    p: float                    # seconds between 16th-note lines
    anchor: float
    st: _Stats                  # how it sits in every window
    base: bool = False          # the single-tempo grid (as given, or refitted on one side)

    @property
    def on(self) -> np.ndarray:                  # the windows it explains
        return self.st.aligned

    def explains(self, idx: np.ndarray, tol: float) -> float:
        """Share of the windows `idx` this grid is within `tol` of."""
        if len(idx) == 0:
            return 0.0
        return float((self.st.confident[idx] & (np.abs(self.st.lag[idx]) <= tol)).mean())


def _need(pauses: list, win: _Windows, lo: int, hi: int) -> int:
    """Windows a stretch must hold a grid for: fewer when a pause borders it."""
    a, b = win.span(lo)[0], win.span(hi)[1]
    near = any(a - 2 * win.block_s <= e and s <= b + 2 * win.block_s for s, e in pauses)
    return PAUSE_WINDOWS if near else STEP_WINDOWS


def _open_runs(win: _Windows, covered: np.ndarray, pauses: list,
               dead: np.ndarray | None = None) -> list:
    """Runs of windows no grid explains that are long enough to hold one."""
    open_ = ~covered & win.enough
    if dead is not None:
        open_ &= ~dead
    return [(a, b) for a, b in _runs(open_) if b - a + 1 >= _need(pauses, win, a, b)]


def _other_grids(win: _Windows, grids: list, ref: np.ndarray | None, pauses: list) -> list:
    """Grids for the stretches no grid so far explains: the same tempo
    re-seated first (a pause, a cut bar), then another tempo. With no `ref`
    (the single tempo explains nothing, see `analyze`) any tempo in ABS_BPM is
    looked for and each grid is matched against its own seed's profile."""
    found = []
    covered = grids[0].on.copy()
    dead = np.zeros(win.n_win, bool)
    while len(grids) + len(found) < MAX_GRIDS:
        runs = _open_runs(win, covered, pauses, dead)
        if not runs:
            break
        lo, hi = max(runs, key=lambda r: r[1] - r[0])
        # seed on the busiest window of the run
        s = lo + int(np.argmax(win.count[lo:hi + 1]))
        m = win.onsets_in(s, s)
        center = 0.5 * sum(win.span(s))
        tt, ww = win.t[m], win.w[m]
        # the tempo to stay near: that of the nearest stretch already explained
        known = (grids if ref is not None else []) + found
        g_p = min(known, key=lambda g: _distance(g.on, s)).p if known else grids[0].p
        grown = None
        if ref is None and not found:
            p_a, a_a, f_a = _local_grid(tt, ww, center, 15.0 / ABS_BPM[1], 15.0 / ABS_BPM[0])
            if p_a is not None and f_a >= LOCAL_FIT_MIN:
                grown = _claim(win, _grow(win, p_a, a_a, s, None, ~covered), covered, pauses)
            if grown is None:
                dead[s] = True
                continue
            found.append(grown)
            covered |= grown.on
            continue
        if ref is None:
            ref_now = found[0].st.ref              # the first stretch found sets the profile
        else:
            ref_now = ref
        p_n, a_n, f_n = _local_grid(tt, ww, center, g_p * (1 - SAME_TEMPO),
                                    g_p * (1 + SAME_TEMPO))
        if p_n is not None and f_n >= LOCAL_FIT_MIN:
            grown = _claim(win, _grow(win, p_n, a_n, s, ref_now), covered, pauses)
        if grown is None:
            # the same tempo does not hold here whatever its phase: another one?
            # Of the tempos the seed's onsets bunch on, take the one whose 8th
            # notes catch the most and whose beat the music repeats at — a
            # grid at 3/2 the real tempo has a line under every 8th note too,
            # and only those two tell it from the real one.
            f8_n = _seat(tt - center, ww, 2 * p_n)[1] if p_n is not None else 0.0
            t_lo, t_hi = win.span(lo)[0], win.span(hi)[1]
            best, best_score = None, -np.inf
            for p_w, a_w, f_w in _local_grid(tt, ww, center, g_p * WIDE_RANGE[0],
                                             g_p * WIDE_RANGE[1], every=5):
                if f_w < LOCAL_FIT_MIN or abs(p_w / g_p - 1.0) <= SAME_TEMPO:
                    continue
                f8_w = _seat(tt - center, ww, 2 * p_w)[1]
                if f8_w < max(f8_n, 0.0) + 0.05:
                    continue
                score = f8_w + _periodicity(win.env, t_lo, t_hi, (4 * p_w,))[0]
                if score > best_score:
                    best, best_score = (p_w, a_w), score
            if best is not None:
                grown = _claim(win, _grow(win, best[0], best[1], s, ref_now, ~covered),
                               covered, pauses)
        if grown is None:
            dead[s] = True                       # nothing holds here; try the rest
            continue
        found.append(grown)
        covered |= grown.on
    return found


def _claim(win: _Windows, grown, covered: np.ndarray, pauses: list):
    """A grown grid, if it explains enough windows nothing else does."""
    if grown is None:
        return None
    new = grown.on & ~covered
    if not new.any() or new.sum() < _need(pauses, win, *_extent(new)):
        return None
    return grown


def _extent(mask: np.ndarray) -> tuple:
    i = np.flatnonzero(mask)
    return (int(i[0]), int(i[-1])) if len(i) else (0, 0)


def _distance(on: np.ndarray, i: int) -> int:
    idx = np.flatnonzero(on)
    return int(np.abs(idx - i).min()) if len(idx) else 10 ** 9


def _grow(win: _Windows, p: float, anchor: float, seed: int, ref: np.ndarray,
          allowed: np.ndarray | None = None):
    """Seat a seed grid the way the song's profile sits on its own grid, then
    extend it over every neighbouring window it stays aligned in, refitting as
    it goes -> _Grid or None. `allowed`, if given, are the only windows it may
    claim."""
    best = None
    if ref is None:
        only = np.zeros(win.n_win, bool)
        only[seed] = True
        ref = win.stats(p, anchor, core=only).ref
    for it in range(12):
        st = win.stats(p, anchor, ref)
        if it == 0:
            # the seed was seated on its tallest peak; the song's profile may
            # have its tallest peak elsewhere (a swung note). Match shapes.
            if not st.confident[seed]:
                return None
            anchor += float(st.lag[seed])
            st = win.stats(p, anchor, ref)
        ok = st.aligned if allowed is None else st.aligned & allowed
        if not ok[seed]:
            near = np.flatnonzero(ok[max(0, seed - 2):seed + 3])
            if len(near) == 0:
                return best
            seed = max(0, seed - 2) + int(near[np.argmin(np.abs(near - min(2, seed)))])
        # the run around the seed, bridging up to 3 windows that say nothing
        a = b = seed
        for step in (-1, 1):
            i, gap, edge = seed + step, 0, seed
            while 0 <= i < win.n_win and gap <= 3:
                if ok[i]:
                    edge, gap = i, 0
                else:
                    gap += 2 if st.off[i] else 1
                i += step
            if step < 0:
                a = edge
            else:
                b = edge
        mask = np.zeros(win.n_win, bool)
        mask[a:b + 1] = ok[a:b + 1]
        if best is not None and (mask == best.on).all():
            break
        # the grid explains the run it grew over, and nothing it did not reach
        best = _Grid(p, anchor, _Stats(st.lag, st.conf, st.fit, st.confident, mask,
                                       st.off, st.ref))
        p, anchor, _ = _tune(win, p, anchor, ref, _inner(mask), rounds=1)
    return best


def _label(grids: list) -> np.ndarray:
    """Which grid each window belongs to: the labelling with the fewest
    windows left unexplained, a change of grid costing SWITCH_PENALTY of them
    (Viterbi). A window no grid explains costs nothing under any."""
    m = np.stack([g.on for g in grids])                  # (G, W)
    known = m.any(0)
    cost = np.where(m, 0.0, np.where(known[None, :], 1.0, 0.0))
    cost[0] -= 1e-3                                      # ties go to the single-tempo grid
    ng, nw = cost.shape
    acc = cost[:, 0].copy()
    back = np.zeros((nw, ng), np.int64)
    for i in range(1, nw):
        j = int(acc.argmin())
        switch = acc[j] + SWITCH_PENALTY
        take = switch < acc
        back[i] = np.where(take, j, np.arange(ng))
        acc = np.where(take, switch, acc) + cost[:, i]
    lab = np.zeros(nw, np.int64)
    lab[-1] = int(acc.argmin())
    for i in range(nw - 1, 0, -1):
        lab[i - 1] = back[i, lab[i]]
    return lab


def _segments(labels: np.ndarray) -> list:
    """[(first window, last window, grid)] of each run of one label."""
    out, a = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[a]:
            out.append((a, i - 1, int(labels[a])))
            a = i
    return out


def _phase_gap(x: "Region", y: "Region", t: float) -> float:
    """How far y's 16th-note grid is from x's at time `t` (seconds, signed,
    within half a 16th)."""
    p = 0.125 * (x.period + y.period)
    d = (t - y.anchor) % (y.period / 4) - (t - x.anchor) % (x.period / 4)
    return float(_wrap(d, p))


def _regions(win: _Windows, grids: list, segs: list, pauses: list,
             p0: float, offset0: float, env: np.ndarray, lost: bool = False) -> list:
    """A Region per labelled stretch, each grid refitted on its own stretch's
    onsets and snapped to a round BPM where the fit allows; then the ones that
    are not different enough from their neighbour to be worth an event are
    folded back into it (see `_worth`). `p0`, `offset0` is the pipeline's own
    grid: a region that turns out to be on it keeps it to the last digit."""
    out = []
    for a, b, g in segs:
        grid = grids[g]
        idx = np.flatnonzero(grid.on[a:b + 1]) + a
        if len(idx) == 0:
            continue
        mask = np.zeros(win.n_win, bool)
        mask[idx] = True
        t_lo, t_hi = win.span(int(idx[0]))[0], win.span(int(idx[-1]))[1]
        mid = 0.5 * (t_lo + t_hi)
        q, qa, _ = _tune(win, grid.p, grid.anchor, grid.st.ref, _inner(mask), rounds=5)
        q, qa = _round_tempo(win, q, qa, grid.st.ref, _inner(mask))
        qa = qa + np.round((mid - qa) / q) * q               # a line mid-stretch
        kept = False
        near = offset0 + np.round((mid - offset0) / p0) * p0     # the pipeline's line mid-stretch
        if abs(q / p0 - 1.0) * (t_hi - t_lo) <= 0.005 and abs(
                float(_wrap(qa - near, p0))) <= 0.005:
            q, qa, kept = p0, offset0, True              # agrees: leave that grid alone
        own = win.onsets_of(mask)
        r = Region(float(t_lo), float(t_hi), 4.0 * q, float(qa),
                   round(win.fit_over(own, q, qa), 3), kept)
        r.windows = int(len(idx))
        r._idx, r._grid = idx, grid                      # what it was found from
        if not kept and lost:
            # the single tempo explains nothing in this song, so there is
            # nothing to compare a stretch's tempo with: it must simply put
            # its own 8th notes on the grid
            if _fit8(win, own, q, qa) < LOST_FIT_MIN:
                continue
        elif not kept and abs(q / p0 - 1.0) > TEMPO_MIN:
            # Another tempo must be something the song's own tempo cannot do
            # at ANY phase: on this stretch's onsets, clearly more of them on
            # an 8th-note line (where a wrong tempo has nowhere to hide) than
            # the old tempo manages at its best seat.
            tt, ww = win.t[own] - mid, win.w[own]
            f_own = _fit8(win, own, q, qa)
            f_old = max(_seat(tt, ww, 2 * p0)[1], 0.0)
            if f_own < f_old + STRETCH_GAIN or f_own < 1.5 * f_old:
                continue
            if abs(q / p0 - 1.0) > SAME_TEMPO:
                # ...and the music must REPEAT at the new beat, not the old
                # one. A cross-rhythm over an unchanged beat can put its notes
                # on some other tempo's grid; it cannot make the drums repeat
                # at that tempo's beat.
                per_new, per_old = _periodicity(env, t_lo, t_hi, (4 * q, 4 * p0))
                if per_new < max(per_old, 0.0) + PERIODICITY_GAIN:
                    continue
        out.append(r)
    if not out:
        return []
    # the same grid on both sides of a stretch that was dropped is one region
    i = 1
    while i < len(out):
        x, y = out[i - 1], out[i]
        if (x.kept and y.kept) or (abs(x.period / y.period - 1.0) <= 1e-4 and abs(
                _phase_gap(x, y, 0.5 * (x.t_end + y.t_start))) <= 0.004):
            x.t_end, x.windows = y.t_end, x.windows + y.windows
            x._idx = np.concatenate([x._idx, y._idx])
            del out[i]
        else:
            i += 1
    # Two neighbours are distinct when the smaller of them is worth an event
    # next to the larger. Where a pair is not, the smaller goes — the song's
    # own grid, where it was kept, never does — and whatever is then side by
    # side on one grid becomes one region.
    while len(out) > 1:
        weak = None
        for i in range(len(out) - 1):
            x, y = out[i], out[i + 1]
            small, big = (x, y) if (x.windows, y.kept) <= (y.windows, x.kept) else (y, x)
            if small.kept and not big.kept:
                small, big = big, small
            if _worth(win, small, big, pauses):
                continue
            j = i if small is x else i + 1
            if weak is None or out[j].windows < out[weak].windows:
                weak = j
        if weak is None:
            break
        del out[weak]
        i = 1
        while i < len(out):
            x, y = out[i - 1], out[i]
            same = (x.kept and y.kept) or (abs(x.period / y.period - 1.0) <= 1e-4 and abs(
                _phase_gap(x, y, 0.5 * (x.t_end + y.t_start))) <= 0.004)
            if same:
                x.t_end, x.windows = y.t_end, x.windows + y.windows
                x._idx = np.concatenate([x._idx, y._idx])
                del out[i]
            else:
                i += 1
    if len(out) == 1 and not out[0].kept:
        # one grid after all, but not the pipeline's: only worth changing if
        # the two drift apart by more than a step's worth over the song
        r = out[0]
        rel = abs(r.period / (4 * p0) - 1.0)
        drift = rel * (r.t_end - r.t_start)
        gap = abs(float(_wrap(r.anchor - offset0, p0)))
        if rel > SAME_TEMPO or (drift < STEP_MIN_S and gap < STEP_MIN_S):
            return []           # another tempo for the whole song is not this module's call
    _seat_beats(win, out)
    return out


def _fit8(win: _Windows, mask: np.ndarray, p: float, anchor: float) -> float:
    """Onset energy within HIT_S of the 8th-note grid, above chance, at the
    better of the grid's two 8th-note seats."""
    if not mask.any():
        return 0.0
    t, w = win.t[mask], win.w[mask]
    chance = min(0.95, HIT_S / p)
    best = 0.0
    for a in (anchor, anchor + p):
        e = np.abs(_wrap(t - a, 2 * p))
        best = max(best, float(w[e <= HIT_S].sum() / (w.sum() + 1e-12)))
    return (best - chance) / (1 - chance)


def _periodicity(env: np.ndarray, lo: float, hi: float, periods) -> list:
    """How strongly the onset envelope between `lo` and `hi` seconds repeats
    at 1, 2 and 4 times each period (autocorrelation, mean of the three; 1.0
    would be perfectly periodic)."""
    a = max(0, int(lo / GF.HOP_S))
    b = min(len(env), int(np.ceil(hi / GF.HOP_S)))
    e = np.asarray(env[a:b], np.float64)
    if len(e) < 16:
        return [0.0 for _ in periods]
    e = e - e.mean()
    n = len(e)
    f = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (ac[0] + 1e-12)
    lags = np.arange(n) * GF.HOP_S
    return [float(np.mean([np.interp(k * p, lags, ac, right=0.0) for k in (1, 2, 4)]))
            for p in periods]


def _beat_profile(win: _Windows, r: "Region", lo: float, hi: float) -> np.ndarray:
    """Onset energy round the BEAT circle of a region's grid, 48 bins, over the
    onsets in [lo, hi) — the accent pattern: where in the beat the loud things
    are."""
    m = (win.t >= lo) & (win.t < hi)
    ph = ((win.t[m] - r.anchor) / r.period) % 1.0
    h = np.bincount(np.minimum((ph * 48).astype(np.int64), 47), weights=win.w[m],
                    minlength=48)[None, :]
    return _shape(h, r.period / 48)[0]


def _seat_beats(win: _Windows, regions: list) -> None:
    """Give every region a beat seat: which of its 16th-note lines is a beat.

    The pipeline's own grid has one where it was kept. Every other region
    takes its seat from a neighbour that has one: across a plain tempo change
    the beat carries on, so the seat that continues it; where the tempo is the
    same and the grid merely moved (a pause, a cut bar), the seat that puts the
    loud things where the neighbour has them. Both are first answers — the
    pipeline asks the rhythm model about each region afterwards."""
    order = sorted(range(len(regions)), key=lambda i: (not regions[i].kept, i))
    seated: set = set()
    for i in order:
        r = regions[i]
        if r.kept or not seated:
            seated.add(i)
            continue
        j = min(seated, key=lambda k: abs(k - i))
        ref = regions[j]
        step = r.period / 4.0
        if abs(r.period / ref.period - 1.0) <= TEMPO_MIN:
            want = _beat_profile(win, ref, ref.t_start, ref.t_end)
            base = _beat_profile(win, r, r.t_start, r.t_end)
            # moving the anchor k sixteenths later turns the profile back 12k bins
            k = int(np.argmax([float(np.roll(base, -12 * k) @ want) for k in range(4)]))
        else:
            tb = 0.5 * (ref.t_end + r.t_start) if i > j else 0.5 * (r.t_end + ref.t_start)
            fx = ((tb - ref.anchor) / ref.period) % 1.0
            k = min(range(4), key=lambda k: abs(
                _wrap((tb - r.anchor - k * step) / r.period - fx, 1.0)))
        r.anchor += k * step
        seated.add(i)


def _worth(win: _Windows, r: "Region", q: "Region", pauses: list) -> bool:
    """Is region `r` different enough from its neighbour `q` to be an event?

    Only if `q`'s grid does not explain `r`'s windows: carried on into `r`'s
    stretch it must leave most of them off the grid. Two grids fitted to two
    stretches of one steady song always differ a little, and that difference
    is not an event.

    Beyond that: another tempo is an event; the same tempo re-seated only by a
    clear step — more than STEP_MIN_S (PAUSE_STEP_MIN_S across a pause) — held
    for long enough, longer still when the step is a triplet's width (a swung
    section looks like that)."""
    first, second = (q, r) if q.t_start <= r.t_start else (r, q)
    tb = 0.5 * (first.t_end + second.t_start)
    paused = any(s <= max(first.t_end, second.t_start) + win.block_s
                 and e >= min(first.t_end, second.t_start) - win.block_s
                 for s, e in pauses)
    need = PAUSE_WINDOWS if paused else STEP_WINDOWS
    if r.windows < need:
        return False
    if q._grid.explains(r._idx, PAUSE_STEP_MIN_S if paused else ALIGN_TOL_S) > 0.25:
        return False
    if abs(r.period / q.period - 1.0) > TEMPO_MIN:
        return True
    gap = abs(_phase_gap(first, second, tb))
    p = 0.125 * (r.period + q.period)
    if paused:
        return gap >= PAUSE_STEP_MIN_S
    if gap < STEP_MIN_S:
        return False
    if abs(gap - p / 3) <= 0.008:
        return r.windows >= TRIPLET_WINDOWS
    return True


def _phase_cost(win: _Windows, st: _Stats | None, regions: list | None) -> float:
    """Mean phase error of the song's windows, in seconds, capped at
    COST_CAP_S; a window that cannot be read (too smeared to match) costs the
    cap. Under one grid's stats, or under a list of regions, each window judged
    by the grid of the region its centre lies in."""
    busy = win.enough
    if regions is None:
        cost = np.where(st.confident, np.minimum(np.abs(st.lag), COST_CAP_S), COST_CAP_S)
        return float(cost[busy].mean()) if busy.any() else COST_CAP_S
    centres = win.t0 + (np.arange(win.n_win) + WINDOW_BLOCKS / 2.0) * win.block_s
    edges = [-np.inf] + [0.5 * (a.t_end + b.t_start)
                         for a, b in zip(regions, regions[1:])] + [np.inf]
    cost = np.full(win.n_win, COST_CAP_S)
    for r, lo, hi in zip(regions, edges, edges[1:]):
        m = (centres >= lo) & (centres < hi)
        if not m.any():
            continue
        rs = win.stats(r.period / 4.0, r.anchor, r._grid.st.ref)
        cost[m] = np.where(rs.confident[m], np.minimum(np.abs(rs.lag[m]), COST_CAP_S),
                           COST_CAP_S)
    return float(cost[busy].mean()) if busy.any() else COST_CAP_S


# --------------------------------------------------------------------------- #
# Joining two regions
# --------------------------------------------------------------------------- #
def _join(win: _Windows, x: Region, y: Region, pauses: list) -> dict:
    """Where grid `x` ends and grid `y` begins, and whether a pause lies there.

    The moment is found onset by onset: between the regions' fitted stretches,
    the cut that leaves the most onset energy on `x`'s grid before it and on
    `y`'s after it."""
    lo, hi = min(x.t_end, y.t_start) - win.block_s, max(x.t_end, y.t_start) + win.block_s
    lo, hi = max(lo, x.t_start), min(hi, y.t_end)
    m = (win.t >= lo) & (win.t <= hi)
    ts, ws = win.t[m], win.w[m]
    px, py = x.period / 4.0, y.period / 4.0              # 16th-note lines
    ex, ey = np.abs(_wrap(ts - x.anchor, px)), np.abs(_wrap(ts - y.anchor, py))
    on_x, on_y = ex <= HIT_S, ey <= HIT_S
    last_x, first_y = lo, hi
    pause = None
    if len(ts):
        # how much nearer each onset is to x's grid than to y's; the cut that
        # leaves the most of that before it
        gain = np.concatenate([[0.0], np.cumsum(ws * np.clip(ey - ex, -HIT_S, HIT_S))])
        k = int(np.flatnonzero(gain >= gain.max() - 1e-12)[-1])   # cut after onset k-1
        # a pause in the zone is where a band stops and starts again: cut
        # there unless the onsets say clearly otherwise
        span = float(gain.max() - gain.min()) + 1e-12
        for s0, s1 in pauses:
            if lo < s0 and s1 < hi:
                kp = int(np.searchsorted(ts, 0.5 * (s0 + s1)))
                if gain[kp] >= gain.max() - 0.25 * span and (
                        pause is None or (s1 - s0) > (pause[1] - pause[0])):
                    pause, k = (s0, s1), kp
        before = np.flatnonzero(on_x[:k])
        after = np.flatnonzero(on_y[k:]) + k
        last_x = float(ts[before[-1]]) if len(before) else lo
        first_y = float(ts[after[0]]) if len(after) else hi
    if pause is None:
        for s0, s1 in pauses:
            if s1 > last_x - 0.05 and s0 < first_y + 0.05:
                if pause is None or (s1 - s0) > (pause[1] - pause[0]):
                    pause = (s0, s1)
    else:
        last_x, first_y = min(last_x, pause[0]), max(first_y, pause[1])
    return {"last_x": last_x, "first_y": first_y, "pause": pause}


def _assemble(regions: list, joins: list, offset_hint: float) -> tuple:
    """Number the beats -> (time of beat 0, bpms, stops, events, region beats).

    Beat 0 is the first region's beat line nearest `offset_hint` (the
    single-tempo grid's beat 0), so the map starts where that grid did. BPMs
    and stops are rounded to the three decimals the simfile carries, and the
    numbering is built from the rounded values: what is written is what the
    features were warped on."""
    r0 = regions[0]
    t0 = r0.anchor + np.round((offset_hint - r0.anchor) / r0.period) * r0.period
    bpms = [(0.0, round(r0.bpm, 3))]
    stops, events, spans = [], [], []
    beat = 0.0                                   # beat number of `t_line`
    t_line = float(t0)                           # a beat line of the current region
    first = 0.0
    for i, (x, y) in enumerate(zip(regions, regions[1:])):
        j = joins[i]
        px = 60.0 / bpms[-1][1]                  # the period as written
        py = 60.0 / round(y.bpm, 3)

        if j["pause"] is not None:
            s0, s1 = j["pause"]
            # the last 8th line of x before the silence; the first beat of y
            # after it. Beats run on y's grid from the stop to that beat, and
            # the stop takes up what is left of the gap.
            k = np.floor((s0 + 0.010 - t_line) / (px / 2) + 1e-9)
            b_a = beat + k * 0.5
            t_a = t_line + k * (px / 2)
            frac = float(np.ceil(b_a - 1e-9) - b_a)          # to the next whole beat
            least = frac if frac > 0.25 else 1.0
            t_b = y.line_at_or_after(max(s1 - 0.030, t_a + least * py))
            while t_b - t_a < least * py - 1e-6:
                t_b += y.period
            d = t_b - t_a
            m = frac + np.floor((d / py - frac) + 1e-6)
            stop = max(0.0, d - m * py)
            if abs(bpms[-1][1] - round(y.bpm, 3)) > 1e-9:
                bpms.append((float(b_a), round(y.bpm, 3)))
            if stop >= MIN_STOP_S:
                stops.append((float(b_a), round(stop, 3)))
            else:
                stop = 0.0
            events.append({"kind": "pause", "time_s": round(float(t_a), 3),
                           "beat": float(b_a), "stop_s": round(stop, 3),
                           "silence_s": [round(s0, 3), round(s1, 3)],
                           "bpm_before": round(x.bpm, 3), "bpm_after": round(y.bpm, 3)})
            spans.append((first, float(b_a)))
            beat, t_line = float(b_a + m), float(t_a + round(stop, 3) + m * py)
            first = beat
            continue
        # No pause: x's grid runs to its last 8th-note line before the break,
        # y's from its first beat after it, and the beats between the two are
        # played at whatever tempo fits them in. The join is kept as short as
        # it can be — the music changes grid at one moment, and every beat
        # given to the join is a beat on neither.
        k = np.floor((j["last_x"] + 0.030 - t_line) / (px / 2) + 1e-9) * 0.5
        t_a = t_line + k * px
        frac = float(np.ceil(beat + k - 1e-9) - (beat + k))       # to the next whole beat
        t_b = y.line_at_or_after(max(j["first_y"] - 0.030, t_a + 0.25 * min(px, py)))
        d = t_b - t_a
        n = frac + max(0.0, np.round(d / (0.5 * (px + py)) - frac))
        if n < 0.5:
            n += 1.0
        b_a = beat + k
        conn = 60.0 * n / d
        shift_ms = round(float((d - n * py) * 1000.0), 1)
        if abs(conn / round(y.bpm, 3) - 1.0) <= 5e-4:
            # y's grid passes through x's last beat: change tempo there
            bpms.append((float(b_a), round(y.bpm, 3)))
            t_b_written = t_a + n * py
        else:
            bpms.append((float(b_a), round(conn, 3)))
            bpms.append((float(b_a + n), round(y.bpm, 3)))
            t_b_written = t_a + n * 60.0 / round(conn, 3)
        tempo = abs(y.bpm / x.bpm - 1.0) > 2.5e-3
        events.append({"kind": "tempo" if tempo else "phase", "time_s": round(float(t_a), 3),
                       "beat": float(b_a), "beats": float(n), "shift_ms": shift_ms,
                       "bpm_before": round(x.bpm, 3), "bpm_after": round(y.bpm, 3)})
        spans.append((first, float(b_a)))
        beat, t_line = float(b_a + n), float(t_b_written)
        first = beat
    spans.append((first, float("inf")))
    # drop BPM entries that repeat the one before them
    clean = [bpms[0]]
    for b, v in bpms[1:]:
        if abs(v - clean[-1][1]) > 1e-9:
            if abs(b - clean[-1][0]) < 1e-9:
                clean[-1] = (b, v)
            else:
                clean.append((b, v))
    return float(t0), clean, stops, events, spans
