"""timing.py — beat <-> seconds for a song. The one place that arithmetic lives.

A chart is written in beats and played in seconds, and the two are related by
the song's timing data: the time of beat 0, a BPM for every stretch of beats
and the stops (pauses) between them. Everything that needs one from the other
— the feature warp, the QA metrics, the previews, the simfile writer, the
benchmark — goes through `TimingMap`, so a song with a tempo change or a
mid-song pause is timed the same way everywhere.

Most songs have one tempo and no stops. For those an analysis dict carries only
`bpm` and `offset`, exactly as it always has, and `beats_to_times` uses the
same expression the callers used before this module existed, so their results
do not change by a single bit. A song with a tempo map carries it as

    analysis["tempo_map"] = {"bpms": [[beat, bpm], ...], "stops": [[beat, seconds], ...]}

with `analysis["offset"]` still the time of beat 0 and `analysis["bpm"]` the
tempo most of the song is in (what a UI shows as "the" BPM).
"""
from __future__ import annotations

import numpy as np

TICKS_PER_BEAT = 12             # the 48-tick grid; neural.codec asserts the same
TICKS_PER_MEASURE = 48


class TimingMap:
    """Beat <-> seconds for a song: an offset, BPM segments and stops/delays."""

    def __init__(self, offset_s: float, bpms, stops=(), delays=()):
        """`offset_s` is the time of beat 0 (i.e. -#OFFSET). `bpms` is
        [(beat, bpm), ...]; `stops`/`delays` are [(beat, seconds), ...]."""
        bpms = sorted((float(b), float(v)) for b, v in bpms)
        if not bpms:
            raise ValueError("TimingMap needs at least one BPM")
        if bpms[0][0] > 0:
            bpms.insert(0, (0.0, bpms[0][1]))
        self.offset_s = float(offset_s)
        self.bpm_beats = np.array([b for b, _ in bpms])
        self.bpm_vals = np.array([v for _, v in bpms])
        spb = 60.0 / self.bpm_vals
        seg = np.diff(self.bpm_beats) * spb[:-1]
        self._seg_t = np.concatenate([[0.0], np.cumsum(seg)])
        # A stop pauses AFTER the note on its beat; a delay pauses BEFORE it.
        self._stops = self._pauses(stops)
        self._delays = self._pauses(delays)
        self._inverse = None

    @staticmethod
    def _pauses(items):
        items = sorted((float(b), float(s)) for b, s in items if float(s) != 0.0)
        beats = np.array([b for b, _ in items])
        cum = np.concatenate([[0.0], np.cumsum([s for _, s in items])])
        return beats, cum

    @classmethod
    def constant(cls, bpm: float, offset_s: float) -> "TimingMap":
        return cls(offset_s, [(0.0, bpm)])

    def shifted(self, delta_s: float) -> "TimingMap":
        out = TimingMap.__new__(TimingMap)
        out.__dict__.update(self.__dict__)
        out.offset_s = self.offset_s + float(delta_s)
        return out

    @property
    def is_constant(self) -> bool:
        return (len(self.bpm_vals) == 1 and len(self._stops[0]) == 0
                and len(self._delays[0]) == 0)

    # -- the map as plain data ------------------------------------------- #
    @property
    def bpms(self) -> list:
        return [(float(b), float(v)) for b, v in zip(self.bpm_beats, self.bpm_vals)]

    @property
    def stops(self) -> list:
        beats, cum = self._stops
        return [(float(b), float(s)) for b, s in zip(beats, np.diff(cum))]

    @property
    def delays(self) -> list:
        beats, cum = self._delays
        return [(float(b), float(s)) for b, s in zip(beats, np.diff(cum))]

    def to_dict(self) -> dict:
        """The JSON form an analysis dict carries as `tempo_map`."""
        out = {"bpms": [[b, v] for b, v in self.bpms],
               "stops": [[b, s] for b, s in self.stops]}
        if len(self._delays[0]):
            out["delays"] = [[b, s] for b, s in self.delays]
        return out

    # -- beats -> seconds ------------------------------------------------- #
    def time_at(self, beats) -> np.ndarray:
        b = np.asarray(beats, np.float64)
        j = np.clip(np.searchsorted(self.bpm_beats, b, side="right") - 1,
                    0, len(self.bpm_vals) - 1)
        t = self._seg_t[j] + (b - self.bpm_beats[j]) * 60.0 / self.bpm_vals[j]
        sb, sc = self._stops
        if len(sb):
            t = t + sc[np.searchsorted(sb, b, side="left")]
        db, dc = self._delays
        if len(db):
            t = t + dc[np.searchsorted(db, b, side="right")]
        return t + self.offset_s

    def tick_times(self, n_ticks: int) -> np.ndarray:
        return self.time_at(np.arange(n_ticks) / TICKS_PER_BEAT)

    def bpm_at(self, beats) -> np.ndarray:
        j = np.clip(np.searchsorted(self.bpm_beats, np.asarray(beats, np.float64),
                                    side="right") - 1, 0, len(self.bpm_vals) - 1)
        return self.bpm_vals[j]

    # -- seconds -> beats ------------------------------------------------- #
    def _knots(self):
        """Where the map bends: every BPM change, stop and delay, as (beat, the
        time the music reaches it, the time it leaves it, BPM from there on)."""
        if self._inverse is None:
            k = np.unique(np.concatenate([self.bpm_beats, self._stops[0],
                                          self._delays[0]]))
            note = self.time_at(k)                       # after a delay, before a stop
            sb, sc = self._stops
            db, dc = self._delays
            stop = np.zeros(len(k))
            delay = np.zeros(len(k))
            if len(sb):
                stop = sc[np.searchsorted(sb, k, side="right")] - sc[np.searchsorted(sb, k, side="left")]
            if len(db):
                delay = dc[np.searchsorted(db, k, side="right")] - dc[np.searchsorted(db, k, side="left")]
            self._inverse = (k, note - delay, note + stop, self.bpm_at(k))
        return self._inverse

    def beat_at(self, times) -> np.ndarray:
        """The beat being played at each time. During a stop or a delay that is
        the beat the music is waiting on; before beat 0 the first tempo is run
        backwards, as `time_at` does."""
        t = np.asarray(times, np.float64)
        k, arrive, leave, bpm = self._knots()
        j = np.clip(np.searchsorted(arrive, t, side="right") - 1, 0, len(k) - 1)
        b = k[j] + np.maximum(t - leave[j], 0.0) * bpm[j] / 60.0
        early = t < arrive[0]
        if np.any(early):
            b = np.where(early, k[0] + (t - arrive[0]) * bpm[0] / 60.0, b)
        return b

    def ticks_for_duration(self, duration_s: float) -> int:
        """Ticks needed to cover the audio, rounded up to a whole measure."""
        lo, hi = 0.0, 4.0
        while self.time_at(hi) < duration_s and hi < 1e6:
            hi *= 2
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.time_at(mid) < duration_s:
                lo = mid
            else:
                hi = mid
        n = int(np.ceil(hi * TICKS_PER_BEAT))
        return max(TICKS_PER_MEASURE,
                   int(np.ceil(n / TICKS_PER_MEASURE)) * TICKS_PER_MEASURE)

    def dominant_bpm(self, duration_s: float) -> float:
        """The tempo the song spends the most time in, up to `duration_s`."""
        if len(self.bpm_vals) == 1:
            return float(self.bpm_vals[0])
        edges = np.append(self.time_at(self.bpm_beats), max(duration_s, self.offset_s))
        spans = np.clip(np.diff(np.minimum(edges, max(duration_s, self.offset_s))), 0.0, None)
        total: dict = {}
        for v, s in zip(self.bpm_vals, spans):
            total[float(v)] = total.get(float(v), 0.0) + float(s)
        return max(total, key=lambda v: total[v])


# --------------------------------------------------------------------------- #
# Plain-function forms
# --------------------------------------------------------------------------- #
def beat_to_time(beat, bpms, stops=(), offset: float = 0.0):
    """Seconds at which `beat` is played. `bpms` is [(beat, bpm), ...], `stops`
    [(beat, seconds), ...] and `offset` the time of beat 0 (the negative of a
    simfile's #OFFSET). `beat` may be a number or an array."""
    out = TimingMap(offset, bpms, stops).time_at(beat)
    return float(out) if np.ndim(out) == 0 else out


def time_to_beat(time_s, bpms, stops=(), offset: float = 0.0):
    """The beat being played at `time_s`: the inverse of `beat_to_time`."""
    out = TimingMap(offset, bpms, stops).beat_at(time_s)
    return float(out) if np.ndim(out) == 0 else out


def has_tempo_map(analysis: dict) -> bool:
    return bool(analysis.get("tempo_map"))


def timing_of(analysis: dict) -> TimingMap:
    """The TimingMap of an analysis dict (see the module docstring)."""
    tm = analysis.get("tempo_map")
    if tm:
        return TimingMap(analysis["offset"], tm["bpms"], tm.get("stops", ()),
                         tm.get("delays", ()))
    return TimingMap.constant(analysis["bpm"], analysis["offset"])


def beats_to_times(analysis: dict, beats) -> np.ndarray:
    """Seconds for an array of beats under an analysis dict's timing. For a
    single-tempo song this is `offset + beat / bpm * 60`, the expression every
    caller used before a song could carry a tempo map."""
    b = np.asarray(beats, np.float64)
    if not analysis.get("tempo_map"):
        return analysis["offset"] + b / analysis["bpm"] * 60
    return timing_of(analysis).time_at(b)


def times_to_beats(analysis: dict, times) -> np.ndarray:
    t = np.asarray(times, np.float64)
    if not analysis.get("tempo_map"):
        return (t - analysis["offset"]) * analysis["bpm"] / 60.0
    return timing_of(analysis).beat_at(t)
