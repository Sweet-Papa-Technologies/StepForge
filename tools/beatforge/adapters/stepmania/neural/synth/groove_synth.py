"""groove_synth.py — synthesized tracks with ground-truth charts (the
`synth_groove` source, REQ-SFV15-03).

    python -m beatforge.adapters.stepmania.neural.synth.groove_synth \
        --out-dir build/corpus/synth_groove [--num-tracks 300] [--seed N] [--workers 4]

The rhythm network learns from a few hundred human-charted songs, and two of
its habits trace back to what those songs cannot teach. When a mix carries a
loud off-beat hat or bass it cannot always tell whether the grid is seated on
the beats or between them, and its Hard charts almost never use 16th notes.
Both need examples where the answer is known exactly, so this module renders
the audio itself: every drum hit is placed by us, logged, and charted.

Five archetypes each aim at one weakness (off-beat-heavy EDM, syncopated funk,
8th grooves with 16th fills, triplet grooves, driving 16th stamina tracks). A
track is arranged as notes on the 48-tick grid, rendered to four stems that sum
to the mix exactly, and charted at five difficulties from the log of what was
rendered: no chart note sits where nothing sounds. The shards carry the same
features as a compiled human song plus exact stem channels.

Density is held under `grammar.BUDGETS[tier].max_nps_4s`. For Beginner through
Hard no 4-second window exceeds it. Challenge is held to it as
`conditioning.sustained_nps` — the figure the network is conditioned on —
because a continuous 16th stream at 150 BPM or more is already 10 notes/s and
the stamina tracks exist to show such streams.

The mix is balanced against real songs as the feature code sees them (stem
levels, pitch-height contour, spectral centroid): a sustained pad bed under
the band, a kick and bass that do not swamp the low end, a mastering soft-clip,
and a balance drawn per track. Drums alone over sub-bass would be too easy —
the network has to hear transients through a mid-range bed.

Training-only: `shards.write_source_manifest` puts every track in "train".
numpy + scipy only.
"""
from __future__ import annotations

import argparse
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.signal

from ...grammar import ALL_DIFFICULTIES, BUDGETS
from .. import codec as C
from .. import conditioning as cond
from .. import grid_features as GF
from ..train import corpus
from . import shards

SOURCE = "synth_groove"
ARCHETYPES = ("edm_offbeat_trap", "syncopated_funk_pop", "hard_16th_fills",
              "triplet_groove", "stamina_drive")
#: Tracks per archetype in the default 300-track set.
DEFAULT_COUNTS = {"edm_offbeat_trap": 75, "syncopated_funk_pop": 65,
                  "hard_16th_fills": 80, "triplet_groove": 35, "stamina_drive": 45}
BPM_RANGE = {"edm_offbeat_trap": (120, 145), "syncopated_funk_pop": (95, 132),
             "hard_16th_fills": (115, 165), "triplet_groove": (95, 155),
             "stamina_drive": (150, 195)}
DEFAULT_SEED = 20261119

BAR = C.G
BEAT = C.TICKS_PER_BEAT
SR = GF.SR
#: Hits are humanized by at most this much. Slightly inside the 1.5 ms the
#: charts are specified against, so rounding to a sample cannot exceed it.
MAX_JITTER_S = 0.00145
_TAIL_S = 2.5                       # room after the final hit for it to ring out
_PEAK = 0.98

_ENERGY = {"intro": 1, "verse": 2, "breakdown": 0, "chorus": 3, "outro": 1}
_STEM_OF = {"kick": "drums", "snare": "drums", "clap": "drums", "hat_closed": "drums",
            "hat_open": "drums", "crash": "drums", "bass": "bass", "lead": "vocals",
            "chord": "other", "pad": "other"}
_DRUMS = frozenset(v for v, s in _STEM_OF.items() if s == "drums")


class Hit(NamedTuple):
    """One rendered sound: where it was written, and when it actually starts."""
    tick: int
    voice: str
    velocity: float
    time_s: float


class Section(NamedTuple):
    name: str
    bar0: int
    bars: int
    energy: int                     # 0 breakdown .. 3 chorus / drop


@dataclass
class Track:
    """One synthesized song: what was rendered, the log of it, and its charts."""
    song_id: str
    archetype: str
    bpm: float
    offset_s: float                 # time of tick 0 = beat 1 of bar 1
    n_ticks: int
    seed: int
    audio: np.ndarray               # mono float32 at GF.SR, the sum of `stems`
    stems: dict                     # GF.STEM_NAMES -> float32, same length
    events: list                    # [Hit], in tick order
    sections: tuple                 # (Section, ...)
    charts: dict                    # difficulty -> codec.ChartEvents


class _Note(NamedTuple):
    tick: int
    voice: str
    vel: float
    pitches: tuple = ()             # MIDI numbers, tonal voices only
    dur: int = 0                    # ticks, tonal voices only


def _schedule() -> tuple:
    """DEFAULT_COUNTS dealt out evenly (smooth weighted round-robin), so any
    prefix of the 300 is close to the same proportions."""
    total = sum(DEFAULT_COUNTS.values())
    credit = dict.fromkeys(ARCHETYPES, 0)
    out = []
    for _ in range(total):
        for a in ARCHETYPES:
            credit[a] += DEFAULT_COUNTS[a]
        best = max(ARCHETYPES, key=credit.get)
        credit[best] -= total
        out.append(best)
    return tuple(out)


_SCHEDULE = _schedule()


def archetype_for(index: int) -> str:
    """The archetype of track `index` — a function of the index alone."""
    return _SCHEDULE[index % len(_SCHEDULE)]


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


# --------------------------------------------------------------------------- #
# Voices
# --------------------------------------------------------------------------- #
def _filt(x: np.ndarray, lo: float = 0.0, hi: float = 0.0, order: int = 2) -> np.ndarray:
    """Butterworth high-pass (`lo`), low-pass (`hi`) or band-pass (both), Hz."""
    nyq = SR / 2
    if lo and hi:
        wn, kind = [lo / nyq, min(hi / nyq, 0.98)], "bandpass"
    elif lo:
        wn, kind = lo / nyq, "highpass"
    else:
        wn, kind = min(hi / nyq, 0.98), "lowpass"
    return scipy.signal.sosfilt(scipy.signal.butter(order, wn, kind, output="sos"), x)


def _shot(x: np.ndarray) -> np.ndarray:
    """Finish a one-shot: fade the last 5 ms and normalize to peak 1."""
    x = x.copy()
    m = min(len(x), int(0.005 * SR))
    x[-m:] *= np.linspace(1.0, 0.0, m)
    return x / (np.abs(x).max() + 1e-12)


def _seconds(dur_s: float) -> np.ndarray:
    return np.arange(int(dur_s * SR)) / SR


def _kick(rng) -> np.ndarray:
    t = _seconds(rng.uniform(0.16, 0.26))
    f_hi, f_lo = rng.uniform(140.0, 180.0), rng.uniform(40.0, 52.0)
    sweep_s = rng.uniform(0.045, 0.080)
    freq = f_lo + (f_hi - f_lo) * np.exp(-3.0 * t / sweep_s)
    # a short tail: most of the kick's energy is spent while it is still
    # sweeping down, not as sub-bass afterwards
    body = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / rng.uniform(0.035, 0.07))
    click = _filt(rng.standard_normal(len(t)), 2000.0, 6000.0)
    click *= np.exp(-t / rng.uniform(0.002, 0.005))
    return _shot(body + rng.uniform(0.3, 0.55) * click / np.abs(click).max())


def _snare(rng) -> np.ndarray:
    t = _seconds(rng.uniform(0.2, 0.32))
    body = np.sin(2 * np.pi * rng.uniform(180.0, 240.0) * t)
    body *= np.exp(-t / rng.uniform(0.03, 0.06))
    rattle = _filt(rng.standard_normal(len(t)), 800.0, 7000.0)
    rattle *= np.exp(-t / rng.uniform(0.05, 0.11))
    return _shot(rng.uniform(0.4, 0.8) * body + rattle / np.abs(rattle).max())


def _clap(rng) -> np.ndarray:
    t = _seconds(rng.uniform(0.2, 0.3))
    env = np.exp(-t / 0.007)                       # the first hand, at full height
    for k in (1, 2):
        i = int((0.009 * k + rng.uniform(0.0, 0.003)) * SR)
        env[i:] = np.maximum(env[i:], (0.85 - 0.1 * k) * np.exp(-t[:len(t) - i] / 0.007))
    env[i:] = np.maximum(env[i:], 0.6 * np.exp(-t[:len(t) - i] / rng.uniform(0.05, 0.1)))
    noise = _filt(rng.standard_normal(len(t)), 800.0, 7000.0)
    body = np.sin(2 * np.pi * rng.uniform(180.0, 240.0) * t) * np.exp(-t / 0.03)
    return _shot(noise / np.abs(noise).max() * env + 0.15 * body)


def _cymbal(rng, freqs: np.ndarray, cutoff: float, decay_s: float, dur_s: float) -> np.ndarray:
    """Inharmonic squares plus noise, high-passed: hats and the crash."""
    t = _seconds(dur_s)
    metal = np.sign(np.sin(2 * np.pi * freqs[:, None] * t[None, :])).mean(0)
    x = _filt(metal + 0.6 * rng.standard_normal(len(t)), lo=cutoff, order=4)
    return _shot(x * np.exp(-t / decay_s))


def _drum_kit(rng) -> dict:
    freqs = rng.uniform(2200.0, 5400.0, 6)
    cutoff = rng.uniform(6000.0, 8500.0)
    return {
        "kick": _kick(rng), "snare": _snare(rng), "clap": _clap(rng),
        "hat_closed": _cymbal(rng, freqs, cutoff, rng.uniform(0.02, 0.05), 0.16),
        "hat_open": _cymbal(rng, freqs, cutoff, rng.uniform(0.1, 0.22), 0.6),
        "crash": _cymbal(rng, freqs * rng.uniform(0.6, 0.8), rng.uniform(3000.0, 5000.0),
                         rng.uniform(0.25, 0.45), 1.4),
    }


_WAVES = {"saw": lambda p: 2.0 * p - 1.0,
          "pulse": lambda p: np.where(p < 0.5, 1.0, -1.0),
          "tri": lambda p: 4.0 * np.abs(p - 0.5) - 1.0}


@dataclass(frozen=True)
class _Timbre:
    """A tonal voice: detuned oscillators through a low-pass whose cutoff
    falls from `cut_hi` to `cut_lo`, then an ADSR."""
    shape: str
    detune: tuple                   # cents, one oscillator each
    phases: tuple
    cut_lo: float
    cut_hi: float
    sweep_s: float
    attack_s: float
    decay_s: float
    sustain: float
    release_s: float
    sub: float = 0.0                # level of an added sine at the fundamental


def _timbre(voice: str, rng) -> _Timbre:
    u = rng.uniform
    spread = u(5.0, 16.0)
    detune = _pick(rng, ((-spread, spread), (-spread, 0.0, spread)))
    if voice == "bass":
        return _Timbre(_pick(rng, ("saw", "saw", "pulse")), _pick(rng, ((0.0,), (-6.0, 6.0))),
                       (0.0, 0.5), u(350, 900), u(1200, 3000), u(0.04, 0.15),
                       0.002, u(0.15, 0.4), u(0.5, 0.9), 0.02, sub=u(0.0, 0.3))
    phases = tuple(rng.random(len(detune)))
    shape = _pick(rng, ("saw", "pulse"))
    if voice == "lead":
        return _Timbre(shape, detune, phases, u(1500, 3000), u(3500, 7000), u(0.03, 0.12),
                       u(0.001, 0.003), u(0.08, 0.25), u(0.3, 0.8), u(0.03, 0.08))
    if voice == "chord":
        return _Timbre(shape, detune, phases, u(900, 1800), u(2500, 5000), u(0.03, 0.12),
                       u(0.0015, 0.003), u(0.1, 0.3), u(0.3, 0.7), u(0.04, 0.1))
    cutoff = u(1400, 4500)                         # pad: no sweep, slow attack
    return _Timbre(shape, detune, phases, cutoff, cutoff, 1.0,
                   u(0.03, 0.12), 1.0, 1.0, u(0.15, 0.4))


def _tone(tb: _Timbre, pitches: tuple, n_dur: int) -> np.ndarray:
    """One note or chord of `n_dur` samples plus its release."""
    n = n_dur + int(tb.release_s * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for midi in pitches:
        f = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        for cents, phase in zip(tb.detune, tb.phases):
            x += _WAVES[tb.shape]((f * 2.0 ** (cents / 1200.0) * t + phase) % 1.0)
        x += tb.sub * len(tb.detune) * np.sin(2 * np.pi * f * t)
    y = _filt(x, hi=tb.cut_lo)
    if tb.cut_hi > tb.cut_lo:
        y += (_filt(x, hi=tb.cut_hi) - y) * np.exp(-t / tb.sweep_s)
    env = np.minimum(t / tb.attack_s, 1.0) * (tb.sustain + (1.0 - tb.sustain)
                                              * np.exp(-t / tb.decay_s))
    env *= np.clip((n - np.arange(n)) / max(1, n - n_dur), 0.0, 1.0)
    y *= env
    return y / (np.abs(y).max() + 1e-12)


#: Mix balances as dB offsets for (drums, bass, lead, pads and chords), and how
#: often each is drawn: even, bed-heavy, drum-forward, lead-forward, sparse.
_BALANCES = (((0.0, 0.0, 0.0, 0.0), 3), ((-1.0, -1.0, 0.0, 2.5), 3),
             ((2.5, 0.0, -1.0, -1.5), 2), ((0.0, -1.0, 5.0, -1.0), 2),
             ((1.0, 0.0, -2.0, -3.0), 1))


def _levels(archetype: str, rng) -> dict:
    """Peak level of every voice. Real mixes differ in what is up front, so a
    track draws one of a few balances and then drifts a little from it."""
    u = rng.uniform
    lv = {"kick": u(0.5, 0.7), "snare": u(0.7, 0.95), "clap": u(0.65, 0.85),
          "hat_closed": u(0.55, 0.8), "hat_open": u(0.55, 0.8), "crash": u(0.45, 0.6),
          "bass": u(0.13, 0.22), "lead": u(0.22, 0.34), "chord": u(0.26, 0.4),
          "pad": u(0.26, 0.4)}
    if archetype == "edm_offbeat_trap":
        # the point of the archetype: the off-beat hat out-shouts whatever
        # lands on the beat, so the kick and the backbeat sit back
        lv["hat_closed"] = lv["hat_open"] = u(0.55, 0.7)
        lv["kick"] *= 0.8
        lv["snare"] *= 0.5
        lv["clap"] *= 0.5
    balance = _pick(rng, [b for b, weight in _BALANCES for _ in range(weight)])
    for stem, db in zip(GF.STEM_NAMES, balance):
        gain = 10.0 ** ((db + u(-1.2, 1.2)) / 20.0)
        for voice in lv:
            if _STEM_OF[voice] == stem:
                lv[voice] *= gain
    return lv


# --------------------------------------------------------------------------- #
# Arrangement: sections -> notes on the grid
# --------------------------------------------------------------------------- #
def _plan_sections(rng, bars: int | None) -> tuple:
    """Intro, then verse / breakdown / chorus cycles, then outro, in 4- and
    8-bar blocks. The default is two cycles totalling 48-64 bars."""
    if bars is None:
        while True:
            plan = [("intro", _pick(rng, (4, 8)))]
            for _ in range(2):
                plan += [("verse", 8), ("breakdown", _pick(rng, (4, 8))),
                         ("chorus", _pick(rng, (8, 16)))]
            plan.append(("outro", _pick(rng, (4, 8))))
            if 48 <= sum(n for _, n in plan) <= 64:
                break
    else:
        if bars < 24 or bars % 4:
            raise ValueError("bars must be a multiple of 4, at least 24")
        plan, left, i = [("intro", 4)], bars - 8, 0
        while left > 0:
            name, n = (("verse", 4), ("breakdown", 4), ("chorus", 8))[i % 3]
            plan.append((name, min(n, left)))
            left -= plan[-1][1]
            i += 1
        plan.append(("outro", 4))
    out, bar0 = [], 0
    for name, n in plan:
        out.append(Section(name, bar0, n, _ENERGY[name]))
        bar0 += n
    return tuple(out)


_SCALES = {"minor": (0, 2, 3, 5, 7, 8, 10), "major": (0, 2, 4, 5, 7, 9, 11)}
#: Four-chord loops as scale degrees.
_PROGRESSIONS = {"minor": ((0, 5, 2, 6), (0, 3, 5, 4), (0, 6, 5, 6), (0, 2, 5, 3)),
                 "major": ((0, 4, 5, 3), (0, 5, 3, 4), (1, 4, 0, 5), (0, 3, 5, 4))}


def _traits(archetype: str, rng) -> dict:
    """What stays fixed for a whole track."""
    mode = "minor" if rng.random() < 0.65 else "major"
    style = _pick(rng, ("trance", "dnb", "rock"))
    clap_odds = {"edm_offbeat_trap": 0.6, "stamina_drive": 1.0 if style == "trance" else 0.1}
    return {
        "scale": _SCALES[mode], "key": int(rng.integers(12)),
        "prog": _pick(rng, _PROGRESSIONS[mode]), "harm_bars": _pick(rng, (1, 2)),
        "arp": tuple(int(i) for i in rng.permutation(4)),
        "backbeat": "clap" if rng.random() < clap_odds.get(archetype, 0.2) else "snare",
        "style": style,
        "offhat": _pick(rng, ("hat_open", "hat_open", "hat_closed")),
        "hat16": rng.random() < 0.3,
        "bass": _pick(rng, ((6,), (6,), (6,), (3, 6, 9), (6, 9))),
        "bed_in_intro": rng.random() < 0.5,
    }


def _chord(tr: dict, bar: int) -> tuple:
    """The triad sounding in `bar`, voiced inside one octave from G3."""
    d = tr["prog"][(bar // tr["harm_bars"]) % len(tr["prog"])]
    return tuple(sorted(55 + (tr["key"] + tr["scale"][(d + k) % 7] - 55) % 12
                        for k in (0, 2, 4)))


def _root(tr: dict, bar: int) -> int:
    d = tr["prog"][(bar // tr["harm_bars"]) % len(tr["prog"])]
    return 36 + (tr["key"] + tr["scale"][d]) % 12


def _bed(chord: tuple) -> tuple:
    """A pad voicing: the triad over its root, doubled an octave up, so the
    bed fills the mid-range the way a real arrangement does."""
    return (chord[0] - 12,) + chord + tuple(p + 12 for p in chord)


# A bar pattern is {voice: [(tick in bar, velocity, duration ticks, role)]}.
# `role` picks the pitch: a bass interval, or a step through the arpeggio.
def _at(ticks, vel: float = 1.0, dur: int = 0) -> list:
    return [(int(t), vel, dur, i) for i, t in enumerate(ticks)]


def _pulse(step: int, accents: tuple, dur: int = 0, roles: tuple | None = None) -> list:
    return [(t, accents[i % len(accents)], dur, i if roles is None else roles[i % len(roles)])
            for i, t in enumerate(range(0, BAR, step))]


def _melody(rng, pool, n: int, dur: int) -> list:
    ticks = sorted(rng.choice(np.asarray(pool), n, replace=False))
    return [(int(t), 0.8, dur, int(rng.integers(4))) for t in ticks]


_OFFBEATS = (6, 18, 30, 42)


def _groove_edm(rng, e: int, tr: dict) -> dict:
    if e == 0:
        g = {"bass": _at((0,), 0.8, 40), "lead": _melody(rng, (0, 12, 18, 24, 36, 42), 4, 5)}
        if rng.random() < 0.5:
            g["hat_closed"] = _at(_OFFBEATS, 0.5)
        return g
    g = {"kick": _at((0, 12, 24, 36)), tr["offhat"]: _at(_OFFBEATS),
         "bass": [(beat + t, 0.9, 3, 0) for beat in range(0, BAR, BEAT) for t in tr["bass"]]}
    if e >= 2:
        g[tr["backbeat"]] = _at((12, 36), 0.95)
    if e == 3:
        if tr["hat16"]:
            g.setdefault("hat_closed", []).extend(_at(range(3, BAR, 6), 0.3))
        g["lead"] = _pick(rng, (_pulse(6, (0.9, 0.7), 4), _at((0, 9, 18, 30, 42), 0.9, 5),
                                _at(_OFFBEATS, 0.9, 4)))
        if rng.random() < 0.5:
            g["chord"] = _at(_OFFBEATS, 0.8, 4)
    return g


_FUNK_KICKS = ((0, 18, 33), (0, 18, 30, 33), (0, 9, 18, 33), (0, 18, 33, 42))
_FUNK_BASS = (((0, 5, 0), (18, 5, 0), (33, 3, 1)),
              ((0, 5, 0), (9, 3, 1), (18, 5, 0), (30, 3, 0), (33, 3, 1)),
              ((0, 5, 0), (6, 3, 1), (18, 5, 0), (27, 3, 0), (33, 3, 1), (42, 5, 2)))
_FUNK_STABS = ((6, 21, 30), (0, 9, 18, 42), (9, 21, 33, 45), (6, 18, 33))


def _groove_funk(rng, e: int, tr: dict) -> dict:
    if e == 0:
        return {"bass": _at((0, 18), 0.8, 10),
                "chord": _at(_pick(rng, _FUNK_STABS), 0.7, 3),
                "lead": _melody(rng, range(0, BAR, 6), 3, 5)}
    g = {"kick": _at(_pick(rng, _FUNK_KICKS) if e >= 2 else (0, 18)),
         "bass": [(t, 0.9, d, r) for t, d, r in _pick(rng, _FUNK_BASS)]}
    hats = (_pulse(3, (1.0, 0.5, 0.75, 0.5)) if rng.random() < (0.0, 0.1, 0.25, 0.6)[e]
            else _pulse(6, (1.0, 0.7)))
    if rng.random() < 0.5:
        hats = [h for h in hats if h[0] != 42]
        g["hat_open"] = _at((42,), 0.9)
    g["hat_closed"] = hats
    if e >= 2:
        g[tr["backbeat"]] = _at((12, 36), 0.95)
        ghosts = rng.choice((9, 21, 27, 45), int(rng.integers(3)), replace=False)
        g.setdefault("snare", []).extend(_at(sorted(ghosts), 0.35))
        g["chord"] = _at(_pick(rng, _FUNK_STABS), 0.85, 3)
    if e == 3:
        g["lead"] = _melody(rng, range(0, BAR, 6), 4, 5)
    return g


_ROCK_KICKS = ((0, 24), (0, 18, 24), (0, 24, 30), (0, 12, 24, 36), (0, 24, 42))


def _groove_hard16(rng, e: int, tr: dict) -> dict:
    if e == 0:
        return {"bass": _at((0,), 0.8, 40), "lead": _melody(rng, range(0, BAR, 6), 4, 5)}
    hats = _pulse(6, (1.0, 0.7))
    g = {"kick": _at(_pick(rng, _ROCK_KICKS)),
         "bass": _pulse(6, (0.9, 0.8), 5, _pick(rng, ((0,), (0, 1), (0, 0, 0, 2))))}
    if rng.random() < 0.4:
        hats = [h for h in hats if h[0] != 42]
        g["hat_open"] = _at((42,), 0.9)
    g["hat_closed"] = hats
    if e >= 2:
        g[tr["backbeat"]] = _at((12, 36), 0.95)
    if e == 3:
        if rng.random() < 0.7:
            g["lead"] = _melody(rng, range(0, BAR, 6), 3, 5)
        if rng.random() < 0.5:
            g["chord"] = _at((0, 24), 0.8, 20)
    return g


_SWING = tuple(t for t in range(BAR) if t % BEAT in (0, 8))
_SHUFFLE_KICKS = ((0, 24), (0, 20, 24), (0, 24, 44), (0, 12, 24, 36))
_SHUFFLE_BASS = (tuple((t, 7 if t % BEAT == 0 else 3, 0) for t in _SWING),
                 ((0, 10, 0), (12, 10, 2), (24, 10, 1), (36, 7, 2), (44, 3, 0)),
                 ((0, 7, 0), (8, 3, 0), (24, 7, 0), (32, 3, 1)))
_TRIPLET_STABS = ((8, 20, 32, 44), (0, 8, 24, 32), (0, 4, 8, 24, 28, 32), (8, 12, 32, 36))
_TRIPLET_LEADS = ((36, 40, 44), (12, 16, 20, 36, 40, 44), (24, 28, 32, 36, 40, 44))


def _groove_triplet(rng, e: int, tr: dict) -> dict:
    if e == 0:
        return {"bass": _at((0, 24), 0.8, 20), "lead": _melody(rng, _SWING, 4, 4)}
    g = {"kick": _at(_pick(rng, _SHUFFLE_KICKS)),
         "bass": [(t, 0.9, d, r) for t, d, r in _pick(rng, _SHUFFLE_BASS)]}
    if e == 3 and rng.random() < 0.5:
        g["hat_closed"] = _pulse(4, (1.0, 0.45, 0.7))
    else:
        g["hat_closed"] = [(t, 1.0 if t % BEAT == 0 else 0.65, 0, 0) for t in _SWING]
    if e >= 2:
        g[tr["backbeat"]] = _at((12, 36), 0.95)
        g["chord"] = _at(_pick(rng, _TRIPLET_STABS), 0.85, 3)
    if e == 3:
        g["lead"] = _at(_pick(rng, _TRIPLET_LEADS), 0.9, 3)
    return g


_DRIVE_KICKS = {"trance": ((0, 12, 24, 36),), "dnb": ((0, 30), (0, 18, 30)),
                "rock": ((0, 18, 24), (0, 6, 24, 30), (0, 12, 24, 36))}


def _groove_stamina(rng, e: int, tr: dict) -> dict:
    trance = tr["style"] == "trance"
    if e == 0:
        g = {"bass": _at((0,), 0.8, 40), "lead": _pulse(6, (0.65, 0.5), 4)}
        if rng.random() < 0.5:
            g["hat_closed"] = _pulse(6, (0.6, 0.4))
        return g
    g = {"kick": _at(_pick(rng, _DRIVE_KICKS[tr["style"]]))}
    if e == 3:
        g["hat_closed"] = _pulse(3, (1.0, 0.55, 0.8, 0.55))
        g["bass"] = ([(beat + t, 0.9, 2, 0) for beat in range(0, BAR, BEAT) for t in (3, 6, 9)]
                     if trance else _pulse(3, (1.0, 0.8, 0.9, 0.8), 2, (0,)))
        g["lead"] = (_pulse(3, (0.7, 0.5, 0.6, 0.5), 2) if rng.random() < 0.35
                     else _pulse(6, (0.7, 0.55), 4))
    else:
        g["hat_closed"] = _pulse(6, (1.0, 0.7))
        g["bass"] = _at(_OFFBEATS, 0.9, 4) if trance else _pulse(6, (0.9, 0.8), 4, (0,))
        if e == 2 and rng.random() < 0.5:
            g["lead"] = _melody(rng, range(0, BAR, 6), 3, 5)
    if e >= 2:
        g[tr["backbeat"]] = _at((12, 36), 0.95)
    return g


_GROOVES = {"edm_offbeat_trap": _groove_edm, "syncopated_funk_pop": _groove_funk,
            "hard_16th_fills": _groove_hard16, "triplet_groove": _groove_triplet,
            "stamina_drive": _groove_stamina}
#: Per archetype: fill subdivision (ticks), fill lengths in notes, odds of a
#: fill at a phrase end, odds of a lead arpeggio figure in any other bar.
_FILLS = {"edm_offbeat_trap": (3, (4, 8), 0.6, 0.15),
          "syncopated_funk_pop": (3, (4, 6, 8), 0.8, 0.0),
          "hard_16th_fills": (3, (4, 6, 8, 12), 1.0, 0.45),
          "triplet_groove": (4, (3, 6, 12), 1.0, 0.0),
          "stamina_drive": (3, (8, 12, 16), 0.8, 0.2)}
#: Archetypes whose breakdown ends in a two-bar snare roll into the drop.
_BUILD_ROLLS = ("edm_offbeat_trap", "stamina_drive")
_FILL_MUTES = ("hat_closed", "hat_open", "snare", "clap")


def _figures(archetype: str, rng, sections: tuple, si: int, k: int) -> list:
    """The fills and arpeggio figures of bar `k` of section `si`, as
    (voice, step, notes, end tick in bar). A figure runs up to `end`, and the
    beat it lands on is where its chart run finishes."""
    sec = sections[si]
    last = k == sec.bars - 1
    into_drop = si + 1 < len(sections) and sections[si + 1].energy == 3
    step, sizes, fill_odds, arp_odds = _FILLS[archetype]
    if sec.energy == 0:
        if into_drop and archetype in _BUILD_ROLLS:
            if last:
                return [("snare", 3, 16, BAR)]
            return [("snare", 6, 8, BAR)] if k == sec.bars - 2 else []
        return [("snare", step, _pick(rng, sizes), BAR)] if last else []
    if (k + 1) % 4 == 0 and (sec.energy >= 2 or last) and rng.random() < fill_odds:
        return [("snare", step, _pick(rng, sizes), BAR)]
    if sec.energy >= 2 and k % 4 == 1 and rng.random() < 0.35 * fill_odds:
        return [("snare", step, sizes[0], BAR)]
    if sec.energy >= 2 and rng.random() < arp_odds:
        return [("lead", step, _pick(rng, (4, 8)), _pick(rng, (24, BAR)))]
    return []


def _arrange(archetype: str, rng, sections: tuple) -> list:
    """Sections -> every note of the track, sorted by tick."""
    tr = _traits(archetype, rng)
    n_bars = sections[-1].bar0 + sections[-1].bars
    notes = []
    accents = {n_bars * BAR}                       # downbeats that get a crash
    for si, sec in enumerate(sections):
        pattern = _GROOVES[archetype](rng, sec.energy, tr)
        if sec.energy != 1 or tr["bed_in_intro"]:
            pattern["pad"] = _at((0,))
        if sec.energy == 3:
            accents.add(sec.bar0 * BAR)
        for k in range(sec.bars):
            bar = sec.bar0 + k
            hits = {v: list(items) for v, items in pattern.items()}
            for voice, step, n, end in _figures(archetype, rng, sections, si, k):
                start = end - step * n
                for v in (_FILL_MUTES if voice == "snare" else (voice,)):
                    hits[v] = [h for h in hits.get(v, ()) if not start <= h[0] < end]
                hits[voice].extend((start + step * i, 0.6 + 0.4 * (i + 1) / n, step - 1, i)
                                   for i in range(n))
                if voice == "snare" and end == BAR:
                    accents.add((bar + 1) * BAR)
            chord = _chord(tr, bar)
            arp = chord + (chord[0] + 12,)
            for voice, items in hits.items():
                for t, vel, dur, role in items:
                    tick = bar * BAR + t
                    if voice == "pad":
                        if k and bar % tr["harm_bars"]:
                            continue               # the chord has not changed
                        span = min(tr["harm_bars"] - bar % tr["harm_bars"], sec.bars - k)
                        notes.append(_Note(tick, voice, vel, _bed(chord), span * BAR - 2))
                    elif voice == "bass":
                        notes.append(_Note(tick, voice, vel,
                                           (_root(tr, bar) + (0, 12, 7)[role % 3],), dur))
                    elif voice == "lead":
                        pitch = arp[tr["arp"][role % 4]] + 12
                        notes.append(_Note(tick, voice, vel, (pitch,), dur))
                    elif voice == "chord":
                        notes.append(_Note(tick, voice, vel, chord, dur))
                    else:
                        notes.append(_Note(tick, voice, vel))
    kicks = {n.tick for n in notes if n.voice == "kick"}
    for tick in sorted(accents):
        notes.append(_Note(tick, "crash", 0.9))
        if tick not in kicks:
            notes.append(_Note(tick, "kick", 1.0))
    end = n_bars * BAR                             # the final hit, left to ring
    notes += [_Note(end, "bass", 0.9, (_root(tr, 0),), 24),
              _Note(end, "chord", 0.9, _chord(tr, 0), 24)]
    return sorted(notes, key=lambda n: (n.tick, n.voice))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _reverb_ir(rng) -> np.ndarray:
    """A Schroeder reverb as an impulse response: eight parallel combs
    (averaged) into four all-passes. It has to be diffuse and stay quiet: the
    features log-compress magnitudes, so a distinct slap-back echo only 25 dB
    under a snare reads as an onset of its own, 20-50 ms after the real one."""
    rt60 = rng.uniform(0.3, 0.8)
    room = rng.uniform(0.7, 1.2)
    ir = np.zeros(int(1.3 * rt60 * SR))
    for delay_ms in (25.3, 26.9, 29.0, 30.7, 32.2, 33.8, 35.3, 36.7):
        d = int(delay_ms * room * SR / 1000.0)
        echoes = np.arange(d, len(ir), d)
        gain = 10.0 ** (-3.0 * d / (rt60 * SR))
        ir[echoes] += 0.125 * gain ** np.arange(1, len(echoes) + 1)
    for delay_ms in (12.6, 10.0, 7.7, 5.1):
        d = int(delay_ms * SR / 1000.0)
        b, a = np.zeros(d + 1), np.zeros(d + 1)
        b[0], b[-1], a[0], a[-1] = -0.5, 1.0, 1.0, -0.5
        ir = scipy.signal.lfilter(b, a, ir)
    return ir


def _bus_gain(mix: np.ndarray, drive: float) -> np.ndarray:
    """The gain curve of a mastering soft-clipper: what `tanh` does to `mix`
    at each sample, as a factor. Applied to every stem alike it keeps them
    summing to the (clipped) mix, and it does what loudness mastering does to
    any released record: the drum peaks give way and the body of the track
    comes up under them. It acts within the peak itself, so nothing swells
    back afterwards to be mistaken for an onset."""
    x = drive * mix / np.abs(mix).max()
    return np.divide(np.tanh(x), x, out=np.ones_like(x), where=np.abs(x) > 1e-9)


def _render(notes: list, archetype: str, bpm: float, offset_s: float, n_samples: int,
            rng) -> tuple:
    """Notes -> (four float64 stems, the event log)."""
    kit = _drum_kit(rng)
    timbres = {v: _timbre(v, rng) for v in ("bass", "lead", "chord", "pad")}
    levels = _levels(archetype, rng)
    tick_s = 60.0 / bpm / BEAT
    stems = {name: np.zeros(n_samples) for name in GF.STEM_NAMES}
    tones: dict = {}
    events = []
    jitter = rng.uniform(-MAX_JITTER_S, MAX_JITTER_S, len(notes))
    human = rng.uniform(0.85, 1.15, len(notes))
    for note, dt, h in zip(notes, jitter, human):
        if note.voice in kit:
            shot = kit[note.voice]
        else:
            key = (note.voice, note.pitches, int(round(0.9 * note.dur * tick_s * SR)))
            if key not in tones:
                tones[key] = _tone(timbres[note.voice], note.pitches, key[2])
            shot = tones[key]
        i = int(round((offset_s + note.tick * tick_s + dt) * SR))
        vel = float(note.vel * h)
        buf = stems[_STEM_OF[note.voice]]
        j = min(n_samples, i + len(shot))
        buf[i:j] += levels[note.voice] * vel * shot[:j - i]
        events.append(Hit(note.tick, note.voice, vel, i / SR))

    if archetype == "edm_offbeat_trap":
        # side-chain: every kick pushes the bass and pads down, and they swell
        # back in time for the off-beat
        t = _seconds(0.35)
        dip = np.minimum(t / 0.004, 1.0) * np.exp(-t / rng.uniform(0.05, 0.09))
        dip *= rng.uniform(0.6, 0.9)
        duck = np.zeros(n_samples)
        for e in events:
            if e.voice == "kick":
                i = int(round(e.time_s * SR))
                j = min(n_samples, i + len(dip))
                duck[i:j] = np.maximum(duck[i:j], dip[:j - i])
        stems["bass"] *= 1.0 - duck
        stems["other"] *= 1.0 - duck

    # glue: reverb sends (the bass stays dry), one EQ tilt, a soft clip, a clean tail
    ir = _reverb_ir(rng)
    wet = rng.uniform(0.1, 0.3)
    for name, send in (("drums", 2.0), ("vocals", 1.2), ("other", 0.8)):
        stems[name] += wet * send * scipy.signal.oaconvolve(stems[name], ir)[:n_samples]
    shelf = 10.0 ** (rng.uniform(-4.0, 4.0) / 20.0) - 1.0
    b, a = scipy.signal.butter(1, 1200.0 / (SR / 2), "highpass")
    fade = np.linspace(1.0, 0.0, int(0.05 * SR))
    for buf in stems.values():
        buf += shelf * scipy.signal.lfilter(b, a, buf)
    gain = _bus_gain(sum(stems.values()), rng.uniform(1.5, 2.5))
    for buf in stems.values():
        buf *= gain
        buf[-len(fade):] *= fade
    return stems, events


# --------------------------------------------------------------------------- #
# Charts from the event log
# --------------------------------------------------------------------------- #
#: How much a hit of each voice asks for a step. Pads have no attack and are
#: never a reason for a note.
_WEIGHT = {"kick": 1.0, "snare": 1.0, "clap": 0.9, "crash": 0.8, "lead": 0.7, "bass": 0.6,
           "chord": 0.6, "hat_open": 0.45, "hat_closed": 0.3}
_STRONG = frozenset(("kick", "snare", "clap", "bass"))           # carry a quarter pulse
_ACCENT = frozenset(("kick", "snare", "clap", "lead", "chord"))  # earn an Easy 8th
_FIGURE = _DRUMS | {"lead"}                                      # may carry a 16th / 12th
#: A 16th or 12th note needs at least this much under it: a ghost hat alone
#: is too faint to chart.
_RUN_MIN_SALIENCE = 0.3
_RUN_PRIORITY = 100.0


@dataclass(frozen=True)
class _Tier:
    target: tuple                   # base-layer notes/s by section energy 0..3
    runs: dict                      # fine step (ticks) -> run lengths in notes, longest first
    fine: tuple = (0, 0)            # range the count of off-8th-grid notes is drawn from
    jumps: tuple = (0.0, 0.0)       # jump share range
    holds: int = 0
    hold_ticks: tuple = ()


_TIERS = {
    "beginner": _Tier((0.0, 1.0, 1.65, 1.85), {}),
    "easy": _Tier((0.9, 1.6, 2.1, 2.65), {}, holds=4, hold_ticks=(24, 48)),
    "medium": _Tier((1.8, 3.0, 3.9, 4.5), {}, jumps=(0.06, 0.11), holds=10,
                    hold_ticks=(24, 36)),
    "hard": _Tier((2.5, 4.5, 5.8, 6.6), {3: (9, 5, 3), 4: (7, 4)}, fine=(20, 33),
                  jumps=(0.04, 0.08), holds=6, hold_ticks=(12, 24)),
    "challenge": _Tier((3.0, 5.5, 7.0, 8.0), {3: (17, 13, 9, 5, 3), 4: (13, 7, 4)},
                       fine=(56, 97), jumps=(0.05, 0.09), holds=4, hold_ticks=(12,)),
}


class _Log:
    """The event log as the chart builder reads it: what sounds on each tick."""

    def __init__(self, events: list, sustain: dict, sections: tuple,
                 timing: GF.TimingMap, triplet: bool):
        self.voices: dict = {}
        self.salience: dict = {}
        for e in events:
            if e.voice in _WEIGHT:
                self.voices.setdefault(e.tick, set()).add(e.voice)
                self.salience[e.tick] = (self.salience.get(e.tick, 0.0)
                                         + _WEIGHT[e.voice] * e.velocity)
        self.sustain = sustain                     # tick -> ticks a pad or long note lasts
        self.sections = sections
        self.timing = timing
        self.end = (sections[-1].bar0 + sections[-1].bars) * BAR
        self.tick_s = 60.0 / float(timing.bpm_vals[0]) / BEAT
        #: The off-beat of the 8th layer (swung in a triplet groove) and the
        #: spacing of the notes finer than it.
        self.off8, self.step = (8, 4) if triplet else (6, 3)

    def on_base(self, tick: int) -> bool:
        return tick % BEAT in (0, self.off8)

    def carries_fine(self, tick: int) -> bool:
        return (bool(self.voices.get(tick, frozenset()) & _FIGURE)
                and self.salience[tick] >= _RUN_MIN_SALIENCE)

    def times_s(self, ticks) -> np.ndarray:
        return self.timing.time_at(np.asarray(sorted(ticks)) / BEAT)


def _base_layer(log: _Log, diff: str, rng) -> dict:
    """The groove of a tier: per bar, the best-supported ticks of its grid, as
    many as the section's energy and the tempo allow. Returns tick -> priority."""
    target = _TIERS[diff].target
    prio = {}
    for sec in log.sections:
        shape = rng.random(BAR) * 0.3              # one phrase shape per section
        owed = 0.0
        for bar in range(sec.bar0, sec.bar0 + sec.bars):
            # a fractional notes-per-bar target is met over several bars
            owed += target[sec.energy] * BAR * log.tick_s
            n_max = int(owed + 1e-9)
            owed -= n_max
            cands = []
            for r in range(BAR):
                tick = bar * BAR + r
                voices = log.voices.get(tick)
                if voices is None:
                    continue
                quarter = r % BEAT == 0
                if diff == "beginner":
                    ok = quarter and bool(voices & _STRONG)
                elif diff == "easy":
                    ok = quarter or (r % 6 == 0 and sec.energy >= 2 and bool(voices & _ACCENT))
                else:
                    ok = log.on_base(tick)
                if ok:
                    place = (8 if r == 0 else 6 if r == 24 else 4) if quarter else 0
                    cands.append((place + log.salience[tick] + shape[r], tick))
            for p, tick in sorted(cands, reverse=True)[:n_max]:
                prio[tick] = p
    if log.end in log.voices:
        prio[log.end] = 10.0
    return prio


def _pick_runs(log: _Log, diff: str, rng) -> set:
    """16th (or 12th) runs for a tier: figures that end on a beat — 3, 5 or 9
    notes at Hard, longer ones too at Challenge — with every fine note on a
    rendered drum or arpeggio hit. Best-supported first, until the tier's
    count of fine notes is reached."""
    tier = _TIERS[diff]
    lengths = tier.runs[log.step]
    cands = []
    for end in range(BEAT, log.end + 1, BEAT):
        if end not in log.voices:
            continue
        for n in lengths:
            body = range(end - log.step * (n - 1), end, log.step)
            fine = [t for t in body if not log.on_base(t)]
            if body[0] >= 0 and all(t in log.voices for t in body) and all(
                    log.carries_fine(t) for t in fine):
                score = (min(log.salience[t] for t in fine) + 0.3 * (end % 24 == 0)
                         + 0.3 * rng.random())
                cands.append((score, end, n, len(fine)))
                break
    target = int(rng.integers(*tier.fine))
    plentiful = sum(c[3] for c in cands) >= 4 * target
    taken, total, ticks = [], 0, set()
    for _, end, n, _ in sorted(cands, reverse=True):
        if total >= target:
            break
        if plentiful and n > lengths[-1] and rng.random() < 0.4:
            n = lengths[lengths.index(n) + 1]      # vary the figure when there is room to
        start = end - log.step * (n - 1)
        if any(start - 6 < e and s - 6 < end for s, e in taken):
            continue                               # runs stay separate figures
        body = range(start, end + 1, log.step)
        taken.append((start, end))
        total += sum(not log.on_base(t) for t in body)
        ticks.update(body)
    return ticks


def _thin(prio: dict, tick_s: float, cap: int) -> None:
    """Drop the least important note of the fullest 4-second window until no
    window holds more than `cap` notes."""
    ticks = np.array(sorted(prio))
    while len(ticks):
        t = ticks * tick_s
        # a note exactly 4 s after another counts as inside, so float fuzz in
        # anyone else's arithmetic cannot push a window over
        last = np.searchsorted(t, t + cond.NPS_WINDOW_S + 1e-6, side="right")
        count = last - np.arange(len(t))
        i = int(count.argmax())
        if count[i] <= cap:
            return
        drop = min(ticks[i:i + count[i]].tolist(), key=lambda k: (prio[k], -k))
        del prio[drop]
        ticks = ticks[ticks != drop]


def _add_streams(log: _Log, prio: dict, cap_nps: float) -> set:
    """Continuous 16ths through the drops of a stamina track: the longest
    block of each drop that the sustained-rate budget still admits."""
    added = set()
    for sec in log.sections:
        if sec.energy != 3:
            continue
        for bars in (16, 8, 4):
            if bars > sec.bars:
                continue
            span = range(sec.bar0 * BAR, (sec.bar0 + bars) * BAR + 1, 3)
            if not all(log.on_base(t) and t in log.voices or log.carries_fine(t)
                       for t in span):
                continue
            if cond.sustained_nps(log.times_s(set(prio) | added | set(span))) <= cap_nps:
                added.update(span)
                break
    for t in added:
        prio[t] = _RUN_PRIORITY
    return added


def _pick_holds(log: _Log, diff: str, prio: dict, figures: set, rng) -> dict:
    """Downbeat notes under a sustained pad or long note become holds. Only
    quarter notes stay inside one (the free foot takes them). Returns
    head tick -> length in ticks."""
    tier, budget = _TIERS[diff], BUDGETS[diff]
    limit = min(tier.holds, budget.max_holds, int(0.9 * budget.hold_share[1] * len(prio)))
    heads = [t for t in sorted(prio) if t % BAR == 0 and t in log.sustain]
    holds: dict = {}
    for i in rng.permutation(len(heads)):
        head = heads[int(i)]
        if len(holds) >= limit:
            break
        fits = [d for d in tier.hold_ticks if d <= log.sustain[head]]
        if not fits or any(abs(head - h) < 2 * BAR for h in holds):
            continue
        dur = _pick(rng, fits)
        if any(t in figures for t in range(head, head + dur + 1)):
            continue
        for t in range(head + 1, head + dur):
            if t % BEAT and t in prio:
                del prio[t]
        holds[head] = dur
    return holds


def _pick_jumps(log: _Log, diff: str, ticks: list, holds: dict, rng) -> set:
    """Jump accents: crashes first, then kicks on downbeats, then other kick
    and backbeat hits. Never next to a 16th and never under a hold."""
    lo, hi = _TIERS[diff].jumps
    want = int(round(rng.uniform(lo, hi) * len(ticks))) if hi else 0
    held = {t for head, dur in holds.items() for t in range(head, head + dur)}
    cands = []
    for i, tick in enumerate(ticks):
        voices = log.voices[tick]
        if tick % BEAT or tick in held or not voices & {"kick", "crash", "snare", "clap"}:
            continue
        if (i and tick - ticks[i - 1] < 4) or (i + 1 < len(ticks) and ticks[i + 1] - tick < 4):
            continue
        weight = (3.0 if "crash" in voices else 0.5 if "kick" not in voices
                  else 2.0 if tick % BAR == 0 else 1.0)
        cands.append((weight + rng.random(), tick))
    return {tick for _, tick in sorted(cands, reverse=True)[:want]}


_SINGLES = np.array([t for t in C.SINGLE_TOKENS if t % 4 != C.CROSS])
_JUMPS = np.array([8 + i for i, pair in enumerate(C.JUMP_PAIRS) if C.CROSS not in pair])


def _foot_walk(ticks: list, jumps: set, holds: dict, tick_ms: float, rng) -> C.ChartEvents:
    """Give every row real panels by walking the codec's own state machine and
    taking only its cheapest moves: feet alternate and never cross; while one
    foot holds, the other takes the notes. The codec's annotator can then
    always find an assignment at least as comfortable as this one."""
    cost = C.tables()["cost"]
    state = C.start_state(int(rng.integers(2)))
    release = [0, 0]                               # per foot: the tick its hold ends
    rows, prev = [], None
    for tick in ticks:
        gap_ms = 1e9 if prev is None else (tick - prev) * tick_ms
        lock = int(tick < release[C.LEFT]) | int(tick < release[C.RIGHT]) << 1
        jump = tick in jumps
        toks = _JUMPS if jump else _SINGLES
        c = cost[int(C.dt_bucket(gap_ms)), lock, state, toks]
        # a jump may move one foot; a step takes a strictly cheapest option
        tok = int(rng.choice(toks[c <= c.min() + int(jump)]))
        pl, pr, foot, pa, ps = C.apply_token(*C.state_tuple(state), tok)
        state = C.state_index(pl, pr, foot)
        if tick in holds:
            rows.append((tick, [(max(pa, ps), C.HOLD, holds[tick])]))
            release[foot] = tick + holds[tick]
        else:
            rows.append((tick, [(p, C.TAP, 0) for p in (pa, ps) if p >= 0]))
        prev = tick
    return C.ChartEvents.from_rows(rows)


def _chart(log: _Log, diff: str, streams: bool, rng) -> C.ChartEvents:
    """One difficulty: groove, then runs, the density budget, streams, holds,
    jumps, and panels last."""
    cap = BUDGETS[diff].max_nps_4s
    prio = _base_layer(log, diff, rng)
    figures = set()
    if _TIERS[diff].runs:
        figures = _pick_runs(log, diff, rng)
        prio.update(dict.fromkeys(figures, _RUN_PRIORITY))
    _thin(prio, log.tick_s, int(cap * cond.NPS_WINDOW_S))
    if streams:
        figures |= _add_streams(log, prio, cap)
    holds = _pick_holds(log, diff, prio, figures, rng)
    ticks = sorted(prio)
    jumps = _pick_jumps(log, diff, ticks, holds, rng)
    return _foot_walk(ticks, jumps, holds, log.tick_s * 1000.0, rng)


# --------------------------------------------------------------------------- #
# Tracks and shards
# --------------------------------------------------------------------------- #
def render_track(index: int, seed: int, archetype: str | None = None,
                 bars: int | None = None) -> Track:
    """Track `index` of the set seeded by `seed`: audio, stems, event log and
    five charts. A function of (seed, index) and the two overrides alone.
    `bars` shortens the arrangement (tests); the default is 48-64 bars."""
    archetype = archetype or archetype_for(index)
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype '{archetype}'")
    rng = np.random.default_rng([seed, index])
    lo, hi = BPM_RANGE[archetype]
    bpm = round(float(rng.uniform(lo, hi)), 1)
    offset_s = round(float(rng.uniform(0.05, 1.5)), 4)
    sections = _plan_sections(rng, bars)
    notes = _arrange(archetype, rng, sections)
    timing = GF.TimingMap.constant(bpm, offset_s)
    n_samples = int((float(timing.time_at(notes[-1].tick / BEAT)) + _TAIL_S) * SR)
    stems, events = _render(notes, archetype, bpm, offset_s, n_samples, rng)

    # one gain for everything, then the mix is literally the sum of the stems
    gain = _PEAK / np.abs(sum(stems.values())).max()
    stems = {name: (buf * gain).astype(np.float32) for name, buf in stems.items()}
    audio = sum(stems[name] for name in GF.STEM_NAMES)

    sustain = {n.tick: n.dur for n in notes if n.dur >= 18}
    log = _Log(events, sustain, sections, timing, archetype == "triplet_groove")
    streams = {"challenge"} if archetype == "stamina_drive" else set()
    charts = {diff: _chart(log, diff, diff in streams, rng) for diff in ALL_DIFFICULTIES}
    return Track(f"{SOURCE}_{index:04d}", archetype, bpm, offset_s,
                 timing.ticks_for_duration(n_samples / SR), int(seed), audio, stems,
                 events, sections, charts)


def _meter(diff: str, times_s: np.ndarray) -> int:
    lo, hi = BUDGETS[diff].meter_range
    return int(np.clip(round(cond.effective_meter(cond.sustained_nps(times_s))), lo, hi))


def track_shard(track: Track) -> tuple:
    """(arrays, meta) for `corpus.save_shard`, with the features computed as
    the corpus compiler computes them for a human song."""
    # `GF.load_audio` peak-normalizes; the stems take the same gain
    peak = float(np.abs(track.audio).max())
    y = (track.audio / peak).astype(np.float32)
    stems = {name: (buf / peak).astype(np.float32) for name, buf in track.stems.items()}
    timing = GF.TimingMap.constant(track.bpm, track.offset_s)
    ff = GF.frame_features(y)
    core, tick_times, dt_ms = GF.warp_core(ff, timing, track.n_ticks)
    arrays = {"core": core.astype(np.float16), "dt_ms": dt_ms.astype(np.float32),
              "stems": GF.stem_features(stems, timing, track.n_ticks).astype(np.float16),
              "fingerprint": corpus.spectral_fingerprint(ff.level)}
    rows = []
    for j, diff in enumerate(ALL_DIFFICULTIES):
        ev = track.charts[diff]
        times_ms = timing.time_at(ev.tick / BEAT) * 1000.0
        rows.append(corpus.pack_chart(arrays, j, ev, times_ms, _meter(diff, times_ms / 1000.0),
                                      diff, has_stems=True))
    meta = shards.base_meta(
        track.song_id, track.song_id, SOURCE, title=f"{track.archetype} {track.song_id[-4:]}",
        artist="StepForge synth", n_ticks=track.n_ticks, duration_s=len(y) / SR,
        has_stems=True, archetype=track.archetype, bpm=track.bpm, seed=track.seed,
        tick0_time_s=float(tick_times[0]), timing=corpus.timing_facts(timing),
        sections=[[s.name, s.bar0, s.bars] for s in track.sections])
    meta["charts"] = rows
    return arrays, meta


def _work(job: tuple) -> dict:
    index, seed, out_dir = job
    t0 = time.time()
    track = render_track(index, seed)
    arrays, meta = track_shard(track)
    corpus.save_shard(Path(out_dir) / "songs" / f"{track.song_id}.npz", arrays, meta)
    return {"key": track.song_id, "archetype": track.archetype, "bpm": track.bpm,
            "duration_s": meta["duration_s"], "seconds": round(time.time() - t0, 2)}


def generate(out_dir, num_tracks: int = 300, seed: int = DEFAULT_SEED, workers: int = 4,
             log=print) -> dict:
    """Render `num_tracks` tracks into `<out_dir>/songs/` and index them.
    Track i depends on (seed, i) only, so any worker count gives the same
    files. Returns the manifest."""
    out_dir = Path(out_dir)
    jobs = [(i, seed, str(out_dir)) for i in range(num_tracks)]
    t0 = time.time()

    def report(rows) -> None:
        for n, row in enumerate(rows, 1):
            log(f"[{SOURCE}] {n}/{num_tracks} {row['key']} {row['archetype']} "
                f"{row['bpm']} bpm {row['duration_s']:.0f}s audio in {row['seconds']}s")

    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            report(pool.imap_unordered(_work, jobs))
    else:
        report(map(_work, jobs))
    log(f"[{SOURCE}] {num_tracks} tracks in {time.time() - t0:.0f}s")
    return shards.write_source_manifest(out_dir, SOURCE, log=log)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-tracks", type=int, default=sum(DEFAULT_COUNTS.values()))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    generate(args.out_dir, args.num_tracks, args.seed, args.workers)


if __name__ == "__main__":
    main()
