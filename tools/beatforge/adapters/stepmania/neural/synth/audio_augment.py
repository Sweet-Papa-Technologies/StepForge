"""audio_augment.py — acoustically different views of each TRAINING song, with
the human charts untouched.

    python -m beatforge.adapters.stepmania.neural.synth.audio_augment \
        --real-dir build/v15_corpus/real --stems-dir build/stems_cache \
        --out-dir build/v15_corpus/aug_stem --variants-per-song 7 --workers 4

A few hundred songs is little audio for the rhythm network to generalize from,
and the charts are the expensive part. Each variant re-renders a training
song from its four stems and recomputes every feature channel; the charts ride
along on the same ticks. Seven recipes, seeded per (song, variant):

    0  remix     stem gains redrawn, vocals muted (an instrumental)
    1  remix     stem gains redrawn
    2  pitch     bass / vocals / other shifted down 1-3 semitones, drums
                 left alone; a gentle tonal tilt, sometimes soft compression
    3  pitch     the same, up
    4  tempo     everything 4-8 % slower; BPM, offset and stops rescaled
    5  tempo     everything 4-8 % faster
    6  degrade   mild gain jitter, +-1 semitone, then a lossy-codec round trip
                 and soft clipping: the rip a user actually feeds the app

Drums are never pitch-shifted: their transients are what the flux channels
measure. A tempo change keeps tick k at the same musical position, so only
`dt_ms`, the note times and the charts' measured density change — and those
are recomputed, never scaled by hand.

Leak rule: variants are built only from songs `shards.training_parents` lists
(split "train" in the real manifest) and carry the parent's `song_id`.

Pitch and tempo changes use the `rubberband` command-line tool (its R3 "fine"
engine keeps a click within 0.5 ms of where it belongs). Without it, tempo
falls back to plain resampling — exact in time, with pitch following tempo —
and pitch variants keep their tilt but not their shift; `--require-rubberband`
turns that fallback into an error for a real run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.signal

from .. import codec as C
from .. import grid_features as GF
from ..train import corpus
from . import shards, stems_cache

SR = GF.SR
VARIANTS = 7
KINDS = ("remix", "remix", "pitch", "pitch", "tempo", "tempo", "degrade")
#: Variants mild enough to stand in for the clean song late in training.
MILD_KINDS = ("remix",)
GAIN_RANGE = {"drums": (0.65, 1.40), "bass": (0.40, 1.35), "vocals": (0.00, 1.35),
              "other": (0.35, 1.35)}
HARMONIC = ("bass", "vocals", "other")
TEMPO_SLOW, TEMPO_FAST = (0.92, 0.96), (1.04, 1.08)
MAX_TILT_DB = 4.0
#: The features stop at 11 kHz, so a codec's 16 kHz cutoff is invisible to the
#: network. The low-pass fallback therefore cuts where the features can see it.
DEGRADE_LOWPASS_HZ = (6500.0, 10500.0)
DEGRADE_KBPS = (48, 64, 80, 96)


# --------------------------------------------------------------------------- #
# Signal tools
# --------------------------------------------------------------------------- #
def have_rubberband() -> bool:
    return shutil.which("rubberband") is not None


def _rubberband(y: np.ndarray, args: list) -> np.ndarray:
    import soundfile as sf
    with tempfile.TemporaryDirectory(prefix="sfaug_") as d:
        src, dst = Path(d) / "in.wav", Path(d) / "out.wav"
        sf.write(str(src), np.asarray(y, np.float32), SR, subtype="FLOAT")
        subprocess.run(["rubberband", "-q", "--fine", *args, str(src), str(dst)],
                       check=True, capture_output=True)
        out, _ = sf.read(str(dst), dtype="float32")
    return out if out.ndim == 1 else out.mean(axis=1)


def time_stretch(y: np.ndarray, rate: float, backend: str = "auto") -> tuple:
    """Play `y` `rate` times faster. Returns (audio of length round(len/rate),
    backend used): 'rubberband' keeps the pitch, 'resample' lets it follow."""
    n_out = int(round(len(y) / rate))
    if backend == "auto":
        backend = "rubberband" if have_rubberband() else "resample"
    if backend == "rubberband":
        out = _rubberband(y, ["--tempo", f"{rate:.8f}"])
    elif backend == "resample":
        fr = Fraction(1.0 / rate).limit_denominator(2000)
        out = scipy.signal.resample_poly(np.asarray(y, np.float64), fr.numerator,
                                         fr.denominator)
        # the rational approximation is off by parts per million: put the last
        # sample where it belongs so a long song does not drift
        if len(out) != n_out and len(out) > 1:
            out = np.interp(np.linspace(0, len(out) - 1, n_out), np.arange(len(out)), out)
    else:
        raise ValueError(f"unknown stretch backend '{backend}'")
    return stems_cache.fit_length(np.asarray(out, np.float32), n_out), backend


def pitch_shift(y: np.ndarray, semitones: float, backend: str = "auto") -> tuple:
    """Shift pitch, keep duration. Returns (audio, backend used); with no
    rubberband the audio comes back unchanged and the backend is 'none'."""
    if semitones == 0:
        return np.asarray(y, np.float32), "none"
    if backend == "auto":
        backend = "rubberband" if have_rubberband() else "none"
    if backend == "none":
        return np.asarray(y, np.float32), "none"
    out = _rubberband(y, ["--pitch", f"{semitones:.4f}"])
    return stems_cache.fit_length(out, len(y)), "rubberband"


def _shelf_sos(gain_db: float, f0: float, high: bool) -> np.ndarray:
    """RBJ cookbook shelving biquad (slope 1) as one second-order section."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / SR
    cw, alpha = np.cos(w0), np.sin(w0) / 2.0 * np.sqrt(2.0)
    s = 2.0 * np.sqrt(a) * alpha
    if high:
        b = (a * ((a + 1) + (a - 1) * cw + s), -2 * a * ((a - 1) + (a + 1) * cw),
             a * ((a + 1) + (a - 1) * cw - s))
        den = ((a + 1) - (a - 1) * cw + s, 2 * ((a - 1) - (a + 1) * cw),
               (a + 1) - (a - 1) * cw - s)
    else:
        b = (a * ((a + 1) - (a - 1) * cw + s), 2 * a * ((a - 1) - (a + 1) * cw),
             a * ((a + 1) - (a - 1) * cw - s))
        den = ((a + 1) + (a - 1) * cw + s, -2 * ((a - 1) + (a + 1) * cw),
               (a + 1) + (a - 1) * cw - s)
    return np.array([[b[0] / den[0], b[1] / den[0], b[2] / den[0], 1.0,
                      den[1] / den[0], den[2] / den[0]]])


def tonal_tilt(y: np.ndarray, low_db: float, high_db: float) -> np.ndarray:
    """A low shelf at 200 Hz and a high shelf at 4 kHz. Causal, so the group
    delay is far below a millisecond at these gains: transients stay put."""
    sos = np.concatenate([_shelf_sos(low_db, 200.0, False), _shelf_sos(high_db, 4000.0, True)])
    return scipy.signal.sosfilt(sos, np.asarray(y, np.float64)).astype(np.float32)


def soft_clip(y: np.ndarray, drive: float) -> np.ndarray:
    """tanh waveshaper; `drive` is the gain into it relative to the peak."""
    peak = float(np.max(np.abs(y))) or 1.0
    return (np.tanh(drive * y / peak) * (peak / np.tanh(drive))).astype(np.float32)


def lowpass(y: np.ndarray, cutoff_hz: float) -> np.ndarray:
    sos = scipy.signal.butter(6, min(cutoff_hz, 0.49 * SR), fs=SR, output="sos")
    return scipy.signal.sosfiltfilt(sos, np.asarray(y, np.float64)).astype(np.float32)


def _align(ref: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """Undo a constant delay of `y` against `ref` (a codec's padding), found
    by cross-correlation within +-max_lag samples."""
    n = min(len(ref), len(y), SR * 30)
    a, b = np.asarray(ref[:n], np.float64), np.asarray(y[:n], np.float64)
    corr = scipy.signal.correlate(b, a, mode="full", method="fft")
    mid = n - 1
    lag = int(np.argmax(corr[mid - max_lag:mid + max_lag + 1])) - max_lag
    out = y[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag, y.dtype), y])
    return stems_cache.fit_length(np.asarray(out, np.float32), len(ref))


def lossy_roundtrip(y: np.ndarray, kbps: int) -> tuple:
    """Through an MP3 encoder and back, re-aligned to the input. Returns
    (audio, 'mp3') or (None, None) when ffmpeg cannot do it."""
    import soundfile as sf
    if shutil.which("ffmpeg") is None:
        return None, None
    with tempfile.TemporaryDirectory(prefix="sfaug_") as d:
        src, mp3, dst = Path(d) / "in.wav", Path(d) / "x.mp3", Path(d) / "out.wav"
        sf.write(str(src), np.asarray(y, np.float32), SR, subtype="PCM_16")
        quiet = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        r1 = subprocess.run([*quiet, "-i", str(src), "-codec:a", "libmp3lame",
                             "-b:a", f"{kbps}k", str(mp3)], capture_output=True)
        r2 = subprocess.run([*quiet, "-i", str(mp3), "-ar", str(SR), "-ac", "1", str(dst)],
                            capture_output=True) if r1.returncode == 0 else r1
        if r2.returncode != 0 or not dst.exists():
            return None, None
        out, _ = sf.read(str(dst), dtype="float32")
    return _align(y, out, int(0.08 * SR)), "mp3"


# --------------------------------------------------------------------------- #
# Recipes
# --------------------------------------------------------------------------- #
def variant_rng(key: str, k: int, seed: int) -> np.random.Generator:
    return np.random.default_rng((seed, zlib.crc32(key.encode()), k))


def recipe(key: str, k: int, seed: int = 0) -> dict:
    """The parameters of variant `k` of song `key`: a plain dict, decided
    before any audio is touched, so a variant can be reproduced or audited."""
    if not 0 <= k < VARIANTS:
        raise ValueError(f"variant {k} is outside 0..{VARIANTS - 1}")
    rng = variant_rng(key, k, seed)
    kind = KINDS[k]
    r = {"variant": k, "kind": kind, "gains": {n: 1.0 for n in GF.STEM_NAMES},
         "pitch": 0, "rate": 1.0, "tilt_db": [0.0, 0.0], "drive": 0.0,
         "degrade": None}
    if kind == "remix":
        r["gains"] = {n: round(float(rng.uniform(*GAIN_RANGE[n])), 3) for n in GF.STEM_NAMES}
        if k == 0:
            r["gains"]["vocals"] = 0.0
    elif kind == "pitch":
        r["pitch"] = int(rng.integers(1, 4)) * (-1 if k == 2 else 1)
        r["tilt_db"] = [round(float(v), 2) for v in rng.uniform(-MAX_TILT_DB, MAX_TILT_DB, 2)]
        if rng.random() < 0.5:
            r["drive"] = round(float(rng.uniform(1.2, 2.0)), 2)
    elif kind == "tempo":
        r["rate"] = round(float(rng.uniform(*(TEMPO_SLOW if k == 4 else TEMPO_FAST))), 4)
    else:
        r["gains"] = {n: round(float(rng.uniform(0.8, 1.2)), 3) for n in GF.STEM_NAMES}
        r["pitch"] = int(rng.choice([-1, 1]))
        r["drive"] = round(float(rng.uniform(1.5, 2.5)), 2)
        r["degrade"] = {"kbps": int(rng.choice(DEGRADE_KBPS)),
                        "lowpass_hz": round(float(rng.uniform(*DEGRADE_LOWPASS_HZ)), 0)}
    return r


def render(stems: dict, rec: dict, backend: str = "auto") -> tuple:
    """Apply a recipe to four stems. Returns (mix, stems, info): the mix is
    peak-normalized the way `GF.load_audio` would deliver it and the stems
    carry the same gain, so the features see what a real file would give."""
    info: dict = {"stretch": None, "pitch": None, "codec": None}
    out = {}
    for name in GF.STEM_NAMES:
        y = np.asarray(stems[name], np.float32) * float(rec["gains"][name])
        if rec["pitch"] and name in HARMONIC and np.any(y):
            y, info["pitch"] = pitch_shift(y, rec["pitch"], backend if backend != "resample"
                                           else "none")
        if rec["rate"] != 1.0:
            y, info["stretch"] = time_stretch(y, rec["rate"], backend)
        out[name] = y
    mix = np.sum([out[n] for n in GF.STEM_NAMES], axis=0).astype(np.float32)
    if any(rec["tilt_db"]):
        mix = tonal_tilt(mix, *rec["tilt_db"])
        out = {n: tonal_tilt(y, *rec["tilt_db"]) for n, y in out.items()}
    if rec["degrade"]:
        coded, info["codec"] = lossy_roundtrip(mix, rec["degrade"]["kbps"])
        if coded is None:
            coded, info["codec"] = lowpass(mix, rec["degrade"]["lowpass_hz"]), "lowpass"
        mix = coded
        # the stems only feed the stem channels: band-limit them like the mix
        out = {n: lowpass(y, rec["degrade"]["lowpass_hz"]) for n, y in out.items()}
    if rec["drive"]:
        mix = soft_clip(mix, rec["drive"])
    peak = float(np.max(np.abs(mix)))
    if peak <= 0:
        raise ValueError("variant rendered to silence")
    return mix / peak, {n: y / peak for n, y in out.items()}, info


# --------------------------------------------------------------------------- #
# Shards
# --------------------------------------------------------------------------- #
def aug_key(parent_key: str, k: int) -> str:
    return f"{parent_key}__aug{k}"


def chart_events(arrays: dict, j: int) -> C.ChartEvents:
    """A chart's raw rows, as stored in a shard, back as ChartEvents."""
    raw, rawd = arrays[f"c{j}_raw"], arrays[f"c{j}_rawd"]
    return C.ChartEvents(arrays[f"c{j}_tick"].astype(np.int32), raw[0], raw[1], raw[2],
                         raw[3], rawd[0], rawd[1],
                         arrays[f"c{j}_mines"].astype(np.int32).reshape(-1, 2))


def build_variant(meta: dict, arrays: dict, kept: list, stems: dict, k: int,
                  song_id: str, seed: int = 0, backend: str = "auto") -> tuple:
    """Variant `k` of one real shard -> (arrays, meta) of its `aug_stem` shard.
    `kept` lists the chart indices the real manifest trains on; only those are
    carried over, renumbered from 0 (`parent_j` remembers where each came from).
    `song_id` is the parent's id in the real manifest, which the variant keeps."""
    rec = recipe(meta["key"], k, seed)
    mix, st, info = render(stems, rec, backend)
    timing = corpus.timing_from_facts(meta["timing"], rate=rec["rate"])
    n_ticks = int(meta["n_ticks"])
    ff = GF.frame_features(mix)
    core, tick_times, dt_ms = GF.warp_core(ff, timing, n_ticks)
    out = {"core": core.astype(np.float16), "dt_ms": dt_ms.astype(np.float32),
           "stems": GF.stem_features(st, timing, n_ticks).astype(np.float16),
           "fingerprint": corpus.spectral_fingerprint(ff.level)}
    rows = []
    by_j = {c["j"]: c for c in meta["charts"]}
    for new_j, j in enumerate(kept):
        ev = chart_events(arrays, j)
        times_ms = timing.time_at(ev.tick / C.TICKS_PER_BEAT) * 1000.0
        src = by_j[j]
        row = corpus.pack_chart(out, new_j, ev, times_ms, src["meter"], src["difficulty"],
                                has_stems=True, difficulty_raw=src.get("difficulty_raw"),
                                description=src.get("description", ""),
                                credit=src.get("credit", ""))
        row["parent_j"] = j
        rows.append(row)
    new = shards.base_meta(
        aug_key(meta["key"], k), song_id, "aug_stem",
        title=meta["title"], artist=meta["artist"], n_ticks=n_ticks,
        duration_s=len(mix) / SR, has_stems=True,
        parent={"source": "real", "key": meta["key"]}, variant=k, kind=rec["kind"],
        recipe=rec, backends=info, tick0_time_s=float(tick_times[0]),
        timing=corpus.timing_facts(timing))
    new["charts"] = rows
    return out, new


def _one_song(job: tuple) -> dict:
    real_dir, stems_dir, out_dir, key, song_id, kept, variants, seed, backend = job
    row = {"key": key, "written": 0, "error": None}
    try:
        meta, arrays = corpus.load_song(Path(real_dir) / "songs" / f"{key}.npz")
        if "timing" not in meta:
            raise ValueError("shard has no timing block (compile it with this version)")
        want = [k for k in variants
                if not (Path(out_dir) / "songs" / f"{aug_key(key, k)}.npz").exists()]
        if not want:
            return row
        stems = stems_cache.load_stems(stems_dir, key)
        if stems is None:
            raise ValueError("no cached stems")
        for k in want:
            a, m = build_variant(meta, arrays, kept, stems, k, song_id, seed, backend)
            corpus.save_shard(Path(out_dir) / "songs" / f"{m['key']}.npz", a, m)
            row["written"] += 1
    except Exception as e:                      # one bad song must not stop a run
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return row


def augment_corpus(real_dir, stems_dir, out_dir, variants_per_song: int = VARIANTS,
                   workers: int = 4, seed: int = 0, limit: int | None = None,
                   require_rubberband: bool = False, log=print) -> dict:
    if not have_rubberband():
        if require_rubberband:
            raise RuntimeError("rubberband is not installed (apt install rubberband-cli "
                               "/ brew install rubberband)")
        log("[aug] rubberband not found: tempo variants resample (pitch follows "
            "tempo) and pitch variants keep only their tonal tilt")
    out_dir = Path(out_dir)
    (out_dir / "songs").mkdir(parents=True, exist_ok=True)
    parents = shards.training_parents(real_dir)
    keys = sorted(parents)[:limit] if limit else sorted(parents)
    variants = list(range(min(variants_per_song, VARIANTS)))
    jobs = [(str(real_dir), str(stems_dir), str(out_dir), k, parents[k]["song_id"],
             [c["j"] for c in parents[k]["charts"] if c["keep"]], variants, seed, "auto")
            for k in keys]
    log(f"[aug] {len(jobs)} training song(s) x {len(variants)} variant(s), "
        f"{workers} worker(s)")
    t0, written, errors = time.time(), 0, []

    def consume(rows):
        nonlocal written
        for i, row in enumerate(rows, 1):
            written += row["written"]
            if row["error"]:
                errors.append((row["key"], row["error"]))
                log(f"[aug] FAILED {row['key']}: {row['error']}")
            if i % 10 == 0 or i == len(jobs):
                rate = i / max(1e-9, time.time() - t0)
                log(f"[aug] {i}/{len(jobs)} songs, {written} shards  "
                    f"eta {(len(jobs) - i) / max(rate, 1e-9) / 60:.0f} min")

    if workers <= 1:
        consume(map(_one_song, jobs))
    else:
        import multiprocessing as mp
        with mp.Pool(workers, maxtasksperchild=20) as pool:
            consume(pool.imap_unordered(_one_song, jobs))
    manifest = shards.write_source_manifest(out_dir, "aug_stem", log)
    (out_dir / "errors.json").write_text(json.dumps(errors))
    return {"songs": len(jobs), "shards": manifest["n_songs"], "errors": len(errors)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--stems-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variants-per-song", type=int, default=VARIANTS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--require-rubberband", action="store_true")
    a = ap.parse_args(argv)
    res = augment_corpus(a.real_dir, a.stems_dir, a.out_dir, a.variants_per_song,
                         a.workers, a.seed, a.limit, a.require_rubberband)
    print(f"[aug] done: {res}")


if __name__ == "__main__":
    main()
