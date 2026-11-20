"""corpus.py — compile StepMania / ITGmania song folders into training shards
(spec §7, REQ-SFN-06).

    python -m beatforge.adapters.stepmania.neural.train.corpus compile \
        --packs-dir /path/to/Songs --out-dir build/neural_corpus --workers 4
    python -m beatforge.adapters.stepmania.neural.train.corpus manifest \
        --out-dir build/neural_corpus

`compile` is the expensive pass (it decodes audio) and is restartable: one
`songs/<key>.npz` per simfile plus a row in `index.jsonl`, and folders already
in the index are skipped. It applies only the filters that can never be undone
by a threshold change — no dance-single chart, no decodable audio, a duration
outside 30-600 s, negative BPMs or warps. Everything that is a judgement
(sync quality, stops, BPM range, how awkward the chart is to stand on) is
MEASURED here and stored, and decided later by `manifest`, so tightening a
filter never means decoding 14,000 songs again. For the same reason each shard
keeps the raw rows of every chart next to its tokens: the codec's foot model
can change without touching the audio.

`manifest` groups songs that share audio (same file hash, same normalized
title + artist, or a near-identical spectral fingerprint) under one `song_id`
and splits by that id, so the Easy and Challenge charts of a song — or the same
song repackaged in two packs — can never straddle train and validation.

`compile --stems-dir <cache>` reads a 4-stem cache written by
`neural.synth.stems_cache` and stores the 16 stem channels in each shard;
songs the cache does not hold compile without them. `manifest --split-manifest
<file>` takes every song's id, split and per-chart verdict from an earlier
manifest instead of deciding them again — that is how a recompiled corpus
keeps a locked train / validation split.

No PyTorch here: this runs on the machine that holds the packs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

from .. import codec as C
from .. import conditioning as cond
from .. import grid_features as GF

CORPUS_VERSION = 1
AUDIO_EXTS = (".ogg", ".mp3", ".wav", ".flac", ".opus")
MIN_DURATION_S, MAX_DURATION_S = 30.0, 600.0
MAX_TICKS = 48000
SNAP_TOL_MS = 15.0
DIFFICULTY_SLOTS = ("beginner", "easy", "medium", "hard", "challenge", "edit")
SPLITS = (("train", 0.85), ("val", 0.075), ("test", 0.075))

# manifest-time quality gate (spec §7 "Deterministic Quality Gate")
MAX_STOPS = 8
BPM_RANGE = (50.0, 320.0)
MIN_NOTES = 25
MAX_SUSTAINED_NPS = 22.0
MAX_HOLD_SHARE = 0.40
MAX_HAND_SHARE = 0.02
MAX_OFFGRID_SHARE = 0.02
#: Notes-on-transients contrast (grid_features.estimate_sync). Measured on a
#: trial of real packs: loose-but-usable charts sit near 1.35, tight ones 1.7-3.
MIN_SYNC_CONTRAST = 1.25
#: Mean foot-model discomfort per transition that a clean chart of each slot
#: costs; a chart is rejected above DISCOMFORT_FACTOR times it. Set from the
#: corpus itself (19,806 charts): 3x these sits between the 97th and 99th
#: percentile of each slot, i.e. it removes the tail of charts no foot model
#: can make sense of (keyboard dumps filed as pad charts) and nothing else.
DISCOMFORT_CEILING = {"beginner": 1.0, "easy": 1.3, "medium": 1.5, "hard": 2.0,
                      "challenge": 3.0, "edit": 3.0}
DISCOMFORT_FACTOR = 3.0
MAX_LOCK_VIOLATION_SHARE = 0.03


# --------------------------------------------------------------------------- #
# Simfile reading
# --------------------------------------------------------------------------- #
def _find_simfile(folder: Path) -> Path | None:
    files = sorted(p for p in folder.iterdir() if p.is_file())
    for ext in (".ssc", ".sm"):
        hits = [p for p in files if p.suffix.lower() == ext]
        if hits:
            return hits[0]
    return None


def _find_audio(folder: Path, music_tag: str | None) -> Path | None:
    if music_tag:
        p = folder / music_tag.strip()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            return p
        want = music_tag.strip().lower()
        for q in folder.iterdir():                 # case-insensitive match
            if q.is_file() and q.name.lower() == want:
                return q
    hits = sorted(q for q in folder.iterdir()
                  if q.is_file() and q.suffix.lower() in AUDIO_EXTS)
    return max(hits, key=lambda q: q.stat().st_size) if hits else None


def _timing_of(sf, chart=None):
    """(TimingMap, facts) from a simfile's timing tags, or raise ValueError
    naming the gimmick that makes the song unusable."""
    from simfile.timing import TimingData
    td = TimingData(sf, chart)
    bpms = [(float(b.beat), float(b.value)) for b in td.bpms]
    if not bpms:
        raise ValueError("no_bpms")
    if any(v <= 0 for _, v in bpms):
        raise ValueError("negative_bpm")
    if len(getattr(td, "warps", []) or []):
        raise ValueError("warps")
    stops = [(float(s.beat), float(s.value)) for s in td.stops]
    delays = [(float(s.beat), float(s.value)) for s in (getattr(td, "delays", []) or [])]
    if any(v < 0 for _, v in stops + delays):
        raise ValueError("negative_stop")
    timing = GF.TimingMap(-float(td.offset), bpms, stops, delays)
    vals = [v for _, v in bpms]
    facts = {"offset": float(td.offset), "bpm_min": min(vals), "bpm_max": max(vals),
             "n_bpm_changes": len(bpms) - 1, "n_stops": len(stops) + len(delays),
             "bpms": bpms[:64], "stops": (stops + delays)[:64]}
    return timing, facts


def _timesig_ok(sf) -> bool:
    ts = (sf.get("TIMESIGNATURES") or "").strip()
    if not ts:
        return True
    for part in ts.split(","):
        bits = part.strip().split("=")
        if len(bits) == 3 and (bits[1].strip(), bits[2].strip()) != ("4", "4"):
            return False
    return True


def chart_notes_to_ticks(chart, timing: GF.TimingMap):
    """A chart's notes on the 48-tick grid: (triples, offgrid_dropped, snapped)."""
    return C.simfile_chart_to_ticks(chart, timing.bpm_at, SNAP_TOL_MS)


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", s)       # (remix), [full ver.], ...
    return re.sub(r"[^0-9a-zÀ-￿]+", "", s)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spectral_fingerprint(level: np.ndarray, segments: int = 16) -> np.ndarray:
    """A coarse, encoder-tolerant fingerprint: mean z-scored mel per band over
    `segments` equal slices of the track -> (segments * 20,) float16. Two
    encodes of one master land almost on top of each other; a different song
    of the same length does not."""
    mel = level[:, :20]
    edges = np.linspace(0, len(mel), segments + 1).astype(int)
    fp = np.stack([mel[a:max(b, a + 1)].mean(0) for a, b in zip(edges[:-1], edges[1:])])
    fp = fp - fp.mean()
    return (fp / (np.linalg.norm(fp) + 1e-9)).astype(np.float16).ravel()


# --------------------------------------------------------------------------- #
# One song
# --------------------------------------------------------------------------- #
def song_key(folder: Path) -> str:
    return hashlib.sha1(folder.name.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def process_song(folder, out_dir, with_demucs: bool = False, stems_dir=None) -> dict:
    """Compile one song folder. Returns the index row; never raises."""
    folder, out_dir = Path(folder), Path(out_dir)
    row = {"folder": folder.name, "key": song_key(folder), "ok": False}
    t0 = time.time()
    try:
        return _process_song(folder, out_dir, row, with_demucs, stems_dir)
    except Exception as e:                       # one bad folder must not stop a run
        row["reason"] = f"error:{type(e).__name__}:{str(e)[:160]}"
        return row
    finally:
        row["seconds"] = round(time.time() - t0, 2)


def timing_facts(timing: GF.TimingMap) -> dict:
    """A TimingMap as plain JSON, complete (the `bpms` / `stops` rows of the
    index are capped at 64 entries; this is not), so a shard can be put back on
    its exact grid — or on that grid at another tempo — without the simfile."""
    sb, sc = timing._stops
    db, dc = timing._delays
    return {"offset_s": float(timing.offset_s),
            "bpms": [[float(b), float(v)] for b, v in zip(timing.bpm_beats, timing.bpm_vals)],
            "stops": [[float(b), float(v)] for b, v in zip(sb, np.diff(sc))],
            "delays": [[float(b), float(v)] for b, v in zip(db, np.diff(dc))]}


def timing_from_facts(facts: dict, rate: float = 1.0) -> GF.TimingMap:
    """Inverse of `timing_facts`. `rate` plays the song `rate` times faster:
    every tempo is multiplied by it and every duration divided, so tick k is
    the same musical position in both."""
    return GF.TimingMap(facts["offset_s"] / rate,
                        [(b, v * rate) for b, v in facts["bpms"]],
                        [(b, v / rate) for b, v in facts["stops"]],
                        [(b, v / rate) for b, v in facts["delays"]])


def pack_chart(arrays: dict, j: int, ev: C.ChartEvents, times_ms: np.ndarray,
               meter: int, slot: str, *, has_stems: bool = False,
               difficulty_raw: str | None = None, description: str = "",
               credit: str = "", st: C.IngestStats | None = None,
               offgrid: int = 0, snapped: int = 0) -> dict:
    """Encode one chart into `arrays` (the `c<j>_*` entries of a shard) and
    return its row for the shard's meta. Shared by the compiler and by every
    generator of synthetic or derived charts, so they cannot drift apart."""
    times_ms = np.asarray(times_ms, np.float64)
    enc, info = C.encode(ev, times_ms)
    style = cond.compute_style_vector(enc, times_ms / 1000.0, meter, has_stems=has_stems)
    arrays.update({
        f"c{j}_tick": ev.tick.astype(np.int32),
        f"c{j}_tok": enc.token, f"c{j}_flav": enc.flavor,
        f"c{j}_da": enc.dur_alt.astype(np.int32),
        f"c{j}_ds": enc.dur_same.astype(np.int32),
        f"c{j}_tms": times_ms.astype(np.float32),
        f"c{j}_mines": ev.mines.astype(np.int32).reshape(-1, 2),
        # raw rows, so a codec change never needs the audio again
        f"c{j}_raw": np.stack([ev.p0, ev.p1, ev.k0, ev.k1]).astype(np.int8),
        f"c{j}_rawd": np.stack([ev.d0, ev.d1]).astype(np.int32),
        f"c{j}_style": style,
    })
    n = len(ev)
    st = st or C.IngestStats(notes=n, rows=n)
    return {
        "j": j, "difficulty": slot if slot in DIFFICULTY_SLOTS else "edit",
        "difficulty_raw": difficulty_raw if difficulty_raw is not None else slot,
        "meter": int(meter), "description": (description or "")[:80],
        "credit": (credit or "")[:80],
        "f_init": int(enc.f_init), "n_events": n,
        "n_jumps": int((enc.token >= 8).sum()),
        "n_holds": int((enc.flavor > 0).sum()), "n_mines": int(len(ev.mines)),
        "sustained_nps": round(cond.sustained_nps(times_ms / 1000.0), 3),
        "peak_nps": round(cond.peak_nps(times_ms / 1000.0), 3),
        "hold_share": round(float((enc.flavor > 0).mean()), 4) if n else 0.0,
        "hand_share": round(st.hand_share, 4), "hands": st.hands,
        "offgrid_dropped": offgrid, "snapped": snapped,
        "ingest_dropped": st.dropped, "open_holds": st.open_holds,
        "mean_cost": round(info.mean_cost, 3),
        "lock_violations": info.lock_violations, "spins": info.spins,
        "last_tick": int(ev.tick[-1]) if n else 0,
    }


def save_shard(path, arrays: dict, meta: dict) -> None:
    """Write one shard atomically: arrays + the JSON meta as a byte array."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(arrays)
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _process_song(folder: Path, out_dir: Path, row: dict, with_demucs: bool,
                  stems_dir=None) -> dict:
    import simfile
    sm_path = _find_simfile(folder)
    if sm_path is None:
        row["reason"] = "no_simfile"
        return row
    sf = simfile.open(str(sm_path), strict=False)
    charts = [c for c in sf.charts
              if (c.stepstype or "").strip().lower() == "dance-single"]
    if not charts:
        row["reason"] = "no_dance_single"
        return row
    try:
        timing, facts = _timing_of(sf)
    except ValueError as e:
        row["reason"] = f"gimmick:{e}"
        return row
    audio_path = _find_audio(folder, sf.music)
    if audio_path is None:
        row["reason"] = "no_audio"
        return row
    try:
        y = GF.load_audio(str(audio_path))
    except Exception as e:
        row["reason"] = f"audio_decode:{type(e).__name__}"
        return row
    duration = len(y) / GF.SR
    if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
        row["reason"] = "duration"
        row["duration_s"] = round(duration, 1)
        return row

    # ---- charts onto the grid ------------------------------------------- #
    parsed = []
    for chart in charts:
        try:
            _, cfacts = _timing_of(sf, chart)
        except ValueError:
            continue
        if (cfacts["offset"], cfacts["bpms"], cfacts["stops"]) != (
                facts["offset"], facts["bpms"], facts["stops"]):
            continue                              # split timing: not this grid
        triples, offgrid, snapped = chart_notes_to_ticks(chart, timing)
        ev, st = C.events_from_notes(triples)
        if len(ev) == 0:
            continue
        parsed.append((chart, ev, st, offgrid, snapped))
    if not parsed:
        row["reason"] = "no_usable_chart"
        return row

    # ---- audio features, then seat the grid on the transients ------------ #
    ff = GF.frame_features(y)
    all_ticks = np.unique(np.concatenate([ev.tick for _, ev, *_ in parsed]))
    note_times = timing.time_at(all_ticks / C.TICKS_PER_BEAT)
    sync = GF.estimate_sync(ff.onset_envelope, note_times)
    shift = sync["shift_s"] if sync["score"] >= 1.03 * sync["score_at_zero"] else 0.0
    # notes sound `shift` later than the simfile says -> move the grid with them
    timing = timing.shifted(shift)
    last_tick = int(all_ticks.max())
    n_ticks = max(timing.ticks_for_duration(duration),
                  int(np.ceil((last_tick + 1) / C.G)) * C.G)
    if n_ticks > MAX_TICKS:
        row["reason"] = "too_many_ticks"
        return row
    core, tick_times, dt_ms = GF.warp_core(ff, timing, n_ticks)

    arrays = {"core": core.astype(np.float16), "dt_ms": dt_ms.astype(np.float32),
              "fingerprint": spectral_fingerprint(ff.level)}
    stems = None
    if stems_dir is not None:
        from ..synth import stems_cache
        # the cache holds stems of the DECODED file; the mix was peak-normalized
        # on load, so the stems take the same gain and still sum to it
        stems = stems_cache.load_stems(stems_dir, row["key"], n_samples=len(y),
                                       match_mix=y)
    elif with_demucs:
        stems = separate_stems(str(audio_path))
    has_stems = stems is not None
    if has_stems:
        arrays["stems"] = GF.stem_features(stems, timing, n_ticks).astype(np.float16)

    chart_meta = []
    for j, (chart, ev, st, offgrid, snapped) in enumerate(parsed):
        times_ms = timing.time_at(ev.tick / C.TICKS_PER_BEAT) * 1000.0
        try:
            meter = int(float(chart.meter))
        except (TypeError, ValueError):
            meter = 0
        chart_meta.append(pack_chart(
            arrays, j, ev, times_ms, meter, (chart.difficulty or "").strip().lower(),
            has_stems=has_stems, difficulty_raw=chart.difficulty,
            description=getattr(chart, "description", "") or "",
            credit=getattr(chart, "credit", "") or "", st=st, offgrid=offgrid,
            snapped=snapped))

    title = sf.titletranslit or sf.title or folder.name
    artist = sf.artisttranslit or sf.artist or ""
    meta = {
        "corpus_version": CORPUS_VERSION, "codec_version": C.CODEC_VERSION,
        "key": row["key"], "folder": folder.name, "simfile": sm_path.name,
        "audio": audio_path.name, "audio_sha256": file_sha256(audio_path),
        "title": title, "artist": artist, "credit": sf.credit or "",
        "name_key": normalize_name(title) + "|" + normalize_name(artist),
        "duration_s": round(duration, 2), "n_ticks": int(n_ticks),
        "tick0_time_s": float(tick_times[0]),
        "sync": {k: round(v, 4) for k, v in sync.items()},
        "sync_shift_applied_s": round(shift, 4),
        "timesig_44": _timesig_ok(sf), "has_stems": has_stems,
        "source_type": "real", "timing": timing_facts(timing),
        **{k: facts[k] for k in ("offset", "bpm_min", "bpm_max", "n_bpm_changes",
                                 "n_stops")},
        "charts": chart_meta,
    }
    save_shard(out_dir / "songs" / f"{row['key']}.npz", arrays, meta)
    row.update(ok=True, n_charts=len(chart_meta), n_ticks=int(n_ticks),
               has_stems=has_stems,
               duration_s=round(duration, 1), sync=round(sync["contrast"], 2),
               shift_ms=round(shift * 1000))
    return row


def separate_stems(audio_path: str) -> dict:
    """4-stem Demucs separation -> mono arrays at GF.SR (training-only path)."""
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.audio import AudioFile
        from demucs.pretrained import get_model
    except ImportError as e:                     # pragma: no cover - optional dep
        raise RuntimeError("--with-demucs needs `pip install demucs torch`") from e
    import scipy.signal
    model = _demucs_model(get_model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wav = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate,
                                     channels=model.audio_channels)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, wav[None].to(dev), device=dev, split=True,
                          overlap=0.1, progress=False)[0].cpu().numpy()
    out = out * float(ref.std() + 1e-8) + float(ref.mean())   # back to the file's level
    stems = {}
    g = int(np.gcd(model.samplerate, GF.SR))
    for name, src in zip(model.sources, out):
        mono = src.mean(0)
        stems[name] = scipy.signal.resample_poly(
            mono, GF.SR // g, model.samplerate // g).astype(np.float32)
    return stems


_DEMUCS = None


def _demucs_model(get_model):
    global _DEMUCS
    if _DEMUCS is None:
        _DEMUCS = get_model("htdemucs").eval()
    return _DEMUCS


def load_song(path) -> tuple:
    """songs/<key>.npz -> (meta dict, arrays dict)."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
    return meta, arrays


# --------------------------------------------------------------------------- #
# compile
# --------------------------------------------------------------------------- #
def iter_song_folders(packs_dir: Path):
    """Every directory under `packs_dir` that directly holds a .sm/.ssc —
    handles both Songs/<pack>/<song>/ and a flat directory of song folders."""
    for root, dirs, files in os.walk(packs_dir):
        dirs.sort()
        if any(f.lower().endswith((".sm", ".ssc")) for f in files):
            dirs[:] = []                          # a song folder: do not descend
            yield Path(root)


def _work(args):
    return process_song(*args)


def compile_corpus(packs_dir, out_dir, workers: int = 4, with_demucs: bool = False,
                   limit: int | None = None, log=print, stems_dir=None) -> dict:
    packs_dir, out_dir = Path(packs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.jsonl"
    done = set()
    if index.exists():
        with open(index, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["folder"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [p for p in iter_song_folders(packs_dir) if p.name not in done]
    if limit:
        todo = todo[:limit]
    log(f"[corpus] {len(done)} already indexed, {len(todo)} to do, "
        f"{workers} worker(s)")
    stats = {"ok": 0, "rejected": 0}
    t0 = time.time()
    jobs = [(str(p), str(out_dir), with_demucs, stems_dir) for p in todo]

    def consume(rows):
        with open(index, "a", encoding="utf-8") as f:
            for i, row in enumerate(rows, 1):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                stats["ok" if row["ok"] else "rejected"] += 1
                if i % 50 == 0 or i == len(jobs):
                    rate = i / max(1e-9, time.time() - t0)
                    log(f"[corpus] {i}/{len(jobs)}  ok={stats['ok']} "
                        f"rejected={stats['rejected']}  {rate:.2f} songs/s  "
                        f"eta {(len(jobs) - i) / max(rate, 1e-9) / 60:.0f} min")

    if workers <= 1 or with_demucs:
        consume(map(_work, jobs))
    else:
        import multiprocessing as mp
        with mp.Pool(workers, maxtasksperchild=200) as pool:
            consume(pool.imap_unordered(_work, jobs, chunksize=2))
    return stats


# --------------------------------------------------------------------------- #
# reencode: re-run the codec on stored raw rows (no audio needed)
# --------------------------------------------------------------------------- #
def reencode_song(path) -> int:
    """Re-annotate every chart of one shard from its raw rows with the current
    codec, refreshing tokens, style vectors and the foot-model statistics the
    quality gate reads. Returns the number of charts rewritten."""
    path = Path(path)
    meta, arrays = load_song(path)
    for c in meta["charts"]:
        j = c["j"]
        raw, rawd = arrays[f"c{j}_raw"], arrays[f"c{j}_rawd"]
        ev = C.ChartEvents(arrays[f"c{j}_tick"], raw[0], raw[1], raw[2], raw[3],
                           rawd[0], rawd[1], arrays[f"c{j}_mines"])
        tms = arrays[f"c{j}_tms"].astype(np.float64)
        enc, info = C.encode(ev, tms)
        arrays[f"c{j}_tok"], arrays[f"c{j}_flav"] = enc.token, enc.flavor
        arrays[f"c{j}_da"] = enc.dur_alt.astype(np.int32)
        arrays[f"c{j}_ds"] = enc.dur_same.astype(np.int32)
        arrays[f"c{j}_style"] = cond.compute_style_vector(
            enc, tms / 1000.0, c["meter"], has_stems=meta.get("has_stems", False))
        c.update(f_init=int(enc.f_init), mean_cost=round(info.mean_cost, 3),
                 lock_violations=info.lock_violations, spins=info.spins,
                 n_jumps=int((enc.token >= 8).sum()))
    meta["codec_version"] = C.CODEC_VERSION
    meta["codec_costs"] = codec_fingerprint()
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return len(meta["charts"])


def codec_fingerprint() -> str:
    """Hash of the foot model's cost table: two shards annotated under
    different cost numbers have different conventions even at one codec version."""
    return hashlib.sha1(C.tables()["cost"].tobytes()).hexdigest()[:12]


def reencode_corpus(out_dir, workers: int = 4, log=print) -> int:
    files = sorted(f for f in (Path(out_dir) / "songs").glob("*.npz")
                   if not f.name.endswith(".tmp.npz"))
    log(f"[reencode] {len(files)} shards, {workers} worker(s), "
        f"cost table {codec_fingerprint()}")
    if workers <= 1:
        n = sum(reencode_song(f) for f in files)
    else:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            n = sum(pool.imap_unordered(reencode_song, files, chunksize=16))
    log(f"[reencode] {n} charts re-annotated")
    return n


# --------------------------------------------------------------------------- #
# manifest: quality gate + leak-free splits
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def chart_verdict(song: dict, chart: dict) -> str | None:
    """None if the chart may be trained on, else the reason it may not."""
    n = chart["n_events"]
    if n < MIN_NOTES:
        return "too_few_notes"
    if chart["sustained_nps"] > MAX_SUSTAINED_NPS:
        return "nps"
    if chart["hold_share"] > MAX_HOLD_SHARE:
        return "hold_share"
    if chart["hand_share"] > MAX_HAND_SHARE:
        return "hands"
    if chart["offgrid_dropped"] > MAX_OFFGRID_SHARE * (n + chart["offgrid_dropped"]):
        return "off_grid"
    if chart["mean_cost"] > DISCOMFORT_FACTOR * DISCOMFORT_CEILING[chart["difficulty"]]:
        return "discomfort"
    if chart["lock_violations"] > MAX_LOCK_VIOLATION_SHARE * n:
        return "lock_violations"
    if not 1 <= chart["meter"] <= 25:
        return "meter"
    # a rating that contradicts the chart: "1" on a 7 notes/s chart is a
    # placeholder someone never filled in, and the slot label is as suspect
    nps = chart["sustained_nps"]
    if nps > chart["meter"] + 3.5 or (chart["meter"] >= 8 and nps < 1.5):
        return "meter_density"
    return None


def song_verdict(song: dict) -> str | None:
    if song["n_stops"] > MAX_STOPS:
        return "stops"
    if song["bpm_min"] < BPM_RANGE[0] or song["bpm_max"] > BPM_RANGE[1]:
        return "bpm_range"
    if not song["timesig_44"]:
        return "time_signature"
    if song["sync"]["contrast"] < MIN_SYNC_CONTRAST:
        return "sync"
    return None


def assign_song_ids(metas: list, fingerprints: np.ndarray | None,
                    fp_threshold: float = 0.97, dur_tol_s: float = 1.5) -> list:
    """Group songs that share audio; returns one `song_id` per input.

    Three signals, any of which merges: identical audio bytes, identical
    normalized title + artist, or near-identical spectral fingerprint at the
    same duration (the same master re-encoded for another pack)."""
    n = len(metas)
    uf = _UnionFind(n)
    for field in ("audio_sha256", "name_key"):
        seen: dict = {}
        for i, m in enumerate(metas):
            k = m.get(field)
            if not k or (field == "name_key" and k.startswith("|")):
                continue
            if k in seen:
                uf.union(seen[k], i)
            else:
                seen[k] = i
    if fingerprints is not None and n > 1:
        fp = np.asarray(fingerprints, np.float32)
        dur = np.array([m["duration_s"] for m in metas])
        order = np.argsort(dur)
        sd = dur[order]
        hi = np.searchsorted(sd, sd + dur_tol_s, side="right")
        for a in range(n):
            if hi[a] - a <= 1:
                continue
            sims = fp[order[a + 1:hi[a]]] @ fp[order[a]]
            for b in np.nonzero(sims >= fp_threshold)[0]:
                uf.union(int(order[a]), int(order[a + 1 + b]))
    roots = [uf.find(i) for i in range(n)]
    ids = {}
    out = []
    for i, r in enumerate(roots):
        if r not in ids:
            ids[r] = "s" + hashlib.sha1(metas[r]["audio_sha256"].encode()).hexdigest()[:12]
        out.append(ids[r])
    return out


def split_of(song_id: str) -> str:
    """Deterministic split from the song id alone (85 / 7.5 / 7.5)."""
    u = int(hashlib.sha1(("split:" + song_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, share in SPLITS:
        acc += share
        if u < acc:
            return name
    return SPLITS[-1][0]


def apply_locked_split(manifest: dict, locked: dict, allow_missing: bool = False) -> dict:
    """Overwrite every song's id, split and chart verdicts with those of
    `locked` (an earlier manifest over the same song folders). A song the
    locked manifest does not list is dropped; a chart that no longer matches
    the one the verdict was given to is an error, not a guess. So is a song
    the locked manifest uses that was not compiled, unless `allow_missing`
    (a deliberately partial corpus, e.g. a test fixture)."""
    by_key = {s["key"]: s for s in locked["songs"]}
    for s in manifest["songs"]:
        ref = by_key.get(s["key"])
        if ref is None:
            s["reject"] = s["reject"] or "not_in_locked_split"
            for c in s["charts"]:
                c["keep"], c["reject"] = False, "not_in_locked_split"
            continue
        s["song_id"], s["split"], s["reject"] = ref["song_id"], ref["split"], ref["reject"]
        want = {c["j"]: c for c in ref["charts"]}
        for c in s["charts"]:
            r = want.get(c["j"])
            if r is None or (r["difficulty"], r["n_events"]) != (c["difficulty"],
                                                                 c["n_events"]):
                raise ValueError(
                    f"locked split: chart {c['j']} of '{s['title']}' ({s['key']}) "
                    f"no longer matches the manifest it was locked in")
            c["keep"], c["reject"] = r["keep"], r["reject"]
    missing = set(by_key) - {s["key"] for s in manifest["songs"]}
    used = [k for k in missing if any(c["keep"] for c in by_key[k]["charts"])]
    if used and not allow_missing:
        raise ValueError(f"locked split: {len(used)} song(s) it trains or validates "
                         f"on were not compiled, e.g. {sorted(used)[:3]}")
    reasons: dict = {}
    by_split = {name: {"songs": 0, "charts": 0} for name, _ in SPLITS}
    kept = 0
    for s in manifest["songs"]:
        k = sum(c["keep"] for c in s["charts"])
        kept += k
        for c in s["charts"]:
            if c["reject"]:
                reasons[c["reject"]] = reasons.get(c["reject"], 0) + 1
        if k:
            by_split[s["split"]]["songs"] += 1
            by_split[s["split"]]["charts"] += k
    manifest.update(n_charts_kept=kept, splits=by_split,
                    n_song_ids=len({s["song_id"] for s in manifest["songs"]}),
                    rejections=dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
                    locked_split=True)
    return manifest


def build_manifest(out_dir, log=print, split_manifest=None,
                   allow_missing: bool = False) -> dict:
    out_dir = Path(out_dir)
    files = sorted((out_dir / "songs").glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    metas, fps = [], []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            metas.append(json.loads(bytes(z["meta"]).decode("utf-8")))
            fps.append(z["fingerprint"].astype(np.float32))
    ids = assign_song_ids(metas, np.stack(fps) if fps else None)
    songs, reasons = [], {}
    n_charts = n_kept = 0
    for m, sid in zip(metas, ids):
        sv = song_verdict(m)
        charts = []
        for c in m["charts"]:
            n_charts += 1
            v = sv or chart_verdict(m, c)
            if v:
                reasons[v] = reasons.get(v, 0) + 1
            else:
                n_kept += 1
            charts.append({"j": c["j"], "difficulty": c["difficulty"],
                           "meter": c["meter"], "n_events": c["n_events"],
                           "keep": v is None, "reject": v})
        songs.append({"key": m["key"], "song_id": sid, "split": split_of(sid),
                      "title": m["title"], "artist": m["artist"],
                      "n_ticks": m["n_ticks"], "duration_s": m["duration_s"],
                      "reject": sv, "charts": charts})
    by_split = {s: {"songs": 0, "charts": 0} for s, _ in SPLITS}
    for s in songs:
        k = sum(c["keep"] for c in s["charts"])
        if k:
            by_split[s["split"]]["songs"] += 1
            by_split[s["split"]]["charts"] += k
    manifest = {"corpus_version": CORPUS_VERSION, "codec_version": C.CODEC_VERSION,
                "n_songs": len(songs), "n_song_ids": len(set(ids)),
                "n_charts": n_charts, "n_charts_kept": n_kept,
                "rejections": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
                "splits": by_split, "source_type": "real", "songs": songs}
    if split_manifest:
        manifest = apply_locked_split(manifest, json.loads(Path(split_manifest).read_text()),
                                      allow_missing)
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    log(f"[manifest] {len(songs)} songs -> {manifest['n_song_ids']} song ids; "
        f"{manifest['n_charts_kept']}/{n_charts} charts kept; "
        f"splits {manifest['splits']}; rejections {manifest['rejections']}"
        + ("; split locked" if split_manifest else ""))
    return manifest


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog="neural.train.corpus", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    cp = sub.add_parser("compile", help="decode + encode song folders into shards")
    cp.add_argument("--packs-dir", required=True)
    cp.add_argument("--out-dir", default="build/neural_corpus")
    cp.add_argument("--workers", type=int, default=4)
    cp.add_argument("--with-demucs", action="store_true")
    cp.add_argument("--stems-dir", help="4-stem cache from neural.synth.stems_cache")
    cp.add_argument("--split-manifest",
                    help="(no-subcommand form) lock ids, splits and verdicts to this manifest")
    cp.add_argument("--split-allow-missing", action="store_true",
                    help="accept a corpus that holds only part of the locked manifest")
    cp.add_argument("--limit", type=int)
    mp_ = sub.add_parser("manifest", help="quality gate + song-level splits")
    mp_.add_argument("--out-dir", default="build/neural_corpus")
    mp_.add_argument("--split-manifest",
                     help="take ids, splits and chart verdicts from this manifest")
    mp_.add_argument("--split-allow-missing", action="store_true")
    rp = sub.add_parser("reencode", help="re-run the codec on stored raw rows")
    rp.add_argument("--out-dir", default="build/neural_corpus")
    rp.add_argument("--workers", type=int, default=4)
    # Spec §7 invokes this module with no subcommand; treat that as `compile`
    # followed by `manifest`.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("compile", "manifest", "reencode", "-h", "--help"):
        a = cp.parse_args(argv)
        compile_corpus(a.packs_dir, a.out_dir, a.workers, a.with_demucs, a.limit,
                       stems_dir=a.stems_dir)
        build_manifest(a.out_dir, split_manifest=a.split_manifest,
                       allow_missing=a.split_allow_missing)
        return
    a = ap.parse_args(argv)
    if a.cmd == "compile":
        compile_corpus(a.packs_dir, a.out_dir, a.workers, a.with_demucs, a.limit,
                       stems_dir=a.stems_dir)
    elif a.cmd == "manifest":
        build_manifest(a.out_dir, split_manifest=a.split_manifest,
                       allow_missing=a.split_allow_missing)
    elif a.cmd == "reencode":
        reencode_corpus(a.out_dir, a.workers)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
