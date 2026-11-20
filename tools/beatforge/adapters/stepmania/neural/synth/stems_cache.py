"""stems_cache.py — separate each song into drums / bass / vocals / other once,
and keep the result where the corpus compiler and the augmenter can read it.

    python -m beatforge.adapters.stepmania.neural.synth.stems_cache \
        --packs-dir /path/to/Songs --out-dir build/stems_cache \
        [--only-manifest manifest.json] [--model htdemucs]

Layout: `<out>/<key>/{drums,bass,vocals,other}.wav` (mono float32 at the
feature sample rate) and `<out>/<key>/song.json`; `<out>/manifest.json` lists
every song with its sample count and reconstruction SNR. `<key>` is the
corpus's song key (`corpus.song_key`), so a shard finds its stems by name.

The stems are stored in the level of the mix AS THE FEATURE CODE LOADS IT
(`grid_features.load_audio`: mono, resampled, peak-normalized). They therefore
sum to that signal as closely as the separator allows, with their relative
levels intact, and exactly as many samples.

Needs `demucs` (requirements-synth.txt). Restartable: a song whose `song.json`
exists is skipped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.signal

from .. import grid_features as GF

STEM_NAMES = GF.STEM_NAMES
#: Largest length difference (samples at GF.SR) accepted between a cached stem
#: and the mix it belongs to before it is padded / trimmed: 1 ms.
MAX_LENGTH_DRIFT = int(round(0.001 * GF.SR))


def pick_device(name: str = "auto") -> str:
    if name and name != "auto":
        return name
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float32)
    g = int(np.gcd(int(sr_in), int(sr_out)))
    return scipy.signal.resample_poly(y, sr_out // g, sr_in // g, axis=-1).astype(np.float32)


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Pad with zeros or trim so that len(y) == n."""
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - len(y), y.dtype)])


def reconstruction_snr(mix: np.ndarray, stems: dict) -> float:
    """10 log10(|mix|^2 / |mix - sum(stems)|^2) in dB."""
    total = np.sum([stems[n] for n in STEM_NAMES], axis=0)
    err = float(np.sum((mix - total) ** 2))
    return float(10.0 * np.log10(float(np.sum(mix ** 2)) / max(err, 1e-20)))


_MODELS: dict = {}


def _model(name: str):
    if name not in _MODELS:
        try:
            from demucs.pretrained import get_model
        except ImportError as e:               # pragma: no cover - optional dep
            raise RuntimeError("the stem cache needs demucs: "
                               "pip install -r requirements-synth.txt") from e
        _MODELS[name] = get_model(name).eval()
    return _MODELS[name]


def separate(audio_path, model: str = "htdemucs", device: str = "auto") -> tuple:
    """One audio file -> (mix, {stem: mono float32 at GF.SR}), both in the
    level of `GF.load_audio(audio_path)`, all the same length as it."""
    import soundfile as sf
    import torch
    from demucs.apply import apply_model

    net = _model(model)
    dev = pick_device(device)
    raw, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    mono = _resample(raw.mean(axis=1), sr, GF.SR)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak <= 0:
        raise ValueError("silent audio")
    wav = raw.T                                              # (channels, samples)
    if wav.shape[0] == 1:
        wav = np.repeat(wav, net.audio_channels, axis=0)
    wav = _resample(wav[:net.audio_channels], sr, net.samplerate)
    x = torch.from_numpy(np.ascontiguousarray(wav))
    ref = x.mean(0)
    mean, std = float(ref.mean()), float(ref.std()) + 1e-8
    with torch.no_grad():
        out = apply_model(net, ((x - mean) / std)[None].to(dev), device=dev, split=True,
                          overlap=0.25, progress=False)[0]
    out = (out * std + mean).cpu().numpy()                   # back to the file's level
    got = {}
    for name, src in zip(net.sources, out):
        y = _resample(src.mean(0), net.samplerate, GF.SR)
        if abs(len(y) - len(mono)) > MAX_LENGTH_DRIFT:
            raise ValueError(f"stem '{name}' is {len(y) - len(mono)} samples off the mix")
        got[name] = (fit_length(y, len(mono)) / peak).astype(np.float32)
    missing = [n for n in STEM_NAMES if n not in got]
    if missing:
        raise ValueError(f"separator '{model}' has no {missing} source")
    return (mono / peak).astype(np.float32), {n: got[n] for n in STEM_NAMES}


def save_stems(out_dir, key: str, mix: np.ndarray, stems: dict, info: dict) -> dict:
    """Write one song's stems and its `song.json` (last, so its presence means
    the four files are complete). Returns the json row."""
    import soundfile as sf
    d = Path(out_dir) / key
    d.mkdir(parents=True, exist_ok=True)
    for name in STEM_NAMES:
        y = np.asarray(stems[name], np.float32)
        if len(y) != len(mix):
            raise ValueError(f"stem '{name}' has {len(y)} samples, the mix {len(mix)}")
        if not np.isfinite(y).all():
            raise ValueError(f"stem '{name}' is not finite")
        sf.write(str(d / f"{name}.wav"), y, GF.SR, subtype="FLOAT")
    row = {"key": key, "sr": GF.SR, "n_samples": int(len(mix)),
           "duration_s": round(len(mix) / GF.SR, 3),
           "snr_db": round(reconstruction_snr(mix, stems), 2), **info}
    tmp = d / "song.json.tmp"
    tmp.write_text(json.dumps(row, ensure_ascii=False))
    tmp.replace(d / "song.json")
    return row


def has_stems(stems_dir, key: str) -> bool:
    return (Path(stems_dir) / key / "song.json").exists()


def load_stems(stems_dir, key: str, n_samples: int | None = None,
               match_mix: np.ndarray | None = None) -> dict | None:
    """The four stems of song `key` as mono float32 at GF.SR, or None if the
    cache does not hold it. With `n_samples`, each is checked to be within
    1 ms of that length and then padded / trimmed to it exactly. `match_mix`
    (the mix as `GF.load_audio` returned it) is used only to refuse a cache
    entry that plainly belongs to different audio."""
    import soundfile as sf
    d = Path(stems_dir) / key
    if not (d / "song.json").exists():
        return None
    stems = {}
    for name in STEM_NAMES:
        y, sr = sf.read(str(d / f"{name}.wav"), dtype="float32")
        if sr != GF.SR:
            raise ValueError(f"{d / name}.wav is at {sr} Hz, not {GF.SR}")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if n_samples is not None:
            if abs(len(y) - n_samples) > MAX_LENGTH_DRIFT:
                raise ValueError(f"cached stem {key}/{name} is {len(y) - n_samples} "
                                 f"samples off its song")
            y = fit_length(y, n_samples)
        stems[name] = y
    if match_mix is not None and len(match_mix):
        # float64: float32 products through Apple's Accelerate raise spurious
        # floating-point warnings on finite data
        mix = np.asarray(match_mix, np.float64)
        total = np.sum([fit_length(stems[n], len(mix)) for n in STEM_NAMES],
                       axis=0).astype(np.float64)
        denom = float(np.linalg.norm(mix) * np.linalg.norm(total))
        if denom <= 0 or float(np.dot(mix, total)) / denom < 0.5:
            raise ValueError(f"cached stems of {key} do not match its audio")
    return stems


def write_manifest(out_dir) -> dict:
    out_dir = Path(out_dir)
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*/song.json"))]
    snr = [r["snr_db"] for r in rows]
    manifest = {"sr": GF.SR, "stems": list(STEM_NAMES), "n_songs": len(rows),
                "hours": round(sum(r["duration_s"] for r in rows) / 3600.0, 2),
                "snr_db_median": round(float(np.median(snr)), 2) if snr else None,
                "snr_db_min": round(float(np.min(snr)), 2) if snr else None,
                "songs": rows}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
    return manifest


def validate(out_dir) -> list:
    """Problems found in a cache (an empty list means it is sound): a missing
    or unreadable stem, a length that differs from the recorded one, or
    non-finite samples."""
    import soundfile as sf
    problems = []
    for p in sorted(Path(out_dir).glob("*/song.json")):
        row = json.loads(p.read_text())
        for name in STEM_NAMES:
            f = p.parent / f"{name}.wav"
            try:
                y, sr = sf.read(str(f), dtype="float32")
            except Exception as e:
                problems.append(f"{row['key']}/{name}: {type(e).__name__}")
                continue
            if sr != row["sr"] or len(y) != row["n_samples"]:
                problems.append(f"{row['key']}/{name}: {len(y)} samples at {sr} Hz, "
                                f"expected {row['n_samples']} at {row['sr']}")
            elif not np.isfinite(y).all():
                problems.append(f"{row['key']}/{name}: not finite")
    return problems


def build_cache(packs_dir, out_dir, model: str = "htdemucs", device: str = "auto",
                only_keys: set | None = None, limit: int | None = None,
                log=print) -> dict:
    """Separate every song folder under `packs_dir` (those in `only_keys`, if
    given) that the cache does not hold yet."""
    import simfile

    from ..train import corpus
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for folder in corpus.iter_song_folders(Path(packs_dir)):
        key = corpus.song_key(folder)
        if only_keys is not None and key not in only_keys:
            continue
        if not has_stems(out_dir, key):
            todo.append((folder, key))
    if limit:
        todo = todo[:limit]
    dev = pick_device(device)
    log(f"[stems] {len(todo)} song(s) to separate with {model} on {dev}")
    done = failed = 0
    t0 = time.time()
    for folder, key in todo:
        try:
            sm = corpus._find_simfile(folder)
            music = simfile.open(str(sm), strict=False).music if sm else None
            audio = corpus._find_audio(folder, music)
            if audio is None:
                raise ValueError("no audio")
            mix, stems = separate(audio, model, dev)
            row = save_stems(out_dir, key, mix, stems,
                             {"folder": folder.name, "audio": audio.name, "model": model})
            done += 1
            if done % 10 == 0 or done == len(todo):
                rate = done / max(1e-9, time.time() - t0)
                log(f"[stems] {done}/{len(todo)}  last snr {row['snr_db']} dB  "
                    f"eta {(len(todo) - done) / max(rate, 1e-9) / 60:.0f} min")
        except Exception as e:                  # one bad file must not stop a run
            failed += 1
            log(f"[stems] FAILED {folder.name}: {type(e).__name__}: {str(e)[:160]}")
    manifest = write_manifest(out_dir)
    log(f"[stems] cache holds {manifest['n_songs']} songs, {manifest['hours']} h; "
        f"median reconstruction SNR {manifest['snr_db_median']} dB; {failed} failed")
    return {"done": done, "failed": failed, "songs": manifest["n_songs"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packs-dir", required=True)
    ap.add_argument("--out-dir", default="build/stems_cache")
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--only-manifest",
                    help="separate only the songs this corpus manifest trains or "
                         "validates on")
    ap.add_argument("--workers", type=int, default=1,
                    help="accepted for symmetry with the other tools; separation "
                         "runs one song at a time on one device")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--validate", action="store_true",
                    help="check the cache instead of building it")
    a = ap.parse_args(argv)
    if a.validate:
        problems = validate(a.out_dir)
        print(f"[stems] {len(problems)} problem(s)")
        for line in problems[:50]:
            print("  " + line)
        raise SystemExit(1 if problems else 0)
    keys = None
    if a.only_manifest:
        m = json.loads(Path(a.only_manifest).read_text())
        keys = {s["key"] for s in m["songs"] if any(c["keep"] for c in s["charts"])}
    build_cache(a.packs_dir, a.out_dir, a.model, a.device, keys, a.limit)


if __name__ == "__main__":
    main()
