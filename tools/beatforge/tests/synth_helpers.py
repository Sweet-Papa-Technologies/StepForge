"""Builders shared by the v1.5 synthesis / training tests: tiny song folders
made of clicks, a fake 4-stem cache for them, and a compiled multi-source
corpus root. Everything is synthetic and deterministic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from beatforge.adapters.stepmania.neural import grid_features as GF
from beatforge.adapters.stepmania.neural.synth import stems_cache
from beatforge.adapters.stepmania.neural.train import corpus

SR = GF.SR
BPM = 150.0
OFFSET_S = 0.25                     # time of beat 0
BARS = 22                           # 35 s at 150 BPM: above the 30 s minimum
REST_BARS = (8, 9)                  # silent in the audio and in every chart
RUN_BARS = (3, 7, 13, 17, 21)       # bars whose second half is a 16th run
PANELS = "LDUR"
STEM_SPLIT = {"drums": 0.5, "bass": 0.2, "vocals": 0.2, "other": 0.1}
TIERS = (("Beginner", 1), ("Easy", 3), ("Medium", 5), ("Hard", 8), ("Challenge", 10))


def chart_ticks(tier: str, bars: int = BARS) -> list:
    """Note ticks of the fixture chart of `tier` (48 ticks per bar)."""
    step = {"Beginner": 24, "Easy": 12, "Medium": 12, "Hard": 6, "Challenge": 6}[tier]
    ticks = []
    for bar in range(bars):
        if bar in REST_BARS:
            continue
        ticks += [bar * 48 + k for k in range(0, 48, step)]
        if tier == "Medium":
            ticks += [bar * 48 + 30, bar * 48 + 42]          # two off-beat 8ths
        if tier == "Challenge" and bar in RUN_BARS:
            ticks += [bar * 48 + k for k in range(27, 48, 6)]
    return sorted(set(ticks))


def _notes_block(tier: str, bars: int) -> str:
    ticks = chart_ticks(tier, bars)
    rows = {}
    for n, t in enumerate(ticks):
        row = ["0"] * 4
        row[n % 4] = "1"
        if tier in ("Hard", "Challenge") and t % 96 == 0:
            row = list("1001")                               # a jump on every other bar
        rows[t] = "".join(row)
    if tier in ("Medium", "Hard", "Challenge"):
        # one freeze in bar 1: the head replaces that tick's tap, the tail a beat later
        head = 48
        rows[head] = "2000"
        rows[head + 24] = "3" + rows.get(head + 24, "0000")[1:]
        if rows[head + 24][1:] == "000":
            rows[head + 24] = "3100"
    measures = []
    for bar in range(bars):
        measures.append("\n".join(rows.get(bar * 48 + k, "0000") for k in range(48)))
    return "\n,\n".join(measures)


def clicks(bars: int = BARS, bpm: float = BPM, seed: int = 0) -> np.ndarray:
    """Clicks on every 8th (louder on the beat) and on the 16ths of the run
    bars; the rest bars are silent."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    y = np.zeros(int((OFFSET_S + bars * 4 * beat + 0.5) * SR), np.float32)
    env = np.exp(-np.arange(int(0.03 * SR)) / (0.004 * SR)).astype(np.float32)
    for bar in range(bars):
        if bar in REST_BARS:
            continue
        ticks = set(range(0, 48, 6))
        if bar in RUN_BARS:
            ticks |= set(range(27, 48, 6))
        for k in sorted(ticks):
            i = int(round((OFFSET_S + (bar * 48 + k) / 12.0 * beat) * SR))
            amp = 1.0 if k % 12 == 0 else 0.55
            y[i:i + len(env)] += amp * env * rng.standard_normal(len(env)).astype(np.float32)
    return (0.8 * y / np.abs(y).max()).astype(np.float32)


def make_song_folder(root: Path, name: str, seed: int = 0, bpm: float = BPM,
                     bars: int = BARS) -> Path:
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(str(folder / "song.wav"), clicks(bars, bpm, seed), SR, subtype="PCM_16")
    parts = [f"#VERSION:0.83;\n#TITLE:{name};\n#ARTIST:test {seed};\n#MUSIC:song.wav;\n"
             f"#OFFSET:{-OFFSET_S:.3f};\n#BPMS:0.000={bpm:.3f};\n#STOPS:;\n"]
    for tier, meter in TIERS:
        parts.append(f"#NOTEDATA:;\n#STEPSTYPE:dance-single;\n#DIFFICULTY:{tier};\n"
                     f"#METER:{meter};\n#NOTES:\n{_notes_block(tier, bars)}\n;\n")
    (folder / f"{name}.ssc").write_text("".join(parts))
    return folder


def fake_stems(packs: Path, stems_dir: Path) -> None:
    """A stem cache whose four stems are fixed fractions of the mix: they sum
    to it exactly, which is all the compiler and the augmenter rely on."""
    for folder in corpus.iter_song_folders(Path(packs)):
        mix = GF.load_audio(str(folder / "song.wav"))
        stems = {n: (mix * g).astype(np.float32) for n, g in STEM_SPLIT.items()}
        stems_cache.save_stems(stems_dir, corpus.song_key(folder), mix, stems,
                               {"folder": folder.name, "audio": "song.wav", "model": "fake"})
    stems_cache.write_manifest(stems_dir)


def make_corpus_root(tmp: Path, train: int = 2, val: int = 1, stems: bool = True) -> Path:
    """`tmp/corpus/real` compiled from `train + val` click songs (with stem
    channels), every chart kept, the split forced by name. Returns the root;
    the song folders are in `tmp/packs`, the stem cache in `tmp/stems`."""
    packs, root = Path(tmp) / "packs", Path(tmp) / "corpus"
    names = [f"train{i}" for i in range(train)] + [f"val{i}" for i in range(val)]
    for i, name in enumerate(names):
        # a different tempo (and so a different length) per song: the manifest
        # merges songs whose audio fingerprints match, and identical click
        # tracks would all become one song id
        make_song_folder(packs, name, seed=i, bpm=BPM + 9.0 * i)
    if stems:
        fake_stems(packs, Path(tmp) / "stems")
    corpus.compile_corpus(packs, root / "real", workers=1, log=lambda *_: None,
                          stems_dir=str(Path(tmp) / "stems") if stems else None)
    manifest = corpus.build_manifest(root / "real", log=lambda *_: None)
    for s in manifest["songs"]:
        s["split"] = "val" if s["title"].startswith("val") else "train"
        s["reject"] = None
        for c in s["charts"]:
            c["keep"], c["reject"] = True, None
    manifest["splits"] = {"train": {"songs": train, "charts": 5 * train},
                          "val": {"songs": val, "charts": 5 * val},
                          "test": {"songs": 0, "charts": 0}}
    (root / "real" / "manifest.json").write_text(json.dumps(manifest))
    return root


def real_keys(root: Path, split: str) -> list:
    m = json.loads((Path(root) / "real" / "manifest.json").read_text())
    return sorted(s["key"] for s in m["songs"] if s["split"] == split)
