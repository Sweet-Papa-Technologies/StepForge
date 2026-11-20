"""shards.py — what the synthetic and derived sources share: the meta block of
a shard, and the manifest of a source folder.

A source folder looks like a compiled corpus (`songs/<key>.npz`,
`manifest.json`) so the same loader reads it. Three kinds of shard exist:

    with audio       `core`, `dt_ms` (and `stems`) + charts      real, aug_stem,
                                                                 synth_groove
    attached         charts only, plus `meta["parent"]` naming   synth_thin,
                     the shard whose audio they sit on           synth_style_grounded
    symbolic         charts only, no audio at all                synth_style

Every shard of a non-real source is a TRAINING shard: `write_source_manifest`
gives each song the split "train" and refuses anything else.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import codec as C
from ..train import corpus
from . import SOURCES, TRAIN_ONLY

#: A synthetic chart with fewer notes than this is not worth a training item.
MIN_NOTES = 16


def base_meta(key: str, song_id: str, source_type: str, *, title: str = "",
              artist: str = "", n_ticks: int = 0, duration_s: float = 0.0,
              has_stems: bool = False, has_audio: bool = True,
              parent: dict | None = None, **extra) -> dict:
    """The meta block every non-real shard carries. `parent` is
    {"source": <folder>, "key": <shard key>} for derived shards."""
    if source_type not in SOURCES:
        raise ValueError(f"unknown source type '{source_type}'")
    meta = {"corpus_version": corpus.CORPUS_VERSION, "codec_version": C.CODEC_VERSION,
            "codec_costs": corpus.codec_fingerprint(),
            "key": key, "song_id": song_id, "source_type": source_type,
            "title": title, "artist": artist, "n_ticks": int(n_ticks),
            "duration_s": round(float(duration_s), 2), "has_stems": bool(has_stems),
            "has_audio": bool(has_audio), "charts": []}
    if parent is not None:
        meta["parent"] = {"source": parent["source"], "key": parent["key"]}
    meta.update(extra)
    return meta


def read_meta(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(bytes(z["meta"]).decode("utf-8"))


def shard_files(source_dir) -> list:
    return sorted(f for f in (Path(source_dir) / "songs").glob("*.npz")
                  if not f.name.endswith(".tmp.npz"))


def write_source_manifest(source_dir, source_type: str, log=print) -> dict:
    """Index a non-real source folder. Every chart with at least MIN_NOTES
    notes is kept; every song is a training song."""
    if source_type not in TRAIN_ONLY:
        raise ValueError(f"'{source_type}' is not a train-only source")
    source_dir = Path(source_dir)
    songs, n_charts, n_kept = [], 0, 0
    for f in shard_files(source_dir):
        m = read_meta(f)
        if m.get("source_type") != source_type:
            raise ValueError(f"{f.name} is a '{m.get('source_type')}' shard in a "
                             f"'{source_type}' folder")
        charts = []
        for c in m["charts"]:
            keep = c["n_events"] >= MIN_NOTES
            n_charts += 1
            n_kept += keep
            charts.append({"j": c["j"], "difficulty": c["difficulty"],
                           "meter": c["meter"], "n_events": c["n_events"],
                           "keep": bool(keep), "reject": None if keep else "too_few_notes"})
        row = {"key": m["key"], "song_id": m["song_id"], "split": "train",
               "title": m.get("title", ""), "artist": m.get("artist", ""),
               "n_ticks": m["n_ticks"], "duration_s": m["duration_s"],
               "reject": None, "charts": charts}
        for k in ("parent", "variant", "archetype"):
            if k in m:
                row[k] = m[k]
        songs.append(row)
    manifest = {"corpus_version": corpus.CORPUS_VERSION, "codec_version": C.CODEC_VERSION,
                "source_type": source_type, "n_songs": len(songs),
                "n_song_ids": len({s["song_id"] for s in songs}),
                "n_charts": n_charts, "n_charts_kept": n_kept,
                "splits": {"train": {"songs": len(songs), "charts": n_kept},
                           "val": {"songs": 0, "charts": 0},
                           "test": {"songs": 0, "charts": 0}},
                "songs": songs}
    (source_dir / "manifest.json").write_text(json.dumps(manifest))
    log(f"[{source_type}] {len(songs)} shards, {n_kept}/{n_charts} charts kept")
    return manifest


def training_parents(real_dir) -> dict:
    """{key: manifest row} for the real songs a derived source may be built
    from: split 'train' and at least one kept chart. Everything derived goes
    through this, which is what keeps validation songs out of it."""
    manifest = json.loads((Path(real_dir) / "manifest.json").read_text())
    return {s["key"]: s for s in manifest["songs"]
            if s["split"] == "train" and any(c["keep"] for c in s["charts"])}


def check_no_leak(corpus_root, log=print) -> dict:
    """Assert the split invariants of a multi-source corpus root and return
    the counts: no non-real source holds a validation or test song, none of
    its songs carries a held-out song's id, and every derived shard descends
    from a training song and carries that parent's id. Raises ValueError on
    any violation."""
    root = Path(corpus_root)
    real = json.loads((root / "real" / "manifest.json").read_text())
    held_ids = {s["song_id"] for s in real["songs"] if s["split"] != "train"}
    known = {("real", s["key"]): s for s in real["songs"]}
    counts: dict = {"real": {sp: v["songs"] for sp, v in real["splits"].items()}}
    manifests = {}
    for source in TRAIN_ONLY:
        mp = root / source / "manifest.json"
        if mp.exists():
            manifests[source] = json.loads(mp.read_text())
            known.update({(source, s["key"]): s for s in manifests[source]["songs"]})
    for source, m in manifests.items():
        bad = [s["key"] for s in m["songs"] if s["split"] != "train"]
        if bad:
            raise ValueError(f"{source}: {len(bad)} shard(s) outside the training split")
        for s in m["songs"]:
            par = s.get("parent")
            if par:
                p = known.get((par["source"], par["key"]))
                if p is None or p["split"] != "train":
                    raise ValueError(f"{source}/{s['key']} descends from "
                                     f"{par['source']}/{par['key']}, which is not a "
                                     f"training song")
                if s["song_id"] != p["song_id"]:
                    raise ValueError(f"{source}/{s['key']} does not carry its "
                                     f"parent's song id")
            if s["song_id"] in held_ids:
                raise ValueError(f"{source}/{s['key']} carries the id of a held-out song")
        counts[source] = len(m["songs"])
    log(f"[corpus] split check passed: {counts}")
    return counts
