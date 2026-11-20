"""colab_bundle.py — pack a multi-source corpus root into tarballs for a GPU
machine, and unpack them there.

    # where the corpus was built
    python -m beatforge.adapters.stepmania.neural.synth.colab_bundle pack \
        --corpus-root build/v15_corpus --out-dir build/colab_upload
    # on the training machine (local SSD, never a network drive)
    python -m beatforge.adapters.stepmania.neural.synth.colab_bundle unpack \
        --bundle-dir /content/drive/MyDrive/StepForge/v15_upload \
        --corpus-root /content/v15_corpus

One tarball per source folder plus `bundle.json` (shard and chart counts,
split counts, the codec's cost-table fingerprint, a SHA-256 per tarball).
`pack` refuses a corpus that breaks the split rules (`shards.check_no_leak`);
`unpack` verifies every checksum and runs the same check again on what it
unpacked, so a corpus cannot arrive on the trainer in a state the builder
would not have shipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

from ..train import corpus
from . import SOURCES, shards

BUNDLE_FILE = "bundle.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(corpus_root, out_dir, compress: bool = True, log=print) -> dict:
    root, out_dir = Path(corpus_root), Path(out_dir)
    counts = shards.check_no_leak(root, log)
    out_dir.mkdir(parents=True, exist_ok=True)
    real = json.loads((root / "real" / "manifest.json").read_text())
    info = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "codec_version": real["codec_version"],
            "codec_costs": corpus.codec_fingerprint(),
            "splits": real["splits"], "sources": {}}
    for source in SOURCES:
        d = root / source
        if not (d / "manifest.json").exists():
            continue
        m = json.loads((d / "manifest.json").read_text())
        name = f"{source}.tar.gz" if compress else f"{source}.tar"
        tmp = out_dir / (name + ".part")
        with tarfile.open(tmp, "w:gz" if compress else "w", **(
                {"compresslevel": 3} if compress else {})) as tar:
            tar.add(d / "manifest.json", arcname=f"{source}/manifest.json")
            for f in shards.shard_files(d):
                tar.add(f, arcname=f"{source}/songs/{f.name}")
        tmp.replace(out_dir / name)
        size = (out_dir / name).stat().st_size
        info["sources"][source] = {"file": name, "sha256": _sha256(out_dir / name),
                                   "bytes": size, "shards": m["n_songs"],
                                   "charts": m["n_charts_kept"]}
        log(f"[bundle] {name}: {m['n_songs']} shards, {size / 1e6:.0f} MB")
    info["checked"] = counts
    (out_dir / BUNDLE_FILE).write_text(json.dumps(info, indent=1))
    total = sum(s["bytes"] for s in info["sources"].values())
    log(f"[bundle] {len(info['sources'])} tarball(s), {total / 1e9:.2f} GB in {out_dir}")
    return info


def unpack(bundle_dir, corpus_root, log=print) -> dict:
    bundle_dir, root = Path(bundle_dir), Path(corpus_root)
    info = json.loads((bundle_dir / BUNDLE_FILE).read_text())
    if info["codec_costs"] != corpus.codec_fingerprint():
        raise ValueError(f"the bundle was encoded with cost table {info['codec_costs']}, "
                         f"this code has {corpus.codec_fingerprint()}: run "
                         f"`corpus reencode` on each source after unpacking")
    root.mkdir(parents=True, exist_ok=True)
    for source, row in info["sources"].items():
        path = bundle_dir / row["file"]
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"{row['file']} is damaged (checksum mismatch)")
        with tarfile.open(path) as tar:
            tar.extractall(root, filter="data")
        got = len(shards.shard_files(root / source))
        if got != row["shards"]:
            raise ValueError(f"{source}: unpacked {got} shards, expected {row['shards']}")
        log(f"[bundle] {source}: {got} shards")
    shards.check_no_leak(root, log)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--corpus-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-compress", action="store_true")
    u = sub.add_parser("unpack")
    u.add_argument("--bundle-dir", required=True)
    u.add_argument("--corpus-root", required=True)
    # the runbook's short form (no subcommand) packs
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("pack", "unpack", "-h", "--help"):
        argv = ["pack", *argv]
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        pack(a.corpus_root, a.out_dir, not a.no_compress)
    else:
        unpack(a.bundle_dir, a.corpus_root)


if __name__ == "__main__":
    main()
