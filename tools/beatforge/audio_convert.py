"""Audio import without ffmpeg.

libsndfile (bundled in the `soundfile` wheels on every platform) decodes WAV,
FLAC, Ogg Vorbis/Opus, AIFF and, since 1.1, MP3, and encodes Ogg Vorbis. That
covers what most users import, so the desktop app can work on a machine with
no ffmpeg; formats libsndfile cannot read (AAC/M4A, video containers) still
need ffmpeg.
"""
from __future__ import annotations

from pathlib import Path

#: Frames per block: bounds memory on long files.
BLOCK = 1 << 18


def to_ogg(src, out, quality: float = 0.6) -> dict:
    """Write `src` to `out` as Ogg Vorbis. Returns {seconds, samplerate, channels}.
    `quality` is libsndfile's VBR quality (0..1)."""
    import soundfile as sf
    src, out = Path(src), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with sf.SoundFile(str(src)) as fin:
        sr, ch, frames = fin.samplerate, min(fin.channels, 2), fin.frames
        with sf.SoundFile(str(tmp), "w", samplerate=sr, channels=ch,
                          format="OGG", subtype="VORBIS") as fout:
            try:
                fout.compression_level = 1.0 - quality
            except Exception:                    # noqa: BLE001 — older soundfile
                pass
            while True:
                block = fin.read(BLOCK, dtype="float32", always_2d=True)
                if not len(block):
                    break
                if block.shape[1] > 2:
                    block = block[:, :2]
                fout.write(block)
    tmp.replace(out)
    return {"seconds": round(frames / sr, 3) if sr else 0.0, "samplerate": sr,
            "channels": ch}
