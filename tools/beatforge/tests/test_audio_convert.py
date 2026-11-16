"""ffmpeg-free audio import (beatforge.audio_convert)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from beatforge.audio_convert import to_ogg

TOOLS = Path(__file__).resolve().parents[2]


def tone(path, sr=44100, seconds=1.5, channels=2, subtype="PCM_16", fmt="WAV"):
    t = np.arange(int(sr * seconds)) / sr
    y = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), np.stack([y] * channels, axis=1), sr, format=fmt, subtype=subtype)


def test_wav_and_flac_become_readable_ogg(tmp_path):
    for fmt, sub in (("WAV", "PCM_16"), ("FLAC", "PCM_16")):
        src = tmp_path / f"in.{fmt.lower()}"
        tone(src, fmt=fmt, subtype=sub)
        info = to_ogg(src, tmp_path / f"{fmt}.ogg")
        y, sr = sf.read(str(tmp_path / f"{fmt}.ogg"))
        assert sr == 44100 and y.shape[1] == 2
        assert abs(len(y) / sr - 1.5) < 0.05 and info["seconds"] == 1.5
        assert not (tmp_path / f"{fmt}.ogg.part").exists()


def test_more_than_two_channels_are_folded_to_stereo(tmp_path):
    tone(tmp_path / "six.wav", channels=6)
    assert to_ogg(tmp_path / "six.wav", tmp_path / "six.ogg")["channels"] == 2


def test_cli_reports_json_and_fails_cleanly(tmp_path):
    tone(tmp_path / "a.wav")
    run = lambda *a: subprocess.run([sys.executable, "-m", "beatforge", "to-ogg", *a],
                                    cwd=TOOLS, capture_output=True, text=True,
                                    env={"PYTHONPATH": str(TOOLS), "PATH": ""})
    ok = run("--in", str(tmp_path / "a.wav"), "--out", str(tmp_path / "a.ogg"))
    assert ok.returncode == 0 and json.loads(ok.stdout.splitlines()[-1])["ok"] is True
    bad = run("--in", str(tmp_path / "missing.wav"), "--out", str(tmp_path / "b.ogg"))
    assert bad.returncode == 1 and json.loads(bad.stdout.splitlines()[-1])["ok"] is False
