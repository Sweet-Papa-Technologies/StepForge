"""The chart timeline preview (<difficulty>.notefield.png) is drawn with
Pillow; matplotlib is no longer bundled with the app."""
from __future__ import annotations

import numpy as np
from PIL import Image

from beatforge.adapters.stepmania.neural.pipeline import _notefield_png
from beatforge.adapters.stepmania.quantize import Placement


def test_notefield_png_draws_taps_holds_and_sections(tmp_path):
    # Real placements: taps carry hold_beats=None, holds a length in beats.
    notes = [Placement(beat=float(i), panels=(i % 4,)) for i in range(32)]
    notes += [Placement(beat=40.0, panels=(1, 2), kind="hold", hold_beats=2.0)]
    times = np.array([0.5 + p.beat * 0.5 for p in notes])
    out = tmp_path / "hard.notefield.png"
    _notefield_png(out, notes, times, 120.0, 0.5, [{"start_bar": 0}, {"start_bar": 8}])
    im = Image.open(out).convert("RGB")
    assert im.size == (1760, 330)
    colors = {c for _, c in im.getcolors(maxcolors=1 << 16)}
    assert (0x2D, 0xE2, 0xE6) in colors      # taps
    assert (0xFF, 0x2D, 0x95) in colors      # the hold
    assert (0xB1, 0x4C, 0xFF) in colors      # section lines


def test_notefield_png_without_notes(tmp_path):
    out = tmp_path / "empty.notefield.png"
    _notefield_png(out, [], np.array([]), 120.0, 0.0, [])
    assert out.is_file()
