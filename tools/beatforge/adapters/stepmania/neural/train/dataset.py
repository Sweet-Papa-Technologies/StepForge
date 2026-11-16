"""dataset.py — song-split PyTorch datasets over a compiled corpus (spec §7).

The split is read from `manifest.json`, which assigns it per `song_id`; nothing
here can move a chart between splits. Mirror augmentation is not implemented
because it would be a no-op: the codec's tokens are already identical for a
chart and its mirror.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import codec as C
from .. import conditioning as cond
from .. import event_features as EF
from .. import grid_features as GF

CROP_TICKS = 32 * C.G                 # 32 measures
MAX_EVENTS = 512
P_DROP_STEMS = 0.50
P_DROP_FIELD = 0.25
P_DROP_ALL = 0.25


class Chart:
    __slots__ = ("song", "j", "difficulty", "meter", "tick", "tok", "flav", "da",
                 "ds", "tms", "mines", "style", "uid")


class Song:
    __slots__ = ("key", "song_id", "core", "dt_ms", "stems", "n_ticks", "charts")


def load_corpus(root, split: str, max_songs: int | None = None,
                reencode: bool = False, log=print) -> list:
    """Load every song of `split` that has at least one kept chart into RAM."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    songs = []
    entries = [s for s in manifest["songs"]
               if s["split"] == split and any(c["keep"] for c in s["charts"])]
    if max_songs:
        entries = entries[:max_songs]
    for n, entry in enumerate(entries, 1):
        with np.load(root / "songs" / f"{entry['key']}.npz", allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            s = Song()
            s.key, s.song_id = entry["key"], entry["song_id"]
            s.core, s.dt_ms = z["core"], z["dt_ms"]
            s.stems = z["stems"] if "stems" in z.files else None
            s.n_ticks = len(s.core)
            s.charts = []
            for ce, cm in zip(entry["charts"], meta["charts"]):
                if not ce["keep"]:
                    continue
                j = cm["j"]
                c = Chart()
                c.song, c.j = s, j
                c.difficulty, c.meter = cm["difficulty"], cm["meter"]
                c.tick = z[f"c{j}_tick"].astype(np.int64)
                c.tms = z[f"c{j}_tms"].astype(np.float64)
                c.mines = z[f"c{j}_mines"]
                if reencode:
                    raw, rawd = z[f"c{j}_raw"], z[f"c{j}_rawd"]
                    ev = C.ChartEvents(z[f"c{j}_tick"], raw[0], raw[1], raw[2], raw[3],
                                       rawd[0], rawd[1], c.mines)
                    enc, _ = C.encode(ev, c.tms)
                    c.tok, c.flav = enc.token, enc.flavor
                    c.da, c.ds = enc.dur_alt, enc.dur_same
                    c.style = cond.compute_style_vector(enc, c.tms / 1000.0, c.meter,
                                                        has_stems=s.stems is not None)
                else:
                    c.tok, c.flav = z[f"c{j}_tok"], z[f"c{j}_flav"]
                    c.da, c.ds = z[f"c{j}_da"], z[f"c{j}_ds"]
                    c.style = z[f"c{j}_style"].astype(np.float32)
                c.uid = zlib.crc32(f"{s.key}:{j}".encode())
                s.charts.append(c)
        songs.append(s)
        if n % 1000 == 0:
            log(f"[data] {split}: {n}/{len(entries)} songs")
    log(f"[data] {split}: {len(songs)} songs, {sum(len(s.charts) for s in songs)} charts")
    return songs


def _cond_vector(chart: Chart, train: bool, rng, has_stems: bool) -> np.ndarray:
    style = cond.training_style(chart.style)
    style[9] = 1.0 if has_stems else 0.0
    drop = []
    if train:
        if rng.random() < P_DROP_ALL:
            drop = list(range(1, 9))
        else:
            drop = [i for i in range(1, 9) if rng.random() < P_DROP_FIELD]
    return cond.cond19(style, drop)


def _window_start(chart: Chart, n_ticks: int, crop: int, train: bool, rng) -> int:
    """A measure-aligned crop start. Random in training, fixed (derived from
    the chart id) in evaluation so validation numbers are comparable."""
    span = max(0, (n_ticks - crop) // C.G)
    if span == 0:
        return 0
    k = int(rng.integers(0, span + 1)) if train else chart.uid % (span + 1)
    return k * C.G


class StageADataset(Dataset):
    """One item = one chart, cropped to `crop` ticks of its song."""

    def __init__(self, songs: list, crop: int = CROP_TICKS, train: bool = True,
                 seed: int = 0):
        self.charts = [c for s in songs for c in s.charts]
        self.crop, self.train, self.seed = crop, train, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.charts)

    def __getitem__(self, i: int) -> dict:
        c = self.charts[i]
        s = c.song
        rng = np.random.default_rng((self.seed, self.epoch, i) if self.train
                                    else (self.seed, i))
        a = _window_start(c, s.n_ticks, self.crop, self.train, rng)
        b = a + self.crop
        use_stems = s.stems is not None and not (self.train and rng.random() < P_DROP_STEMS)
        x = GF.assemble(s.core, s.dt_ms, s.stems if use_stems else None, a, b)
        valid = np.zeros(self.crop, bool)
        valid[:max(0, min(b, s.n_ticks) - a)] = True

        onset = np.zeros(self.crop, np.float32)
        jump = np.zeros(self.crop, np.float32)
        hold_type = np.zeros(self.crop, np.int64)
        hold_dur = np.zeros(self.crop, np.int64)
        mine = np.zeros(self.crop, np.float32)
        lo, hi = np.searchsorted(c.tick, [a, b])
        t = c.tick[lo:hi] - a
        onset[t] = 1.0
        jump[t] = c.tok[lo:hi] >= 8
        hold_type[t] = C.flavor6(c.flav[lo:hi])
        dur = np.maximum(c.da[lo:hi], c.ds[lo:hi])
        hold_dur[t] = C.dur_bucket(np.maximum(dur, 1))
        if len(c.mines):
            mt = c.mines[:, 0]
            mine[mt[(mt >= a) & (mt < b)] - a] = 1.0
        return {"x": torch.from_numpy(x),
                "cond": torch.from_numpy(_cond_vector(c, self.train, rng, use_stems)),
                "valid": torch.from_numpy(valid), "onset": torch.from_numpy(onset),
                "jump": torch.from_numpy(jump), "hold_type": torch.from_numpy(hold_type),
                "hold_dur": torch.from_numpy(hold_dur), "mine": torch.from_numpy(mine)}


class StageBDataset(Dataset):
    """One item = the notes of one chart inside a `crop`-tick window (at most
    `max_events` of them), with the audio context around each note."""

    def __init__(self, songs: list, crop: int = CROP_TICKS, max_events: int = MAX_EVENTS,
                 train: bool = True, seed: int = 0, min_events: int = 4):
        self.charts = [c for s in songs for c in s.charts if len(c.tick) >= min_events]
        self.crop, self.max_events = crop, max_events
        self.train, self.seed, self.min_events = train, seed, min_events
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.charts)

    def __getitem__(self, i: int) -> dict:
        c = self.charts[i]
        s = c.song
        rng = np.random.default_rng((self.seed, self.epoch, i) if self.train
                                    else (self.seed, i))
        # start the window at a random note so sparse charts are never empty
        n = len(c.tick)
        first = int(rng.integers(0, n)) if self.train else c.uid % n
        a = (int(c.tick[first]) // C.G) * C.G
        lo = int(np.searchsorted(c.tick, a))
        hi = int(np.searchsorted(c.tick, a + self.crop))
        hi = min(hi, lo + self.max_events)
        if hi - lo < self.min_events:              # tail of the chart: back up
            lo = max(0, hi - self.max_events)
            a = (int(c.tick[lo]) // C.G) * C.G
        ticks = c.tick[lo:hi]
        b = int(ticks[-1]) + 1
        use_stems = s.stems is not None and not (self.train and rng.random() < P_DROP_STEMS)
        m = EF.CTX_HALF
        x = GF.assemble(s.core, s.dt_ms, s.stems if use_stems else None, a - m, b + m)
        ctx = EF.event_context(x, ticks - (a - m))
        # scalars are computed on the WHOLE chart, then sliced, so a freeze that
        # began before the window still counts as in progress inside it
        scal = EF.event_scalars(c.tick, c.tms, c.tok >= 8, c.flav, c.da, c.ds)[lo:hi]
        dt = np.empty(hi - lo)
        dt[0] = 1e9 if lo == 0 else c.tms[lo] - c.tms[lo - 1]
        dt[1:] = np.diff(c.tms[lo:hi])
        return {"ctx": torch.from_numpy(ctx), "scalars": torch.from_numpy(scal),
                "tokens": torch.from_numpy(c.tok[lo:hi].astype(np.int64)),
                "dtb": torch.from_numpy(C.dt_bucket(dt).astype(np.int64)),
                "cond": torch.from_numpy(_cond_vector(c, self.train, rng, use_stems))}


def collate_b(items: list) -> dict:
    n = max(len(it["tokens"]) for it in items)
    b = len(items)
    k, d = items[0]["ctx"].shape[1:]
    out = {"ctx": torch.zeros(b, n, k, d), "scalars": torch.zeros(b, n, EF.N_SCALARS),
           "tokens": torch.zeros(b, n, dtype=torch.long),
           "dtb": torch.full((b, n), C.N_DT - 1, dtype=torch.long),
           "pad": torch.zeros(b, n, dtype=torch.bool),
           "cond": torch.stack([it["cond"] for it in items])}
    for i, it in enumerate(items):
        m = len(it["tokens"])
        out["ctx"][i, :m], out["scalars"][i, :m] = it["ctx"], it["scalars"]
        out["tokens"][i, :m], out["dtb"][i, :m] = it["tokens"], it["dtb"]
        out["pad"][i, :m] = True
    return out


def set_epoch(loader, epoch: int) -> None:
    """Re-seed a dataset's random crops for a new epoch (works through a
    DataLoader with persistent workers disabled)."""
    loader.dataset.epoch = epoch
