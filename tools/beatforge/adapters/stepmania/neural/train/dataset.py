"""dataset.py — song-split PyTorch datasets over a compiled corpus (spec §7).

The split is read from `manifest.json`, which assigns it per `song_id`; nothing
here can move a chart between splits. Mirror augmentation is not implemented
because it would be a no-op: the codec's tokens are already identical for a
chart and its mirror.

A corpus is either one folder (`load_corpus`) or a ROOT of source folders
(`load_sources`; see `neural/synth/__init__.py`): the human charts in `real/`
plus training-only sources derived from them or synthesized. `MixedDataset`
draws each training item from one source according to a schedule of shares
that may change with the epoch. Validation never goes through it: it is built
from the `val` split of `real/` alone, and `load_sources` refuses a training
-only source that lists anything else.
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
#: Sources under a corpus root that only ever train (mirrors synth.TRAIN_ONLY;
#: repeated here so the trainers do not import the synthesis package).
TRAIN_ONLY_SOURCES = ("aug_stem", "synth_groove", "synth_thin", "synth_style",
                      "synth_style_grounded", "synth_cf", "synth_cf_stream")
#: Sources whose shards borrow another shard's audio (loaded after the rest).
ATTACHED_SOURCES = ("synth_thin", "synth_style_grounded", "synth_cf")
#: The style sliders the step network alone decides (crossovers, jacks,
#: footswitches). On a counterfactual chart they are never dropped: its rhythm
#: and audio are shared with the other variants of its group, so these three
#: numbers are the only thing that tells the variants apart.
FOOTING_FIELDS = (cond.INDEX["crossover_share"], cond.INDEX["jack_share"],
                  cond.INDEX["footswitch_share"])
#: Relative sampling weight of a thinned chart whose tier the song already has.
COMPANION_WEIGHT = 0.5


class Chart:
    __slots__ = ("song", "j", "difficulty", "meter", "tick", "tok", "flav", "da",
                 "ds", "tms", "mines", "style", "uid", "weight", "group", "variant")


class Song:
    """`core` is None for a symbolic song (step patterns with no audio)."""
    __slots__ = ("key", "song_id", "core", "dt_ms", "stems", "n_ticks", "charts",
                 "source", "kind")


def _read_charts(z, song: Song, entry: dict, meta: dict, reencode: bool) -> None:
    for ce, cm in zip(entry["charts"], meta["charts"]):
        if not ce["keep"]:
            continue
        j = cm["j"]
        c = Chart()
        c.song, c.j = song, j
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
                                                has_stems=song.stems is not None)
        else:
            c.tok, c.flav = z[f"c{j}_tok"], z[f"c{j}_flav"]
            c.da, c.ds = z[f"c{j}_da"], z[f"c{j}_ds"]
            c.style = z[f"c{j}_style"].astype(np.float32)
        c.uid = zlib.crc32(f"{song.key}:{j}".encode())
        c.weight = COMPANION_WEIGHT if cm.get("companion") else 1.0
        # counterfactual charts: the base rhythm they share, and which footing
        c.group, c.variant = cm.get("rhythm_group"), cm.get("variant")
        song.charts.append(c)


def load_corpus(root, split: str, max_songs: int | None = None,
                reencode: bool = False, log=print, parents: dict | None = None,
                source: str | None = None) -> list:
    """Load every song of `split` that has at least one kept chart into RAM.

    `parents` maps (source, key) to an already loaded Song; a shard that names
    a parent (charts with no audio of their own) borrows that song's features,
    and is skipped when the parent is not there. A shard with neither audio
    nor a parent loads as a symbolic song."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    source = source or manifest.get("source_type") or "real"
    songs = []
    entries = [s for s in manifest["songs"]
               if s["split"] == split and any(c["keep"] for c in s["charts"])
               and not (source == "synth_style" and s.get("grounded"))]
    if max_songs:
        entries = entries[:max_songs]
    orphans = 0
    for n, entry in enumerate(entries, 1):
        with np.load(root / "songs" / f"{entry['key']}.npz", allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            s = Song()
            s.key, s.song_id, s.source = entry["key"], entry["song_id"], source
            s.kind = meta.get("kind")
            s.charts = []
            par = meta.get("parent")
            if "core" in z.files:
                s.core, s.dt_ms = z["core"], z["dt_ms"]
                s.stems = z["stems"] if "stems" in z.files else None
                s.n_ticks = len(s.core)
            elif par is not None:
                p = (parents or {}).get((par["source"], par["key"]))
                if p is None:
                    orphans += 1
                    continue
                s.core, s.dt_ms, s.stems, s.n_ticks = p.core, p.dt_ms, p.stems, p.n_ticks
            else:
                s.core = s.dt_ms = s.stems = None
                s.n_ticks = int(meta["n_ticks"])
            _read_charts(z, s, entry, meta, reencode)
        songs.append(s)
        if n % 1000 == 0:
            log(f"[data] {source} {split}: {n}/{len(entries)} songs")
    log(f"[data] {source} {split}: {len(songs)} songs, "
        f"{sum(len(s.charts) for s in songs)} charts"
        + (f" ({orphans} skipped: parent not loaded)" if orphans else ""))
    return songs


def load_sources(corpus_root, names, max_songs: int | None = None,
                 reencode: bool = False, log=print) -> dict:
    """Load a multi-source corpus root -> {"val": [...], <source>: [...]}.

    `real` is always loaded (train and val). Each other name in `names` is
    loaded if its folder exists, train split only, after checking that its
    manifest holds nothing else and that none of its songs carries the id of
    a validation song."""
    root = Path(corpus_root)
    out = {"real": load_corpus(root / "real", "train", max_songs, reencode, log,
                               source="real"),
           "val": load_corpus(root / "real", "val", max_songs, reencode, log,
                              source="real")}
    held = {s.song_id for s in out["val"]}
    real_manifest = json.loads((root / "real" / "manifest.json").read_text())
    held |= {s["song_id"] for s in real_manifest["songs"] if s["split"] != "train"}
    parents = {("real", s.key): s for s in out["real"]}
    # sources with audio first, so attached shards can find any of them
    order = sorted((n for n in names if n != "real"),
                   key=lambda n: n in ATTACHED_SOURCES)
    for name in order:
        if name not in TRAIN_ONLY_SOURCES:
            raise ValueError(f"unknown corpus source '{name}'")
        d = root / name
        if not (d / "manifest.json").exists():
            log(f"[data] {name}: not in this corpus root, skipped")
            continue
        m = json.loads((d / "manifest.json").read_text())
        bad = [s["key"] for s in m["songs"] if s["split"] != "train"]
        leaked = [s["key"] for s in m["songs"] if s["song_id"] in held]
        if bad or leaked:
            raise ValueError(f"{name}: {len(bad)} shard(s) outside the training split, "
                             f"{len(leaked)} carrying a held-out song id")
        out[name] = load_corpus(d, "train", max_songs, reencode, log, parents, name)
        parents.update({(name, s.key): s for s in out[name] if s.core is not None})
    return out


def _cond_vector(chart: Chart, train: bool, rng, has_stems: bool,
                 keep: tuple = ()) -> np.ndarray:
    """The conditioning vector of a chart. In training each slider is hidden
    at random (so the network also works with sliders left to it), except the
    ones listed in `keep`."""
    style = cond.training_style(chart.style)
    style[9] = 1.0 if has_stems else 0.0
    drop = []
    if train:
        if rng.random() < P_DROP_ALL:
            drop = list(range(1, 9))
        else:
            drop = [i for i in range(1, 9) if rng.random() < P_DROP_FIELD]
        drop = [i for i in drop if i not in keep]
    return cond.cond19(style, drop)


def _window_start(chart: Chart, n_ticks: int, crop: int, train: bool, rng) -> int:
    """A measure-aligned crop start. Random in training, fixed (derived from
    the chart id) in evaluation so validation numbers are comparable."""
    span = max(0, (n_ticks - crop) // C.G)
    if span == 0:
        return 0
    k = int(rng.integers(0, span + 1)) if train else chart.uid % (span + 1)
    return k * C.G


class _ChartDataset(Dataset):
    """Shared by both stages: the chart list, per-chart sampling weights and
    the per-item random generator."""

    charts: list
    train: bool
    seed: int
    epoch: int

    def _rng(self, i: int):
        return np.random.default_rng((self.seed, self.epoch, i) if self.train
                                     else (self.seed, i))

    def __len__(self) -> int:
        return len(self.charts)

    def __getitem__(self, i: int) -> dict:
        return self.item(i, self._rng(i))

    def item(self, i: int, rng) -> dict:
        raise NotImplementedError

    @property
    def weights(self) -> np.ndarray:
        return np.array([c.weight for c in self.charts], np.float64)


#: SpecAugment on the input tensor (v1.5c): what is masked are the acoustic
#: channels only — mel, flux, stems, chroma. The metric-timing channels say
#: where in the bar and at what tempo a tick is; a model that cannot rely on
#: them cannot place anything, so they are never touched.
SPEC_ACOUSTIC = GF.METRIC.start                 # channels [0, 54) are acoustic
SPEC_TIME_MASKS = 2
SPEC_TIME_TICKS = (4, 12)                       # inclusive
SPEC_BAND_MASKS = (1, 2)                        # inclusive
SPEC_BAND_CHANNELS = (2, 5)                     # inclusive


def spec_augment(x: np.ndarray, rng) -> np.ndarray:
    """Zero SPEC_TIME_MASKS random spans of ticks and one or two random bands
    of channels, in the acoustic channels of `x` (ticks, 64) only. In place;
    returns `x`."""
    n = len(x)
    for _ in range(SPEC_TIME_MASKS):
        w = int(rng.integers(SPEC_TIME_TICKS[0], SPEC_TIME_TICKS[1] + 1))
        if n > w:
            a = int(rng.integers(0, n - w + 1))
            x[a:a + w, :SPEC_ACOUSTIC] = 0.0
    for _ in range(int(rng.integers(SPEC_BAND_MASKS[0], SPEC_BAND_MASKS[1] + 1))):
        w = int(rng.integers(SPEC_BAND_CHANNELS[0], SPEC_BAND_CHANNELS[1] + 1))
        c = int(rng.integers(0, SPEC_ACOUSTIC - w + 1))
        x[:, c:c + w] = 0.0
    return x


class StageADataset(_ChartDataset):
    """One item = one chart, cropped to `crop` ticks of its song.

    `stems`: 'drop' zeroes the stem channels on a random `p_drop_stems` of
    training items (and never in evaluation), 'off' always zeroes them — the
    app's situation, which has no separator — and 'on' never does.

    `spec_augment` masks parts of the acoustic input of every training item
    (see `spec_augment`); an evaluation dataset never does, whatever it is
    given."""

    def __init__(self, songs: list, crop: int = CROP_TICKS, train: bool = True,
                 seed: int = 0, p_drop_stems: float = P_DROP_STEMS, stems: str = "drop",
                 spec_augment: bool = False):
        self.charts = [c for s in songs for c in s.charts if s.core is not None]
        self.crop, self.train, self.seed = crop, train, seed
        self.p_drop_stems, self.stems = p_drop_stems, stems
        self.spec_augment = bool(spec_augment) and train
        self.epoch = 0

    def _use_stems(self, s: Song, rng) -> bool:
        if s.stems is None or self.stems == "off":
            return False
        if self.stems == "on" or not self.train:
            return True
        return not rng.random() < self.p_drop_stems

    def item(self, i: int, rng) -> dict:
        c = self.charts[i]
        s = c.song
        a = _window_start(c, s.n_ticks, self.crop, self.train, rng)
        b = a + self.crop
        use_stems = self._use_stems(s, rng)
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
        cv = _cond_vector(c, self.train, rng, use_stems)
        if self.spec_augment:
            # drawn after everything else, so the crop and the conditioning of
            # an item are the ones a run without masking would have seen
            x = spec_augment(np.array(x, np.float32, copy=True), rng)
        return {"x": torch.from_numpy(x), "cond": torch.from_numpy(cv),
                "valid": torch.from_numpy(valid), "onset": torch.from_numpy(onset),
                "jump": torch.from_numpy(jump), "hold_type": torch.from_numpy(hold_type),
                "hold_dur": torch.from_numpy(hold_dur), "mine": torch.from_numpy(mine)}


class StageBDataset(_ChartDataset):
    """One item = the notes of one chart inside a `crop`-tick window (at most
    `max_events` of them), with the audio context around each note.

    `drop_audio` in an item tells the network to put its learned "no audio"
    vector in place of what it heard: always for a symbolic song (there is
    nothing to hear), and on a random `p_drop_audio` of the other training
    items, so that the vector means "choose from the rhythm and the style
    alone" rather than "this is a synthetic pattern"."""

    def __init__(self, songs: list, crop: int = CROP_TICKS, max_events: int = MAX_EVENTS,
                 train: bool = True, seed: int = 0, min_events: int = 4,
                 p_drop_stems: float = P_DROP_STEMS, p_drop_audio: float = 0.0):
        self.charts = [c for s in songs for c in s.charts if len(c.tick) >= min_events]
        self.crop, self.max_events = crop, max_events
        self.train, self.seed, self.min_events = train, seed, min_events
        self.p_drop_stems, self.p_drop_audio = p_drop_stems, p_drop_audio
        self.epoch = 0

    def item(self, i: int, rng) -> dict:
        c = self.charts[i]
        s = c.song
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
        symbolic = s.core is None
        use_stems = (not symbolic and s.stems is not None
                     and not (self.train and rng.random() < self.p_drop_stems))
        if symbolic:
            ctx = np.zeros((hi - lo, EF.CTX_K, GF.D_IN), np.float32)
        else:
            m = EF.CTX_HALF
            x = GF.assemble(s.core, s.dt_ms, s.stems if use_stems else None, a - m, b + m)
            ctx = EF.event_context(x, ticks - (a - m))
        drop_audio = symbolic or (self.train and rng.random() < self.p_drop_audio)
        # scalars are computed on the WHOLE chart, then sliced, so a freeze that
        # began before the window still counts as in progress inside it
        scal = EF.event_scalars(c.tick, c.tms, c.tok >= 8, c.flav, c.da, c.ds)[lo:hi]
        dt = np.empty(hi - lo)
        dt[0] = 1e9 if lo == 0 else c.tms[lo] - c.tms[lo - 1]
        dt[1:] = np.diff(c.tms[lo:hi])
        return {"ctx": torch.from_numpy(ctx), "scalars": torch.from_numpy(scal),
                "tokens": torch.from_numpy(c.tok[lo:hi].astype(np.int64)),
                "dtb": torch.from_numpy(C.dt_bucket(dt).astype(np.int64)),
                "cond": torch.from_numpy(_cond_vector(
                    c, self.train, rng, use_stems,
                    keep=FOOTING_FIELDS if getattr(c, "group", None) else ())),
                "drop_audio": torch.tensor(bool(drop_audio))}


def collate_b(items: list) -> dict:
    n = max(len(it["tokens"]) for it in items)
    b = len(items)
    k, d = items[0]["ctx"].shape[1:]
    out = {"ctx": torch.zeros(b, n, k, d), "scalars": torch.zeros(b, n, EF.N_SCALARS),
           "tokens": torch.zeros(b, n, dtype=torch.long),
           "dtb": torch.full((b, n), C.N_DT - 1, dtype=torch.long),
           "pad": torch.zeros(b, n, dtype=torch.bool),
           "cond": torch.stack([it["cond"] for it in items]),
           "drop_audio": torch.stack([it.get("drop_audio", torch.tensor(False))
                                      for it in items])}
    for i, it in enumerate(items):
        m = len(it["tokens"])
        out["ctx"][i, :m], out["scalars"][i, :m] = it["ctx"], it["scalars"]
        out["tokens"][i, :m], out["dtb"][i, :m] = it["tokens"], it["dtb"]
        out["pad"][i, :m] = True
    return out


class GroupedStageBDataset(StageBDataset):
    """Counterfactual charts, drawn two at a time: two footing variants of the
    SAME rhythm (`Chart.group`), cropped to the same window with the same
    sliders hidden. Next to each other in a batch, the two items differ in
    their arrows and in their three footing sliders and in nothing else, so
    every gradient step contrasts what those sliders mean.

    `groups` lists the chart indices of every rhythm that has at least two
    variants; a rhythm with one is left out (there is nothing to contrast)."""

    def __init__(self, songs: list, *args, **kwargs):
        super().__init__(songs, *args, **kwargs)
        by: dict = {}
        for i, c in enumerate(self.charts):
            if c.group is not None:
                by.setdefault((c.song.key, c.group), []).append(i)
        self.groups = [g for g in by.values() if len(g) >= 2]
        keep = sorted(i for g in self.groups for i in g)
        remap = {old: new for new, old in enumerate(keep)}
        self.charts = [self.charts[i] for i in keep]
        self.groups = [[remap[i] for i in g] for g in self.groups]

    def pair(self, rng) -> tuple:
        """Two chart indices: different variants of one randomly chosen rhythm."""
        g = self.groups[int(rng.integers(len(self.groups)))]
        a, b = rng.choice(len(g), 2, replace=False)
        return g[int(a)], g[int(b)]


class MixedDataset(Dataset):
    """Training items drawn from several datasets by share.

    `parts` maps a name to a `_ChartDataset`; `schedule` is a list of
    (first_epoch, {name: share}) in ascending epoch order, the last entry at
    or before the current epoch applying. Shares of parts that are absent or
    empty are redistributed over the rest. An "epoch" is `epoch_items` draws;
    item i of epoch e is a pure function of (seed, e, i), whatever the worker.
    """

    def __init__(self, parts: dict, schedule: list, epoch_items: int, seed: int = 0):
        self.parts = {k: v for k, v in parts.items() if len(v) > 0}
        if not self.parts:
            raise ValueError("no training data in any source")
        self.schedule = sorted(schedule, key=lambda e: e[0])
        self.epoch_items, self.seed = int(epoch_items), seed
        self.epoch = 0
        self._paired = any(getattr(ds, "groups", None) for ds in self.parts.values())
        self._cdf = {}
        for name, ds in self.parts.items():
            w = ds.weights
            self._cdf[name] = np.cumsum(w / w.sum())

    def shares(self, epoch: int | None = None) -> dict:
        """The shares in force at `epoch`, over the parts that exist."""
        epoch = self.epoch if epoch is None else epoch
        want = self.schedule[0][1]
        for first, sh in self.schedule:
            if epoch >= first:
                want = sh
        live = {k: float(v) for k, v in want.items() if k in self.parts and v > 0}
        if not live:
            raise ValueError(f"the schedule names no available source at epoch {epoch}")
        total = sum(live.values())
        return {k: v / total for k, v in live.items()}

    def __len__(self) -> int:
        return self.epoch_items

    def pick(self, i: int):
        """(part name, index in that part, rng) for item i of this epoch.

        With a grouped part present (`GroupedStageBDataset`), items are drawn
        in pairs (2k, 2k+1): the pair shares its source, and from a grouped
        source it is two variants of one rhythm under one random generator —
        same window, same hidden sliders. The loader must not shuffle and the
        batch size must be even, so the two land in the same batch."""
        sh = self.shares()
        names = list(sh)
        cum = np.cumsum([sh[k] for k in names])
        if self._paired:
            prng = np.random.default_rng((self.seed, self.epoch, i // 2, 1))
            name = names[int(np.searchsorted(cum, prng.random(), side="right")
                             .clip(0, len(names) - 1))]
            ds = self.parts[name]
            if getattr(ds, "groups", None):
                j = ds.pair(prng)[i % 2]
                return name, j, np.random.default_rng((self.seed, self.epoch, i // 2, 2))
            rng = np.random.default_rng((self.seed, self.epoch, i))
            rng.random()
        else:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            name = names[int(np.searchsorted(cum, rng.random(), side="right")
                             .clip(0, len(names) - 1))]
        cdf = self._cdf[name]
        j = int(min(np.searchsorted(cdf, rng.random(), side="right"), len(cdf) - 1))
        return name, j, rng

    def __getitem__(self, i: int) -> dict:
        name, j, rng = self.pick(i)
        ds = self.parts[name]
        ds.epoch = self.epoch
        return ds.item(j, rng)


def set_epoch(loader, epoch: int) -> None:
    """Re-seed a dataset's random crops for a new epoch (works through a
    DataLoader with persistent workers disabled)."""
    loader.dataset.epoch = epoch
