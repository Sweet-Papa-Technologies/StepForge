"""style_rag.py — optional phrase-level style retrieval (spec §8.1).

Slice audio into 4-bar phrases, embed each as a 256-D L2-normalised float16
vector, and keep beside it the style a human charter used over those four bars.
At generation time the phrases of a new song are embedded the same way and the
styles of their nearest neighbours fill in whatever sliders the user left
unset. Lookup is one numpy matrix product — no FAISS, no service.

The embedder is pluggable, and the index records which one built it:

  GridStatEmbedder   (default) pooled statistics of the beat-warped feature
                     tensor. No model, no network; it works from the training
                     shards alone, so an index can be built anywhere.
  EndpointEmbedder   an OpenAI-compatible /v1/embeddings endpoint that accepts
                     audio, e.g. EmbeddingGemma 2. The spec's intended encoder.

Everything here is None-safe: with no index on disk `load_index` returns None
and the engine simply does not use retrieval.
"""
from __future__ import annotations

import base64
import io
import json
import urllib.request
from pathlib import Path

import numpy as np

from . import codec as C
from . import grid_features as GF

DIM = 256
BARS = 4
PHRASE_TICKS = BARS * C.G
INDEX_FILE = "phrase_index.npz"
STYLE_KEYS = ("nps", "stream_share", "jump_share", "hold_share", "crossover_share",
              "jack_share", "footswitch_share", "mine_share")
SLOTS = ("beginner", "easy", "medium", "hard", "challenge", "edit")


def index_path(model_dir) -> Path:
    return Path(model_dir) / INDEX_FILE


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class GridStatEmbedder:
    """256-D phrase vector from the 31 stored feature channels: per channel,
    the mean and standard deviation over the phrase, plus the mean over each
    of its four bars; then the onset-flux profile folded onto one beat."""
    name = "gridstat-v1"

    def embed_core(self, core: np.ndarray) -> np.ndarray:
        """core (PHRASE_TICKS, 31) -> (256,) float32, L2-normalised."""
        c = np.asarray(core, np.float32)
        if len(c) < PHRASE_TICKS:
            c = np.pad(c, ((0, PHRASE_TICKS - len(c)), (0, 0)))
        c = c[:PHRASE_TICKS]
        bars = c.reshape(BARS, C.G, -1).mean(1)                      # (4, 31)
        beat = c[:, 20:26].sum(1).reshape(-1, C.TICKS_PER_BEAT).mean(0)   # (12,)
        v = np.concatenate([c.mean(0), c.std(0), bars.ravel(), beat])      # 198
        out = np.zeros(DIM, np.float32)
        out[:len(v)] = v
        return _unit(out)

    def embed_phrases(self, core: np.ndarray) -> np.ndarray:
        """Every whole 4-bar phrase of a song: (n_phrases, 256)."""
        n = len(core) // PHRASE_TICKS
        if n == 0:
            return np.zeros((0, DIM), np.float32)
        return np.stack([self.embed_core(core[i * PHRASE_TICKS:(i + 1) * PHRASE_TICKS])
                         for i in range(n)])


class EndpointEmbedder:
    """Audio phrases -> an OpenAI-compatible embeddings endpoint, truncated to
    256-D (Matryoshka) and re-normalised. Needs the endpoint to accept audio."""

    def __init__(self, base_url: str, model: str = "google/embeddinggemma-2",
                 api_key: str | None = None, timeout: float = 60.0):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.api_key, self.timeout = api_key, timeout
        self.name = f"endpoint:{model}"

    def embed_audio(self, y: np.ndarray, sr: int = GF.SR) -> np.ndarray:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.asarray(y, np.float32), sr, format="WAV", subtype="PCM_16")
        body = {"model": self.model, "dimensions": DIM, "encoding_format": "float",
                "input": [{"type": "input_audio", "input_audio": {
                    "data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}]}
        req = urllib.request.Request(
            self.base_url + "/embeddings", json.dumps(body).encode(),
            {"Content-Type": "application/json",
             **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            vec = np.asarray(json.loads(r.read())["data"][0]["embedding"], np.float32)
        return _unit(vec[:DIM])

    def hears_audio(self) -> bool:
        """Differential probe: two clips that sound nothing alike must not
        embed to the same vector.

        An endpoint can accept an audio part and silently ignore it — measured
        on the lab's EmbeddingGemma 2 service (2026-11-13), which runs with its
        audio encoder off: a sine, white noise, silence and two songs all came
        back at cosine 1.000. An index built through such an endpoint would be
        one point repeated. Call this before trusting it.
        """
        t = np.arange(GF.SR * 2) / GF.SR
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        noise = (0.2 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)
        return float(self.embed_audio(tone) @ self.embed_audio(noise)) < 0.98

    def embed_phrases_audio(self, y: np.ndarray, phrase_times: np.ndarray) -> np.ndarray:
        """`phrase_times` is (n + 1,) phrase boundaries in seconds."""
        out = []
        for a, b in zip(phrase_times[:-1], phrase_times[1:]):
            out.append(self.embed_audio(y[int(a * GF.SR):int(b * GF.SR)]))
        return np.stack(out) if out else np.zeros((0, DIM), np.float32)


def phrase_styles(tick, tok, flav, times_ms, n_phrases: int) -> np.ndarray:
    """The local style of one chart over each 4-bar phrase: (n_phrases, 8) in
    STYLE_KEYS order, NaN where the phrase has no notes."""
    tick = np.asarray(tick, np.int64)
    tok = np.asarray(tok, np.int64)
    out = np.full((n_phrases, len(STYLE_KEYS)), np.nan, np.float32)
    ph = tick // PHRASE_TICKS
    single = tok < 8
    for i in range(n_phrases):
        m = ph == i
        n = int(m.sum())
        if n == 0:
            continue
        t = np.asarray(times_ms)[m] / 1000.0
        span = max(1.0, float(t[-1] - t[0]) + 0.25)
        ns = max(1, int((m & single).sum()))
        gaps = np.diff(tick[m])
        out[i] = (min(n / span, 16.0) / 16.0,
                  float((gaps <= 3).mean()) if len(gaps) else 0.0,
                  float((tok[m] >= 8).mean()), float((np.asarray(flav)[m] > 0).mean()),
                  float((single & ((tok & 3) == C.CROSS))[m].sum() / ns),
                  0.0, 0.0, 0.0)
    return out


class StyleIndex:
    def __init__(self, vectors, styles, slot, embedder: str):
        self.vectors = np.asarray(vectors, np.float32)
        self.styles = np.asarray(styles, np.float32)
        self.slot = np.asarray(slot, np.int8)
        self.embedder = embedder

    def __len__(self) -> int:
        return len(self.vectors)

    def query(self, vec: np.ndarray, difficulty: str | None = None, k: int = 5) -> dict:
        """Top-k phrases by cosine similarity -> {'style': mean (8,), 'std': (8,),
        'scores': (k,), 'index': (k,)}; None if the index has no usable row."""
        mask = np.ones(len(self), bool)
        if difficulty in SLOTS:
            mask = self.slot == SLOTS.index(difficulty)
        if not mask.any():
            return None
        scores = self.vectors @ np.asarray(vec, np.float32)        # one matmul
        scores = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        st = self.styles[top]
        return {"style": np.nanmean(st, 0), "std": np.nanstd(st, 0),
                "scores": scores[top], "index": top}

    def song_style(self, phrase_vecs: np.ndarray, difficulty: str, k: int = 5) -> dict:
        """Average the retrieved style over every phrase of a song ->
        {slider key: value}, the overrides `conditioning.resolve_style` takes."""
        rows = [r["style"] for v in phrase_vecs
                if (r := self.query(v, difficulty, k)) is not None]
        if not rows:
            return {}
        mean = np.nanmean(np.stack(rows), 0)
        out = {key: float(val) for key, val in zip(STYLE_KEYS, mean) if np.isfinite(val)}
        if "nps" in out:
            out["nps"] *= 16.0
        return out

    def save(self, path) -> None:
        np.savez(path, vectors=self.vectors.astype(np.float16),
                 styles=self.styles.astype(np.float16), slot=self.slot,
                 embedder=np.frombuffer(self.embedder.encode(), np.uint8))


def load_index(model_dir) -> StyleIndex | None:
    p = index_path(model_dir)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return StyleIndex(z["vectors"], z["styles"], z["slot"],
                          bytes(z["embedder"]).decode())


def build_index(corpus_dir, out_path, split: str = "train", max_phrases: int = 50000,
                seed: int = 0, log=print) -> StyleIndex:
    """Build a phrase index from compiled training shards with the built-in
    embedder (train split only, so retrieval can never leak a held-out chart)."""
    corpus_dir = Path(corpus_dir)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    emb = GridStatEmbedder()
    vecs, styles, slots = [], [], []
    for s in manifest["songs"]:
        if s["split"] != split or not any(c["keep"] for c in s["charts"]):
            continue
        with np.load(corpus_dir / "songs" / f"{s['key']}.npz", allow_pickle=False) as z:
            pv = emb.embed_phrases(z["core"])
            for c in s["charts"]:
                if not c["keep"] or c["difficulty"] not in SLOTS:
                    continue
                j = c["j"]
                st = phrase_styles(z[f"c{j}_tick"], z[f"c{j}_tok"], z[f"c{j}_flav"],
                                   z[f"c{j}_tms"], len(pv))
                ok = np.isfinite(st[:, 0])
                vecs.append(pv[ok])
                styles.append(st[ok])
                slots.append(np.full(int(ok.sum()), SLOTS.index(c["difficulty"]), np.int8))
    v, st, sl = np.concatenate(vecs), np.concatenate(styles), np.concatenate(slots)
    if len(v) > max_phrases:
        pick = np.sort(np.random.default_rng(seed).choice(len(v), max_phrases, False))
        v, st, sl = v[pick], st[pick], sl[pick]
    idx = StyleIndex(v, st, sl, emb.name)
    idx.save(out_path)
    log(f"[style-rag] {len(idx)} phrases -> {out_path}")
    return idx


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build the phrase style index.")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="model dir (writes phrase_index.npz)")
    ap.add_argument("--max-phrases", type=int, default=50000)
    a = ap.parse_args(argv)
    build_index(a.corpus, index_path(a.out), max_phrases=a.max_phrases)


if __name__ == "__main__":
    main()
