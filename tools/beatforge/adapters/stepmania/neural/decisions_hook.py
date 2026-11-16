"""decisions_hook.py — optional `/v1/decisions` scorer (spec §8.2).

A Decisions endpoint answers typed questions about an input in one prefill
pass, with no generated tokens. Two uses, both off unless configured:

  * offline corpus filter — score a community chart's pad musicality 0..10 so
    junk can be excluded before training (`score_chart`)
  * macro planner — pick a style preset for a song when the user gave none
    (`choose_preset`)

Disabled by default and None-safe: `DecisionsClient.from_env()` returns None
unless BEATFORGE_DECISIONS_URL is set, and every caller treats None, a network
failure and a malformed reply the same way — as "no opinion".
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

from . import conditioning as cond

ENV_URL = "BEATFORGE_DECISIONS_URL"
ENV_KEY = "BEATFORGE_DECISIONS_API_KEY"
ENV_MODEL = "BEATFORGE_DECISIONS_MODEL"


class DecisionsClient:
    def __init__(self, base_url: str, model: str = "decisions", api_key: str | None = None,
                 timeout: float = 30.0, transport=None):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.api_key, self.timeout = api_key, timeout
        self._transport = transport or self._http

    @classmethod
    def from_env(cls) -> "DecisionsClient | None":
        url = os.environ.get(ENV_URL, "").strip()
        if not url:
            return None
        return cls(url, os.environ.get(ENV_MODEL, "decisions"),
                   os.environ.get(ENV_KEY) or None)

    def _http(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + "/decisions", json.dumps(body).encode(),
            {"Content-Type": "application/json",
             **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def decide(self, questions: list, *, text: str = "", audio_wav: bytes | None = None
               ) -> dict | None:
        """Ask typed questions in one call. Each question is
        {"id", "type": "choice"|"score", "prompt", "choices"|"min"/"max"}.
        Returns {id: answer}, or None if the endpoint could not be used."""
        content = []
        if audio_wav:
            content.append({"type": "input_audio", "input_audio": {
                "data": base64.b64encode(audio_wav).decode(), "format": "wav"}})
        if text:
            content.append({"type": "text", "text": text})
        try:
            reply = self._transport({"model": self.model, "input": content,
                                     "decisions": questions, "max_output_tokens": 0})
            got = {d["id"]: d.get("value", d.get("choice")) for d in reply["decisions"]}
        except Exception:
            return None
        out = {}
        for q in questions:
            v = got.get(q["id"])
            if q["type"] == "choice":
                out[q["id"]] = v if v in q["choices"] else None
            else:
                try:
                    out[q["id"]] = min(q.get("max", 10), max(q.get("min", 0), float(v)))
                except (TypeError, ValueError):
                    out[q["id"]] = None
        return out


def score_chart(client: DecisionsClient | None, summary: str,
                audio_wav: bytes | None = None) -> float | None:
    """Pad musicality 0..10 for a chart described by `summary`; None = no opinion."""
    if client is None:
        return None
    res = client.decide([{"id": "musicality", "type": "score", "min": 0, "max": 10,
                          "prompt": "How musical and playable on a dance pad is this "
                                    "StepMania chart for the given audio? 0 = unplayable "
                                    "keyboard dump, 10 = an excellent pad chart."}],
                        text=summary, audio_wav=audio_wav)
    return None if res is None else res.get("musicality")


def choose_preset(client: DecisionsClient | None, description: str,
                  audio_wav: bytes | None = None) -> str | None:
    """A style preset for a song, or None to leave the sliders to the model."""
    if client is None:
        return None
    res = client.decide([{"id": "preset", "type": "choice",
                          "choices": list(cond.PRESET_NAMES),
                          "prompt": "Which charting style suits this song best?"}],
                        text=description, audio_wav=audio_wav)
    return None if res is None else res.get("preset")
