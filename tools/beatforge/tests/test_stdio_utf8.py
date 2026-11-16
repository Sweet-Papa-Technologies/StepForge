"""The engine writes UTF-8 to stdout/stderr on every platform (Windows Python
defaults piped output to cp1252, which turned "…" into the non-UTF-8 byte 0x85
and stopped the desktop app's log reader mid-run)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2]


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(TOOLS), PYTHONIOENCODING="cp1252")
    env.pop("PYTHONUTF8", None)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, env=env)


def test_progress_text_comes_out_as_utf8_even_when_the_platform_default_is_cp1252():
    res = _run("from beatforge.__main__ import utf8_stdio; utf8_stdio(); "
               "print('grid\\u2026 \\u6d88\\u3048\\u3066\\u3086\\u304f'); "
               "import sys; print('err\\u2014', file=sys.stderr)")
    assert res.returncode == 0, res.stderr
    assert res.stdout.decode("utf-8").splitlines() == ["grid… 消えてゆく"]
    assert res.stderr.decode("utf-8").splitlines() == ["err—"]


def test_without_it_cp1252_output_is_not_utf8():
    """The failure this guards against, so the test above means something."""
    res = _run("print('grid\\u2026')")
    assert res.stdout.rstrip(b"\r\n") == b"grid\x85"


def test_unencodable_text_is_replaced_not_fatal():
    res = _run("from beatforge.__main__ import utf8_stdio; utf8_stdio(); print('a\\udc80b')")
    assert res.returncode == 0, res.stderr
    assert res.stdout.decode("utf-8").strip() == "a?b"
