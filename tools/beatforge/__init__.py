"""BEATFORGE — music-aware track & chart pipeline for OVERDRIVE.

Deterministic-first: all timing truth (tempo, beat grid, onsets, offsets) comes
from DSP (Workstream B) and is cached as JSON. Gemini Flash (Workstream C) hears
the audio and designs charts by *selecting DSP-provided candidates by ID* — it
never emits a raw timestamp. See IMPROVE.MD for the full spec.
"""

__version__ = "1.0.0"

# --------------------------------------------------------------------------- #
# Third-party import noise
#
# `simfile` (the .sm/.ssc library, and the only way we read or write StepMania
# files) requires `fs ~=2.4.15`, and fs 2.4.16 — its final release — calls
# `pkg_resources.declare_namespace()` at import. Modern setuptools warns twice
# about that, on stderr, before the CLI has printed a single line of its own.
#
# There is nothing to fix upstream and nothing to upgrade past: simfile 2.1.1 is
# the latest and still pins fs, fs 2.4.16 is the last pyfilesystem2 release, and
# the warning's own advice ("pin to Setuptools<81") does not apply — it fires on
# 80.10.2, which is what the app venv already has. So the choice is to silence it
# here or to keep printing a warning about someone else's deprecated call that no
# user of this tool can act on.
#
# Matched by MESSAGE, deliberately narrow: a module- or category-wide filter here
# would also swallow the next real deprecation warning out of our own code. Both
# lines are needed — the same call site raises a UserWarning and a
# DeprecationWarning. This must run before anything imports simfile, which it
# does: every importer of simfile is under this package.
# --------------------------------------------------------------------------- #
import warnings as _warnings

_warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API")
_warnings.filterwarnings(
    "ignore", message=r"Deprecated call to `pkg_resources\.declare_namespace")
del _warnings
