"""grammar.py — dance-single note vocabulary, difficulty budgets, closed intent
vocabularies (STEPFORGE spec §4-6)."""
from __future__ import annotations

from dataclasses import dataclass

# dance-single panels, left->right in a .sm/.ssc row.
PANELS = ("L", "D", "U", "R")
PANEL_INDEX = {p: i for i, p in enumerate(PANELS)}
# Panel (x, y) unit positions for foot geometry: L=left, R=right, D=down, U=up.
PANEL_POS = {0: (-1.0, 0.0), 1: (0.0, -1.0), 2: (0.0, 1.0), 3: (1.0, 0.0)}

# Row tokens.
EMPTY, TAP, HOLD_HEAD, TAIL, ROLL_HEAD, MINE = "0", "1", "2", "3", "4", "M"
LEGAL_ROWS = (4, 8, 12, 16, 24, 32, 48, 64, 192)

# Closed intent vocabularies (REQ-SM-06) — anything else is a validation error.
TEXTURES = ("steps", "stream", "jumpstream", "drill", "runningman",
            "jacks_sparse", "stops_breather")
MOVEMENTS = ("static", "drift_L_to_R", "drift_R_to_L", "zigzag", "box")
CROSSOVERS = ("none", "light", "moderate")
KINDS = ("tap", "hold", "roll", "mine")


@dataclass(frozen=True)
class DiffBudget:
    name: str
    sm_difficulty: str          # StepMania DIFFICULTY name
    finest_subdiv: float        # beats
    max_nps_4s: float
    crossover: str              # none|light|moderate  (tolerance)
    footswitch: bool            # allowed?
    jack_limit: int
    jumps: str                  # "downbeats" | "accents" | "free"
    hold_share: tuple           # (lo, hi)
    hold_len_beats: tuple
    meter_range: tuple          # (lo, hi) clamp for METER
    max_run: int = 8            # longest allowed stream (consecutive fast notes)
    #: Max fraction of chart time with BOTH feet pinned by holds. See
    #: validate.cap_two_foot_time — the budget that actually tracks playability.
    max_two_foot_frac: float = 0.02
    #: Hard cap on holds per chart. A RATIO cannot bound this: the worst
    #: chart measured 38 holds at a share of 0.085, comfortably inside its
    #: 0.12 ceiling, because the chart was dense. Holds are counted, not
    #: just shared. See validate.cap_holds.
    max_holds: int = 12
    fine_frac: float = 0.6      # max share of notes off the quarter grid (8ths+)


# Spec §5 budget table (Easy/Medium/Hard are v1; Beginner/Challenge frame them).
# finest_subdiv + max_run encode the DIFFICULTY-as-rhythmic-complexity gradient
# learned from authored DDR (e.g. LOVE SHINE): easy is ~quarters with no streams,
# each tier adds a rhythmic layer, only the top tiers get 16ths and long streams.
#
# --- hold_share: MEASURED, and the floor is deliberately 0.0 ---
#
# These were drafted numbers. STEPMANIA.SPEC.MD's own header records that file as
# written from stated intent rather than from charts, and the two-sided band it
# proposed (10-20% / 5-15% / 5-12%) turned out to describe nothing. Scored over
# the 96 charts per difficulty in tools/chartbench/out/, it accepted 40-49% of
# them — it rejected four out of five DDC charts, and DDC is a model trained on
# thousands of human-authored charts (chartbench/BENCHMARK.md). A gate that fails
# most human-derived work is not measuring quality, it is measuring its own guess.
#
# THE FLOOR IS GONE, at every tier. It has no defensible intent: a chart of pure
# taps is ordinary DDR, and the reference steppers agree — DDC's median hold share
# at beginner is 0.000, and 35-54% of all reference charts sat under the old floor
# at every tier. "This chart could use more holds" is a real note, but it is a
# musical opinion and belongs to the critic, which hears the song. A hard gate
# that fires on half of all known-good charts only teaches people to ignore it.
#
# THE CEILING IS REAL and stays: a hold pins a foot for its duration, so a chart
# that is one-third holds stops being a dance chart.
#
# These ceilings were briefly RAISED to 0.30/0.30/0.25/0.22/0.20 and that was a
# mistake worth recording, because the reasoning looked rigorous. They were fitted
# to maximise the fraction of the chartbench corpus that PASSED — which lands at
# roughly reference p90, and the corpus's p90 is exactly the pathological tail
# this gate exists to catch. Optimising a ceiling for pass rate optimises it
# toward the worst charts you have. The charts that came out were hold-spam.
#
# The values below are back to the drafted originals, which turn out to sit right
# at the p75 of real charts measured per difficulty (0.178 / 0.186 / 0.097 /
# 0.082 / 0.096 across DDC, AutoStepper and Founders Mix #2, n=171). The drafted
# CEILINGS were sound all along; only the floor was invented. Calibrate a ceiling
# against the middle of good practice, not the edge of bad.
#
# `hold_share` alone does NOT capture playability, and a chart can pass it and
# still be miserable — see `max_two_foot_frac` and validate.cap_two_foot_time,
# which bound the thing a player actually feels.
BUDGETS = {
    "beginner": DiffBudget("beginner", "Beginner", 1.0, 2.0, "none", False, 0,
                           "downbeats", (0.0, 0.10), (2.0, 8.0), (1, 2), max_run=1, fine_frac=0.0, max_two_foot_frac=0.02, max_holds=8),
    "easy": DiffBudget("easy", "Easy", 0.5, 3.0, "none", False, 0,
                       "downbeats", (0.0, 0.10), (2.0, 8.0), (2, 4), max_run=2, fine_frac=0.15, max_two_foot_frac=0.03, max_holds=10),
    "medium": DiffBudget("medium", "Medium", 0.5, 5.0, "light", False, 2,
                         "accents", (0.0, 0.08), (1.0, 8.0), (5, 6), max_run=4, fine_frac=0.30, max_two_foot_frac=0.01, max_holds=12),
    "hard": DiffBudget("hard", "Hard", 0.25, 8.0, "moderate", True, 4,
                       "free", (0.0, 0.06), (1.0, 8.0), (7, 9), max_run=8, fine_frac=0.55, max_two_foot_frac=0.01, max_holds=12),
    "challenge": DiffBudget("challenge", "Challenge", 0.25, 10.0, "moderate", True, 5,
                            "free", (0.0, 0.06), (1.0, 8.0), (10, 12), max_run=16, fine_frac=0.70, max_two_foot_frac=0.01, max_holds=12),
}
DIFFICULTIES = ("easy", "medium", "hard")   # spec default set (§13)
ALL_DIFFICULTIES = ("beginner", "easy", "medium", "hard", "challenge")


def grammar_description() -> dict:
    return {
        "panels": list(PANELS),
        "note_types": {"tap": TAP, "hold_head": HOLD_HEAD, "tail": TAIL,
                       "roll_head": ROLL_HEAD, "mine": MINE},
        "legal_row_subdivisions": list(LEGAL_ROWS),
        "textures": list(TEXTURES), "movements": list(MOVEMENTS),
        "crossovers": list(CROSSOVERS), "kinds": list(KINDS),
        "difficulties": list(ALL_DIFFICULTIES),
    }
