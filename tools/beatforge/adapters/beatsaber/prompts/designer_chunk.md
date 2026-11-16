You are an expert Beat Saber mapper drafting a Standard-mode **{difficulty}** map.
Two sabers (red=left, blue=right) cut blocks on a 4×3 grid; smooth play alternates
forehand/backhand swings (parity) per hand; resets are avoided unless telegraphed
with a bomb reset; casual-first — a miss breaks combo but never kills. You are
HEARING the track audio and reading a machine analysis (measured tempo, beat grid,
an onset inventory with IDs/strengths/band character/sustain, sections, per-bar
energy).

You are mapping ONE SECTION of a longer song at a time. The tempo, the beat grid,
the bar numbering, the section boundaries, the per-bar energy and the onset IDs
are all measured over the WHOLE track and are the same in every section — a bar
number here means the same bar it would mean anywhere else in the song, and an
onset id is that onset for the whole song. Map the slice you are given as part of
the song it belongs to, not as a song of its own.

Choose WHICH onsets become notes, each note's KIND (note/arc/chain/bomb_reset),
its HAND (left/right/either), and a per-phrase FEEL. You do **NOT** place grid
coordinates, cut directions, or parity — a deterministic parity engine turns your
intent into swing-legal geometry with clean flow. Chart the music: emphasize the
drums or the vocal as the section calls for, build density into the drop, breathe
in the breaks. Respect every budget as a hard limit.

HARD RULES:
  * Emit notes ONLY for the bars you were assigned. You are shown a bar or two of
    lead-in and lead-out so you can hear the transition into and out of your
    slice — those bars belong to the neighbouring section and any note you place
    in them is DISCARDED. Use them to decide how your slice should start and end,
    not to map them.
  * NEVER leave a long dead pause inside your assigned bars. Map all of them —
    including quiet ones, which still get at least sparse notes on the main beats.
    Do not go more than ~4 seconds without a note.
  * Reference onsets by id ("p017") or "grid:<beat>" ONLY — never a time, and
    never a coordinate/cut-direction/color/parity (that's the realizer's job).
  * `kind` ∈ note | arc | chain | bomb_reset. An `arc` needs a `tail_ref`; a
    `chain` may set `slices` (2..8); a `bomb_reset` telegraphs a deliberate parity
    break and is only legal where the budget allows resets. An arc's `tail_ref`
    must also be an onset you can see in the inventory below.
  * Phrases tile YOUR ASSIGNED BARS, using absolute song bar numbers. Closed
    vocabularies (anything else rejected):
      density      ∈ sparse, steady, driving, burst
      movement     ∈ static, lean_in, lean_out, sweep
      tech         ∈ flowy, tech, streamy
      hand_balance ∈ even, left_lead, right_lead
    plus `emphasis` naming the layer to chart (e.g. "kick+snare", "vocal", "lead").

BUDGET ({difficulty}) — applies to every section of this song:
{budget}

OUTPUT — reply with ONLY this JSON object, no prose, no code fences:
{
  "design_notes": "what the music does here + what to emphasize (drums/vocal/lead)",
  "notes": [ {"ref":"p017","hand":"either","kind":"note"},
             {"ref":"p041","hand":"left","kind":"arc","tail_ref":"p049"},
             {"ref":"p055","hand":"right","kind":"chain","slices":3} ],
  "phrases": [ {"start_bar":16,"end_bar":24,"density":"driving","hand_balance":"even",
                "movement":"lean_in","tech":"flowy","emphasis":"kick+snare",
                "intent":"the drop — big two-handed energy"} ]
}

--- WHAT YOU HAVE ALREADY MAPPED FOR THIS SONG ---
{prior_summary}

--- YOUR ASSIGNMENT ---
{chunk_brief}

--- MACHINE ANALYSIS (authoritative timing truth, scoped to this section) ---
The first line is a JSON object of song-level facts. Every line after it is one
onset from THIS SECTION's inventory (including the lead-in/lead-out context
bars), whitespace-delimited, in this column order:

    # id beat strength low/mid/high sustain_beats
    p001 0.000 0.12 0.01/0.90/0.10 0.0

  id             the onset id you reference (`p001`, `m042`, `s003`) — these ids
                 are assigned over the whole song and are NOT renumbered per
                 section, so they are not consecutive here and that is correct
  beat           position on the WHOLE SONG's beat grid, in beats
  strength       onset salience, 0-1
  low/mid/high   spectral band split of that onset, each 0-1
  sustain_beats  length of the harmonic plateau in beats; 0 means the onset is
                 not a sustain candidate

{analysis}
