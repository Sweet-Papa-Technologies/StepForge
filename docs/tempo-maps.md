# Tempo changes and mid-song pauses

Up to 0.1.0-beta.11 StepForge charted every song on one tempo. That is right
for most music and wrong for some of it: a band that stops and comes back in,
a bar cut short, a tempo change, a drummer playing without a click. On those
songs the chart was in time until that moment and off for the rest of the
song, in the built-in player and in the game alike.

From the next release the engine checks, for every song, that its one tempo
holds from the first bar to the last. Where it does not, the song gets a tempo
map and the simfile carries it:

| What happens in the music | What the simfile gets |
| --- | --- |
| A pause: silence, then the music comes back in off the old grid | a `#STOPS` entry at the pause |
| A bar cut short or stretched, with no silence | a short `#BPMS` segment that carries the beat count across |
| A tempo change | a `#BPMS` entry for the new tempo |
| A pause after which the music comes back in on another 16th-note line: every note still on the grid, the beat no longer where it was | a `#STOPS` entry that puts the beat back |

**Songs that hold one tempo are not touched.** For them the engine writes the
same single `#BPMS` entry, the same notes and the same files as before, byte
for byte. On the 180 benchmark songs, each charted by a person at one tempo,
all 180 are left alone, and the check costs about 3 ms a song.

## How the check works

The song is cut into windows of eight bars. In each, the note onsets are
folded onto the 16th-note circle of the grid and the resulting shape is slid
against the song's own typical shape until the two match; how far it had to
slide is that window's phase error. It is the whole shape that is matched and
not its tallest peak, so a swung, triplet or syncopated section reads as on
the grid as long as the rest of the song is shaped the same way.

A stretch is given a grid of its own only when all of these hold:

- the single tempo is visibly off it for several windows running (more than
  18 ms, or unreadable), not merely a little loose;
- the stretch holds a grid of its own for long enough: about 14 bars, or about
  10 next to a pause;
- the step between the two grids is clear: 30 ms or more, 12 ms across a pause;
- for another tempo: the stretch's 8th notes sit on the new grid clearly better
  than on the old tempo at its best seat, and the music repeats at the new
  beat more than at the old one. A cross-rhythm over an unchanged beat fails
  that last test.

Which of a grid's 16th-note lines is the beat is not something onsets can
say. Each stretch first takes the seat that continues its neighbour's, and
then the rhythm model is asked, stretch by stretch, exactly as it is already
asked once per song.

### After a pause, on a song that keeps its tempo

A band that stops for five 8th notes comes back in on a line of the old grid,
so the check above finds one tempo and nothing wrong with it. But the beat is
now on what used to be an off-beat, and an easy chart would step between the
notes of the music from there to the end. Only the rhythm model can see that,
and it is asked in two steps:

1. **A screen that costs nothing.** The model has already been run on the
   song's grid to seat the beat. Asked for a Beginner chart, it puts its steps
   on the quarter notes where the music's beats are under them. If, on either
   side of a pause, some other 16th-note position holds more than a quarter of
   what the quarter notes hold, that side may be seated wrong. Otherwise the
   pause is left alone and nothing more runs.
2. **The question.** The model is run on the grid moved one, two and three
   16th notes later, and each side of the pause picks the grid it fits best.
   If both sides are sure and they disagree, the song gets a stop at the pause
   that puts the beat back.

Step 2 costs about a second of song preparation (three more runs of the rhythm
network) and only runs when step 1 says so. On the 180 benchmark songs, 32 of
which have a mid-song pause, step 1 never does, so they are prepared in the
same time as before. The question is also only put to songs whose one tempo is
locked from end to end: on a grid that drifts, which line is the beat changes
along the song for a reason no stop can mend.

## How well it does

**Spliced songs, where the answer is known.** Copies of 60 benchmark songs
were spliced three ways, so the true beat times before and after the splice
are known, and run through the whole engine. "In time" means at least 90% of
the true beats are within 20 ms of a 16th-note line of the chart's grid, both
before and after the splice.

| Splice | One tempo (before) | Tempo map |
| --- | --: | --: |
| A 0.75 s silent pause half-way through | 43 of 60 | 55 of 60 |
| Two beats cut out (a 2/4 turnaround bar) | 59 of 60 | 59 of 60 |
| Tempo jumps by 15 BPM | 0 of 60 | 49 of 60 |

The pause row starts at 43 because a 0.75 s pause happens to land close to a
16th-note line at many tempos; the 17 songs where it does not are the ones
that used to go out of time, and 12 of them are now fixed. Cutting two whole
beats leaves the beat grid locked, so there is nothing to fix: the bar lines
after the cut sit two beats off, which StepMania has no plain way to write and
which does not affect timing.

Being in time is not the same as having the beat on the right 16th-note line.
After a tempo jump the beat is seated correctly on 38 of the 60 songs. After a
pause it is on 30 of 60: a pause that lasts a whole number of 16th notes
leaves nothing in the timing to see, and the rhythm model, asked on both sides
of the pause, is sure enough to move the beat on 6 of those songs. All 6 are
moved to the right line, and none of the other 54 gets worse.

**Real songs.** On 20 songs that step artists charted with tempo changes or
stops, three get a tempo map. On those three the share of the artist's beats
within 20 ms of the chart's grid goes from 47%, 39% and 61% to 91%, 87% and
66%. The other 17 are left on one tempo, and are as close to the artist's
beats as before (10 of them were already at 100%: many stops and BPM changes
in hand-made charts are effects, and the music under them never leaves the
grid).

**Songs with one tempo.** All 180 benchmark songs are left alone: the 360
simfiles the engine writes for them are byte-identical to the ones 0.1.0-beta.11
writes, so the benchmark score is unchanged (0.689).

## What it does not catch yet

- **A pause or cut of exactly a 16th or 8th note.** The notes after it are
  still on the grid, so nothing is out of time, but the beat may be seated on
  what is now an off-beat, which matters most on Beginner and Easy. After a
  pause the rhythm model is asked and puts the beat back when it is sure. It
  is deliberately cautious, because a wrong answer moves a beat that was
  right: on the 60 spliced songs it acted 6 times, and 25 more ended with the
  notes in time and the beat still on the wrong line. A cut with no pause is
  not looked at.
- **Small steps.** A step under 30 ms without a pause is left alone: songs
  charted by people at one tempo show steps that size between sections, from
  one instrument's attack to another's.
- **A wrong first guess.** Everything starts from the song's single tempo. If
  that estimate is itself off by a simple ratio, the sections are still found,
  but less reliably.
- **Free time.** A drummer drifting a little is followed in stretches of 14
  bars or more; a tempo that never settles is not.
- **Stream speed limits** use the song's main tempo, so a faster section can
  run slightly over them.

The detector is `tools/beatforge/adapters/stepmania/neural/tempo_map.py`; the
beat-to-seconds arithmetic every part of the engine now shares is
`tools/beatforge/adapters/stepmania/timing.py`.
