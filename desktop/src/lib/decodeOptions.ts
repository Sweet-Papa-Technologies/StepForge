/**
 * Decoder options: what the user asks of a chart beyond the style sliders.
 * The keys are the engine's own (`DecodeOptions` in the neural engine, sent as
 * `--options-json`), as with the style sliders. A key that is ABSENT means the
 * default — never store a placeholder, it would be sent.
 */
export interface DecodeOptions {
  /** No note finer than this: 4ths, 8ths or 16ths. Absent = the difficulty's own. */
  max_subdivision?: 4 | 8 | 16;
  /** Stream mode: unbroken runs at one subdivision with real breaks between. */
  stream?: boolean;
  /** Stream mode: share of the song that is stream (0–1). */
  stream_coverage?: number;
  /** Stream mode: a break is never shorter than this many bars. */
  min_break_bars?: number;
  /** false = no step may end with the feet crossed. */
  crossovers?: boolean;
  /** Bar ranges only: more or fewer notes than the model would write (1 = as is). */
  density?: number;
}

export const SUBDIVISIONS: { value: 4 | 8 | 16 | null; label: string }[] = [
  { value: null, label: "No limit" },
  { value: 4, label: "4ths" },
  { value: 8, label: "8ths" },
  { value: 16, label: "16ths" },
];
export const BREAK_BARS = [1, 2, 4, 8];
export const DEFAULT_STREAM_COVERAGE = 0.75;
export const DEFAULT_MIN_BREAK_BARS = 2;

/** Keep only what the engine accepts; anything else (an old settings file, a
 *  hand-edited one) is dropped rather than sent. */
export function cleanOptions(raw: unknown): DecodeOptions {
  const out: DecodeOptions = {};
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return out;
  const r = raw as Record<string, unknown>;
  if (r.max_subdivision === 4 || r.max_subdivision === 8 || r.max_subdivision === 16)
    out.max_subdivision = r.max_subdivision;
  if (r.stream === true) out.stream = true;
  if (typeof r.stream_coverage === "number" && r.stream_coverage > 0 && r.stream_coverage <= 1)
    out.stream_coverage = r.stream_coverage;
  if (typeof r.min_break_bars === "number" && Number.isInteger(r.min_break_bars) && r.min_break_bars >= 1 && r.min_break_bars <= 64)
    out.min_break_bars = r.min_break_bars;
  if (r.crossovers === false) out.crossovers = false;
  if (typeof r.density === "number" && r.density >= 0.25 && r.density <= 2) out.density = r.density;
  return out;
}

/** What to send: only what differs from the engine's defaults, and the Stream
 *  settings only with Stream mode on. Null when there is nothing to send. */
export function optionsJson(raw: DecodeOptions): string | null {
  const o = cleanOptions(raw);
  const out: DecodeOptions = {};
  if (o.max_subdivision) out.max_subdivision = o.max_subdivision;
  if (o.stream) {
    out.stream = true;
    out.stream_coverage = o.stream_coverage ?? DEFAULT_STREAM_COVERAGE;
    out.min_break_bars = o.min_break_bars ?? DEFAULT_MIN_BREAK_BARS;
  }
  if (o.crossovers === false) out.crossovers = false;
  if (o.density !== undefined && Math.abs(o.density - 1) > 1e-6) out.density = Number(o.density.toFixed(2));
  return Object.keys(out).length ? JSON.stringify(out) : null;
}

/** Notes per second of an unbroken stream at `subdivision` and `bpm`. */
export function streamRate(bpm: number, subdivision: 4 | 8 | 16): number {
  return (bpm / 60) * (subdivision / 4);
}
