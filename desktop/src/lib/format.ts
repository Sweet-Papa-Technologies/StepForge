export function fmtDuration(sec: number | null | undefined): string {
  if (sec == null || !isFinite(sec)) return "--:--";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function fmtBpm(bpm: number | null | undefined): string {
  if (bpm == null) return "—";
  return bpm.toFixed(2);
}

export function fmtSize(bytes: number): string {
  if (!bytes) return "—";
  const mb = bytes / (1024 * 1024);
  if (mb >= 1) return `${mb.toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

export function newJobId(prefix = "BF"): string {
  const n = Math.floor(performance.now() * 1000) % 100000;
  return `${prefix}-${n.toString().padStart(5, "0")}`;
}

/** Deterministic 0..1 pseudo-values from a string — for placeholder waveforms. */
export function hashBars(seed: string, count: number): number[] {
  let h = 2166136261;
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    h ^= seed.charCodeAt(i % seed.length) + i * 31;
    h = Math.imul(h, 16777619);
    out.push(0.15 + (Math.abs(h % 1000) / 1000) * 0.85);
  }
  return out;
}
