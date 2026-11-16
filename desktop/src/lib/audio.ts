import { convertFileSrc } from "@tauri-apps/api/core";

export interface Peaks {
  /** Per-bucket max absolute amplitude, 0..1. */
  data: Float32Array;
  duration: number;
}

const cache = new Map<string, Peaks>();

/**
 * Decode an audio file and reduce it to `buckets` peak amplitudes so the
 * waveform reflects the real track rather than a placeholder.
 */
export async function loadPeaks(filePath: string, buckets = 1200): Promise<Peaks> {
  const key = `${filePath}#${buckets}`;
  const hit = cache.get(key);
  if (hit) return hit;

  const res = await fetch(convertFileSrc(filePath));
  const bytes = await res.arrayBuffer();
  const Ctx: typeof AudioContext =
    (window as any).AudioContext || (window as any).webkitAudioContext;
  const ctx = new Ctx();
  try {
    const buf = await ctx.decodeAudioData(bytes);
    const ch = buf.getChannelData(0);
    const per = Math.max(1, Math.floor(ch.length / buckets));
    const data = new Float32Array(buckets);
    let max = 0;
    for (let i = 0; i < buckets; i++) {
      const start = i * per;
      const end = Math.min(ch.length, start + per);
      let peak = 0;
      for (let j = start; j < end; j++) {
        const v = ch[j] < 0 ? -ch[j] : ch[j];
        if (v > peak) peak = v;
      }
      data[i] = peak;
      if (peak > max) max = peak;
    }
    if (max > 0) for (let i = 0; i < buckets; i++) data[i] /= max;
    const peaks: Peaks = { data, duration: buf.duration };
    cache.set(key, peaks);
    return peaks;
  } finally {
    ctx.close();
  }
}
