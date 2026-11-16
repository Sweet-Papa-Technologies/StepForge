<script setup lang="ts">
/**
 * The signature visualization: the real decoded waveform with the engine's
 * measured beat grid, onset candidates and structural sections drawn over it.
 * Everything here comes from `build/analysis/<base>.analysis.json` — nothing is
 * illustrative.
 *
 * Performance notes (this used to be the app's worst hot spot):
 *  - the expensive layers (waveform + hundreds of onsets) are rendered ONCE into
 *    an offscreen canvas and blitted; only the playhead is redrawn per frame;
 *  - the playhead repaint is coalesced through requestAnimationFrame;
 *  - `peaks` is a shallowRef so Vue never proxies the Float32Array, and the
 *    analysis object is watched shallowly — deep-watching it meant traversing
 *    every onset on every tick.
 */
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import { loadPeaks } from "../lib/audio";
import type { Analysis } from "../lib/types";

const props = defineProps<{
  audioPath: string | null;
  analysis: Analysis | null;
  playhead?: number; // seconds
}>();
const emit = defineEmits<{ (e: "seek", seconds: number): void }>();

const canvas = ref<HTMLCanvasElement | null>(null);
const wrap = ref<HTMLDivElement | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);
const peaks = shallowRef<Float32Array | null>(null);
const audioDuration = ref(0);

let layer: HTMLCanvasElement | null = null; // cached static layer
let cssW = 0;
let cssH = 0;
let rafId = 0;
let ro: ResizeObserver | null = null;

const duration = computed(() => props.analysis?.duration_s || audioDuration.value || 0);

const SECTION_COLORS = [
  "rgba(139,92,246,0.30)",
  "rgba(16,185,129,0.30)",
  "rgba(59,130,246,0.30)",
  "rgba(245,158,11,0.30)",
];

/** Colour by structural role, falling back to a rotating palette. */
function sectionColor(name: string, i: number): string {
  const l = name.toLowerCase();
  if (l.includes("chorus") || l.includes("drop")) return "rgba(244,63,94,0.32)";
  if (l.includes("bridge")) return "rgba(245,158,11,0.32)";
  if (l.includes("intro") || l.includes("outro")) return "rgba(59,130,246,0.32)";
  if (l.includes("verse")) return "rgba(16,185,129,0.32)";
  return SECTION_COLORS[i % SECTION_COLORS.length];
}

/** Sections are bar-indexed; convert to seconds via bpm (4 beats per bar).
 *  `role_guess` carries the engine's structural label (intro/verse/chorus/…). */
function sectionsOf(a: Analysis | null) {
  const secs = a?.sections ?? [];
  const bpm = a?.bpm ?? 0;
  const offset = a?.offset ?? 0;
  if (!secs.length || !bpm) return [];
  const barSec = (4 * 60) / bpm;
  return secs.map((s: any, i: number) => ({
    name: (s.role_guess || s.name || `S${i}`).toString(),
    start: offset + (s.start_bar ?? 0) * barSec,
    end: offset + (s.end_bar ?? 0) * barSec,
  }));
}

async function load() {
  peaks.value = null;
  error.value = null;
  audioDuration.value = 0;
  if (!props.audioPath) {
    renderStatic();
    return;
  }
  const requested = props.audioPath;
  loading.value = true;
  try {
    const p = await loadPeaks(requested, 1400);
    if (props.audioPath !== requested) return; // selection moved on
    peaks.value = p.data;
    audioDuration.value = p.duration;
  } catch (e) {
    if (props.audioPath === requested) error.value = `Could not decode audio: ${e}`;
  } finally {
    if (props.audioPath === requested) loading.value = false;
    renderStatic();
  }
}

/** Redraw the expensive layers into the offscreen cache. */
function renderStatic() {
  const box = wrap.value;
  const view = canvas.value;
  if (!box || !view) return;
  const w = box.clientWidth;
  const h = box.clientHeight;
  if (!w || !h) return;

  const dpr = window.devicePixelRatio || 1;
  cssW = w;
  cssH = h;
  view.width = w * dpr;
  view.height = h * dpr;
  view.style.width = `${w}px`;
  view.style.height = `${h}px`;

  if (!layer) layer = document.createElement("canvas");
  layer.width = w * dpr;
  layer.height = h * dpr;

  const ctx = layer.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const a = props.analysis;
  const dur = duration.value;
  const xOf = (t: number) => (dur ? (t / dur) * w : 0);

  // --- section bands ---
  const secs = sectionsOf(a);
  ctx.font = "10px 'JetBrains Mono Variable', monospace";
  secs.forEach((s, i) => {
    const x0 = xOf(s.start);
    const w0 = Math.max(1, xOf(s.end) - x0);
    ctx.fillStyle = sectionColor(s.name, i);
    ctx.fillRect(x0, 0, w0, 18);
    if (w0 > 34) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, 0, w0, 18);
      ctx.clip();
      ctx.fillStyle = "#e4e1e6";
      ctx.fillText(s.name, x0 + 4, 13);
      ctx.restore();
    }
  });

  const top = secs.length ? 22 : 4;
  const mid = top + (h - top) / 2;
  const half = (h - top) / 2 - 4;

  // --- bar lines ---
  const bpm = a?.bpm ?? 0;
  if (bpm > 0 && dur) {
    const step = (60 / bpm) * 4;
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let t = a?.offset ?? 0; t <= dur; t += step) {
      const x = Math.round(xOf(t)) + 0.5;
      ctx.moveTo(x, top);
      ctx.lineTo(x, h);
    }
    ctx.stroke();
  }

  // --- waveform (single path) ---
  const p = peaks.value;
  if (p && p.length && half > 0) {
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, "#bc13fe");
    grad.addColorStop(1, "#00eefc");
    ctx.strokeStyle = grad;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const stride = Math.max(1, Math.ceil(p.length / Math.max(1, w)));
    for (let i = 0; i < p.length; i += stride) {
      let amp = 0;
      for (let j = i; j < Math.min(p.length, i + stride); j++) if (p[j] > amp) amp = p[j];
      const x = (i / p.length) * w;
      const y = amp * half;
      ctx.moveTo(x, mid - y);
      ctx.lineTo(x, mid + y);
    }
    ctx.stroke();
  }

  // --- onset markers (batched into two paths, not one stroke each) ---
  const onsets = a?.onsets;
  if (onsets && onsets.length && dur) {
    for (const sustain of [false, true]) {
      ctx.strokeStyle = sustain ? "rgba(255,185,96,0.85)" : "rgba(235,178,255,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      let drew = false;
      for (const o of onsets) {
        if (!!o.sustain !== sustain) continue;
        const x = Math.round(xOf(o.time)) + 0.5;
        ctx.moveTo(x, h - 14);
        ctx.lineTo(x, h);
        drew = true;
      }
      if (drew) ctx.stroke();
    }
  }

  paint();
}

/** Blit the cached layer and draw only the playhead. */
function paint() {
  const view = canvas.value;
  if (!view || !layer) return;
  const ctx = view.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, view.width, view.height);
  ctx.drawImage(layer, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const dur = duration.value;
  if (props.playhead != null && dur > 0) {
    const x = Math.round((props.playhead / dur) * cssW) + 0.5;
    ctx.strokeStyle = "#00eefc";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, cssH);
    ctx.stroke();
  }
}

function schedulePaint() {
  if (rafId) return;
  rafId = requestAnimationFrame(() => {
    rafId = 0;
    paint();
  });
}

function onClick(e: MouseEvent) {
  const box = wrap.value;
  if (!box || !duration.value) return;
  const r = box.getBoundingClientRect();
  emit("seek", ((e.clientX - r.left) / r.width) * duration.value);
}

// Shallow watches only — deep-watching `analysis` traverses every onset.
watch(() => props.audioPath, load);
watch(() => props.analysis, renderStatic);
watch(() => props.playhead, schedulePaint);

onMounted(() => {
  load();
  let lastW = 0;
  let lastH = 0;
  ro = new ResizeObserver(() => {
    const box = wrap.value;
    if (!box) return;
    // Ignore no-op observations; only a real size change needs a re-render.
    if (box.clientWidth === lastW && box.clientHeight === lastH) return;
    lastW = box.clientWidth;
    lastH = box.clientHeight;
    renderStatic();
  });
  if (wrap.value) ro.observe(wrap.value);
});

onBeforeUnmount(() => {
  if (rafId) cancelAnimationFrame(rafId);
  ro?.disconnect();
  layer = null;
});
</script>

<template>
  <div ref="wrap" class="relative w-full h-full cursor-crosshair" @click="onClick">
    <canvas ref="canvas" class="absolute inset-0"></canvas>
    <div
      v-if="loading"
      class="absolute inset-0 flex items-center justify-center font-label-mono text-label-mono text-on-surface-variant"
    >
      Decoding audio…
    </div>
    <div
      v-else-if="!audioPath"
      class="absolute inset-0 flex items-center justify-center font-label-mono text-label-mono text-on-surface-variant"
    >
      No .ogg audio for this track
    </div>
    <div
      v-else-if="error"
      class="absolute inset-0 flex items-center justify-center font-label-mono text-label-mono text-error px-6 text-center"
    >
      {{ error }}
    </div>
    <div
      v-else-if="!analysis"
      class="absolute bottom-2 left-0 right-0 text-center font-label-mono text-[11px] text-on-surface-variant"
    >
      Not analyzed yet. Generating charts analyzes the song first (tempo, beat grid, onsets).
    </div>
  </div>
</template>
