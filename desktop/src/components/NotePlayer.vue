<script setup lang="ts">
/**
 * A small ITGmania-style chart player: a 4-lane notefield scrolling up to
 * receptors, driven by the song audio so you can *watch and hear* the chart.
 *
 * Deliberately simple — it previews, it doesn't score. Notes come straight from
 * the emitted simfile (`get_chart_notes`), so what you see is what the game
 * would play. Canvas rendering keeps it cheap even for dense charts.
 *
 * Assist: the engine's `.assist.ogg` is the song *already mixed* with tick
 * transients (ffmpeg `amix` of track + clicks), so "assist on" swaps the source
 * rather than stacking a second element — one clock, no drift. The swap keeps
 * position and play state, so toggling mid-song is seamless.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useMediaSrc } from "../lib/mediaSrc";
import type { Chart, ChartNote } from "../lib/types";
import { fmtDuration } from "../lib/format";

const props = withDefaults(
  defineProps<{
    audioPath: string | null;
    assistPath?: string | null;
    chart: Chart | null;
    autoplay?: boolean;
    /** Bumped when the assist track was re-rendered at the SAME path (a section
     *  re-roll), so the webview fetches it again instead of replaying its cache. */
    rev?: number;
  }>(),
  { autoplay: true, assistPath: null, rev: 0 },
);

const canvas = ref<HTMLCanvasElement | null>(null);
const wrap = ref<HTMLDivElement | null>(null);
const audio = ref<HTMLAudioElement | null>(null);

const playing = ref(false);
const current = ref(0);
const duration = ref(0);
/** Pixels per second of scroll — the usual "speed mod" knob. */
const speed = ref(320);
/** Assist ticks default on when the engine rendered them. */
const assist = ref(true);

let raf = 0;
let ro: ResizeObserver | null = null;
/** Non-reactive render state; touching refs per frame is wasted work. */
let notes: ChartNote[] = [];
let cols = 4;
let hitFlash: number[] = [];
let cursor = 0;
let autoplayWhenReady = false;
/** Canvas CSS size, maintained by the ResizeObserver (never inside rAF). */
let cssW = 0;
let cssH = 0;

const songUrl = useMediaSrc(() => props.audioPath);
const assistUrl = useMediaSrc(() => props.assistPath, () => props.rev);
const hasAssist = computed(() => !!assistUrl.value);
const activeUrl = computed(() =>
  assist.value && assistUrl.value ? assistUrl.value : songUrl.value,
);

const COL_COLORS = ["#ff5d8f", "#00eefc", "#c084fc", "#7dd3a0"];
const COL_ROT = [Math.PI, Math.PI / 2, -Math.PI / 2, 0]; // left, down, up, right
const RECEPTOR_Y = 62;

function resetChart() {
  notes = props.chart?.notes ?? [];
  cols = props.chart?.columns || 4;
  hitFlash = new Array(cols).fill(-1);
  cursor = 0;
}

/* ------------------------------- audio ---------------------------------- */
/** Source is managed imperatively so a swap can preserve position + state. */
function setSource(url: string, keepTime = 0, resume = false) {
  const a = audio.value;
  if (!a) return;
  if (!url) {
    a.removeAttribute("src");
    a.load();
    return;
  }
  const onMeta = () => {
    a.removeEventListener("loadedmetadata", onMeta);
    duration.value = a.duration || 0;
    if (keepTime > 0) a.currentTime = Math.min(keepTime, a.duration || keepTime);
    if (resume) void a.play().then(() => (playing.value = true)).catch(() => (playing.value = false));
  };
  a.addEventListener("loadedmetadata", onMeta);
  a.src = url;
  a.load();
}

async function play() {
  const a = audio.value;
  if (!a || !activeUrl.value) return;
  try {
    await a.play();
    playing.value = true;
  } catch {
    playing.value = false; // autoplay blocked — user can press play
  }
}
function toggle() {
  const a = audio.value;
  if (!a || !activeUrl.value) return;
  if (a.paused) void play();
  else {
    a.pause();
    playing.value = false;
  }
}
function restart() {
  const a = audio.value;
  if (!a) return;
  a.currentTime = 0;
  current.value = 0;
  cursor = 0;
  hitFlash.fill(-1);
  void play();
}
function seek(e: MouseEvent) {
  const a = audio.value;
  if (!a || !duration.value) return;
  const el = e.currentTarget as HTMLElement;
  const r = el.getBoundingClientRect();
  const t = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * duration.value;
  a.currentTime = t;
  current.value = t;
  resyncCursor(t);
}
/** Rebuild the flash cursor after an arbitrary seek. */
function resyncCursor(t: number) {
  cursor = 0;
  while (cursor < notes.length && notes[cursor].time <= t) cursor++;
}

// Toggling assist keeps position and play state — one clock, no drift.
watch(activeUrl, (url) => {
  const a = audio.value;
  const keep = a ? a.currentTime : 0;
  let resume = a ? !a.paused : false;
  if (url && autoplayWhenReady) {
    resume = true;
    autoplayWhenReady = false;
  }
  setSource(url, keep, resume);
});

/** Start a chart from the top. Where the audio address arrives a moment later
 *  (see lib/mediaSrc), the autoplay is remembered until it does. */
function startChart() {
  autoplayWhenReady = !!props.autoplay && !activeUrl.value;
  setSource(activeUrl.value, 0, props.autoplay);
}

watch(
  () => props.chart,
  () => {
    resetChart();
    assist.value = true; // fresh difficulty -> assist back on
    startChart();
  },
);

/* ------------------------------ rendering -------------------------------- */
/** Size the backing store from the observer only — never inside the rAF loop,
 *  where a transient 0/!0 layout read could feed back into element size. */
function resize() {
  const box = wrap.value;
  const c = canvas.value;
  if (!box || !c) return;
  const w = Math.max(1, Math.floor(box.clientWidth));
  const h = Math.max(1, Math.floor(box.clientHeight));
  if (w === cssW && h === cssH) return;
  cssW = w;
  cssH = h;
  const dpr = window.devicePixelRatio || 1;
  c.style.width = `${w}px`;
  c.style.height = `${h}px`;
  c.width = Math.round(w * dpr);
  c.height = Math.round(h * dpr);
}

function drawArrow(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, color: string, rot: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rot);
  ctx.fillStyle = color;
  const s = size / 2;
  ctx.beginPath();
  ctx.moveTo(s, 0);
  ctx.lineTo(0, -s);
  ctx.lineTo(0, -s * 0.45);
  ctx.lineTo(-s, -s * 0.45);
  ctx.lineTo(-s, s * 0.45);
  ctx.lineTo(0, s * 0.45);
  ctx.lineTo(0, s);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function frame() {
  raf = requestAnimationFrame(frame);
  const c = canvas.value;
  const a = audio.value;
  if (!c || !cssW || !cssH) return;
  const ctx = c.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const t = a ? a.currentTime : 0;
  current.value = t;

  const laneW = Math.min(74, cssW / (cols + 1));
  const fieldW = laneW * cols;
  const x0 = (cssW - fieldW) / 2;
  const size = laneW * 0.72;

  ctx.fillStyle = "rgba(255,255,255,0.02)";
  for (let i = 0; i < cols; i++) ctx.fillRect(x0 + i * laneW + 2, 0, laneW - 4, cssH);

  // beat lines
  const bpm = props.chart?.bpm ?? 0;
  const beat0 = props.chart?.beat0Time ?? 0;
  if (bpm > 0) {
    const spb = 60 / bpm;
    const first = Math.floor((t - beat0) / spb);
    const span = Math.ceil((cssH - RECEPTOR_Y) / (spb * speed.value)) + 2;
    for (let b = first; b < first + span; b++) {
      const y = RECEPTOR_Y + (beat0 + b * spb - t) * speed.value;
      if (y < RECEPTOR_Y || y > cssH) continue;
      ctx.fillStyle = b % 4 === 0 ? "rgba(255,255,255,0.13)" : "rgba(255,255,255,0.05)";
      ctx.fillRect(x0, y, fieldW, 1);
    }
  }

  // receptors
  for (let i = 0; i < cols; i++) {
    const cx = x0 + i * laneW + laneW / 2;
    const flash = hitFlash[i] >= 0 ? Math.max(0, 1 - (t - hitFlash[i]) / 0.16) : 0;
    ctx.save();
    ctx.strokeStyle = flash > 0 ? "#ffffff" : "rgba(255,255,255,0.35)";
    ctx.lineWidth = 2 + flash * 3;
    ctx.beginPath();
    ctx.arc(cx, RECEPTOR_Y, (size / 2) * 0.95, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    if (flash > 0) {
      ctx.save();
      ctx.globalAlpha = flash * 0.5;
      ctx.fillStyle = COL_COLORS[i % 4];
      ctx.beginPath();
      ctx.arc(cx, RECEPTOR_Y, size * 0.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  // notes — culled at the receptor line so nothing flies past it
  const lookahead = (cssH - RECEPTOR_Y) / speed.value + 0.2;
  for (let i = 0; i < notes.length; i++) {
    const n = notes[i];
    const isHold = n.kind === "hold" || n.kind === "roll";
    const tail = isHold ? (n.endTime ?? n.time) : n.time;
    if (tail < t) continue; // fully consumed
    if (n.time > t + lookahead) break;
    const cx = x0 + n.column * laneW + laneW / 2;
    const y = RECEPTOR_Y + (n.time - t) * speed.value;
    const color = COL_COLORS[n.column % 4];

    if (isHold) {
      // Body shrinks into the receptor as it's held.
      const yEnd = RECEPTOR_Y + (tail - t) * speed.value;
      const top = Math.max(RECEPTOR_Y, Math.min(y, yEnd));
      const bottom = Math.max(y, yEnd);
      if (bottom > top) {
        ctx.save();
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = color;
        ctx.fillRect(cx - size * 0.22, top, size * 0.44, bottom - top);
        ctx.restore();
      }
      if (n.time >= t) drawArrow(ctx, cx, y, size, color, COL_ROT[n.column % 4]);
    } else if (n.time >= t) {
      if (n.kind === "mine") {
        ctx.fillStyle = "#ff6b6b";
        ctx.beginPath();
        ctx.arc(cx, y, size * 0.3, 0, Math.PI * 2);
        ctx.fill();
      } else {
        drawArrow(ctx, cx, y, size, color, COL_ROT[n.column % 4]);
      }
    }
  }

  // flash receptors as notes reach them
  while (cursor < notes.length && notes[cursor].time <= t) {
    const n = notes[cursor];
    if (n.kind !== "mine") hitFlash[n.column % cols] = n.time;
    cursor++;
  }
}

onMounted(() => {
  resetChart();
  resize();
  ro = new ResizeObserver(resize);
  if (wrap.value) ro.observe(wrap.value);
  startChart();
  raf = requestAnimationFrame(frame);
});
onBeforeUnmount(() => {
  cancelAnimationFrame(raf);
  ro?.disconnect();
  audio.value?.pause();
});
</script>

<template>
  <div class="w-full h-full flex flex-col min-h-0">
    <div ref="wrap" class="flex-1 relative bg-surface-container-lowest min-h-0 overflow-hidden">
      <canvas ref="canvas" class="absolute top-0 left-0 block"></canvas>
      <audio
        ref="audio"
        @timeupdate="current = ($event.target as HTMLAudioElement).currentTime"
        @durationchange="duration = ($event.target as HTMLAudioElement).duration || 0"
        @ended="playing = false"
      ></audio>
      <div
        v-if="!chart"
        class="absolute inset-0 flex items-center justify-center font-label-mono text-label-mono text-on-surface-variant"
      >
        No playable chart for this difficulty
      </div>
      <div
        v-else-if="!songUrl && !assistUrl"
        class="absolute inset-0 flex items-center justify-center font-label-mono text-label-mono text-on-surface-variant"
      >
        Song audio not found
      </div>
      <div
        v-if="chart"
        class="absolute top-2 right-3 font-label-mono text-[10px] text-on-surface-variant bg-surface/70 rounded px-2 py-1"
      >
        {{ chart.stepstype }} · {{ chart.difficulty }} · meter {{ chart.meter }} · {{ chart.notes.length }} arrows
      </div>
    </div>

    <!-- transport -->
    <div class="h-14 bg-surface-container border-t border-outline-variant flex items-center px-gutter gap-4 shrink-0">
      <button
        class="w-9 h-9 rounded-full flex items-center justify-center transition-colors disabled:opacity-40"
        :class="activeUrl ? 'bg-primary-container text-on-primary-container hover:bg-primary' : 'bg-surface-variant text-on-surface-variant'"
        :disabled="!activeUrl"
        @click="toggle"
      >
        <span class="material-symbols-outlined icon-fill">{{ playing ? "pause" : "play_arrow" }}</span>
      </button>
      <button
        class="w-9 h-9 rounded-full flex items-center justify-center text-on-surface-variant hover:bg-surface-variant disabled:opacity-40"
        :disabled="!activeUrl"
        title="Restart"
        @click="restart"
      >
        <span class="material-symbols-outlined">refresh</span>
      </button>
      <span class="font-label-mono text-label-mono text-on-surface-variant w-12 text-right">{{ fmtDuration(current) }}</span>
      <div class="flex-1 h-2 bg-surface-variant rounded-full relative overflow-hidden cursor-pointer" @click="seek">
        <div
          class="absolute left-0 top-0 h-full bg-primary-container rounded-full"
          :style="{ width: (duration ? (current / duration) * 100 : 0) + '%' }"
        ></div>
      </div>
      <span class="font-label-mono text-label-mono text-on-surface-variant w-12">{{ fmtDuration(duration) }}</span>

      <!-- Assist ticks: on by default, seamless mid-song toggle -->
      <label
        class="flex items-center gap-2 shrink-0"
        :class="hasAssist ? 'cursor-pointer' : 'opacity-40 cursor-not-allowed'"
        :title="hasAssist ? 'Play the assist-tick mix (clicks on every note)' : 'No assist track was rendered'"
      >
        <span class="font-label-mono text-[10px] uppercase tracking-wider text-on-surface-variant">Assist</span>
        <div class="relative">
          <input v-model="assist" :disabled="!hasAssist" class="sr-only peer" type="checkbox" />
          <div
            class="w-9 h-5 bg-surface-variant rounded-full peer peer-checked:bg-primary-container after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-on-surface after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:after:translate-x-full"
          ></div>
        </div>
      </label>

      <label class="flex items-center gap-2 shrink-0">
        <span class="font-label-mono text-[10px] uppercase tracking-wider text-on-surface-variant">Speed</span>
        <input v-model.number="speed" type="range" min="120" max="700" step="20" class="w-24 accent-[#bc13fe]" />
      </label>
    </div>
  </div>
</template>
