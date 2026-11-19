<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { useMediaSrc } from "../lib/mediaSrc";
import { api } from "../lib/api";
import { useLibrary } from "../stores/library";
import { useEngine } from "../stores/engine";
import type { Analysis } from "../lib/types";
import { fmtDuration } from "../lib/format";
import AnalysisCanvas from "../components/AnalysisCanvas.vue";
import EngineBanner from "../components/EngineBanner.vue";

const props = defineProps<{ base: string }>();
const router = useRouter();
const route = useRoute();
const lib = useLibrary();
const engine = useEngine();

// shallowRef: the parsed analysis holds hundreds of onsets plus energy/beat
// arrays — deep reactivity over it is pure overhead and made the view crawl.
const analysis = shallowRef<Analysis | null>(null);
const audioPath = ref<string | null>(null);
const backend = ref<"local" | "colab">("local");
const force = ref(false);
const jobId = ref<string | null>(null);

const audioEl = ref<HTMLAudioElement | null>(null);
const playing = ref(false);
const playhead = ref(0);

const track = computed(() => lib.byBase(props.base));
const job = computed(() => engine.job(jobId.value));
const running = computed(() => job.value?.running ?? false);
const failure = computed(() => engine.failureOf(jobId.value));

/** Any running job for this track — analysis must finish before charting, and
 *  a chart run must finish before re-analyzing. */
const busy = computed(() => engine.busyFor(track.value?.id));
const busyReason = computed(() =>
  busy.value?.kind === "analyze"
    ? "Analysis is still running…"
    : busy.value
      ? "A chart run is still in progress…"
      : "",
);
const audioUrl = useMediaSrc(() => audioPath.value);

const duration = computed(() => analysis.value?.duration_s ?? track.value?.durationSec ?? 0);

async function load() {
  if (!lib.tracks.length) await lib.refresh();
  lib.select(props.base);
  // Re-attach to a run that is still going (or the last one) so leaving and
  // coming back doesn't lose the job — the process itself never stopped.
  const existing = engine.jobFor("analyze", lib.byBase(props.base)?.id);
  if (existing) jobId.value = existing.id;
  const [a, p] = await Promise.all([api.getAnalysis(props.base), api.trackAudioPath(props.base)]);
  analysis.value = a;
  audioPath.value = p;
}

async function reanalyze() {
  const t = track.value;
  if (!t) return;
  const j = await engine.startAnalyze(t.id, backend.value, force.value);
  jobId.value = j.id;
}

function toggle() {
  const a = audioEl.value;
  if (!a || !audioUrl.value) return;
  if (a.paused) {
    a.play();
    playing.value = true;
  } else {
    a.pause();
    playing.value = false;
  }
}
function seek(sec: number) {
  if (audioEl.value) audioEl.value.currentTime = sec;
  playhead.value = sec;
}

watch(
  () => job.value?.running,
  async (isRunning, was) => {
    if (was && !isRunning) {
      analysis.value = await api.getAnalysis(props.base);
      await lib.refresh();
    }
  },
);
/** Arriving with `?run=1` (the Library's Analyze action) starts immediately. */
async function maybeAutoRun() {
  if (route.query.run !== "1") return;
  // Consume the flag so a refresh/back-nav doesn't re-trigger the run.
  await router.replace({ name: "analyze", params: { base: props.base } });
  if (busy.value || !lib.engineReady || !track.value) return;
  await reanalyze();
}

watch(() => props.base, async () => {
  await load();
  await maybeAutoRun();
});
onMounted(async () => {
  await load();
  await maybeAutoRun();
});
onBeforeUnmount(() => audioEl.value?.pause());
</script>

<template>
  <div class="h-full flex flex-col bg-background">
    <header class="h-20 border-b border-outline-variant bg-surface flex items-center justify-between px-margin-desktop shrink-0">
      <div class="flex flex-col min-w-0">
        <h2 class="font-headline-lg text-headline-lg text-on-surface tracking-tight truncate">
          {{ track?.title ?? base }}
        </h2>
        <p class="font-body-md text-on-surface-variant flex items-center gap-2">
          <span class="material-symbols-outlined text-sm">person</span>
          {{ track?.artist || "Unregistered — set metadata in Library" }}
        </p>
      </div>
      <div class="flex items-center gap-3">
        <label class="flex items-center gap-2 font-label-mono text-label-mono text-on-surface-variant cursor-pointer" title="Ignore the analysis cache">
          <input v-model="force" type="checkbox" class="accent-[#bc13fe]" /> force
        </label>
        <button
          class="px-4 py-2 rounded border border-outline text-on-surface hover:bg-surface-variant transition-colors font-label-mono text-label-mono flex items-center gap-2 disabled:opacity-50"
          :disabled="!!busy || !lib.engineReady"
          :title="busyReason"
          @click="reanalyze"
        >
          <span class="material-symbols-outlined text-sm" :class="{ spinner: running }">
            {{ running ? "progress_activity" : "graphic_eq" }}
          </span>
          {{ running ? "Analyzing…" : analysis ? "Re-analyze" : "Analyze" }}
        </button>
        <button
          class="px-6 py-2 rounded bg-primary-container text-on-primary-container hover:bg-primary transition-colors font-label-mono text-label-mono font-bold flex items-center gap-2 shadow-[0_0_15px_rgba(188,19,254,0.3)] disabled:opacity-40 disabled:shadow-none"
          :disabled="!!busy"
          :title="busyReason || 'Configure and run a chart generation'"
          @click="router.push({ name: 'generate', params: { base } })"
        >
          {{ busy ? busyReason : "Generate charts" }}
          <span class="material-symbols-outlined text-sm" :class="{ spinner: !!busy }">
            {{ busy ? "progress_activity" : "arrow_forward" }}
          </span>
        </button>
      </div>
    </header>

    <EngineBanner />

    <div class="flex-1 flex overflow-hidden">
      <div class="flex-1 flex flex-col relative min-w-0">
        <div class="flex-1 bg-surface-container-lowest relative overflow-hidden">
          <AnalysisCanvas :audio-path="audioPath" :analysis="analysis" :playhead="playhead" @seek="seek" />

          <div
            v-if="job && (running || job.lines.length)"
            class="absolute bottom-3 left-3 right-3 max-h-32 overflow-y-auto rounded-lg bg-[#0a0a0c]/92 border border-outline-variant p-3 font-label-mono text-[11px] text-[#8b949e] z-40"
          >
            <p v-for="(l, i) in job.lines.slice(-8)" :key="i" :class="{ 'text-error': l.stream === 'stderr' }">
              &gt; {{ l.line }}
            </p>
          </div>
        </div>

        <div v-if="failure" class="px-4 py-2 bg-error-container/20 border-y border-error/40 text-error text-[12px] font-label-mono whitespace-pre-wrap max-h-24 overflow-y-auto">
          {{ failure }}
        </div>

        <!-- Stats strip: measured DSP truth -->
        <div class="h-10 bg-surface border-y border-outline-variant flex items-center px-4 font-label-mono text-label-mono text-on-surface-variant gap-5 shrink-0 overflow-x-auto">
          <div class="flex items-center gap-2"><span class="text-on-surface">BPM:</span><span class="text-secondary-fixed">{{ analysis?.bpm?.toFixed(2) ?? "—" }}</span></div>
          <div class="w-px h-4 bg-outline-variant"></div>
          <div class="flex items-center gap-2"><span class="text-on-surface">Offset:</span>{{ analysis?.offset != null ? analysis.offset.toFixed(3) + "s" : "—" }}</div>
          <div class="w-px h-4 bg-outline-variant"></div>
          <div class="flex items-center gap-2"><span class="text-on-surface">Duration:</span>{{ fmtDuration(duration) }}</div>
          <div class="w-px h-4 bg-outline-variant"></div>
          <div class="flex items-center gap-2"><span class="text-on-surface">Onsets:</span>{{ analysis?.onsets?.length ?? "—" }}</div>
          <div class="w-px h-4 bg-outline-variant"></div>
          <div class="flex items-center gap-2">
            <span :class="(analysis?.sustain_available ?? 0) > 0 ? 'text-green-400' : 'text-on-surface-variant'">
              Sustains: {{ analysis?.sustain_available ?? "—" }}
            </span>
          </div>
          <div class="w-px h-4 bg-outline-variant"></div>
          <div class="flex items-center gap-2"><span class="text-on-surface">Backend:</span>{{ analysis?.beat_backend ?? "—" }}</div>
          <div class="w-px h-4 bg-outline-variant"></div>
          <div class="flex items-center gap-2"><span class="text-on-surface">Stems:</span>{{ analysis?.stem_source ?? "—" }}</div>
        </div>

        <!-- Transport for the source audio -->
        <div class="h-16 bg-surface-container-low flex items-center px-6 gap-4 shrink-0">
          <audio
            ref="audioEl"
            :src="audioUrl || undefined"
            @timeupdate="playhead = ($event.target as HTMLAudioElement).currentTime"
            @ended="playing = false"
          ></audio>
          <button
            class="p-3 rounded-full transition-colors disabled:opacity-40"
            :class="audioUrl ? 'bg-primary-container text-on-primary-container hover:bg-primary' : 'bg-surface-variant text-on-surface-variant'"
            :disabled="!audioUrl"
            @click="toggle"
          >
            <span class="material-symbols-outlined icon-fill">{{ playing ? "pause" : "play_arrow" }}</span>
          </button>
          <span class="font-label-mono text-label-mono text-on-surface-variant">{{ fmtDuration(playhead) }}</span>
          <div class="flex-1 h-2 bg-surface-variant rounded-full relative overflow-hidden">
            <div class="absolute left-0 top-0 h-full bg-primary-container rounded-full" :style="{ width: (duration ? (playhead / duration) * 100 : 0) + '%' }"></div>
          </div>
          <span class="font-label-mono text-label-mono text-on-surface-variant">{{ fmtDuration(duration) }}</span>
        </div>
      </div>

      <div class="w-64 bg-surface border-l border-outline-variant shrink-0 flex flex-col p-4">
        <h3 class="font-headline-md text-body-lg font-semibold text-on-surface mb-6 border-b border-outline-variant pb-2">Analysis Settings</h3>
        <div class="mb-6">
          <label class="font-label-mono text-label-mono text-on-surface-variant uppercase block mb-2">DSP Backend</label>
          <div class="bg-surface-container-highest p-1 rounded-md flex">
            <button
              v-for="b in (['local', 'colab'] as const)"
              :key="b"
              class="flex-1 py-1.5 px-2 text-center font-label-mono text-label-mono rounded transition-colors"
              :class="backend === b ? 'bg-surface-container text-primary shadow-sm border border-outline-variant' : 'text-on-surface-variant hover:text-on-surface'"
              @click="backend = b"
            >
              {{ b === "local" ? "Local CPU" : "GPU / Colab" }}
            </button>
          </div>
          <p class="text-[11px] text-on-surface-variant mt-2">
            {{ backend === "local" ? "numpy/scipy, HPSS only." : "madmom beat-tracking + Demucs stems. Fails loudly if Colab is unreachable." }}
          </p>
        </div>

        <div class="mb-6">
          <label class="font-label-mono text-label-mono text-on-surface-variant uppercase block mb-2">Timeline Legend</label>
          <div class="space-y-2 font-body-md text-[13px]">
            <div class="flex items-center gap-3"><div class="w-3 h-3 rounded-sm" style="background: linear-gradient(90deg,#bc13fe,#00eefc)"></div><span class="text-on-surface">Waveform</span></div>
            <div class="flex items-center gap-3"><div class="w-3 h-3 rounded-sm bg-[#ebb2ff]/60"></div><span class="text-on-surface">Onset candidate</span></div>
            <div class="flex items-center gap-3"><div class="w-3 h-3 rounded-sm bg-[#ffb960]/85"></div><span class="text-on-surface">Sustain candidate</span></div>
            <div class="flex items-center gap-3"><div class="w-3 h-3 rounded-sm bg-white/10 border border-outline-variant"></div><span class="text-on-surface">Bar line</span></div>
            <div class="flex items-center gap-3"><div class="w-3 h-3 rounded-sm bg-blue-500/40"></div><span class="text-on-surface">Section band</span></div>
          </div>
        </div>

        <div v-if="analysis?.sanity_warnings?.length" class="mb-4 p-3 rounded-lg border border-tertiary-container/50 bg-tertiary-container/10">
          <p class="font-label-mono text-label-mono text-tertiary-fixed-dim mb-1">Sanity warnings</p>
          <p v-for="(w, i) in analysis.sanity_warnings" :key="i" class="text-[11px] text-on-surface-variant">{{ w }}</p>
        </div>

        <div class="mt-auto p-4 bg-surface-container-low border border-outline-variant rounded-lg">
          <p class="font-body-md text-xs text-on-surface-variant">
            {{ analysis ? "Analyzed — the grid and onsets above are what the charts are built on." : "Run Analyze to measure tempo, beat grid and onset candidates." }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>
