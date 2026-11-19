<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { useLibrary } from "../stores/library";
import { useEngine, GEN_STAGES } from "../stores/engine";
import { useSettings } from "../stores/settings";
import { useNeural } from "../stores/neural";
import { neuralProblem } from "../lib/engineProblems";
import { api } from "../lib/api";
import type { ChartEngine, NeuralStyleField, Target } from "../lib/types";
import StagePipeline from "../components/StagePipeline.vue";
import EngineBanner from "../components/EngineBanner.vue";

const props = defineProps<{ base?: string }>();
const router = useRouter();
const lib = useLibrary();
const engine = useEngine();
const settings = useSettings();
const neural = useNeural();

const baseId = computed(() => props.base ?? lib.selectedBase ?? "");
const track = computed(() => lib.byBase(baseId.value));

// Beat Saber (SABERFORGE) is implemented in the engine but not supported by the
// app yet, so it is deliberately absent here; the Rust side rejects it too.
// The LLM-only "native" beatmap target is retired too: StepForge charts
// StepMania / ITGmania with its onboard model.
const TARGETS: { key: Target; label: string; icon: string; blurb: string }[] = [
  { key: "stepforge", label: "StepMania", icon: "music_note", blurb: "ITGmania .ssc/.sm song folder" },
];

const target = ref<Target>("stepforge");
const availableDiffs = ref<string[]>([]);
const selectedDiffs = ref<string[]>([]);
const rights = ref(false);
const jobId = ref<string | null>(null);

const job = computed(() => engine.job(jobId.value));
const running = computed(() => job.value?.running ?? false);
const failure = computed(() => engine.failureOf(jobId.value));
const isNative = computed(() => target.value === "native");
const needsRights = computed(() => target.value === "saberforge");

// --- Chart engine (StepMania target only) --------------------------------- //
// StepForge charts with its onboard model only. The engine still has the LLM
// designer and the DSP-only path, but the app no longer offers them.
const ENGINES: { key: ChartEngine; label: string; hint: string }[] = [
  { key: "neural", label: "StepForge Flow", hint: "Two small models that run on this machine" },
];
/** What the toggle is set to. Starts as the saved choice; with none saved it
 *  becomes neural once `neural-status` says the model can run (see onMounted). */
const engineChoice = ref<ChartEngine>("neural");
/** Set once the user clicks the toggle, so a slow status probe arriving later
 *  cannot move it from under them. */
let enginePicked = false;
/** The engine this run will actually use. The native `chart` path has no choice
 *  of engine — it always uses the designer. */
const chartEngine = computed<ChartEngine>(() => (isNative.value ? "llm" : engineChoice.value));
const isNeural = computed(() => chartEngine.value === "neural");
const deterministic = computed(() => chartEngine.value === "deterministic");

/** An explicit pick is remembered. The status-driven default is not saved, so
 *  an install that gains the weights later starts defaulting to them. */
function pickEngine(e: ChartEngine) {
  enginePicked = true;
  engineChoice.value = e;
  settings.value.chartEngine = e;
  void settings.save();
}

// --- Neural style: a preset and/or individual sliders --------------------- //
/** Preset to send: only one the engine actually lists. */
const activePreset = computed(() => {
  const p = settings.value.neuralStylePreset;
  return p && neural.presets.includes(p) ? p : null;
});
function presetLabel(p: string) {
  const s = p.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** A slider is "auto" until the user turns it on: the key is then ABSENT from
 *  `neuralStyle` and from the request, and the model infers that trait from the
 *  audio. Turning one off remembers where it was for this visit. */
const parked: Record<string, number> = {};
function styleIsSet(key: string) {
  return typeof settings.value.neuralStyle[key] === "number";
}
function clampToField(f: NeuralStyleField, v: number) {
  return Math.min(f.max, Math.max(f.min, v));
}
/** Where an untouched slider sits when it is first turned on. */
function midpoint(f: NeuralStyleField) {
  const steps = Math.round((f.max - f.min) / 2 / f.step);
  return clampToField(f, Number((f.min + steps * f.step).toFixed(4)));
}
function styleValue(f: NeuralStyleField) {
  const v = settings.value.neuralStyle[f.key];
  return typeof v === "number" ? clampToField(f, v) : (parked[f.key] ?? midpoint(f));
}
function toggleStyle(f: NeuralStyleField, on: boolean) {
  const next = { ...settings.value.neuralStyle };
  if (on) next[f.key] = parked[f.key] ?? midpoint(f);
  else {
    if (typeof next[f.key] === "number") parked[f.key] = next[f.key];
    delete next[f.key];
  }
  settings.value.neuralStyle = next;
  void settings.save();
}
function setStyle(f: NeuralStyleField, raw: string) {
  const n = Number(raw);
  if (!Number.isFinite(n)) return;
  settings.value.neuralStyle = { ...settings.value.neuralStyle, [f.key]: clampToField(f, n) };
}
/** Shares read as percentages; anything else (notes per second) as a number. */
function styleLabel(f: NeuralStyleField, v: number) {
  if (f.key.endsWith("_share")) return `${Math.round(v * 100)}%`;
  return v.toFixed(f.step < 1 ? Math.min(2, String(f.step).split(".")[1]?.length ?? 1) : 0);
}
/** Only the sliders the user set, and only ones the engine knows — a key left
 *  in settings from another version is dropped rather than sent. */
const styleJson = computed(() => {
  const out: Record<string, number> = {};
  for (const f of neural.fields) {
    const v = settings.value.neuralStyle[f.key];
    if (typeof v === "number" && Number.isFinite(v)) out[f.key] = clampToField(f, v);
  }
  return Object.keys(out).length ? JSON.stringify(out) : null;
});

// --- Saved style presets --------------------------------------------------- //
const namingPreset = ref(false);
const presetName = ref("");
const userPresetNames = computed(() => Object.keys(settings.value.neuralUserPresets).sort());
/** The sliders that are set and that this engine knows, as a plain object. */
function currentStyle(): Record<string, number> {
  const out: Record<string, number> = {};
  for (const f of neural.fields) {
    const v = settings.value.neuralStyle[f.key];
    if (typeof v === "number" && Number.isFinite(v)) out[f.key] = clampToField(f, v);
  }
  return out;
}
/** Is the screen currently showing exactly this saved preset? */
function isActiveUserPreset(name: string) {
  const p = settings.value.neuralUserPresets[name];
  if (!p || (p.preset || "") !== (settings.value.neuralStylePreset || "")) return false;
  const now = currentStyle();
  const keys = new Set([...Object.keys(now), ...Object.keys(p.style)]);
  for (const k of keys) if (now[k] !== p.style[k]) return false;
  return true;
}
function saveUserPreset() {
  const name = presetName.value.trim();
  if (!name) return;
  settings.value.neuralUserPresets = {
    ...settings.value.neuralUserPresets,
    [name]: { preset: settings.value.neuralStylePreset || "", style: currentStyle() },
  };
  namingPreset.value = false;
  presetName.value = "";
  void settings.save();
}
function applyUserPreset(name: string) {
  const p = settings.value.neuralUserPresets[name];
  if (!p) return;
  // A built-in preset this engine no longer lists falls back to "let the model decide".
  settings.value.neuralStylePreset = neural.presets.includes(p.preset) ? p.preset : "";
  settings.value.neuralStyle = { ...p.style };
  void settings.save();
}
function deleteUserPreset(name: string) {
  const next = { ...settings.value.neuralUserPresets };
  delete next[name];
  settings.value.neuralUserPresets = next;
  void settings.save();
}

/** Why a neural run cannot start, if it cannot — with the real cause, so an
 *  engine that failed to start is never described as an old one. */
const neuralIssue = computed(() => (isNeural.value ? neuralProblem(neural.status) : null));
const neuralBlocker = computed(() => {
  if (!isNeural.value) return null;
  if (!neural.status) return "Checking for the StepForge Flow model…";
  const p = neuralIssue.value;
  return p ? `${p.title}. See Settings for details.` : null;
});

/** A run of any kind already in flight for this track (e.g. an Analyze started
 *  on the previous screen). Chart runs must not race it. */
const otherJob = computed(() => {
  const b = engine.busyFor(track.value?.id);
  return b && b.id !== jobId.value ? b : undefined;
});

const blockers = computed(() => {
  const out: string[] = [];
  if (!baseId.value) out.push("Select a track in the Library first.");
  else if (track.value && !track.value.isOgg) out.push("This track still needs converting to .ogg.");
  if (!lib.engineReady) out.push("The engine environment is not ready.");
  if (!selectedDiffs.value.length) out.push("Pick at least one difficulty.");
  if (needsRights.value && !rights.value) out.push("Acknowledge you hold the rights to this audio.");
  if (neuralBlocker.value) out.push(neuralBlocker.value);
  if (otherJob.value)
    out.push(
      otherJob.value.kind === "analyze"
        ? "Analysis is still running for this track — wait for it to finish."
        : "Another run is already in progress for this track.",
    );
  return out;
});
const canGenerate = computed(() => !blockers.value.length && !running.value);

async function loadDiffs(resetSelection = true) {
  availableDiffs.value = await api.difficultiesFor(target.value);
  if (resetSelection) selectedDiffs.value = await api.defaultDifficultiesFor(target.value);
}

function toggleDiff(d: string) {
  if (isNative.value) {
    // `chart --difficulty` takes exactly one argparse choice.
    selectedDiffs.value = [d];
    return;
  }
  selectedDiffs.value = selectedDiffs.value.includes(d)
    ? selectedDiffs.value.filter((x) => x !== d)
    : [...selectedDiffs.value, d];
}

// Explicit arg: watch would otherwise pass the new target as `resetSelection`.
watch(target, () => loadDiffs(true));

async function generate() {
  const t = track.value;
  if (!t || !canGenerate.value) return;
  const j = await engine.startGenerate({
    trackId: t.id,
    target: target.value,
    difficulties: selectedDiffs.value,
    deterministic: deterministic.value,
    iHaveRights: rights.value,
    backend: "local",
    // Only the StepMania target has an engine; the style belongs to neural.
    engine: target.value === "stepforge" ? chartEngine.value : undefined,
    stylePreset: isNeural.value ? activePreset.value : undefined,
    styleJson: isNeural.value ? styleJson.value : undefined,
  });
  jobId.value = j.id;
}

watch(
  () => job.value?.running,
  async (isRunning, was) => {
    if (was && !isRunning && job.value?.success) await lib.refresh();
  },
);

/** Render a bounded window — the store keeps more than the console shows. */
const consoleLines = computed(() => {
  const l = job.value?.lines;
  if (!l) return [];
  return l.length > 200 ? l.slice(-200) : l;
});

const stageDetails = computed<Record<number, string>>(() => {
  const j = job.value;
  if (!j) return {};
  const last = [...j.lines].reverse().find((l) => l.stream === "stdout" && l.line.trim());
  return last ? { [Math.min(j.stage, GEN_STAGES.length - 1)]: last.line.trim() } : {};
});

onMounted(async () => {
  if (!lib.tracks.length) await lib.refresh();
  if (baseId.value) lib.select(baseId.value);
  if (!settings.loaded) await settings.load();
  // The saved engine applies at once. With none saved, today's default (the
  // designer) stands until the status probe says the neural model can run —
  // not awaited, because it starts a Python process and nothing else here
  // should wait on it.
  const savedEngine = ENGINES.find((e) => e.key === settings.value.chartEngine)?.key;
  if (savedEngine) engineChoice.value = savedEngine;
  void neural.load().then(() => {
    if (!savedEngine && !enginePicked && neural.ready) engineChoice.value = "neural";
  });
  // Re-attach to an in-flight (or the last) generation for this track so the
  // pipeline and console survive navigating away and back.
  const existing = engine.jobFor("generate", track.value?.id);
  if (existing) jobId.value = existing.id;
  // Keep the chosen difficulties while a run is in flight.
  await loadDiffs(!existing?.running);
});
</script>

<template>
  <div class="h-full flex flex-col md:flex-row overflow-hidden">
    <!-- Config -->
    <section class="flex-1 border-r border-outline-variant bg-surface-container-lowest overflow-y-auto flex flex-col">
      <EngineBanner />
      <div class="p-margin-desktop flex flex-col gap-7 flex-1">
        <header>
          <h2 class="font-headline-lg text-headline-lg text-on-surface">Chart Configuration</h2>
          <p class="font-body-md text-on-surface-variant mt-2 flex items-center gap-2">
            <span class="material-symbols-outlined text-sm">music_note</span>
            {{ track?.title ?? "No track selected" }}
            <span v-if="track?.artist" class="text-on-surface-variant">— {{ track.artist }}</span>
            <router-link
              v-if="track?.isOgg"
              :to="{ name: 'analyze', params: { base: track.base } }"
              class="ml-2 text-[12px] text-secondary-container hover:underline flex items-center gap-1"
              title="Tempo, beat grid and onsets. Generating analyzes the song automatically when it needs to."
            >
              <span class="material-symbols-outlined text-[14px]">graphic_eq</span> View analysis
            </router-link>
          </p>
        </header>

        <div v-if="TARGETS.length > 1" class="space-y-3">
          <label class="font-label-mono text-label-mono text-on-surface-variant uppercase block">Target Format</label>
          <div class="grid gap-3" :class="TARGETS.length === 2 ? 'grid-cols-2' : 'grid-cols-3'">
            <button
              v-for="t in TARGETS"
              :key="t.key"
              type="button"
              class="flex flex-col items-center justify-center p-4 rounded-lg gap-1 transition-colors text-center"
              :class="
                target === t.key
                  ? 'bg-primary-container border-2 border-primary-container text-on-primary-container shadow-[0_0_15px_rgba(188,19,254,0.3)]'
                  : 'bg-surface-container border border-outline-variant hover:border-primary-container text-on-surface'
              "
              @click="target = t.key"
            >
              <span class="material-symbols-outlined text-2xl">{{ t.icon }}</span>
              <span class="font-label-mono text-label-mono">{{ t.label }}</span>
            </button>
          </div>
          <p class="text-[11px] text-on-surface-variant">{{ TARGETS.find((t) => t.key === target)?.blurb }}</p>
        </div>

        <div class="space-y-3">
          <label class="font-label-mono text-label-mono text-on-surface-variant uppercase block">
            {{ isNative ? "Difficulty (one per run)" : "Difficulties to Generate" }}
          </label>
          <div class="flex flex-wrap gap-2">
            <button
              v-for="d in availableDiffs"
              :key="d"
              type="button"
              class="px-4 py-2 rounded-full border font-body-md text-body-md capitalize transition-colors"
              :class="
                selectedDiffs.includes(d)
                  ? 'bg-surface-variant border-primary text-primary'
                  : 'border-outline-variant text-on-surface hover:bg-surface-variant'
              "
              @click="toggleDiff(d)"
            >
              {{ d }}
            </button>
          </div>
        </div>

        <div class="space-y-3 bg-surface-container p-4 rounded-xl border border-outline-variant">
          <label class="font-label-mono text-label-mono text-on-surface-variant uppercase block mb-2">Chart style</label>
          <div v-if="ENGINES.length > 1" class="flex items-center gap-1 bg-surface p-1 rounded-lg border border-outline-variant">
            <button
              v-for="e in ENGINES"
              :key="e.key"
              type="button"
              class="flex-1 py-2 rounded-md font-label-mono text-label-mono transition-colors disabled:opacity-40"
              :class="chartEngine === e.key ? 'bg-surface-container-high text-on-surface border border-outline-variant' : 'text-on-surface-variant hover:text-on-surface'"
              :disabled="isNative && e.key !== 'llm'"
              :title="isNative && e.key !== 'llm' ? 'The native chart path always uses the designer' : e.hint"
              @click="pickEngine(e.key)"
            >
              {{ e.label }}
            </button>
          </div>

          <!-- Neural style. Nothing here is required: with no preset and every
               slider on auto the model reads the style off the audio. -->
          <div v-if="isNeural" class="space-y-3" :class="ENGINES.length > 1 ? 'pt-4 border-t border-outline-variant mt-4' : ''">
            <div v-if="neuralIssue" class="p-3 rounded-lg border border-error/40 bg-error-container/20">
              <p class="font-label-mono text-label-mono text-error mb-1">{{ neuralIssue.title }}</p>
              <p v-if="neuralIssue.detail" class="text-[11px] text-on-surface-variant whitespace-pre-wrap break-words">{{ neuralIssue.detail }}</p>
              <p class="text-[11px] text-on-surface-variant mt-2">
                <router-link :to="{ name: 'settings' }" class="underline hover:text-on-surface">Open Settings</router-link>
                ·
                <button type="button" class="underline hover:text-on-surface disabled:opacity-50" :disabled="neural.loading" @click="neural.refresh()">
                  {{ neural.loading ? "Checking…" : "Check again" }}
                </button>
              </p>
            </div>

            <div>
              <label class="font-label-mono text-label-mono text-on-surface-variant uppercase block mb-2">Style preset</label>
              <select
                v-model="settings.value.neuralStylePreset"
                class="w-full bg-surface-container-lowest border border-outline-variant text-on-surface rounded-lg p-3 font-body-md focus:border-secondary outline-none disabled:opacity-40"
                @change="settings.save()"
              >
                <option value="">Let the model decide</option>
                <option v-for="p in neural.presets" :key="p" :value="p">{{ presetLabel(p) }}</option>
              </select>
            </div>

            <!-- Your own presets: the preset above plus whichever sliders are set -->
            <div>
              <label class="font-label-mono text-label-mono text-on-surface-variant uppercase block mb-2">Your presets</label>
              <div class="flex flex-wrap items-center gap-2">
                <span
                  v-for="name in userPresetNames"
                  :key="name"
                  class="inline-flex items-center rounded-full border text-[12px]"
                  :class="isActiveUserPreset(name) ? 'border-primary text-primary bg-surface-variant' : 'border-outline-variant text-on-surface'"
                >
                  <button type="button" class="pl-3 pr-1 py-1 hover:underline" :title="`Load “${name}”`" @click="applyUserPreset(name)">
                    {{ name }}
                  </button>
                  <button
                    type="button"
                    class="pr-2 pl-1 py-1 text-on-surface-variant hover:text-error"
                    :title="`Delete “${name}”`"
                    @click="deleteUserPreset(name)"
                  >
                    <span class="material-symbols-outlined text-[14px] align-middle">close</span>
                  </button>
                </span>
                <template v-if="namingPreset">
                  <input
                    v-model="presetName"
                    class="bg-surface-container-lowest border border-outline-variant rounded px-2 py-1 text-on-surface text-[12px] focus:border-secondary outline-none w-40"
                    placeholder="Preset name"
                    autofocus
                    @keyup.enter="saveUserPreset"
                    @keyup.esc="namingPreset = false"
                  />
                  <button
                    type="button"
                    class="px-3 py-1 rounded border border-outline-variant text-on-surface text-[12px] hover:bg-surface-variant disabled:opacity-40"
                    :disabled="!presetName.trim()"
                    @click="saveUserPreset"
                  >
                    Save
                  </button>
                  <button type="button" class="text-[12px] text-on-surface-variant hover:text-on-surface" @click="namingPreset = false">
                    Cancel
                  </button>
                </template>
                <button
                  v-else
                  type="button"
                  class="px-3 py-1 rounded-full border border-dashed border-outline-variant text-on-surface-variant text-[12px] hover:text-on-surface hover:border-outline flex items-center gap-1"
                  title="Save the preset and sliders as they are now"
                  @click="namingPreset = true"
                >
                  <span class="material-symbols-outlined text-[14px]">add</span> Save preset…
                </button>
              </div>
              <p v-if="presetName.trim() && settings.value.neuralUserPresets[presetName.trim()]" class="text-[11px] text-tertiary mt-1">
                “{{ presetName.trim() }}” exists and will be replaced.
              </p>
            </div>

            <div v-if="neural.fields.length">
              <label class="font-label-mono text-label-mono text-on-surface-variant uppercase block mb-2">Style sliders</label>
              <div class="space-y-1.5">
                <div v-for="f in neural.fields" :key="f.key" class="flex items-center gap-3">
                  <label class="flex items-center gap-2 w-36 shrink-0 cursor-pointer" :title="`Tick to set ${f.label} yourself`">
                    <input
                      type="checkbox"
                      class="accent-[#bc13fe] cursor-pointer"
                      :checked="styleIsSet(f.key)"
                      @change="toggleStyle(f, ($event.target as HTMLInputElement).checked)"
                    />
                    <span class="font-body-md text-[13px] truncate" :class="styleIsSet(f.key) ? 'text-on-surface' : 'text-on-surface-variant'">{{ f.label }}</span>
                  </label>
                  <input
                    type="range"
                    class="flex-1 min-w-0 accent-[#bc13fe] disabled:opacity-30"
                    :min="f.min"
                    :max="f.max"
                    :step="f.step"
                    :value="styleValue(f)"
                    :disabled="!styleIsSet(f.key)"
                    :aria-label="f.label"
                    @input="setStyle(f, ($event.target as HTMLInputElement).value)"
                    @change="settings.save()"
                  />
                  <span class="w-12 shrink-0 text-right font-label-mono text-[11px]" :class="styleIsSet(f.key) ? 'text-on-surface' : 'text-on-surface-variant'">
                    {{ styleIsSet(f.key) ? styleLabel(f, styleValue(f)) : "auto" }}
                  </span>
                </div>
              </div>
              <p class="text-[11px] text-on-surface-variant mt-2">
                Tick a slider to set that trait yourself; an unticked one stays on <span class="font-label-mono">auto</span> and
                the model infers it from the audio. A slider you set overrides the preset for that trait. Every chart is still
                held to its difficulty's limits.
              </p>
            </div>
          </div>
          <p class="text-[11px] text-on-surface-variant pt-3">
            StepForge Flow runs on this computer: no account, no network, no cost.
          </p>
        </div>

        <div class="mt-auto space-y-4 pt-4">
          <label v-if="needsRights" class="flex items-center gap-3 cursor-pointer group">
            <input v-model="rights" type="checkbox" class="w-5 h-5 accent-[#bc13fe] cursor-pointer" />
            <span class="font-body-md text-on-surface-variant group-hover:text-on-surface transition-colors">
              I have the rights to chart this audio material
            </span>
          </label>

          <ul v-if="blockers.length" class="text-[12px] text-tertiary-fixed-dim space-y-0.5">
            <li v-for="b in blockers" :key="b">• {{ b }}</li>
          </ul>

          <button
            type="button"
            class="w-full bg-primary-container hover:bg-[#a60ee5] text-on-primary-container font-headline-lg-mobile text-headline-lg-mobile py-4 rounded-xl transition-all shadow-[0_4px_20px_rgba(188,19,254,0.4)] active:scale-[0.98] flex justify-center items-center gap-3 disabled:opacity-40 disabled:shadow-none disabled:active:scale-100"
            :disabled="!canGenerate"
            @click="generate"
          >
            <span class="material-symbols-outlined" :class="{ spinner: running }">
              {{ running ? "progress_activity" : "auto_awesome" }}
            </span>
            {{ running ? "Generating…" : "Generate Charts" }}
          </button>
        </div>
      </div>
    </section>

    <!-- Live progress -->
    <section class="flex-1 bg-surface flex flex-col min-w-0">
      <header class="p-margin-desktop border-b border-outline-variant flex justify-between items-center bg-surface-container">
        <h2 class="font-headline-md text-headline-md text-on-surface">Generation Pipeline</h2>
        <div
          v-if="job"
          class="font-label-mono text-label-mono px-3 py-1 rounded-full flex items-center gap-2 border"
          :class="
            running
              ? 'text-secondary-container bg-secondary-container/10 border-secondary-container/20'
              : job.success
                ? 'text-green-400 bg-green-500/10 border-green-500/20'
                : 'text-error bg-error-container/20 border-error/20'
          "
        >
          <span class="w-2 h-2 rounded-full" :class="running ? 'bg-secondary-container animate-pulse' : job.success ? 'bg-green-400' : 'bg-error'"></span>
          {{ running ? "Running" : job.success ? "Complete" : "Failed" }}
        </div>
        <div v-else class="font-label-mono text-label-mono text-on-surface-variant">Idle</div>
      </header>

      <div class="p-margin-desktop flex-1 overflow-y-auto flex flex-col gap-6 min-h-0">
        <StagePipeline :stages="GEN_STAGES" :current="job?.stage ?? 0" :running="running" :details="stageDetails" />

        <div v-if="failure" class="rounded-lg border border-error/40 bg-error-container/20 p-3 text-error text-[12px] font-label-mono whitespace-pre-wrap">
          {{ failure }}
        </div>

        <div class="mt-auto rounded-xl bg-[#0a0a0c] border border-outline-variant p-4 font-label-mono text-label-mono flex flex-col min-h-[220px] max-h-[45vh]">
          <div class="flex justify-between items-center mb-3 border-b border-[#2a2a2d] pb-2 text-on-surface-variant text-[10px]">
            <span>PROCESS_LOG · {{ job?.id ?? "—" }}</span>
            <span v-if="job?.costEst != null" class="text-tertiary-fixed-dim">${{ job.costEst.toFixed(3) }} seen in log</span>
          </div>
          <div class="flex-1 overflow-y-auto space-y-0.5 text-[#8b949e]">
            <p v-if="!consoleLines.length" class="text-on-surface-variant/60">&gt; Idle — configure options and press Generate.</p>
            <p
              v-for="(l, i) in consoleLines"
              :key="i"
              :class="l.stream === 'stderr' ? 'text-error' : /DONE|->|complete/i.test(l.line) ? 'text-[#4caf50]' : ''"
            >
              &gt; {{ l.line }}
            </p>
          </div>
        </div>

        <div class="flex justify-center gap-4">
          <button
            v-if="running"
            type="button"
            class="px-6 py-2 rounded-lg border border-error text-error hover:bg-error/10 transition-colors font-body-md"
            @click="engine.cancel(job!.id)"
          >
            Cancel Generation
          </button>
          <button
            v-else-if="job?.success"
            type="button"
            class="px-6 py-2 rounded-lg bg-primary-container text-on-primary-container hover:bg-primary transition-colors font-body-md flex items-center gap-2"
            @click="router.push({ name: 'results', params: { base: baseId } })"
          >
            <span class="material-symbols-outlined text-[18px]">analytics</span> View Results
          </button>
        </div>
      </div>
    </section>
  </div>
</template>
