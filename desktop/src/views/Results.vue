<script setup lang="ts">
import { computed, onMounted, ref, shallowRef, watch } from "vue";
import { useRouter } from "vue-router";
import { open } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { convertFileSrc } from "@tauri-apps/api/core";
import { api } from "../lib/api";
import { useLibrary } from "../stores/library";
import { useEngine } from "../stores/engine";
import type { ResultDetail, ResultSet, Target } from "../lib/types";
import ScoreRing from "../components/ScoreRing.vue";
import GateChip from "../components/GateChip.vue";
import TransportBar from "../components/TransportBar.vue";
import NotePlayer from "../components/NotePlayer.vue";
import ContextMenu, { type MenuItem } from "../components/ContextMenu.vue";
import Spinner from "../components/Spinner.vue";
import SendFeedback from "../components/SendFeedback.vue";
import RhythmOptions from "../components/RhythmOptions.vue";
import { optionsJson, type DecodeOptions } from "../lib/decodeOptions";
import type { Chart } from "../lib/types";

const props = defineProps<{ base?: string }>();
const router = useRouter();
const lib = useLibrary();
const engine = useEngine();

const baseId = computed(() => props.base ?? lib.selectedBase ?? "");
const track = computed(() => lib.byBase(baseId.value));

const sets = ref<ResultSet[]>([]);
const target = ref<Target | null>(null);
const difficulty = ref<string>("");
const detail = ref<ResultDetail | null>(null);
const loading = ref(false);
const message = ref<string | null>(null);

// Player: watch the chart play against the song. STEPFORGE only — it's the only
// target that emits a simfile with real columns.
const mode = ref<"player" | "timeline">("player");
const chart = shallowRef<Chart | null>(null);
const songAudio = shallowRef<string | null>(null);
const chartError = ref<string | null>(null);
const menu = ref<InstanceType<typeof ContextMenu> | null>(null);
const canPlay = computed(() => target.value === "stepforge");

async function loadChart() {
  chart.value = null;
  chartError.value = null;
  if (!canPlay.value || !baseId.value || !difficulty.value) return;
  try {
    const [c, a] = await Promise.all([
      api.getChartNotes(baseId.value, difficulty.value),
      api.trackAudioPath(baseId.value),
    ]);
    chart.value = c;
    songAudio.value = a;
  } catch (e) {
    chartError.value = String(e);
  }
}

function resultMenu(e: MouseEvent) {
  const items: MenuItem[] = [
    { label: mode.value === "player" ? "Show timeline image" : "Show player", icon: "swap_horiz",
      action: () => { mode.value = mode.value === "player" ? "timeline" : "player"; } },
    { label: "Export song folder…", icon: "folder_zip", separatorBefore: true,
      disabled: !detail.value || target.value === "native", action: exportFolder },
    { label: "Reveal output", icon: "folder_open", disabled: !detail.value, action: reveal },
    { label: "Regenerate", icon: "autorenew",
      action: () => void router.push({ name: "generate", params: { base: baseId.value } }) },
  ];
  menu.value?.open(e, items);
}

const activeSet = computed(() => sets.value.find((s) => s.target === target.value));
/** Bumped after a section re-roll. The engine rewrites the preview PNG and the
 *  assist track at the SAME paths, so without a changed URL the webview would
 *  keep showing (and playing) the chart from before the re-roll. */
const artifactRev = ref(0);
const previewUrl = computed(() =>
  detail.value?.previewPng
    ? convertFileSrc(detail.value.previewPng) + (artifactRev.value ? `?v=${artifactRev.value}` : "")
    : "",
);

// --- Edit a section (neural StepMania charts only) ------------------------- //
// `stepforge-inpaint` re-does a range of bars and keeps the rest of the chart:
// either only which arrows its notes are ("arrows"), or the notes themselves,
// written again from the audio with settings of their own ("notes"). Only the
// neural engine can do that, so the control exists only for a result whose
// report says `mode: "neural"`.
const canReroll = computed(() => target.value === "stepforge" && detail.value?.mode === "neural");
const rerollWhat = ref<"arrows" | "notes">("arrows");
/** Settings for the range alone; the rest of the chart keeps its own. */
const rangeOptions = ref<DecodeOptions>({});
const rangeJumps = ref<"" | "none" | "more">("");
const rangeStyleJson = computed(() => {
  if (rerollWhat.value !== "notes" || !rangeJumps.value) return null;
  // "more" asks for the most the difficulty allows; the engine clamps it there
  return JSON.stringify({ jump_share: rangeJumps.value === "none" ? 0 : 0.35 });
});
const rangeOptionsJson = computed(() => {
  const o = rangeOptions.value;
  if (rerollWhat.value === "arrows") return optionsJson({ crossovers: o.crossovers });
  // a stream asked for a range fills the range
  return optionsJson(o.stream ? { ...o, stream_coverage: 1, min_break_bars: 1, density: undefined } : o);
});
const sectionPanel = ref<HTMLElement | null>(null);
const sectionFlash = ref(false);
/** Bring the section editor into view — it lives in the Inspector, where it
 *  was easy to miss. */
function showSectionEditor() {
  sectionPanel.value?.scrollIntoView({ behavior: "smooth", block: "start" });
  sectionFlash.value = true;
  window.setTimeout(() => (sectionFlash.value = false), 1600);
}
function markStart(bar: number) {
  rerollStart.value = bar;
  const e = wholeNumber(rerollEnd.value);
  if (e !== null && e <= bar) rerollEnd.value = "";
  showSectionEditor();
}
function markEnd(bar: number) {
  rerollEnd.value = bar;
  const s = wholeNumber(rerollStart.value);
  if (s !== null && s >= bar) rerollStart.value = "";
  showSectionEditor();
}
/** Number inputs hand back a number, or "" when blank. */
const rerollStart = ref<number | string>("");
const rerollEnd = ref<number | string>("");
const rerollSeed = ref<number | string>("");
const rerollJobId = ref<string | null>(null);
const rerollJob = computed(() => engine.job(rerollJobId.value));
const rerolling = computed(() => rerollJob.value?.running ?? false);
const rerollFailure = computed(() => engine.failureOf(rerollJobId.value));
/** Anything else running on this track (a generation, an analysis). A re-roll
 *  rewrites the song folder, so it must not overlap one. */
const trackBusy = computed(() => {
  const b = engine.busyFor(track.value?.id);
  return b && b.id !== rerollJobId.value ? b : undefined;
});

/** A whole number >= 0, or null. Rejects 1.5 and -1 rather than rounding them
 *  into a bar the user did not ask for. */
function wholeNumber(v: number | string): number | null {
  const s = String(v).trim();
  if (!/^\d+$/.test(s)) return null;
  const n = Number(s);
  return Number.isSafeInteger(n) ? n : null;
}
function isBlank(v: number | string) {
  return String(v).trim() === "";
}
/** How many bars the chart spans, as the engine recorded it in report.json
 *  (`charts[diff].bars`). The real limit: a valid end bar is 1..bars. Null for
 *  a report written before the engine recorded it. */
const reportBars = computed(() => {
  const b = detail.value?.extra?.bars;
  return typeof b === "number" && Number.isInteger(b) && b > 0 ? b : null;
});
/** Fallback for those older reports: the bars the chart's NOTES span — a
 *  measure is 4 beats in the emitted simfile. Only a guide, never a limit: the
 *  song can run on past its last note, and the engine has the final say. */
const notedBars = computed(() => {
  const notes = chart.value?.notes;
  if (!notes?.length) return null;
  let last = 0;
  for (const n of notes) if (n.beat > last) last = n.beat;
  return Math.floor(last / 4) + 1;
});
/** What the inputs offer as their maximum. */
const barCount = computed(() => reportBars.value ?? notedBars.value);
/** What is wrong with the form, if anything. Blank bars are "not filled in
 *  yet" (button disabled, no complaint); anything typed is checked. */
const rerollProblem = computed(() => {
  const s = wholeNumber(rerollStart.value);
  const e = wholeNumber(rerollEnd.value);
  if ((!isBlank(rerollStart.value) && s === null) || (!isBlank(rerollEnd.value) && e === null))
    return "Bars are whole numbers, counted from 0.";
  if (s !== null && e !== null && s >= e)
    return "The end bar must be greater than the start bar (the end bar itself is not included).";
  // Enforced only against the engine's own count, never the estimate from notes.
  const bars = reportBars.value;
  if (bars !== null && ((e !== null && e > bars) || (s !== null && s > bars - 1)))
    return `This chart has ${bars} bars: the start bar can be 0 to ${bars - 1} and the end bar at most ${bars}.`;
  if (!isBlank(rerollSeed.value) && wholeNumber(rerollSeed.value) === null)
    return "The seed is a whole number, or blank for a random one.";
  return null;
});
const rerollReady = computed(
  () =>
    canReroll.value &&
    !rerolling.value &&
    !trackBusy.value &&
    !rerollProblem.value &&
    wholeNumber(rerollStart.value) !== null &&
    wholeNumber(rerollEnd.value) !== null,
);

async function reroll() {
  if (!rerollReady.value) return;
  const j = await engine.startInpaint({
    // The CLI is driven by the catalogue id, which is the base for most tracks.
    trackId: track.value?.id ?? baseId.value,
    difficulty: difficulty.value,
    startBar: wholeNumber(rerollStart.value)!,
    endBar: wholeNumber(rerollEnd.value)!,
    seed: wholeNumber(rerollSeed.value),
    regenerate: rerollWhat.value === "notes",
    styleJson: rangeStyleJson.value,
    optionsJson: rangeOptionsJson.value,
  });
  rerollJobId.value = j.id;
}

// On success the song folder and report.json have been rewritten in place:
// reload the detail and the notes so the player, the timeline image and the
// metrics all show the new section.
watch(
  () => rerollJob.value?.running,
  async (isRunning, was) => {
    if (!was || isRunning || !rerollJob.value?.success) return;
    const summary = [...rerollJob.value.lines].reverse().find((l) => l.line.includes("[inpaint]"));
    message.value = summary?.line.trim() ?? "Section updated.";
    artifactRev.value++;
    await loadDetail();
  },
);

/** Metric rows worth surfacing, using the engine's real metric keys. */
const metricRows = computed(() => {
  const d = detail.value;
  if (!d) return [];
  const m = d.metrics ?? {};
  const rows: { label: string; value: string }[] = [];
  if (d.notes != null) rows.push({ label: "Notes", value: String(d.notes) });
  if (typeof m.event_count === "number") rows.push({ label: "Events", value: String(m.event_count) });
  if (typeof m.peak_nps_4s === "number") rows.push({ label: "Peak NPS (4s)", value: m.peak_nps_4s.toFixed(2) });
  if (typeof m.onset_alignment === "number")
    rows.push({ label: "Onset alignment", value: `${(m.onset_alignment * 100).toFixed(0)}%` });
  if (typeof m.repaired_fraction === "number")
    rows.push({ label: "Repaired", value: `${(m.repaired_fraction * 100).toFixed(1)}%` });
  const e = d.extra ?? {};
  if (e.meter != null) rows.push({ label: "Meter", value: String(e.meter) });
  if (e.options?.max_subdivision) rows.push({ label: "Max subdivision", value: `${e.options.max_subdivision}ths` });
  if (e.stream)
    rows.push({
      label: "Stream",
      value: `${e.stream.subdivision}ths · ${e.stream.stream_bars}/${e.stream.playable_bars} bars · ${Number(e.stream.nps).toFixed(1)}/s`,
    });
  if (e.options?.crossovers === false) rows.push({ label: "Crossovers", value: "off" });
  if (e.jumps != null) rows.push({ label: "Jumps", value: String(e.jumps) });
  if (e.holds != null) rows.push({ label: "Holds", value: String(e.holds) });
  if (e.njs != null) rows.push({ label: "NJS", value: String(e.njs) });
  if (e.bombs != null) rows.push({ label: "Bombs", value: String(e.bombs) });
  if (e.arcs != null) rows.push({ label: "Arcs", value: String(e.arcs) });
  if (e.simulatorClean != null) rows.push({ label: "Simulator", value: e.simulatorClean ? "clean" : "flagged" });
  return rows;
});

/** Difficulties in play order (StepMania, then Beat Saber names), not the
 *  alphabetical order the folder listing gives. */
const DIFFICULTY_ORDER = ["beginner", "easy", "medium", "hard", "challenge", "edit",
  "normal", "expert", "expertplus"];
function difficultyRank(d: string) {
  const i = DIFFICULTY_ORDER.indexOf(d.toLowerCase());
  return i < 0 ? DIFFICULTY_ORDER.length : i;
}

async function loadSets() {
  if (!lib.tracks.length) await lib.refresh();
  if (baseId.value) lib.select(baseId.value);
  const raw = baseId.value ? await api.listResults(baseId.value) : [];
  sets.value = raw.map((s) => ({
    ...s,
    difficulties: [...s.difficulties].sort(
      (a, b) => difficultyRank(a) - difficultyRank(b) || a.localeCompare(b),
    ),
  }));
  const first = sets.value[0];
  target.value = first?.target ?? null;
  difficulty.value = first?.difficulties[0] ?? "";
  await loadDetail();
}

async function loadDetail() {
  if (!target.value || !difficulty.value) {
    detail.value = null;
    return;
  }
  loading.value = true;
  try {
    detail.value = await api.getResult(baseId.value, target.value, difficulty.value);
  } catch (e) {
    detail.value = null;
    message.value = String(e);
  } finally {
    loading.value = false;
  }
  if (!canPlay.value) mode.value = "timeline";
  await loadChart();
}

watch(target, () => {
  difficulty.value = activeSet.value?.difficulties[0] ?? "";
  loadDetail();
});
watch(difficulty, loadDetail);
watch(() => props.base, loadSets);

async function exportFolder() {
  if (!target.value) return;
  if (target.value === "native") {
    message.value = "The native target writes beatmap JSON into build/analysis — use Reveal instead.";
    return;
  }
  const dest = await open({ directory: true, multiple: false, title: "Export song folder to…" });
  if (!dest) return;
  try {
    const out = await api.exportSongFolder(baseId.value, target.value, dest as string);
    message.value = `Exported to ${out}`;
  } catch (e) {
    message.value = `Export failed: ${e}`;
  }
}
async function reveal() {
  const p = detail.value?.previewPng || detail.value?.outDir;
  if (p) await revealItemInDir(p);
}

/** Follow the re-roll still running for THIS track, if there is one (jobs live
 *  in the store, so one survives navigating away and back) — and let go of
 *  another track's job, whose progress and errors do not belong on this page. */
function attachReroll() {
  const existing = engine.jobFor("inpaint", track.value?.id ?? baseId.value);
  rerollJobId.value = existing?.running ? existing.id : null;
}
watch(() => props.base, attachReroll);

onMounted(async () => {
  await loadSets();
  attachReroll();
});
</script>

<template>
  <div class="h-full flex">
    <ContextMenu ref="menu" />
    <main class="flex-1 flex flex-col h-full min-w-0">
      <header class="border-b border-outline-variant bg-surface px-margin-desktop py-4 shrink-0">
        <div class="flex items-center gap-3 mb-3">
          <h2 class="font-headline-lg text-headline-lg text-on-surface truncate">
            {{ track?.title ?? baseId ?? "No track" }}
          </h2>
          <span class="bg-surface-variant text-on-surface-variant px-2 py-1 rounded-sm font-label-mono text-[10px] uppercase tracking-wider flex items-center gap-1">
            <span class="material-symbols-outlined text-[12px]">smart_toy</span> StepForge chart
          </span>
          <span v-if="detail?.humanReview" class="bg-tertiary-container/20 text-tertiary-fixed-dim border border-tertiary-container/40 px-2 py-1 rounded-sm font-label-mono text-[10px] uppercase">
            needs human review
          </span>
          <!-- Feedback: the text is shown before anything opens (SendFeedback). -->
          <div class="ml-auto shrink-0 flex items-center gap-2">
            <SendFeedback
              v-if="detail"
              kind="chart"
              :base="baseId"
              :target="target"
              :difficulty="difficulty"
              :song-title="track?.title ?? baseId"
            />
            <SendFeedback />
          </div>
        </div>

        <div v-if="sets.length" class="flex flex-wrap items-center gap-4">
          <div class="flex gap-1">
            <button
              v-for="s in sets"
              :key="s.target"
              class="px-3 py-1.5 rounded font-label-mono text-label-mono border capitalize transition-all"
              :class="target === s.target ? 'bg-surface-variant text-primary border-primary' : 'text-on-surface-variant border-outline-variant hover:bg-surface-variant'"
              @click="target = s.target"
            >
              {{ s.target }}
              <span v-if="s.mode" class="opacity-60">· {{ s.mode }}</span>
            </button>
          </div>
          <div class="w-px h-5 bg-outline-variant"></div>
          <div class="flex gap-1 flex-wrap">
            <button
              v-for="d in activeSet?.difficulties ?? []"
              :key="d"
              class="px-4 py-1.5 rounded-full font-label-mono text-label-mono border capitalize transition-all"
              :class="difficulty === d ? 'bg-primary-container/10 text-primary border-primary' : 'text-on-surface-variant border-transparent hover:bg-surface-variant hover:border-outline-variant'"
              @click="difficulty = d"
            >
              {{ d }}
            </button>
          </div>
        </div>
        <p v-else class="font-body-md text-on-surface-variant">
          No charts generated for this track yet.
          <button class="text-primary underline" @click="router.push({ name: 'generate', params: { base: baseId } })">Generate some →</button>
        </p>
      </header>

      <!-- Main area: live chart player, or the engine's rendered timeline PNG -->
      <div class="flex-1 flex flex-col min-h-0 bg-surface-container-lowest" @contextmenu="resultMenu">
        <div v-if="detail" class="flex items-center gap-1 px-gutter py-2 border-b border-outline-variant shrink-0">
          <button
            v-for="m in (['player', 'timeline'] as const)"
            :key="m"
            class="px-3 py-1 rounded font-label-mono text-label-mono border transition-colors capitalize"
            :class="mode === m ? 'bg-surface-variant text-primary border-primary' : 'text-on-surface-variant border-transparent hover:bg-surface-variant'"
            :disabled="m === 'player' && !canPlay"
            :title="m === 'player' && !canPlay ? 'Only StepMania charts can be played' : ''"
            @click="mode = m"
          >
            <span class="material-symbols-outlined text-[14px] align-middle mr-1">
              {{ m === "player" ? "play_arrow" : "image" }}
            </span>
            {{ m }}
          </button>
          <button
            v-if="canReroll"
            type="button"
            class="ml-auto px-3 py-1 rounded font-label-mono text-label-mono border border-outline-variant text-on-surface hover:bg-surface-variant transition-colors flex items-center gap-1"
            title="Re-do a range of bars — new arrows, or new notes with their own settings — and keep the rest of the chart"
            @click="showSectionEditor"
          >
            <span class="material-symbols-outlined text-[14px]">autorenew</span>
            Edit a section
          </button>
        </div>

        <div class="flex-1 min-h-0">
          <div v-if="loading" class="h-full flex items-center justify-center gap-2 font-label-mono text-on-surface-variant">
            <Spinner :size="18" /> Loading result…
          </div>

          <!-- Live player -->
          <template v-else-if="mode === 'player' && canPlay">
            <div v-if="chartError" class="h-full flex items-center justify-center text-error font-label-mono text-label-mono px-6 text-center">
              {{ chartError }}
            </div>
            <NotePlayer
              v-else
              :audio-path="songAudio"
              :assist-path="detail?.clickOgg ?? null"
              :chart="chart"
              :autoplay="true"
              :rev="artifactRev"
              :selectable="canReroll"
              :range-start="wholeNumber(rerollStart)"
              :range-end="wholeNumber(rerollEnd)"
              @mark-start="markStart"
              @mark-end="markEnd"
            />
          </template>

          <!-- Rendered timeline PNG -->
          <div v-else class="h-full overflow-auto flex items-center justify-center p-4">
            <img v-if="previewUrl" :src="previewUrl" class="max-w-full max-h-full object-contain rounded-lg border border-outline-variant" alt="Chart timeline preview" />
            <div v-else-if="detail" class="text-center text-on-surface-variant font-body-md">
              <span class="material-symbols-outlined text-4xl opacity-40 block mb-2">image_not_supported</span>
              <p>No preview image was rendered for this difficulty.</p>
              <p class="text-[12px] mt-1">Previews need matplotlib (PNG) and ffmpeg (assist track).</p>
            </div>
            <div v-else class="text-on-surface-variant font-body-md">Select a difficulty to review.</div>
          </div>
        </div>
      </div>

      <!-- Assist track for timeline mode; in player mode the player owns audio
           (and has its own assist toggle), so only one source ever plays. -->
      <div
        v-if="mode === 'timeline' || !canPlay"
        class="h-16 bg-surface-container border-t border-outline-variant flex items-center px-gutter shrink-0"
      >
        <TransportBar :src="detail?.clickOgg" :rev="artifactRev">
          <span class="font-label-mono text-[11px] text-on-surface-variant uppercase tracking-wider shrink-0">
            {{ detail?.clickOgg ? "Assist track — hear notes on the music" : "No assist track rendered" }}
          </span>
        </TransportBar>
      </div>

      <div class="min-h-20 bg-surface-container-high border-t border-outline-variant flex items-center justify-between px-gutter shrink-0 gap-4 py-3">
        <p v-if="message" class="font-label-mono text-[11px] text-on-surface-variant truncate">{{ message }}</p>
        <div v-else></div>
        <div class="flex items-center gap-3">
          <button
            class="px-5 py-2.5 rounded text-on-surface bg-surface border border-outline-variant hover:bg-surface-variant transition-colors font-body-md flex items-center gap-2"
            @click="router.push({ name: 'generate', params: { base: baseId } })"
          >
            <span class="material-symbols-outlined text-[18px]">autorenew</span> Regenerate
          </button>
          <button
            class="px-5 py-2.5 rounded text-on-surface bg-surface border border-outline-variant hover:bg-surface-variant transition-colors font-body-md flex items-center gap-2 disabled:opacity-40"
            :disabled="!detail"
            title="Reveal the output folder so you can open it in GrooveAuthor / ArrowVortex / ChroMapper"
            @click="reveal"
          >
            <span class="material-symbols-outlined text-[18px]">folder_open</span> Reveal output
          </button>
          <button
            class="px-5 py-2.5 rounded bg-primary-container text-white hover:bg-primary transition-colors font-body-md shadow-[0_0_15px_rgba(188,19,254,0.3)] flex items-center gap-2 disabled:opacity-40"
            :disabled="!detail || target === 'native'"
            @click="exportFolder"
          >
            <span class="material-symbols-outlined text-[18px]">folder_zip</span> Export song folder
          </button>
        </div>
      </div>
    </main>

    <aside class="bg-surface-container border-l border-outline-variant w-sidebar-width shrink-0 flex flex-col overflow-y-auto">
      <div class="px-gutter py-4 border-b border-outline-variant">
        <h2 class="font-headline-md text-headline-md text-on-surface">Inspector</h2>
        <p class="font-body-md text-on-surface-variant capitalize">{{ target ?? "—" }} · {{ difficulty || "—" }}</p>
      </div>

      <div v-if="detail" class="p-gutter space-y-7">
        <!-- Edit a section: neural charts only (see `canReroll`). -->
        <section v-if="canReroll" ref="sectionPanel" class="scroll-mt-4">
          <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider mb-2">Edit a section</h3>
          <div
            class="bg-surface-container-low border rounded p-3 space-y-3 transition-colors duration-500"
            :class="sectionFlash ? 'border-primary shadow-[0_0_12px_rgba(188,19,254,0.35)]' : 'border-outline-variant/50'"
          >
            <div class="grid grid-cols-3 gap-2">
              <label class="block">
                <span class="block text-[11px] text-on-surface-variant mb-1">Start bar</span>
                <input
                  v-model="rerollStart"
                  type="number"
                  min="0"
                  step="1"
                  :max="barCount != null ? barCount - 1 : undefined"
                  placeholder="0"
                  :disabled="rerolling"
                  class="w-full bg-surface-container-lowest border border-outline-variant rounded-lg p-2 text-on-surface font-label-mono text-[12px] focus:border-secondary outline-none disabled:opacity-50"
                />
              </label>
              <label class="block">
                <span class="block text-[11px] text-on-surface-variant mb-1">End bar</span>
                <input
                  v-model="rerollEnd"
                  type="number"
                  min="1"
                  step="1"
                  :max="barCount ?? undefined"
                  :placeholder="barCount != null ? String(barCount) : ''"
                  :disabled="rerolling"
                  class="w-full bg-surface-container-lowest border border-outline-variant rounded-lg p-2 text-on-surface font-label-mono text-[12px] focus:border-secondary outline-none disabled:opacity-50"
                />
              </label>
              <label class="block">
                <span class="block text-[11px] text-on-surface-variant mb-1">Seed</span>
                <input
                  v-model="rerollSeed"
                  type="number"
                  min="0"
                  step="1"
                  placeholder="random"
                  :disabled="rerolling"
                  class="w-full bg-surface-container-lowest border border-outline-variant rounded-lg p-2 text-on-surface font-label-mono text-[12px] focus:border-secondary outline-none disabled:opacity-50"
                />
              </label>
            </div>
            <p class="text-[11px] text-on-surface-variant">
              Bars count from 0 and the end bar is not included, so 16 to 24 is eight bars.
              <template v-if="canPlay">
                Or play the chart and press <span class="font-label-mono">Start here</span> and
                <span class="font-label-mono">End here</span> under the player.
              </template>
              {{ " " }}
              <template v-if="reportBars != null">This chart has {{ reportBars }} bars (0 to {{ reportBars - 1 }}).</template>
              <template v-else-if="notedBars != null">This chart's notes run to bar {{ notedBars - 1 }}.</template>
            </p>

            <div class="flex items-center gap-1 bg-surface p-1 rounded-lg border border-outline-variant">
              <button
                v-for="w in ([['arrows', 'New arrows'], ['notes', 'New notes']] as const)"
                :key="w[0]"
                type="button"
                class="flex-1 py-1.5 rounded-md font-label-mono text-[11px] transition-colors disabled:opacity-40"
                :class="rerollWhat === w[0] ? 'bg-surface-container-high text-on-surface border border-outline-variant' : 'text-on-surface-variant hover:text-on-surface'"
                :disabled="rerolling"
                :aria-pressed="rerollWhat === w[0]"
                @click="rerollWhat = w[0]"
              >
                {{ w[1] }}
              </button>
            </div>
            <p class="text-[11px] text-on-surface-variant">
              <template v-if="rerollWhat === 'arrows'">
                Keeps every note where it is and chooses which arrows they are again. The rest of the chart is untouched.
              </template>
              <template v-else>
                Writes these bars again from the music — where the notes fall as well as which arrows they are — with the
                settings below, on top of the ones the chart was generated with. The rest of the chart is untouched.
              </template>
            </p>

            <template v-if="rerollWhat === 'notes'">
              <RhythmOptions v-model="rangeOptions" scope="range" :disabled="rerolling" :bpm="chart?.bpm ?? null" />
              <div class="flex items-center gap-3">
                <span class="w-16 shrink-0 text-[12px] text-on-surface-variant">Jumps</span>
                <select
                  v-model="rangeJumps"
                  :disabled="rerolling || rangeOptions.stream === true"
                  aria-label="Jumps"
                  class="flex-1 min-w-0 bg-surface-container-lowest border border-outline-variant text-on-surface rounded p-1.5 text-[12px] focus:border-secondary outline-none disabled:opacity-40"
                >
                  <option value="">As the chart has</option>
                  <option value="none">None</option>
                  <option value="more">The most allowed</option>
                </select>
              </div>
            </template>
            <label v-else class="flex items-center gap-2 cursor-pointer" :class="{ 'opacity-50': rerolling }">
              <input
                type="checkbox"
                class="accent-[#bc13fe] cursor-pointer"
                :checked="rangeOptions.crossovers === false"
                :disabled="rerolling"
                @change="rangeOptions = { ...rangeOptions, crossovers: ($event.target as HTMLInputElement).checked ? false : undefined }"
              />
              <span class="font-body-md text-[13px] text-on-surface">No crossovers</span>
            </label>

            <p v-if="rerollProblem" class="text-[11px] text-error">{{ rerollProblem }}</p>
            <p v-else-if="trackBusy" class="text-[11px] text-tertiary-fixed-dim">
              Another run is in progress for this track — wait for it to finish.
            </p>
            <button
              type="button"
              class="w-full px-4 py-2 rounded text-on-surface bg-surface border border-outline-variant hover:bg-surface-variant transition-colors font-body-md flex items-center justify-center gap-2 disabled:opacity-40"
              :disabled="!rerollReady"
              @click="reroll"
            >
              <span class="material-symbols-outlined text-[18px]" :class="{ spinner: rerolling }">
                {{ rerolling ? "progress_activity" : "autorenew" }}
              </span>
              {{ rerolling ? "Working…" : rerollWhat === "notes" ? "Rewrite these bars" : "Re-roll the arrows" }}
            </button>
            <div v-if="rerollFailure" class="rounded-lg border border-error/40 bg-error-container/20 p-2 text-error text-[11px] font-label-mono whitespace-pre-wrap">
              {{ rerollFailure }}
            </div>
          </div>
        </section>

        <section class="flex flex-col items-center text-center">
          <template v-if="detail.score != null">
            <!-- The engine's critic scores 0–10 (ship threshold 7). -->
            <ScoreRing :score="detail.score" :max="10" class="mb-3" />
            <p class="font-label-mono text-[10px] uppercase tracking-widest" :class="detail.score >= 7 ? 'text-green-400' : 'text-tertiary-fixed-dim'">
              {{ detail.score >= 7 ? "Above ship threshold (7)" : "Below ship threshold (7)" }}
            </p>
          </template>

          <blockquote v-if="detail.criticSummary" class="text-on-surface-variant font-body-md italic border-l-2 border-primary pl-3 text-left mt-3">
            "{{ detail.criticSummary }}"
          </blockquote>
        </section>

        <section v-if="detail.criticIssues?.length">
          <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider mb-2">Critic Issues</h3>
          <ul class="space-y-2">
            <li v-for="(iss, i) in detail.criticIssues" :key="i" class="bg-surface-container-low border border-outline-variant/50 rounded p-2 text-[12px]">
              <span v-if="iss.where" class="text-primary font-label-mono">{{ iss.where }}: </span>
              <span class="text-on-surface-variant">{{ iss.problem }}</span>
            </li>
          </ul>
        </section>

        <section v-if="detail.gates.length">
          <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider mb-3">Gates</h3>
          <div class="space-y-2">
            <GateChip
              v-for="g in detail.gates"
              :key="g.name"
              :label="g.name.replace(/_/g, ' ')"
              :state="g.pass ? 'pass' : 'fail'"
            />
          </div>
          <p
            v-if="detail.extra?.stream && detail.gates.some((g) => g.name === 'onset_alignment' && !g.pass)"
            class="text-[11px] text-on-surface-variant mt-2"
          >
            Onset alignment asks for a sound under nearly every note. A stream plays every slot of its bars whether or not
            there is one, so in Stream mode this check is expected to fail.
          </p>
        </section>

        <section v-if="metricRows.length">
          <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider mb-3">Chart Metrics</h3>
          <div class="bg-[#0f0f12] border border-outline-variant rounded p-3 font-label-mono text-[12px] text-secondary space-y-1">
            <div v-for="r in metricRows" :key="r.label" class="flex justify-between gap-2">
              <span>{{ r.label }}:</span><span class="text-on-surface">{{ r.value }}</span>
            </div>
          </div>
        </section>

        <section v-if="detail.designNotes" class="border border-outline-variant rounded bg-surface-container-low overflow-hidden">
          <details class="group" open>
            <summary class="flex justify-between items-center font-label-mono text-label-mono font-bold cursor-pointer list-none p-3 hover:bg-surface-variant transition-colors">
              <span class="uppercase tracking-wider text-on-surface">Design Notes</span>
              <span class="transition group-open:rotate-180 material-symbols-outlined text-on-surface-variant">expand_more</span>
            </summary>
            <div class="text-on-surface-variant font-body-md text-[13px] p-3 border-t border-outline-variant bg-surface-container whitespace-pre-wrap">
              {{ detail.designNotes }}
            </div>
          </details>
        </section>
      </div>
      <div v-else class="p-gutter text-on-surface-variant font-body-md text-[13px]">Nothing to inspect yet.</div>
    </aside>
  </div>
</template>
