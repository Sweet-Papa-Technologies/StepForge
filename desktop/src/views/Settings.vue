<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { useSettings } from "../stores/settings";
import { useLibrary } from "../stores/library";
import { useNeural } from "../stores/neural";
import { api } from "../lib/api";
import type { PackInfo } from "../lib/types";
import { engineAdvice, engineProblems, neuralProblem } from "../lib/engineProblems";
import CopyDiagnostics from "../components/CopyDiagnostics.vue";

const settings = useSettings();
const lib = useLibrary();
const neural = useNeural();
const saved = ref(false);
const packs = ref<PackInfo | null>(null);

async function pickDir(field: "engineRoot" | "outputDir" | "songsDir") {
  const dir = await open({ directory: true, multiple: false });
  if (dir) settings.value[field] = dir as string;
}
async function pickFile(field: "pythonPath" | "ffmpegPath") {
  const f = await open({ multiple: false });
  if (f) settings.value[field] = f as string;
}

const rechecking = ref(false);

/** Engine and model problems, each with its real cause (lib/engineProblems). */
const engineIssues = computed(() => (lib.status ? engineProblems(lib.status) : []));
const neuralIssue = computed(() => neuralProblem(neural.status));
/** The model's files are in place — true even when onnxruntime, which runs
 *  them, is what failed (that is not "model not installed"). */
const modelFiles = computed(() => !!neural.status?.weights_present);

// --- Neural engine (StepForge Flow) ---------------------------------------- //
// No credential and nothing to test against a server: the only question is
// whether the weights and onnxruntime are there.
const neuralChecking = ref(false);

async function recheckNeural() {
  neuralChecking.value = true;
  try {
    await settings.save();
    await neural.refresh();
  } finally {
    neuralChecking.value = false;
  }
}

function fmtParams(n: number) {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n);
}
/** The model's own reported numbers. Their names are the trainer's to choose,
 *  so only plain values are listed, as they come. */
const neuralMetrics = computed(() =>
  Object.entries(neural.status?.metrics ?? {}).flatMap(([k, v]) => {
    if (typeof v === "number") return [{ label: k, value: Number.isInteger(v) ? String(v) : v.toFixed(3) }];
    if (typeof v === "string" || typeof v === "boolean") return [{ label: k, value: String(v) }];
    return [];
  }));

/** Settings that change what the engine probes find; the rest never need one. */
const PROBED = ["engineRoot", "pythonPath", "ffmpegPath", "songsDir", "outputDir"] as const;
function probedState() {
  return PROBED.map((k) => settings.value[k]).join("\u0000");
}
let lastProbed = "";

async function save() {
  window.clearTimeout(autoTimer);
  await settings.save();
  saved.value = true;
  setTimeout(() => (saved.value = false), 2000);
  // Only re-probe when a path changed: the probe starts a Python process, and
  // an auto-save after every keystroke pause must not.
  const now = probedState();
  if (now !== lastProbed) {
    lastProbed = now;
    await Promise.all([lib.loadStatus(true), lib.refresh(), neural.refresh()]);
    packs.value = await api.listPacks();
  }
}

/* Auto-save. Edits used to be lost by leaving the page without scrolling back
 * up to Save; now a change is written shortly after the last edit, and on
 * leaving the page. The Save button still saves at once. */
let armed = false;
let autoTimer: number | undefined;
const pending = ref(false);
watch(
  () => settings.value,
  () => {
    if (!armed) return;
    pending.value = true;
    window.clearTimeout(autoTimer);
    autoTimer = window.setTimeout(async () => {
      await save();
      pending.value = false;
    }, 700);
  },
  { deep: true },
);
onBeforeUnmount(() => {
  window.clearTimeout(autoTimer);
  if (pending.value) void settings.save();
});


async function recheck() {
  rechecking.value = true;
  try {
    await lib.loadStatus(true);
  } finally {
    rechecking.value = false;
  }
}

onMounted(async () => {
  if (!settings.loaded) await settings.load();
  packs.value = await api.listPacks();
  await lib.loadStatus();
  void neural.load();
  lastProbed = probedState();
  armed = true;
});
</script>

<template>
  <div class="h-full flex flex-col">
    <header class="h-16 border-b border-outline-variant flex items-center justify-between px-margin-desktop shrink-0 bg-surface">
      <h2 class="font-headline-md text-headline-md text-on-surface">Settings</h2>
      <div class="flex items-center gap-3">
        <span v-if="pending || settings.saving" class="font-label-mono text-label-mono text-on-surface-variant">Saving…</span>
        <span v-else-if="saved" class="font-label-mono text-label-mono text-green-400">Saved ✓</span>
        <span v-else class="font-label-mono text-[11px] text-on-surface-variant">Changes save automatically</span>
        <button
          class="bg-primary-container text-on-primary-container px-6 py-2 rounded-full font-label-mono hover:bg-primary transition-colors flex items-center gap-2 disabled:opacity-50"
          :disabled="settings.saving"
          title="Save now"
          @click="save"
        >
          <span class="material-symbols-outlined text-[18px]">save</span> Save
        </button>
      </div>
    </header>

    <div class="flex-1 overflow-y-auto p-margin-desktop max-w-3xl">
      <!-- Only shown when something is wrong -->
      <section v-if="engineIssues.length" class="mb-10">
        <div class="mt-3 p-3 rounded-lg border border-error/40 bg-error-container/20">
          <p class="font-label-mono text-label-mono text-error mb-1">The charting engine is not working</p>
          <p
            v-for="issue in engineIssues"
            :key="issue"
            class="text-[11px] text-on-surface-variant font-label-mono whitespace-pre-wrap break-words"
          >
            {{ issue }}
          </p>
          <p class="text-[12px] text-on-surface-variant mt-2">{{ engineAdvice(lib.status) }}</p>
        </div>
        <div class="mt-3"><CopyDiagnostics /></div>
      </section>

      <!-- Chart model. Local ONNX models: no credential and
           no network, so this is a status panel rather than a login form. The
           engine itself is chosen per run on the Generate screen. -->
      <section class="mb-10 space-y-4">
        <div class="flex items-center justify-between">
          <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider">
            Chart model
          </h3>
          <button
            class="px-3 py-1 rounded border border-outline-variant text-on-surface hover:bg-surface-variant font-label-mono text-label-mono flex items-center gap-2 disabled:opacity-50"
            :disabled="neuralChecking || neural.loading"
            @click="recheckNeural"
          >
            <span class="material-symbols-outlined text-[14px]" :class="{ spinner: neuralChecking || neural.loading }">refresh</span>
            Re-check
          </button>
        </div>
        <p class="text-[11px] text-on-surface-variant">
          Two small models that chart on this machine — no API key, no network and no per-chart cost. StepMania
          charts only. Pick it per run on the Generate screen.
        </p>

        <div class="grid grid-cols-3 gap-gutter">
          <div class="bg-surface-container border border-outline-variant rounded-lg p-3 flex items-center gap-2">
            <span
              class="material-symbols-outlined"
              :class="!neural.status ? 'text-on-surface-variant' : modelFiles ? 'text-green-400' : 'text-error'"
            >
              {{ !neural.status ? "progress_activity" : modelFiles ? "check_circle" : "cancel" }}
            </span>
            <span class="font-label-mono text-[12px] text-on-surface">
              {{ !neural.status ? "Checking…" : modelFiles ? "Model files present" : "Model files missing" }}
            </span>
          </div>
          <div class="bg-surface-container border border-outline-variant rounded-lg p-3 flex items-center gap-2">
            <span
              class="material-symbols-outlined"
              :class="!neural.status ? 'text-on-surface-variant' : neural.status.onnxruntime ? 'text-green-400' : 'text-error'"
            >
              {{ !neural.status ? "progress_activity" : neural.status.onnxruntime ? "check_circle" : "cancel" }}
            </span>
            <span class="font-label-mono text-[12px] text-on-surface">onnxruntime</span>
          </div>
          <div class="bg-surface-container border border-outline-variant rounded-lg p-3 flex items-center gap-2 min-w-0">
            <span class="font-label-mono text-[12px] text-on-surface-variant shrink-0">Quantization</span>
            <span class="font-label-mono text-[12px] text-on-surface truncate">{{ neural.status?.quantization || "—" }}</span>
          </div>
        </div>

        <div>
          <p class="font-body-md text-body-md text-on-surface">
            {{ neural.status?.display_name || "StepForge Flow" }}
            <span class="text-[11px] text-on-surface-variant">· the model that ships with the app</span>
          </p>
          <p v-if="neural.status?.model_dir" class="text-[11px] mt-1 break-all" :class="modelFiles ? 'text-green-400' : 'text-tertiary-fixed-dim'">
            {{ neural.status.model_dir }} · {{ modelFiles ? "model found" : "no model here" }}
          </p>
          <p class="text-[11px] text-on-surface-variant mt-1">
            Training data and credits: <code class="font-label-mono">MODEL_CARD.md</code> and
            <code class="font-label-mono">CREDITS.csv</code> in that folder.
          </p>
        </div>

        <div v-if="neural.installed && neural.status" class="bg-[#0f0f12] border border-outline-variant rounded p-3 font-label-mono text-[12px] text-secondary space-y-1">
          <div class="flex justify-between gap-2">
            <span>Parameters:</span>
            <span class="text-on-surface">
              <template v-if="neural.status.params">
                rhythm {{ fmtParams(neural.status.params.stage_a) }} · choreography {{ fmtParams(neural.status.params.stage_b) }}
              </template>
              <template v-else>—</template>
            </span>
          </div>
          <div class="flex justify-between gap-2">
            <span>Codec version:</span><span class="text-on-surface">{{ neural.status.codec_version ?? "—" }}</span>
          </div>
          <div class="flex justify-between gap-2">
            <span>Trained with stems:</span><span class="text-on-surface">{{ neural.status.trained_with_stems ? "yes" : "no" }}</span>
          </div>
          <div class="flex justify-between gap-2">
            <span>Style retrieval:</span><span class="text-on-surface">{{ neural.status.style_rag_available ? "available" : "not available" }}</span>
          </div>
          <div v-if="neural.status.presets.length" class="flex justify-between gap-2">
            <span class="shrink-0">Presets:</span><span class="text-on-surface text-right">{{ neural.status.presets.join(", ") }}</span>
          </div>
          <div v-for="m in neuralMetrics" :key="m.label" class="flex justify-between gap-2">
            <span>{{ m.label }}:</span><span class="text-on-surface">{{ m.value }}</span>
          </div>
        </div>

        <div v-if="neuralIssue" class="p-3 rounded-lg border border-error/40 bg-error-container/20">
          <p class="font-label-mono text-label-mono text-error mb-1">{{ neuralIssue.title }}</p>
          <p v-if="neuralIssue.detail" class="text-[11px] text-on-surface-variant whitespace-pre-wrap break-words">{{ neuralIssue.detail }}</p>
          <p v-if="!neural.legacy" class="text-[12px] text-on-surface-variant mt-2">{{ engineAdvice(lib.status) }}</p>
          <div class="mt-2"><CopyDiagnostics /></div>
        </div>
      </section>

      <!-- Game and output -->
      <section class="mb-10 space-y-4">
        <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider">Game &amp; files</h3>
        <div class="space-y-4">
          <div>
            <label class="block font-body-md text-body-md text-on-surface mb-1">
              Game <code class="font-label-mono">Songs</code> folder (for “Install to game”)
            </label>
            <div class="flex gap-2">
              <input
                v-model="settings.value.songsDir"
                :placeholder="packs?.songsDir ?? 'OS default'"
                class="flex-1 bg-surface-container-lowest border border-outline-variant rounded-lg p-2.5 text-on-surface font-label-mono text-[12px] focus:border-secondary outline-none"
              />
              <button class="px-3 rounded-lg border border-outline-variant text-on-surface hover:bg-surface-variant" @click="pickDir('songsDir')">
                <span class="material-symbols-outlined text-[18px]">folder_open</span>
              </button>
            </div>
            <p class="text-[11px] mt-1" :class="packs?.songsDirExists ? 'text-green-400' : 'text-tertiary-fixed-dim'">
              {{ packs?.songsDir ?? "—" }}
              <span v-if="packs">· {{ packs.songsDirExists ? "found" : "not found" }}</span>
            </p>
            <p v-if="packs?.detected?.length" class="text-[11px] text-on-surface-variant mt-1">
              Detected:
              <button
                v-for="d in packs.detected"
                :key="d"
                class="underline hover:text-on-surface mr-2"
                @click="settings.value.songsDir = d"
              >{{ d }}</button>
            </p>
          </div>

          <div>
            <label class="block font-body-md text-body-md text-on-surface mb-1">Output folder (optional)</label>
            <div class="flex gap-2">
              <input v-model="settings.value.outputDir" class="flex-1 bg-surface-container-lowest border border-outline-variant rounded-lg p-2.5 text-on-surface font-label-mono text-[12px] focus:border-secondary outline-none" />
              <button class="px-3 rounded-lg border border-outline-variant text-on-surface hover:bg-surface-variant" @click="pickDir('outputDir')"><span class="material-symbols-outlined text-[18px]">folder_open</span></button>
            </div>
          </div>
        </div>
      </section>

      <!-- Appearance -->
      <section class="mb-10 space-y-4">
        <h3 class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider">Appearance</h3>
        <div>
          <label class="block font-body-md text-body-md text-on-surface mb-1">Theme</label>
          <select v-model="settings.value.theme" class="w-full bg-surface-container-lowest border border-outline-variant rounded-lg p-2.5 text-on-surface font-body-md focus:border-secondary outline-none">
            <option value="dark">Dark</option>
            <option value="light">Light (experimental)</option>
          </select>
        </div>
      </section>

      <!-- Advanced: the engine's location and tools. Release builds bundle all of
           this, so it only matters when running from source. -->
      <details class="mb-10 group">
        <summary class="cursor-pointer select-none font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider flex items-center gap-2">
          <span class="material-symbols-outlined text-[16px] transition-transform group-open:rotate-90">chevron_right</span>
          Advanced
        </summary>
        <div class="mt-4 space-y-4">
          <div class="flex items-center justify-between">
            <span class="text-[12px] text-on-surface-variant">Engine status</span>
            <button
              class="px-3 py-1 rounded border border-outline-variant text-on-surface hover:bg-surface-variant font-label-mono text-label-mono flex items-center gap-2 disabled:opacity-50"
              :disabled="rechecking"
              @click="recheck"
            >
              <span class="material-symbols-outlined text-[14px]" :class="{ spinner: rechecking }">refresh</span>
              Re-check
            </button>
          </div>
        <div class="grid grid-cols-4 gap-gutter">
          <div class="bg-surface-container border border-outline-variant rounded-lg p-3 flex items-center gap-2">
            <span class="material-symbols-outlined" :class="lib.status?.pythonOk ? 'text-green-400' : 'text-error'">
              {{ lib.status?.pythonOk ? "check_circle" : "cancel" }}
            </span>
            <span class="font-label-mono text-[12px] text-on-surface">Python</span>
          </div>
          <div class="bg-surface-container border border-outline-variant rounded-lg p-3 flex items-center gap-2">
            <span class="material-symbols-outlined" :class="lib.status?.beatforgeOk ? 'text-green-400' : 'text-error'">
              {{ lib.status?.beatforgeOk ? "check_circle" : "cancel" }}
            </span>
            <span class="font-label-mono text-[12px] text-on-surface">charting engine</span>
          </div>
          <div class="bg-surface-container border border-outline-variant rounded-lg p-3 flex items-center gap-2">
            <span class="material-symbols-outlined" :class="lib.status?.ffmpegOk ? 'text-green-400' : 'text-tertiary'">
              {{ lib.status?.ffmpegOk ? "check_circle" : "error" }}
            </span>
            <span class="font-label-mono text-[12px] text-on-surface">ffmpeg</span>
          </div>
          <!-- Amber, never red: only the neural engine needs it. -->
          <div
            class="bg-surface-container border border-outline-variant rounded-lg p-3 flex items-center gap-2"
            title="Needed by the neural engine only"
          >
            <span class="material-symbols-outlined" :class="lib.status?.onnxruntimeOk ? 'text-green-400' : 'text-tertiary'">
              {{ lib.status?.onnxruntimeOk ? "check_circle" : "error" }}
            </span>
            <span class="font-label-mono text-[12px] text-on-surface">onnxruntime</span>
          </div>
        </div>
        <p v-if="lib.status?.engineRoot" class="font-label-mono text-[11px] text-on-surface-variant mt-2">
          Engine root: {{ lib.status.engineRoot }}
        </p>
          <p v-if="lib.status?.engineRoot" class="font-label-mono text-[11px] text-on-surface-variant break-all">
            Engine root: {{ lib.status.engineRoot }}
          </p>
        <p v-if="lib.status && !lib.status.ffmpegOk" class="text-[12px] text-tertiary-fixed-dim mt-2">
          ffmpeg was not found ({{ lib.status.ffmpeg }}). WAV, MP3, FLAC and OGG still import without it; other formats
          (M4A/AAC, video files) need it.
        </p>
          <div>
            <label class="block font-body-md text-body-md text-on-surface mb-1">Engine root (folder containing <code class="font-label-mono">tools/</code>)</label>
            <div class="flex gap-2">
              <input v-model="settings.value.engineRoot" class="flex-1 bg-surface-container-lowest border border-outline-variant rounded-lg p-2.5 text-on-surface font-label-mono text-[12px] focus:border-secondary focus:ring-1 focus:ring-secondary outline-none" />
              <button class="px-3 rounded-lg border border-outline-variant text-on-surface hover:bg-surface-variant" @click="pickDir('engineRoot')"><span class="material-symbols-outlined text-[18px]">folder_open</span></button>
            </div>
          </div>
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block font-body-md text-body-md text-on-surface mb-1">Python path</label>
              <div class="flex gap-2">
                <input v-model="settings.value.pythonPath" class="flex-1 bg-surface-container-lowest border border-outline-variant rounded-lg p-2.5 text-on-surface font-label-mono text-[12px] focus:border-secondary outline-none" />
                <button class="px-3 rounded-lg border border-outline-variant text-on-surface hover:bg-surface-variant" @click="pickFile('pythonPath')"><span class="material-symbols-outlined text-[18px]">folder_open</span></button>
              </div>
            </div>
            <div>
              <label class="block font-body-md text-body-md text-on-surface mb-1">ffmpeg path (optional)</label>
              <div class="flex gap-2">
                <input v-model="settings.value.ffmpegPath" class="flex-1 bg-surface-container-lowest border border-outline-variant rounded-lg p-2.5 text-on-surface font-label-mono text-[12px] focus:border-secondary outline-none" />
                <button class="px-3 rounded-lg border border-outline-variant text-on-surface hover:bg-surface-variant" @click="pickFile('ffmpegPath')"><span class="material-symbols-outlined text-[18px]">folder_open</span></button>
              </div>
            </div>
          </div>
        </div>
      </details>
    </div>
  </div>
</template>
