<script setup lang="ts">
/** "Send feedback" / "Report this chart": a button, and the dialog that shows
 *  the exact text of a GitHub issue before the browser opens GitHub's form
 *  with it filled in.
 *
 *  The app sends nothing: the user reads the text here, then again on GitHub,
 *  and submits it there (or does not). The text is built in `feedback.rs` from
 *  the app version, the OS, the model, the Copy diagnostics report and — for a
 *  chart — its difficulty and recorded settings. No audio, no file paths, and
 *  the song's title only if the box is ticked. */
import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";
import { openUrl } from "@tauri-apps/plugin-opener";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import { api } from "../lib/api";
import { feedbackIssue, NEW_ISSUE_PAGE, type FeedbackIssue } from "../lib/feedback";
import { useNeural } from "../stores/neural";
import Spinner from "./Spinner.vue";

const props = withDefaults(
  defineProps<{
    /** `chart` reports the chart named by the other props; `general` is anything else. */
    kind?: "general" | "chart";
    base?: string;
    target?: string | null;
    difficulty?: string;
    /** Offered behind the "Include the song title" box; never sent otherwise. */
    songTitle?: string;
  }>(),
  { kind: "general" },
);

const neural = useNeural();
const isChart = computed(() => props.kind === "chart");
const label = computed(() => (isChart.value ? "Report this chart" : "Send feedback"));

const open = ref(false);
const collecting = ref(false);
const includeTitle = ref(false);
const issue = ref<FeedbackIssue | null>(null);
const error = ref<string | null>(null);
const openFailed = ref(false);
const copied = ref(false);
const panel = ref<HTMLElement | null>(null);

/** The Copy diagnostics text, collected once per opening: the self-test takes
 *  seconds, and ticking the box must not run it again. */
let diagnostics = "";
/** Bumped on every opening and closing, so a slow answer for a dialog that has
 *  since been closed (or rebuilt) is dropped. */
let generation = 0;

/** The installed model's name, as Settings shows it. */
function modelName(): string | null {
  const s = neural.status;
  return s?.display_name ?? (s?.version ? `StepForge Flow v${s.version}` : null);
}

async function build() {
  const mine = generation;
  error.value = null;
  try {
    const built = await feedbackIssue({
      kind: props.kind,
      base: props.base,
      target: props.target ?? undefined,
      difficulty: props.difficulty,
      songTitle: props.songTitle,
      includeTitle: isChart.value && includeTitle.value,
      model: modelName(),
      diagnostics,
    });
    if (mine === generation) issue.value = built;
  } catch (e) {
    if (mine !== generation) return;
    issue.value = null;
    error.value = `The report could not be put together: ${e}`;
  }
}

async function show() {
  const mine = ++generation;
  open.value = true;
  includeTitle.value = false;
  issue.value = null;
  error.value = null;
  openFailed.value = false;
  copied.value = false;
  collecting.value = true;
  window.addEventListener("keydown", onKey);
  void nextTick(() => panel.value?.focus());
  try {
    const [report] = await Promise.all([
      api.engineDiagnostics().catch(() => "Diagnostics could not be collected."),
      neural.load(),
    ]);
    if (mine !== generation) return;
    diagnostics = report;
    await build();
  } finally {
    if (mine === generation) collecting.value = false;
  }
}

function close() {
  generation++;
  open.value = false;
  collecting.value = false;
  window.removeEventListener("keydown", onKey);
}
function onKey(e: KeyboardEvent) {
  if (e.key === "Escape") close();
}
onBeforeUnmount(() => window.removeEventListener("keydown", onKey));

// Ticking (or clearing) the box rewrites the text shown, so what is on screen
// is always what the link carries.
watch(includeTitle, () => {
  if (open.value && !collecting.value) {
    generation++;
    void build();
  }
});

async function openOnGitHub() {
  if (!issue.value) return;
  try {
    await openUrl(issue.value.url);
    close();
  } catch {
    openFailed.value = true;
  }
}

async function copyText() {
  if (!issue.value) return;
  try {
    await writeText(`${issue.value.title}\n\n${issue.value.body}`);
    copied.value = true;
    window.setTimeout(() => (copied.value = false), 2000);
  } catch {
    /* The text is on screen and can be selected by hand. */
  }
}
</script>

<template>
  <button
    type="button"
    class="px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant font-label-mono text-label-mono flex items-center gap-2 whitespace-nowrap"
    :title="
      isChart
        ? 'Open a GitHub issue about this chart. You see the text first; no audio is attached.'
        : 'Open a GitHub issue. You see the text first; nothing is sent from the app.'
    "
    @click="show"
  >
    <span class="material-symbols-outlined text-[14px]">{{ isChart ? "flag" : "chat" }}</span>
    {{ label }}
  </button>

  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-[70] bg-black/60 flex items-center justify-center p-6"
      @click.self="close"
    >
      <div
        ref="panel"
        role="dialog"
        aria-modal="true"
        :aria-label="label"
        tabindex="-1"
        class="w-full max-w-2xl max-h-full flex flex-col rounded-lg bg-surface-container border border-outline-variant shadow-xl outline-none text-on-surface"
      >
        <header class="px-5 py-4 border-b border-outline-variant flex items-center justify-between gap-3 shrink-0">
          <h3 class="font-headline-md text-headline-md text-on-surface">{{ label }}</h3>
          <button
            type="button"
            class="text-on-surface-variant hover:text-on-surface flex items-center"
            aria-label="Close"
            @click="close"
          >
            <span class="material-symbols-outlined">close</span>
          </button>
        </header>

        <div class="px-5 py-4 space-y-3 overflow-y-auto min-h-0">
          <p class="text-[12px] text-on-surface-variant">
            This is the text of a new GitHub issue. “Open on GitHub” opens GitHub’s form in your browser with it filled
            in, where you can add to it or change it. Nothing is sent until you submit it there (that needs a GitHub
            account). No audio, files or folder names are included.
          </p>

          <label v-if="isChart && songTitle" class="flex items-start gap-2 cursor-pointer">
            <input v-model="includeTitle" type="checkbox" class="mt-0.5 w-4 h-4 accent-[#bc13fe] cursor-pointer" />
            <span class="min-w-0">
              <span class="block font-body-md text-[13px] text-on-surface">Include the song title</span>
              <span class="block text-[11px] text-on-surface-variant break-words">
                “{{ songTitle }}” — left out unless you tick this. Issues on GitHub are public.
              </span>
            </span>
          </label>

          <div v-if="collecting" class="flex items-center gap-2 font-label-mono text-label-mono text-on-surface-variant py-6">
            <Spinner :size="16" label="Collecting diagnostics" />
            Collecting diagnostics (this runs the engine self-test and takes a few seconds)…
          </div>

          <template v-else-if="issue">
            <div>
              <p class="font-label-mono text-[10px] uppercase tracking-widest text-on-surface-variant mb-1">Title</p>
              <p class="bg-[#0f0f12] border border-outline-variant rounded px-3 py-2 font-label-mono text-[12px] text-on-surface break-words select-text">
                {{ issue.title }}
              </p>
            </div>
            <div>
              <p class="font-label-mono text-[10px] uppercase tracking-widest text-on-surface-variant mb-1">Text</p>
              <pre
                class="max-h-80 overflow-y-auto whitespace-pre-wrap break-words bg-[#0f0f12] border border-outline-variant rounded p-3 font-label-mono text-[11px] text-on-surface-variant select-text"
              >{{ issue.body }}</pre>
            </div>
            <p v-if="issue.truncated" class="text-[11px] text-tertiary-fixed-dim">
              The diagnostics were shortened to fit in a link (the text says where). If we ask for the rest, Copy
              diagnostics in Settings has the full text.
            </p>
          </template>

          <p v-if="error" class="text-[12px] text-error whitespace-pre-wrap break-words">{{ error }}</p>
          <p v-if="openFailed" class="text-[12px] text-tertiary-fixed-dim break-words select-text">
            Your browser could not be opened with the form filled in. Use “Copy text”, then paste it into a new issue
            at {{ NEW_ISSUE_PAGE }}
          </p>
        </div>

        <footer class="px-5 py-3 border-t border-outline-variant flex items-center justify-end gap-2 shrink-0">
          <button
            type="button"
            class="mr-auto px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant font-label-mono text-label-mono flex items-center gap-2 disabled:opacity-50"
            :disabled="!issue"
            @click="copyText"
          >
            <span class="material-symbols-outlined text-[14px]">{{ copied ? "check" : "content_copy" }}</span>
            {{ copied ? "Copied" : "Copy text" }}
          </button>
          <button
            type="button"
            class="px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant font-label-mono text-label-mono"
            @click="close"
          >
            Cancel
          </button>
          <button
            type="button"
            class="px-4 py-1.5 rounded bg-primary-container text-on-primary-container hover:bg-primary transition-colors font-label-mono text-label-mono flex items-center gap-2 disabled:opacity-50"
            :disabled="!issue"
            @click="openOnGitHub"
          >
            Open on GitHub
            <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
          </button>
        </footer>
      </div>
    </div>
  </Teleport>
</template>
