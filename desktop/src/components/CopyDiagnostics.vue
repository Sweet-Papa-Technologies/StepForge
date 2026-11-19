<script setup lang="ts">
/** Collects the diagnostics report (app, OS, CPU, engine self-test, last error)
 *  and puts it on the clipboard, for pasting into a bug report. The report has
 *  no audio, no song names and no folder names from this computer. */
import { computed, ref } from "vue";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import { api } from "../lib/api";

const state = ref<"idle" | "working" | "copied" | "failed">("idle");
const report = ref("");
const icon = computed(() =>
  state.value === "working" ? "progress_activity" : state.value === "copied" ? "check" : "content_copy",
);

async function copy() {
  if (state.value === "working") return;
  state.value = "working";
  try {
    report.value = await api.engineDiagnostics();
    await writeText(report.value);
    state.value = "copied";
  } catch (e) {
    // The clipboard can refuse; the text is still shown below to select by hand.
    if (!report.value) report.value = `Could not collect diagnostics: ${e}`;
    state.value = "failed";
  }
}
</script>

<template>
  <div class="min-w-0">
    <button
      type="button"
      class="px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant font-label-mono text-label-mono flex items-center gap-2 disabled:opacity-60 whitespace-nowrap"
      :disabled="state === 'working'"
      title="App version, system, engine self-test and the last error. No audio, song names or folder names."
      @click="copy"
    >
      <span class="material-symbols-outlined text-[14px]" :class="{ spinner: state === 'working' }">
        {{ icon }}
      </span>
      {{
        state === "working"
          ? "Running self-test…"
          : state === "copied"
            ? "Copied"
            : "Copy diagnostics"
      }}
    </button>
    <textarea
      v-if="state === 'failed'"
      readonly
      rows="8"
      class="mt-2 w-full bg-[#0f0f12] border border-outline-variant rounded p-2 font-label-mono text-[11px] text-on-surface-variant"
      :value="report"
      @focus="($event.target as HTMLTextAreaElement).select()"
    />
  </div>
</template>
