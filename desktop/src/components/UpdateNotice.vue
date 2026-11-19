<script setup lang="ts">
/** "A newer StepForge is out": the version, its release notes and a button
 *  that opens the release page in the browser. The app downloads and installs
 *  nothing itself. Dismissing hides that version for good.
 *
 *  The notes come from the network, so they are only ever rendered as text. */
import { ref } from "vue";
import { openUrl } from "@tauri-apps/plugin-opener";
import { useUpdate } from "../stores/update";

const update = useUpdate();
const showNotes = ref(true);
/** The browser could not be opened: show the address to copy instead. */
const openFailed = ref(false);

async function openReleasePage() {
  const url = update.notice?.url;
  if (!url) return;
  try {
    await openUrl(url);
    openFailed.value = false;
  } catch {
    openFailed.value = true;
  }
}
</script>

<template>
  <div
    v-if="update.notice"
    class="shrink-0 mx-margin-desktop mt-4 mb-3 rounded-lg border border-secondary-container/30 bg-secondary-container/5 px-4 py-3 flex items-start gap-3"
    role="status"
  >
    <span class="material-symbols-outlined text-secondary-container mt-0.5">update</span>
    <div class="flex-1 min-w-0">
      <p class="font-label-mono text-label-mono text-secondary-container uppercase tracking-wider mb-1">
        Update available
      </p>
      <p class="text-body-md text-on-surface-variant">
        StepForge {{ update.notice.version }} is out. You have {{ update.notice.current }}.
        <button
          v-if="update.notice.notes"
          type="button"
          class="underline hover:text-on-surface ml-1"
          :aria-expanded="showNotes"
          @click="showNotes = !showNotes"
        >
          {{ showNotes ? "Hide release notes" : "Show release notes" }}
        </button>
      </p>
      <pre
        v-if="showNotes && update.notice.notes"
        class="mt-2 max-h-28 overflow-y-auto whitespace-pre-wrap break-words bg-[#0f0f12] border border-outline-variant rounded p-3 font-label-mono text-[11px] text-on-surface-variant"
      >{{ update.notice.notes }}</pre>
      <p class="text-[11px] text-on-surface-variant mt-1.5">
        The button opens the release page in your browser. StepForge does not download or install anything by itself.
        You can turn this check off in Settings.
      </p>
      <p v-if="openFailed" class="text-[11px] text-tertiary-fixed-dim mt-1.5 break-all select-text">
        Your browser could not be opened. The release page is {{ update.notice.url }}
      </p>
    </div>
    <div class="shrink-0 flex items-start gap-2">
      <button
        type="button"
        class="px-3 py-1.5 rounded bg-primary-container text-on-primary-container hover:bg-primary transition-colors font-label-mono text-label-mono flex items-center gap-2 whitespace-nowrap"
        @click="openReleasePage"
      >
        <span class="material-symbols-outlined text-[14px]">download</span>
        Get the update
      </button>
      <button
        type="button"
        class="px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant font-label-mono text-label-mono whitespace-nowrap"
        title="Hide this version. A later release will show a notice again."
        @click="update.dismiss()"
      >
        Dismiss
      </button>
    </div>
  </div>
</template>
