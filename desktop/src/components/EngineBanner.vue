<script setup lang="ts">
/** Surfaces a broken engine environment instead of letting every action fail
 *  with a silent non-zero exit. */
import { computed } from "vue";
import { useLibrary } from "../stores/library";
import Spinner from "./Spinner.vue";

const lib = useLibrary();

/** Still probing, and nothing known yet: say so, rather than "not found". */
const starting = computed(() => lib.statusLoading && !lib.status);

const problems = computed(() => {
  if (starting.value) return [];
  const s = lib.status;
  const out: string[] = [];
  if (!s) return ["Could not locate the StepForge engine. Set the engine root in Settings."];
  if (!s.engineRoot) out.push("Engine root not found — set it in Settings.");
  if (!s.pythonOk) out.push(`Python not runnable (${s.python}).`);
  else if (!s.beatforgeOk)
    out.push(
      `The charting engine or its dependencies did not import${s.importError ? `: ${s.importError}` : ""}. Run: python3 -m pip install -r requirements.txt`,
    );
  return out;
});
</script>

<template>
  <div
    v-if="starting"
    class="mx-margin-desktop mt-4 rounded-lg border border-secondary-container/30 bg-secondary-container/5 px-4 py-2.5 flex items-center gap-3 text-body-md text-on-surface-variant"
  >
    <Spinner :size="18" class="text-secondary-container" label="Starting the charting engine" />
    Starting the charting engine… The first start after installing or updating can take a few seconds.
  </div>
  <div
    v-else-if="problems.length"
    class="mx-margin-desktop mt-4 rounded-lg border border-tertiary-container/60 bg-tertiary-container/10 px-4 py-3 flex items-start gap-3"
  >
    <span class="material-symbols-outlined text-tertiary-fixed-dim mt-0.5">warning</span>
    <div class="flex-1 min-w-0">
      <p class="font-label-mono text-label-mono text-tertiary-fixed-dim uppercase tracking-wider mb-1">
        Engine not ready
      </p>
      <ul class="text-body-md text-on-surface-variant space-y-0.5">
        <li v-for="p in problems" :key="p">• {{ p }}</li>
      </ul>
    </div>
    <router-link
      :to="{ name: 'settings' }"
      class="shrink-0 px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant font-label-mono text-label-mono"
    >
      Settings
    </router-link>
  </div>
</template>
