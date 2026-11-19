<script setup lang="ts">
/** Surfaces a broken engine environment instead of letting every action fail
 *  with a silent non-zero exit. */
import { computed } from "vue";
import { useLibrary } from "../stores/library";
import Spinner from "./Spinner.vue";
import CopyDiagnostics from "./CopyDiagnostics.vue";
import { engineAdvice, engineProblems } from "../lib/engineProblems";

const lib = useLibrary();

/** Still probing, and nothing known yet: say so, rather than "not found". */
const starting = computed(() => lib.statusLoading && !lib.status);

/** Each problem with its real cause (the exception, how Python died), never a
 *  guess: see lib/engineProblems.ts. */
const problems = computed(() => (starting.value ? [] : engineProblems(lib.status)));
const advice = computed(() => engineAdvice(lib.status));
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
        <li v-for="p in problems" :key="p" class="break-words">• {{ p }}</li>
      </ul>
      <p class="text-[12px] text-on-surface-variant mt-1.5">{{ advice }}</p>
    </div>
    <div class="shrink-0 flex items-start gap-2">
      <CopyDiagnostics />
      <router-link
        :to="{ name: 'settings' }"
        class="px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant font-label-mono text-label-mono"
      >
        Settings
      </router-link>
    </div>
  </div>
</template>
