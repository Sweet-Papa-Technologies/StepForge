<script setup lang="ts">
/**
 * A thin animated bar along the top of the window while anything is working:
 * the engine starting, the library loading, a run in progress. It never covers
 * or blocks anything; it only says "busy".
 */
import { computed } from "vue";
import { useBatch } from "../stores/batch";
import { useEngine } from "../stores/engine";
import { useLibrary } from "../stores/library";
import { useNeural } from "../stores/neural";

const lib = useLibrary();
const engine = useEngine();
const neural = useNeural();
const batch = useBatch();

const busy = computed(
  () => lib.loading || lib.statusLoading || neural.loading || engine.anyRunning || batch.running,
);
</script>

<template>
  <div v-show="busy" class="activity-bar" role="progressbar" aria-label="Working" aria-busy="true">
    <div class="activity-bar__run"></div>
  </div>
</template>

<style scoped>
.activity-bar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  z-index: 60;
  overflow: hidden;
  pointer-events: none;
  background: rgba(188, 19, 254, 0.28);
  box-shadow: 0 0 12px rgba(188, 19, 254, 0.55);
}
.activity-bar__run {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 40%;
  background: linear-gradient(90deg, transparent, #e86bff, #00eefc, transparent);
  animation: activity-run 1.4s ease-in-out infinite;
}
@keyframes activity-run {
  from {
    left: -40%;
  }
  to {
    left: 100%;
  }
}
@media (prefers-reduced-motion: reduce) {
  .activity-bar__run {
    animation-duration: 4s;
  }
}
</style>
