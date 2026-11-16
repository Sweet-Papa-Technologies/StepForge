<script setup lang="ts">
import { computed } from "vue";
import type { TrackStatus } from "../lib/types";

const props = defineProps<{ status: TrackStatus }>();

const cfg = computed(() => {
  switch (props.status) {
    case "charted":
      return { label: "Charted", cls: "bg-secondary-container/10 text-secondary-container border-secondary-container/30" };
    case "analyzed":
      return { label: "Analyzed", cls: "bg-green-500/10 text-green-400 border-green-500/30" };
    case "unconverted":
      return { label: "Needs .ogg", cls: "bg-error-container/30 text-error border-error/40" };
    default:
      return { label: "Draft", cls: "bg-tertiary-container/10 text-tertiary-fixed-dim border-tertiary-container/30" };
  }
});
</script>

<template>
  <span
    class="px-2 py-1 rounded text-[10px] font-label-mono border uppercase tracking-wider"
    :class="cfg.cls"
  >
    {{ cfg.label }}
  </span>
</template>
