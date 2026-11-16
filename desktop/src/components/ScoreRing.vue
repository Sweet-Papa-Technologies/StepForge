<script setup lang="ts">
import { computed } from "vue";

const props = withDefaults(defineProps<{ score: number; max?: number }>(), { max: 100 });

const R = 40;
const CIRC = 2 * Math.PI * R;
const offset = computed(() => CIRC * (1 - Math.max(0, Math.min(1, props.score / props.max))));
</script>

<template>
  <div class="relative w-32 h-32">
    <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
      <circle class="text-surface-variant" cx="50" cy="50" fill="none" r="40" stroke="currentColor" stroke-width="8" />
      <circle
        class="text-primary drop-shadow-[0_0_8px_rgba(188,19,254,0.5)]"
        cx="50"
        cy="50"
        fill="none"
        r="40"
        stroke="currentColor"
        stroke-width="8"
        stroke-linecap="round"
        :stroke-dasharray="CIRC"
        :stroke-dashoffset="offset"
      />
    </svg>
    <div class="absolute inset-0 flex flex-col items-center justify-center">
      <span class="font-headline-lg text-headline-lg text-on-surface leading-none">
        {{ max === 10 ? score.toFixed(1) : Math.round(score) }}
      </span>
      <span class="font-label-mono text-[10px] text-on-surface-variant uppercase">/ {{ max }}</span>
    </div>
  </div>
</template>
