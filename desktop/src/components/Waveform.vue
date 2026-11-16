<script setup lang="ts">
import { computed } from "vue";
import { hashBars } from "../lib/format";

const props = withDefaults(
  defineProps<{ seed?: string; bars?: number; dim?: boolean; height?: string }>(),
  { seed: "beatforge", bars: 48, dim: false, height: "100%" },
);

const values = computed(() => hashBars(props.seed, props.bars));
</script>

<template>
  <div class="relative w-full overflow-hidden flex items-center" :style="{ height }">
    <div v-if="!dim" class="absolute inset-0 waveform-bg opacity-30"></div>
    <div class="flex items-end gap-[2px] w-full h-full px-2 z-10" :class="{ 'opacity-30': dim }">
      <div
        v-for="(v, i) in values"
        :key="i"
        class="flex-1 min-w-[2px] rounded-t-sm"
        :class="dim ? 'bg-on-surface' : 'waveform-bar'"
        :style="{ height: Math.round(v * 100) + '%' }"
      ></div>
    </div>
  </div>
</template>
