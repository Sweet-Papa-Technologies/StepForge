<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{ label: string; state: "pass" | "warn" | "fail" }>();

const chip = computed(() => {
  switch (props.state) {
    case "pass":
      return { text: "PASS", cls: "bg-[#004d40] text-[#69f0ae] border-[#00695c]" };
    case "warn":
      return { text: "WARN", cls: "bg-[#ff8f00]/20 text-[#ffd54f] border-[#ff8f00]/50" };
    default:
      return { text: "FAIL", cls: "bg-error-container/40 text-error border-error/50" };
  }
});
</script>

<template>
  <div class="flex items-center justify-between bg-surface-container-low border border-outline-variant/50 rounded px-3 py-2">
    <span class="font-label-mono text-[11px] text-on-surface capitalize">{{ label }}</span>
    <span class="px-2 py-0.5 rounded-sm font-label-mono text-[10px] font-bold tracking-wider border" :class="chip.cls">
      {{ chip.text }}
    </span>
  </div>
</template>
