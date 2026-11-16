<script setup lang="ts">
/** The app's one loading indicator: a spinning ring. Inline, never blocking. */
withDefaults(defineProps<{ size?: number; label?: string }>(), { size: 16, label: "Loading" });
</script>

<template>
  <span
    class="material-symbols-outlined spinner inline-block align-middle leading-none"
    :style="{ fontSize: `${size}px`, width: `${size}px`, height: `${size}px` }"
    role="status"
    :aria-label="label"
  >progress_activity</span>
</template>
