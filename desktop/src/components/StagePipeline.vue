<script setup lang="ts">
defineProps<{ stages: readonly string[]; current: number; running: boolean; details?: Record<number, string> }>();

function state(idx: number, current: number, running: boolean): "done" | "active" | "pending" {
  if (idx < current) return "done";
  if (idx === current) return running ? "active" : idx < current ? "done" : "active";
  return "pending";
}
</script>

<template>
  <div class="flex flex-col gap-1 w-full max-w-md mx-auto">
    <div v-for="(stage, idx) in stages" :key="stage" class="flex items-start gap-4">
      <div class="flex flex-col items-center">
        <div
          class="w-8 h-8 rounded-full flex items-center justify-center border"
          :class="{
            'bg-[#1b5e20] border-[#4caf50] text-[#4caf50]': state(idx, current, running) === 'done',
            'bg-secondary-container/20 border-secondary-container text-secondary-container':
              state(idx, current, running) === 'active',
            'bg-surface-container border-outline-variant text-on-surface-variant':
              state(idx, current, running) === 'pending',
          }"
        >
          <span v-if="state(idx, current, running) === 'done'" class="material-symbols-outlined text-sm font-bold">check</span>
          <span
            v-else-if="state(idx, current, running) === 'active'"
            class="material-symbols-outlined text-sm"
            :class="{ spinner: running }"
            >progress_activity</span
          >
          <div v-else class="w-2 h-2 rounded-full bg-outline-variant"></div>
        </div>
        <div
          v-if="idx < stages.length - 1"
          class="w-px h-8 my-1"
          :class="state(idx, current, running) === 'done' ? 'bg-[#4caf50] opacity-50' : 'bg-outline-variant'"
        ></div>
      </div>
      <div class="pt-1" :class="{ 'opacity-50': state(idx, current, running) === 'pending' }">
        <h4
          class="font-body-lg text-body-lg"
          :class="state(idx, current, running) === 'active' ? 'text-secondary-container font-semibold' : 'text-on-surface'"
        >
          {{ stage }}
        </h4>
        <p v-if="details && details[idx]" class="font-body-md text-body-md text-on-surface-variant">
          {{ details[idx] }}
        </p>
      </div>
    </div>
  </div>
</template>
