<script setup lang="ts">
import { computed } from "vue";
import { useRouter } from "vue-router";
import type { Track } from "../lib/types";
import { fmtDuration } from "../lib/format";
import StatusBadge from "./StatusBadge.vue";
import Waveform from "./Waveform.vue";

const props = defineProps<{ track: Track; selected: boolean; busy?: boolean }>();
const emit = defineEmits<{
  (e: "select", base: string): void;
  (e: "convert", track: Track): void;
  (e: "menu", payload: { event: MouseEvent; track: Track }): void;
}>();
const router = useRouter();

/** The single next step for this track, driven by its real state. */
const primary = computed(() => {
  switch (props.track.status) {
    case "unconverted":
      return { label: "Convert to .ogg", icon: "sync_alt", kind: "convert" as const };
    case "draft":
      return { label: "Analyze", icon: "graphic_eq", kind: "analyze" as const };
    case "analyzed":
      return { label: "Generate", icon: "auto_awesome", kind: "generate" as const };
    default:
      return { label: "Results", icon: "analytics", kind: "results" as const };
  }
});

function go() {
  emit("select", props.track.base);
  const p = { base: props.track.base };
  if (primary.value.kind === "convert") emit("convert", props.track);
  // Kicking off Analyze from the Library should actually start it, not just
  // land on the screen — `run` tells the view to begin immediately.
  else if (primary.value.kind === "analyze") router.push({ name: "analyze", params: p, query: { run: "1" } });
  else if (primary.value.kind === "generate") router.push({ name: "generate", params: p });
  else router.push({ name: "results", params: p });
}
function open() {
  emit("select", props.track.base);
  router.push({ name: "analyze", params: { base: props.track.base } });
}
</script>

<template>
  <div
    class="rounded-xl overflow-hidden relative flex flex-col cursor-pointer transition-transform hover:-translate-y-1"
    :class="
      selected
        ? 'bg-surface-container-high border border-primary shadow-[0_0_15px_rgba(188,19,254,0.15)]'
        : 'bg-surface-container border border-outline-variant hover:border-outline'
    "
    @click="emit('select', track.base)"
    @dblclick="open"
    @contextmenu="emit('menu', { event: $event, track })"
  >
    <div class="p-4 border-b border-outline-variant flex items-start justify-between gap-2">
      <div class="min-w-0">
        <h3 class="font-headline-md text-body-lg text-on-surface font-semibold mb-1 truncate">{{ track.title }}</h3>
        <p class="text-on-surface-variant text-[12px] truncate">{{ track.artist || track.file }}</p>
      </div>
      <StatusBadge :status="track.status" />
    </div>

    <div class="h-20 w-full border-b border-outline-variant">
      <Waveform :seed="track.base" :bars="16" :dim="track.status === 'draft' || track.status === 'unconverted'" />
    </div>

    <div class="p-4 flex flex-col gap-3 flex-1 justify-between">
      <div class="flex justify-between text-[12px] font-label-mono text-on-surface-variant">
        <span class="flex items-center gap-1">
          <span class="material-symbols-outlined text-[14px]">speed</span>
          {{ track.bpm ? track.bpm.toFixed(1) : "—" }} BPM
        </span>
        <span class="flex items-center gap-1">
          <span class="material-symbols-outlined text-[14px]">timer</span>
          {{ fmtDuration(track.durationSec) }}
        </span>
      </div>

      <div v-if="track.targets.length" class="flex flex-wrap gap-1">
        <span
          v-for="t in track.targets"
          :key="t"
          class="px-1.5 py-0.5 rounded-sm bg-surface-variant text-on-surface-variant font-label-mono text-[10px] uppercase"
        >
          {{ t }}
        </span>
      </div>

      <div class="flex gap-2">
        <button
          v-if="track.status !== 'unconverted'"
          class="flex-1 bg-surface-variant hover:bg-surface-bright text-on-surface py-1.5 rounded text-[12px] font-medium transition-colors border border-outline-variant"
          @click.stop="open"
        >
          Open
        </button>
        <button
          class="flex-1 py-1.5 rounded text-[12px] font-medium transition-colors flex items-center justify-center gap-1 disabled:opacity-50"
          :class="
            track.status === 'analyzed' || track.status === 'unconverted'
              ? 'bg-primary-container hover:bg-primary text-on-primary-container hover:text-on-primary'
              : 'bg-surface-variant hover:bg-surface-bright text-on-surface border border-outline-variant'
          "
          :disabled="busy"
          @click.stop="go"
        >
          <span class="material-symbols-outlined text-[14px]" :class="{ spinner: busy }">
            {{ busy ? "progress_activity" : primary.icon }}
          </span>
          {{ busy ? "Converting…" : primary.label }}
        </button>
      </div>
    </div>
  </div>
</template>
