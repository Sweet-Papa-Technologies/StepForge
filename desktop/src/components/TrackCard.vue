<script setup lang="ts">
import { computed } from "vue";
import { useRouter } from "vue-router";
import type { Track } from "../lib/types";
import { fmtDuration } from "../lib/format";
import StatusBadge from "./StatusBadge.vue";
import Waveform from "./Waveform.vue";

const props = defineProps<{ track: Track; selected: boolean; checked?: boolean; busy?: boolean }>();
const emit = defineEmits<{
  (e: "select", base: string): void;
  /** Tick/untick for bulk generation; `range` = shift-click from the last tick. */
  (e: "check", payload: { base: string; range: boolean }): void;
  (e: "convert", track: Track): void;
  (e: "menu", payload: { event: MouseEvent; track: Track }): void;
}>();
const router = useRouter();

/** The single next step for this track, driven by its real state. Analysis
 *  is not a step of its own: generating analyzes the song when it needs it
 *  (first time, or the audio changed). */
const primary = computed(() => {
  switch (props.track.status) {
    case "unconverted":
      return { label: "Convert to .ogg", icon: "sync_alt", kind: "convert" as const };
    case "charted":
      return { label: "Results", icon: "analytics", kind: "results" as const };
    default:
      return { label: "Generate", icon: "auto_awesome", kind: "generate" as const };
  }
});

function go() {
  emit("select", props.track.base);
  const p = { base: props.track.base };
  if (primary.value.kind === "convert") emit("convert", props.track);
  else if (primary.value.kind === "generate") router.push({ name: "generate", params: p });
  else router.push({ name: "results", params: p });
}
/** "Open": the results once there are charts, otherwise the Generate screen. */
function open() {
  emit("select", props.track.base);
  const name = props.track.status === "charted" ? "results" : "generate";
  router.push({ name, params: { base: props.track.base } });
}
function onCardClick(e: MouseEvent) {
  // Cmd/Ctrl-click or Shift-click ticks the card, like a file list.
  if (e.metaKey || e.ctrlKey || e.shiftKey) {
    emit("check", { base: props.track.base, range: e.shiftKey });
    return;
  }
  emit("select", props.track.base);
}
</script>

<template>
  <div
    class="rounded-xl overflow-hidden relative flex flex-col cursor-pointer transition-transform hover:-translate-y-1"
    :class="[
      selected
        ? 'bg-surface-container-high border border-primary shadow-[0_0_15px_rgba(188,19,254,0.15)]'
        : 'bg-surface-container border border-outline-variant hover:border-outline',
      checked ? 'ring-2 ring-secondary-container/70' : '',
    ]"
    @click="onCardClick"
    @dblclick="open"
    @contextmenu="emit('menu', { event: $event, track })"
  >
    <div class="p-4 border-b border-outline-variant flex items-start justify-between gap-2">
      <label
        class="shrink-0 mt-0.5 -ml-1 p-1 rounded cursor-pointer hover:bg-surface-variant"
        :title="track.isOgg ? 'Select for bulk generation (Shift-click a range)' : 'Convert to .ogg before charting'"
        @click.stop
      >
        <input
          type="checkbox"
          class="w-4 h-4 accent-[#00eefc] cursor-pointer align-middle"
          :checked="checked"
          :disabled="!track.isOgg"
          :aria-label="`Select ${track.title}`"
          @click.stop="emit('check', { base: track.base, range: ($event as MouseEvent).shiftKey })"
        />
      </label>
      <div class="min-w-0 flex-1">
        <h3 class="font-headline-md text-body-lg text-on-surface font-semibold mb-1 truncate">{{ track.title }}</h3>
        <p class="text-on-surface-variant text-[12px] truncate">{{ track.artist || track.file }}</p>
      </div>
      <StatusBadge :status="track.status" />
    </div>

    <div class="h-20 w-full border-b border-outline-variant">
      <Waveform :seed="track.base" :bars="16" :dim="track.status === 'unconverted'" />
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
            track.status !== 'charted'
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
