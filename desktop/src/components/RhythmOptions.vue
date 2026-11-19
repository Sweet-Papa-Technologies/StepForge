<script setup lang="ts">
/**
 * The decoder options as controls: a cap on the subdivision, Stream mode, no
 * crossovers — and, for a bar range, more or fewer notes. Used by the Generate
 * screen (the whole chart) and by "Edit a section" on Results (a bar range).
 */
import { computed } from "vue";
import {
  BREAK_BARS, DEFAULT_MIN_BREAK_BARS, DEFAULT_STREAM_COVERAGE, SUBDIVISIONS, streamRate,
  type DecodeOptions,
} from "../lib/decodeOptions";

const props = withDefaults(
  defineProps<{
    modelValue: DecodeOptions;
    /** "chart": a whole chart. "range": a bar range of one (adds the density
     *  slider; a stream then fills the range, so its amount is not asked for). */
    scope?: "chart" | "range";
    disabled?: boolean;
    /** The song's tempo, when known: lets Stream mode say how fast it will be. */
    bpm?: number | null;
  }>(),
  { scope: "chart", disabled: false, bpm: null },
);
const emit = defineEmits<{ (e: "update:modelValue", v: DecodeOptions): void; (e: "commit"): void }>();

function patch(p: Partial<DecodeOptions>, commit = true) {
  const next: DecodeOptions = { ...props.modelValue, ...p };
  // absent = default: never leave a placeholder behind
  for (const k of Object.keys(next) as (keyof DecodeOptions)[]) {
    if (next[k] === undefined || next[k] === null) delete next[k];
  }
  if (next.stream !== true) delete next.stream;
  if (next.crossovers !== false) delete next.crossovers;
  emit("update:modelValue", next);
  if (commit) emit("commit");
}

const subdivision = computed(() => props.modelValue.max_subdivision ?? null);
const coverage = computed(() => props.modelValue.stream_coverage ?? DEFAULT_STREAM_COVERAGE);
const minBreak = computed(() => props.modelValue.min_break_bars ?? DEFAULT_MIN_BREAK_BARS);
const density = computed(() => props.modelValue.density ?? 1);
/** What an unbroken stream at the chosen subdivision comes to at this tempo. */
const rateNote = computed(() => {
  const sub = subdivision.value;
  if (!props.modelValue.stream || !sub || !props.bpm) return null;
  return `${sub}ths at ${Math.round(props.bpm)} BPM are ${streamRate(props.bpm, sub).toFixed(1)} notes per second.`;
});
</script>

<template>
  <div class="space-y-3">
    <div>
      <span class="block text-[11px] text-on-surface-variant mb-1">Max subdivision</span>
      <div class="flex items-center gap-1 bg-surface p-1 rounded-lg border border-outline-variant">
        <button
          v-for="s in SUBDIVISIONS"
          :key="s.label"
          type="button"
          class="flex-1 py-1.5 rounded-md font-label-mono text-[11px] whitespace-nowrap transition-colors disabled:opacity-40"
          :class="subdivision === s.value ? 'bg-surface-container-high text-on-surface border border-outline-variant' : 'text-on-surface-variant hover:text-on-surface'"
          :disabled="disabled"
          :aria-pressed="subdivision === s.value"
          @click="patch({ max_subdivision: s.value ?? undefined })"
        >
          {{ s.label }}
        </button>
      </div>
      <p class="text-[11px] text-on-surface-variant mt-1">
        No note finer than this.
        <template v-if="scope === 'chart'">
          It can only simplify a difficulty: Beginner stays on 4ths and Easy and Medium on 8ths whatever is chosen. 8ths
          also rules out triplets; 16ths keeps to straight 16ths.
        </template>
      </p>
    </div>

    <div>
      <label class="flex items-center gap-2 cursor-pointer" :class="{ 'opacity-50': disabled }">
        <input
          type="checkbox"
          class="accent-[#bc13fe] cursor-pointer"
          :checked="modelValue.stream === true"
          :disabled="disabled"
          @change="patch({ stream: ($event.target as HTMLInputElement).checked })"
        />
        <span class="font-body-md text-[13px] text-on-surface">
          {{ scope === "range" ? "Fill with a stream" : "Stream mode" }}
        </span>
      </label>
      <p class="text-[11px] text-on-surface-variant mt-1">
        <template v-if="scope === 'range'">
          Every slot of these bars at the chosen subdivision, unbroken (the finest the difficulty's speed limit allows
          if none is chosen).
        </template>
        <template v-else>
          Unbroken runs at the chosen subdivision with real breaks between them, instead of placing notes one by one. No
          jumps or freezes inside a stream, and nothing faster than it anywhere. With no subdivision chosen it runs at
          the finest one the difficulty's speed limit allows.
        </template>
        <span v-if="rateNote" class="text-on-surface"> {{ rateNote }}</span>
      </p>
      <div v-if="modelValue.stream && scope === 'chart'" class="mt-2 space-y-2 pl-6">
        <div class="flex items-center gap-3">
          <span class="w-28 shrink-0 text-[12px] text-on-surface-variant">Stream amount</span>
          <input
            type="range"
            class="flex-1 min-w-0 accent-[#bc13fe]"
            min="0.3"
            max="1"
            step="0.05"
            :value="coverage"
            :disabled="disabled"
            aria-label="Stream amount"
            @input="patch({ stream_coverage: Number(($event.target as HTMLInputElement).value) }, false)"
            @change="emit('commit')"
          />
          <span class="w-12 shrink-0 text-right font-label-mono text-[11px] text-on-surface">{{ Math.round(coverage * 100) }}%</span>
        </div>
        <div class="flex items-center gap-3">
          <span class="w-28 shrink-0 text-[12px] text-on-surface-variant">Shortest break</span>
          <select
            class="bg-surface-container-lowest border border-outline-variant text-on-surface rounded p-1.5 text-[12px] focus:border-secondary outline-none"
            :value="minBreak"
            :disabled="disabled"
            aria-label="Shortest break"
            @change="patch({ min_break_bars: Number(($event.target as HTMLSelectElement).value) })"
          >
            <option v-for="b in BREAK_BARS" :key="b" :value="b">{{ b }} {{ b === 1 ? "bar" : "bars" }}</option>
          </select>
          <span class="text-[11px] text-on-surface-variant">
            {{ coverage >= 0.999 ? "At 100% there are no breaks." : "Breaks go where the song is quietest." }}
          </span>
        </div>
      </div>
    </div>

    <label class="flex items-center gap-2 cursor-pointer" :class="{ 'opacity-50': disabled }">
      <input
        type="checkbox"
        class="accent-[#bc13fe] cursor-pointer"
        :checked="modelValue.crossovers === false"
        :disabled="disabled"
        @change="patch({ crossovers: ($event.target as HTMLInputElement).checked ? false : undefined })"
      />
      <span class="font-body-md text-[13px] text-on-surface">No crossovers</span>
    </label>

    <div v-if="scope === 'range'" class="flex items-center gap-3">
      <span class="w-16 shrink-0 text-[12px] text-on-surface-variant">Density</span>
      <input
        type="range"
        class="flex-1 min-w-0 accent-[#bc13fe] disabled:opacity-30"
        min="0.5"
        max="1.5"
        step="0.05"
        :value="density"
        :disabled="disabled || modelValue.stream === true"
        aria-label="Density"
        @input="patch({ density: Number(($event.target as HTMLInputElement).value) }, false)"
        @change="emit('commit')"
      />
      <span class="w-12 shrink-0 text-right font-label-mono text-[11px] text-on-surface">{{ Math.round(density * 100) }}%</span>
    </div>
  </div>
</template>
