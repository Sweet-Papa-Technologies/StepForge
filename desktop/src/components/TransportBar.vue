<script setup lang="ts">
import { ref, watch, onBeforeUnmount, computed } from "vue";
import { useMediaSrc } from "../lib/mediaSrc";
import { fmtDuration } from "../lib/format";

/** `rev` is bumped when the file at `src` was rewritten in place, so the webview
 *  fetches it again instead of replaying its cache. */
const props = defineProps<{ src?: string | null; rev?: number }>();

const audio = ref<HTMLAudioElement | null>(null);
const playing = ref(false);
const current = ref(0);
const duration = ref(0);

const url = useMediaSrc(() => props.src, () => props.rev);

watch(url, () => {
  playing.value = false;
  current.value = 0;
  duration.value = 0;
});

function toggle() {
  const a = audio.value;
  if (!a || !url.value) return;
  if (a.paused) {
    a.play();
    playing.value = true;
  } else {
    a.pause();
    playing.value = false;
  }
}
function onTime() {
  if (audio.value) current.value = audio.value.currentTime;
}
function onMeta() {
  if (audio.value) duration.value = audio.value.duration || 0;
}
function seek(e: MouseEvent) {
  const a = audio.value;
  if (!a || !duration.value) return;
  const el = e.currentTarget as HTMLElement;
  const rect = el.getBoundingClientRect();
  const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  a.currentTime = ratio * duration.value;
}

onBeforeUnmount(() => audio.value?.pause());

const pct = computed(() => (duration.value ? (current.value / duration.value) * 100 : 0));
</script>

<template>
  <div class="flex items-center gap-6 w-full">
    <audio
      ref="audio"
      :src="url || undefined"
      @timeupdate="onTime"
      @loadedmetadata="onMeta"
      @ended="playing = false"
    ></audio>
    <button
      class="w-10 h-10 rounded-full flex items-center justify-center transition-colors disabled:opacity-40"
      :class="url ? 'bg-primary-container text-on-primary-container hover:bg-primary' : 'bg-surface-variant text-on-surface-variant'"
      :disabled="!url"
      :title="url ? 'Play / pause click-track' : 'No click-track available'"
      @click="toggle"
    >
      <span class="material-symbols-outlined icon-fill">{{ playing ? "pause" : "play_arrow" }}</span>
    </button>
    <div class="font-label-mono text-label-mono text-on-surface-variant w-14 text-right">
      {{ fmtDuration(current) }}
    </div>
    <div
      class="flex-1 h-2 bg-surface-variant rounded-full relative overflow-hidden cursor-pointer group"
      @click="seek"
    >
      <div class="absolute left-0 top-0 h-full bg-primary-container rounded-full" :style="{ width: pct + '%' }"></div>
    </div>
    <div class="font-label-mono text-label-mono text-on-surface-variant w-14">
      {{ fmtDuration(duration) }}
    </div>
    <slot />
  </div>
</template>
