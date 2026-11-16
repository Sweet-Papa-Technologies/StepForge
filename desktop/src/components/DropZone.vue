<script setup lang="ts">
import { ref } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { useLibrary } from "../stores/library";

const props = defineProps<{ disabled?: boolean }>();
const lib = useLibrary();
const busy = ref(false);
const emit = defineEmits<{ (e: "imported", base: string): void }>();

async function pick() {
  if (props.disabled) return;
  const selected = await open({
    multiple: true,
    filters: [{ name: "Audio", extensions: ["ogg", "wav", "mp3", "flac", "m4a", "aac"] }],
  });
  if (!selected) return;
  const paths = Array.isArray(selected) ? selected : [selected];
  busy.value = true;
  lib.error = null;
  try {
    let last = "";
    for (const p of paths) last = (await lib.importFrom(p as string)).base;
    if (last) emit("imported", last);
  } catch (e) {
    lib.error = String(e);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <button
    type="button"
    class="w-full h-28 rounded-xl border-2 border-dashed border-outline-variant bg-surface-container hover:border-secondary hover:bg-surface-container-high transition-all duration-300 flex flex-col items-center justify-center group disabled:opacity-50 disabled:hover:border-outline-variant"
    :disabled="busy || disabled"
    @click="pick"
  >
    <div class="w-9 h-9 rounded-full bg-surface-variant flex items-center justify-center mb-2 group-hover:bg-secondary-container group-hover:text-on-secondary-container transition-colors">
      <span class="material-symbols-outlined" :class="{ spinner: busy }">
        {{ busy ? "progress_activity" : "cloud_upload" }}
      </span>
    </div>
    <p class="font-label-mono text-on-surface">
      {{ busy ? "Transcoding to .ogg…" : disabled ? "ffmpeg required to import" : "Click to import audio" }}
    </p>
    <p class="text-[11px] text-on-surface-variant mt-1">.ogg, .wav, .mp3, .flac (.m4a and others need ffmpeg) — converted to .ogg</p>
  </button>
</template>
