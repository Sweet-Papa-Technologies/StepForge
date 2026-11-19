<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef, watch } from "vue";
import { useRouter } from "vue-router";
import { open } from "@tauri-apps/plugin-dialog";
import { useLibrary } from "../stores/library";
import { api } from "../lib/api";
import { fmtBpm, fmtDuration, fmtSize } from "../lib/format";
import { optionsJson } from "../lib/decodeOptions";
import { convertFileSrc } from "@tauri-apps/api/core";
import { useEngine } from "../stores/engine";
import { useSettings } from "../stores/settings";
import { useNeural } from "../stores/neural";
import { useBatch } from "../stores/batch";
import type { Analysis, ChartEngine, PackInfo, SongAssets, Track } from "../lib/types";
import PackCombobox from "../components/PackCombobox.vue";
import DropZone from "../components/DropZone.vue";
import TrackCard from "../components/TrackCard.vue";
import Spinner from "../components/Spinner.vue";
import AnalysisCanvas from "../components/AnalysisCanvas.vue";
import EngineBanner from "../components/EngineBanner.vue";
import ContextMenu, { type MenuItem } from "../components/ContextMenu.vue";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";

const menu = ref<InstanceType<typeof ContextMenu> | null>(null);

/** Pack a given song is filed under, if any. */
function packOf(base: string): string {
  return packInfo.value?.assignments[base] ?? "";
}

/** Right-click actions for a track card, gated on what the track can do. */
function trackMenu({ event, track }: { event: MouseEvent; track: Track }) {
  lib.select(track.base);
  const base = track.base;
  const items: MenuItem[] = [
    {
      label: "Generate charts…",
      icon: "auto_awesome",
      disabled: !track.isOgg,
      action: () => void router.push({ name: "generate", params: { base } }),
    },
    {
      label: "View results",
      icon: "analytics",
      disabled: !track.targets.length,
      action: () => void router.push({ name: "results", params: { base } }),
    },
    {
      label: "View analysis",
      icon: "graphic_eq",
      disabled: !track.isOgg,
      action: () => void router.push({ name: "analyze", params: { base } }),
    },
    {
      label: "Re-analyze now",
      icon: "play_arrow",
      disabled: !track.isOgg || !lib.engineReady,
      action: () => void router.push({ name: "analyze", params: { base }, query: { run: "1" } }),
    },
    {
      label: checked.value.has(base) ? "Deselect" : "Select for bulk generation",
      icon: "done_all",
      separatorBefore: true,
      disabled: !track.isOgg,
      action: () => toggleCheck({ base, range: false }),
    },
    {
      label: "Install song to game",
      icon: "download",
      separatorBefore: true,
      disabled: !track.targets.includes("stepforge"),
      action: () => install(),
    },
    {
      label: packOf(track.base) ? `Install pack “${packOf(track.base)}”` : "Install pack",
      icon: "library_music",
      disabled: !packOf(track.base),
      action: () => install(packOf(track.base)),
    },
    {
      label: "Convert to .ogg",
      icon: "sync_alt",
      separatorBefore: true,
      disabled: track.isOgg,
      action: () => convert(track),
    },
    {
      label: "Reveal in Finder",
      icon: "folder_open",
      action: () => void revealItemInDir(track.path),
    },
    {
      label: "Copy track id",
      icon: "content_copy",
      action: async () => {
        await writeText(track.id);
        notice.value = `Copied "${track.id}"`;
      },
    },
  ];
  menu.value?.open(event, items);
}

const isMac = /mac/i.test(navigator.platform || navigator.userAgent);
const lib = useLibrary();
const engine = useEngine();
const settings = useSettings();
const neural = useNeural();
const batch = useBatch();
const router = useRouter();
const query = ref("");
const converting = ref<string | null>(null);
const notice = ref<string | null>(null);

// Metadata editor — writes TRACK_META so exported simfiles carry a real title.
const editTitle = ref("");
const editArtist = ref("");
const savingMeta = ref(false);

// Real waveform + analysis overlay for the selected track (not a placeholder).
// shallowRef so Vue doesn't deep-proxy the large analysis payload.
const selectedAudio = shallowRef<string | null>(null);
const selectedAnalysis = shallowRef<Analysis | null>(null);
let inspectorTimer: number | undefined;

watch(
  () => lib.selectedBase,
  (base) => {
    const t = lib.selected;
    editTitle.value = t?.title ?? "";
    editArtist.value = t?.artist ?? "";
    selectedAudio.value = null;
    selectedAnalysis.value = null;
    // Debounced: clicking through the grid shouldn't queue an audio decode per card.
    window.clearTimeout(inspectorTimer);
    void loadSongManager(base);
    if (!base || !t) return;
    inspectorTimer = window.setTimeout(async () => {
      const [audio, analysis] = await Promise.all([
        t.isOgg ? api.trackAudioPath(base) : Promise.resolve(null),
        api.getAnalysis(base),
      ]);
      if (lib.selectedBase !== base) return; // selection moved on
      selectedAudio.value = audio;
      selectedAnalysis.value = analysis;
    }, 180);
  },
  { immediate: true },
);

onBeforeUnmount(() => window.clearTimeout(inspectorTimer));

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase();
  if (!q) return lib.tracks;
  return lib.tracks.filter(
    (t) => t.title.toLowerCase().includes(q) || t.artist.toLowerCase().includes(q) || t.base.includes(q),
  );
});

async function importSongs() {
  const selected = await open({
    multiple: true,
    filters: [{ name: "Audio", extensions: ["ogg", "wav", "mp3", "flac", "m4a", "aac"] }],
  });
  if (!selected) return;
  const paths = Array.isArray(selected) ? selected : [selected];
  lib.error = null;
  for (const p of paths) {
    converting.value = p as string;
    try {
      await lib.importFrom(p as string);
    } catch (e) {
      lib.error = String(e);
      break;
    } finally {
      converting.value = null;
    }
  }
}

async function convert(t: Track) {
  converting.value = t.base;
  lib.error = null;
  try {
    await lib.convert(t.path);
    notice.value = `Converted ${t.file} to ${t.base}.ogg`;
  } catch (e) {
    lib.error = String(e);
  } finally {
    converting.value = null;
  }
}

/* ---------------- song manager: packs, artwork, game install -------------- */
const packInfo = ref<PackInfo | null>(null);
const packName = ref("");
const assets = ref<SongAssets | null>(null);
const installing = ref(false);
const artPrompt = ref("");
const artKind = ref<"banner" | "background">("banner");
const artMedia = ref<"image" | "video">("image");
const artJobId = ref<string | null>(null);
const artJobRunning = computed(() => !!artJobId.value && !!engine.job(artJobId.value)?.running);

// Artwork generation needs a Google backend; the OpenAI path has no image model.
// AI artwork went through Google's image models; StepForge uses no online
// service, so the generator is not offered (picking your own images still is).
const canGenerateArt = computed(() => false);

/** Cache-bust so a regenerated file doesn't show the stale image. */
const assetVersion = ref(0);
function assetUrl(kind: "banner" | "background"): string | null {
  const p = kind === "banner" ? assets.value?.banner : assets.value?.background;
  if (!p) return null;
  if (/\.(mp4|mov|webm|avi|mpe?g)$/i.test(p)) return null; // video: no thumbnail
  return `${convertFileSrc(p)}?v=${assetVersion.value}`;
}

async function loadSongManager(base: string | null) {
  if (!settings.loaded) await settings.load();
  // Fetched into a local first: this runs from an immediate watcher above the
  // `packInfo` declaration, and `packInfo.value = await …` would touch the ref
  // before the await, i.e. before it exists.
  const info = await api.listPacks();
  packInfo.value = info;
  // An unfiled song is offered the pack used last, so filing a run of songs
  // into one pack is "save, save, save" rather than retyping the name.
  packName.value = base ? (info.assignments[base] ?? settings.value.lastPack ?? "") : "";
  assets.value = base ? await api.getSongAssets(base) : null;
  assetVersion.value++;
}

async function savePack() {
  const t = lib.selected;
  if (!t) return;
  const name = packName.value.trim();
  packName.value = name;
  await api.setSongPack(t.base, name);
  packInfo.value = await api.listPacks();
  if (name && settings.value.lastPack !== name) {
    settings.value.lastPack = name;
    void settings.save();
  }
  notice.value = name ? `Assigned to pack "${name}"` : "Removed from pack";
}

/** Packs to offer: ones songs are filed under here, and folders that already
 *  exist in the game's Songs directory. */
const packOptions = computed(() => {
  const info = packInfo.value;
  if (!info) return [];
  const counts: Record<string, number> = {};
  for (const p of Object.values(info.assignments)) counts[p] = (counts[p] ?? 0) + 1;
  const inGame = new Set(info.installedPacks ?? []);
  const names = [...new Set([...info.packs, ...inGame])].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
  return names.map((name) => ({
    name,
    note: [counts[name] ? `${counts[name]} here` : "", inGame.has(name) ? "in game" : ""].filter(Boolean).join(" · "),
  }));
});

/* -------------------- selection: tick songs for bulk generation ----------- */
/** Ticked songs (by base). A new Set on every change keeps it simply reactive. */
const checked = ref<Set<string>>(new Set());
let lastChecked: string | null = null;
/** Songs that can be ticked in the current view (converted, search applied). */
const selectable = computed(() => filtered.value.filter((t) => t.isOgg));
const checkedTracks = computed(() => lib.tracks.filter((t) => t.isOgg && checked.value.has(t.base)));
const allChecked = computed(() => selectable.value.length > 0 && selectable.value.every((t) => checked.value.has(t.base)));
const someChecked = computed(() => !allChecked.value && selectable.value.some((t) => checked.value.has(t.base)));

function toggleCheck({ base, range }: { base: string; range: boolean }) {
  const next = new Set(checked.value);
  const list = selectable.value.map((t) => t.base);
  if (range && lastChecked && list.includes(lastChecked) && list.includes(base)) {
    const [a, b] = [list.indexOf(lastChecked), list.indexOf(base)].sort((x, y) => x - y);
    for (const s of list.slice(a, b + 1)) next.add(s);
  } else if (next.has(base)) {
    next.delete(base);
  } else {
    next.add(base);
  }
  lastChecked = base;
  checked.value = next;
}
/** Tick every song in view, or untick them if all are ticked. */
function toggleAll() {
  const next = new Set(checked.value);
  if (allChecked.value) selectable.value.forEach((t) => next.delete(t.base));
  else selectable.value.forEach((t) => next.add(t.base));
  checked.value = next;
}
function clearChecked() {
  checked.value = new Set();
  lastChecked = null;
}
// Forget ticks for songs that are gone (deleted, renamed on disk).
watch(
  () => lib.tracks,
  (tracks) => {
    const alive = new Set(tracks.map((t) => t.base));
    if ([...checked.value].some((b) => !alive.has(b)))
      checked.value = new Set([...checked.value].filter((b) => alive.has(b)));
  },
);
// Cmd/Ctrl+A ticks every song in view; Escape clears. Not while typing.
function onKey(e: KeyboardEvent) {
  const el = e.target as HTMLElement | null;
  if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "a") {
    e.preventDefault();
    const next = new Set(checked.value);
    selectable.value.forEach((t) => next.add(t.base));
    checked.value = next;
  } else if (e.key === "Escape" && checked.value.size && !batchOpen.value) {
    clearChecked();
  }
}
window.addEventListener("keydown", onKey);
onBeforeUnmount(() => window.removeEventListener("keydown", onKey));

/* --------------------- batch: generate the ticked songs ------------------- */
const batchOpen = ref(false);
const batchPack = ref("");
const batchInstall = ref(true);
/** The engine a batch uses: StepForge charts with its onboard model only. */
const batchEngine = computed<ChartEngine>(() => "neural");
const batchEngineLabel = computed(() =>
  batchEngine.value === "neural" ? "StepForge Flow" : batchEngine.value === "deterministic" ? "Deterministic DSP" : "AI Designer",
);
/** What a batch charts: the ticked songs. */
const batchTracks = computed(() => checkedTracks.value);
const batchBlocker = computed(() => {
  if (!lib.engineReady) return "The engine environment is not ready.";
  if (!batchTracks.value.length) return "Tick the songs to chart first.";
  if (batchEngine.value === "llm")
    return "A batch runs only on a local engine, so it can never run up a bill. Choose Neural or Deterministic on the Generate screen.";
  if (batchEngine.value === "neural" && !neural.ready) return "The neural model is not installed. See Settings.";
  if (engine.anyRunning) return "Another run is in progress.";
  return null;
});
async function openBatch() {
  if (!settings.loaded) await settings.load();
  await neural.load();
  if (!packInfo.value) packInfo.value = await api.listPacks();
  batchPack.value = settings.value.lastPack ?? "";
  batchOpen.value = true;
}
async function startBatch() {
  if (batchBlocker.value) return;
  const pack = batchPack.value.trim();
  if (pack && settings.value.lastPack !== pack) {
    settings.value.lastPack = pack;
    void settings.save();
  }
  const style: Record<string, number> = {};
  for (const f of neural.fields) {
    const v = settings.value.neuralStyle[f.key];
    if (typeof v === "number" && Number.isFinite(v)) style[f.key] = Math.min(f.max, Math.max(f.min, v));
  }
  const preset = settings.value.neuralStylePreset;
  batchOpen.value = false;
  lib.error = null;
  const tracks = [...batchTracks.value];
  clearChecked();
  await batch.start({
    tracks,
    engine: batchEngine.value,
    difficulties: await api.difficultiesFor("stepforge"),
    stylePreset: preset && neural.presets.includes(preset) ? preset : null,
    styleJson: Object.keys(style).length ? JSON.stringify(style) : null,
    optionsJson: optionsJson(settings.value.neuralOptions),
    pack: pack || undefined,
    install: batchInstall.value && !!packInfo.value?.songsDirExists,
  });
  await lib.refresh();
  packInfo.value = await api.listPacks();
}

async function pickAsset(kind: "banner" | "background") {
  const t = lib.selected;
  if (!t) return;
  const exts =
    kind === "banner"
      ? ["png", "jpg", "jpeg", "webp"]
      : ["png", "jpg", "jpeg", "webp", "mp4", "mov", "webm", "avi", "mpg"];
  const f = await open({ multiple: false, filters: [{ name: "Artwork", extensions: exts }] });
  if (!f) return;
  try {
    await api.setSongAsset(t.base, kind, f as string);
    assets.value = await api.getSongAssets(t.base);
    assetVersion.value++;
    notice.value = `${kind} set`;
  } catch (e) {
    lib.error = String(e);
  }
}

async function generateArt() {
  const t = lib.selected;
  if (!t) return;
  try {
    const j = await engine.startAssetGen({
      base: t.base,
      kind: artKind.value,
      media: artMedia.value,
      prompt: artPrompt.value,
      trackId: t.id,
    });
    artJobId.value = j.id;
  } catch (e) {
    lib.error = String(e);
  }
}

// When generation finishes, point the simfile at the new file and refresh.
watch(
  () => (artJobId.value ? engine.job(artJobId.value)?.running : undefined),
  async (running, was) => {
    const t = lib.selected;
    if (!(was && !running) || !t) return;
    if (engine.job(artJobId.value!)?.success) {
      try {
        await api.attachGeneratedAsset(t.base, artKind.value);
        assets.value = await api.getSongAssets(t.base);
        assetVersion.value++;
        notice.value = "Artwork generated and attached";
      } catch (e) {
        lib.error = String(e);
      }
    } else {
      lib.error = engine.failureOf(artJobId.value) ?? "Artwork generation failed";
    }
  },
);

/** The pack this song is actually saved under (not the unsaved input value). */
const savedPack = computed(() => {
  const b = lib.selectedBase;
  return b ? (packInfo.value?.assignments[b] ?? "") : "";
});
const packSongCount = computed(() => {
  const p = savedPack.value;
  if (!p || !packInfo.value) return 0;
  return Object.values(packInfo.value.assignments).filter((v) => v === p).length;
});

/** Install one song, or every charted song in `pack`. */
async function install(pack?: string) {
  const t = lib.selected;
  if (!t && !pack) return;
  installing.value = true;
  lib.error = null;
  try {
    const r = pack
      ? await api.installToGame({ pack })
      : await api.installToGame({ base: t!.base });
    notice.value =
      r.installed.length === 1
        ? `Installed to ${r.installed[0]}`
        : `Installed ${r.installed.length} songs to the game`;
  } catch (e) {
    lib.error = String(e);
  } finally {
    installing.value = false;
  }
}

async function saveMeta() {
  const t = lib.selected;
  if (!t) return;
  savingMeta.value = true;
  try {
    await api.registerTrackMeta(t.id, editTitle.value, editArtist.value);
    notice.value = `Registered "${editTitle.value}" — exports will carry this title.`;
    await lib.refresh();
  } catch (e) {
    lib.error = String(e);
  } finally {
    savingMeta.value = false;
  }
}
</script>

<template>
  <div class="h-full flex flex-col">
    <ContextMenu ref="menu" />
    <header class="h-16 border-b border-outline-variant flex items-center justify-between px-margin-desktop shrink-0 bg-surface">
      <div class="flex items-center gap-4 flex-1 max-w-md">
        <div class="relative w-full">
          <span class="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-on-surface-variant">search</span>
          <input
            v-model="query"
            class="w-full bg-surface-container-low border border-outline-variant rounded-full py-2 pl-10 pr-4 text-on-surface focus:outline-none focus:border-secondary focus:ring-1 focus:ring-secondary transition-colors"
            placeholder="Search library..."
            type="text"
          />
        </div>
      </div>
      <div class="flex items-center gap-4">
        <button
          class="w-10 h-10 rounded-full flex items-center justify-center text-on-surface-variant hover:bg-surface-variant transition-colors"
          title="Rescan track folders"
          :disabled="lib.loading"
          @click="lib.refresh()"
        >
          <span class="material-symbols-outlined" :class="{ spinner: lib.loading }">refresh</span>
        </button>

        <button
          class="bg-primary-container text-on-primary-container px-6 py-2 rounded-full font-label-mono hover:bg-primary hover:text-on-primary transition-colors flex items-center gap-2 disabled:opacity-50"
          :disabled="!lib.canImport"
          :title="lib.canImport ? 'Import audio' : lib.statusLoading ? 'Starting the charting engine…' : 'The charting engine is not ready (see the banner)'"
          @click="importSongs"
        >
          <span class="material-symbols-outlined text-[18px]">add</span>
          Import song
        </button>
      </div>
    </header>

    <!-- Batch: set-up, progress, summary -->
    <div v-if="batchOpen" class="border-b border-outline-variant bg-surface-container px-margin-desktop py-3 flex flex-wrap items-center gap-3 text-[12px]">
      <span class="text-on-surface">
        Chart the <b>{{ batchTracks.length }}</b> selected {{ batchTracks.length === 1 ? "song" : "songs" }}
        with <b>{{ batchEngineLabel }}</b>, all five difficulties. Songs are analyzed first if they need it.
      </span>
      <label class="text-on-surface-variant">Pack</label>
      <div class="w-56 flex"><PackCombobox v-model="batchPack" :options="packOptions" placeholder="Leave songs unfiled" /></div>
      <label class="flex items-center gap-2 text-on-surface-variant" :class="{ 'opacity-40': !packInfo?.songsDirExists }">
        <input v-model="batchInstall" type="checkbox" class="accent-[#bc13fe]" :disabled="!packInfo?.songsDirExists" />
        Install to the game
      </label>
      <span v-if="batchBlocker" class="text-tertiary-fixed-dim basis-full">{{ batchBlocker }}</span>
      <div class="ml-auto flex gap-2">
        <button class="px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant" @click="batchOpen = false">Cancel</button>
        <button
          class="px-4 py-1.5 rounded bg-primary-container text-on-primary-container hover:bg-primary disabled:opacity-40"
          :disabled="!!batchBlocker"
          @click="startBatch"
        >
          Start
        </button>
      </div>
    </div>
    <div v-else-if="batch.running" class="border-b border-outline-variant bg-surface-container px-margin-desktop py-3 flex items-center gap-4 text-[12px]">
      <Spinner :size="18" class="text-secondary" label="Generating" />
      <span class="text-on-surface">Generating {{ batch.index }} / {{ batch.total }}</span>
      <span class="text-on-surface-variant truncate">{{ batch.current }}</span>
      <div class="flex-1 h-1 rounded bg-surface-variant overflow-hidden min-w-[80px]">
        <div class="h-full bg-primary-container" :style="{ width: `${(100 * (batch.index - 1)) / Math.max(1, batch.total)}%` }"></div>
      </div>
      <button class="px-3 py-1.5 rounded border border-error text-error hover:bg-error/10" :disabled="batch.cancelled" @click="batch.cancel()">
        {{ batch.cancelled ? "Stopping…" : "Stop" }}
      </button>
    </div>
    <div v-else-if="batch.finished" class="border-b border-outline-variant bg-surface-container px-margin-desktop py-3 flex items-start gap-4 text-[12px]">
      <div class="flex-1">
        <p class="text-on-surface">
          Charted {{ batch.done.length }} of {{ batch.total }} {{ batch.total === 1 ? "track" : "tracks" }}<template v-if="batch.cancelled"> (stopped)</template><template v-if="batch.pack">, filed under “{{ batch.pack }}”</template><template v-if="batch.installed">, {{ batch.installed }} installed to the game</template>.
        </p>
        <ul v-if="batch.failed.length" class="mt-1 text-error space-y-0.5">
          <li v-for="f in batch.failed" :key="f.title">• {{ f.title }}: {{ f.error }}</li>
        </ul>
      </div>
      <button class="px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant" @click="batch.dismiss()">Dismiss</button>
    </div>

    <EngineBanner />

    <div class="flex-1 flex overflow-hidden">
      <div class="flex-1 overflow-y-auto p-margin-desktop scrollbar-hide">
        <DropZone class="mb-6" :disabled="!lib.canImport" @imported="(b: string) => lib.select(b)" />

        <div v-if="notice" class="mb-4 p-3 rounded-lg border border-green-500/40 bg-green-500/10 text-green-300 text-body-md flex justify-between">
          <span>{{ notice }}</span>
          <button class="text-on-surface-variant" @click="notice = null">✕</button>
        </div>
        <div v-if="lib.error" class="mb-6 p-4 rounded-lg border border-error/40 bg-error-container/20 text-error text-body-md whitespace-pre-wrap">
          {{ lib.error }}
        </div>

        <div v-if="lib.loading && !lib.tracks.length" class="text-on-surface-variant font-label-mono flex items-center gap-2 py-6">
          <Spinner :size="18" /> Loading library…
        </div>

        <div v-else-if="!filtered.length" class="text-on-surface-variant font-body-md py-16 text-center">
          <p class="mb-2">No tracks yet.</p>
          <p class="text-[13px]">
            Import an audio file — it is transcoded to <code class="font-label-mono">&lt;name&gt;.ogg</code>, the only
            format the engine reads.
          </p>
        </div>

        <template v-else>
          <!-- Selection toolbar: tick songs (or all of them) to chart in bulk -->
          <div
            class="sticky top-0 z-10 -mx-2 mb-4 px-3 py-2 rounded-xl flex flex-wrap items-center gap-3 text-[12px] transition-colors"
            :class="checked.size ? 'bg-surface-container-high border border-secondary-container/40' : 'bg-surface/90 backdrop-blur border border-transparent'"
          >
            <label class="flex items-center gap-2 cursor-pointer select-none text-on-surface">
              <input
                type="checkbox"
                class="w-4 h-4 accent-[#00eefc] cursor-pointer"
                :checked="allChecked"
                :indeterminate="someChecked"
                :disabled="!selectable.length"
                @change="toggleAll"
              />
              {{ allChecked ? "Deselect all" : "Select all" }}
            </label>
            <span v-if="checked.size" class="font-label-mono text-on-surface-variant">
              {{ checked.size }} selected
              <button class="ml-2 underline hover:text-on-surface" @click="clearChecked">Clear</button>
            </span>
            <span v-else class="text-on-surface-variant">Tick songs to generate several at once (Shift-click for a range, {{ isMac ? "⌘" : "Ctrl+" }}A for all).</span>
            <button
              class="ml-auto bg-primary-container text-on-primary-container px-4 py-1.5 rounded-full font-label-mono hover:bg-primary hover:text-on-primary transition-colors flex items-center gap-2 disabled:opacity-40"
              :disabled="!checked.size || batch.running"
              :title="batch.running ? 'A bulk run is already going' : 'Generate charts for the selected songs'"
              @click="openBatch"
            >
              <span class="material-symbols-outlined text-[16px]">auto_awesome</span>
              Generate {{ checked.size ? `${checked.size} selected` : "selected" }}
            </button>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-gutter pb-8">
            <TrackCard
              v-for="t in filtered"
              :key="t.base"
              :track="t"
              :selected="t.base === lib.selectedBase"
              :checked="checked.has(t.base)"
              :busy="converting === t.base"
              @select="lib.select($event)"
              @check="toggleCheck"
              @convert="convert"
              @menu="trackMenu"
            />
          </div>
        </template>
      </div>

      <aside v-if="lib.selected" class="bg-surface-container border-l border-outline-variant flex flex-col p-gutter w-sidebar-width shrink-0 overflow-y-auto">
        <div class="mb-6 pb-4 border-b border-outline-variant">
          <span class="text-on-surface-variant text-[10px] font-label-mono uppercase tracking-widest mb-1 block">Inspector</span>
          <h2 class="font-headline-md text-headline-md text-on-surface truncate">{{ lib.selected.title }}</h2>
          <p class="text-on-surface-variant text-[11px] mt-1 font-label-mono">id: {{ lib.selected.id }}</p>
        </div>

        <div class="flex-1 flex flex-col gap-6">
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-on-surface-variant text-[10px] font-label-mono uppercase tracking-widest">Audio Profile</span>
              <router-link
                v-if="lib.selected.isOgg"
                :to="{ name: 'analyze', params: { base: lib.selected.base } }"
                class="text-[11px] text-secondary-container hover:underline flex items-center gap-1"
              >
                View analysis <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
              </router-link>
            </div>
            <div class="w-full h-24 bg-surface-container-lowest rounded-lg border border-outline-variant overflow-hidden">
              <AnalysisCanvas :audio-path="selectedAudio" :analysis="selectedAnalysis" />
            </div>
          </div>

          <div>
            <span class="text-on-surface-variant text-[10px] font-label-mono uppercase tracking-widest mb-2 block">Properties</span>
            <ul class="space-y-1">
              <li class="flex justify-between items-center py-1.5 border-b border-outline-variant/50">
                <span class="text-on-surface-variant text-[12px]">BPM</span>
                <span class="text-primary-fixed-dim text-[12px] font-label-mono">{{ fmtBpm(lib.selected.bpm) }}</span>
              </li>
              <li class="flex justify-between items-center py-1.5 border-b border-outline-variant/50">
                <span class="text-on-surface-variant text-[12px]">Duration</span>
                <span class="text-on-surface text-[12px] font-label-mono">{{ fmtDuration(lib.selected.durationSec) }}</span>
              </li>
              <li class="flex justify-between items-center py-1.5 border-b border-outline-variant/50">
                <span class="text-on-surface-variant text-[12px]">Onsets</span>
                <span class="text-on-surface text-[12px] font-label-mono">{{ lib.selected.onsets ?? "—" }}</span>
              </li>
              <li class="flex justify-between items-center py-1.5 border-b border-outline-variant/50">
                <span class="text-on-surface-variant text-[12px]">Sustains</span>
                <span class="text-on-surface text-[12px] font-label-mono">{{ lib.selected.sustainCount ?? "—" }}</span>
              </li>
              <li class="flex justify-between items-center py-1.5 border-b border-outline-variant/50">
                <span class="text-on-surface-variant text-[12px]">File</span>
                <span class="text-on-surface text-[12px] font-label-mono truncate max-w-[140px]" :title="lib.selected.file">
                  {{ lib.selected.file }}
                </span>
              </li>
              <li class="flex justify-between items-center py-1.5 border-b border-outline-variant/50">
                <span class="text-on-surface-variant text-[12px]">Size</span>
                <span class="text-on-surface text-[12px] font-label-mono">{{ fmtSize(lib.selected.sizeBytes) }}</span>
              </li>
            </ul>
          </div>

          <!-- Song manager: pack, artwork, install to the game -->
          <div>
            <span class="text-on-surface-variant text-[10px] font-label-mono uppercase tracking-widest mb-2 block">Song Manager</span>

            <!-- Pack assignment works before charting, so you can file as you go -->
            <label class="block text-[11px] text-on-surface-variant mb-1">Pack</label>
            <div class="flex gap-2 mb-2">
              <PackCombobox v-model="packName" :options="packOptions" placeholder="Unfiled" @commit="savePack" />
              <button
                class="px-2 rounded border border-outline-variant text-on-surface hover:bg-surface-variant"
                title="Save pack"
                @click="savePack"
              >
                <span class="material-symbols-outlined text-[16px]">save</span>
              </button>
            </div>
            <p v-if="packName.trim() !== savedPack" class="text-[10px] text-tertiary-fixed-dim mb-1">
              {{ packName.trim() ? `Not saved yet — save to file this song under “${packName.trim()}”.` : "Save to remove this song from its pack." }}
            </p>
            <p v-if="packSongCount > 1" class="text-[10px] text-on-surface-variant mb-3">
              “{{ savedPack }}” holds {{ packSongCount }} songs.
            </p>
            <div v-else class="mb-3"></div>
          </div>

          <div v-if="lib.selected.targets.includes('stepforge')">
            <!-- Artwork -->
            <label class="block text-[11px] text-on-surface-variant mb-1">Artwork</label>
            <div class="grid grid-cols-2 gap-2 mb-2">
              <div v-for="k in (['banner', 'background'] as const)" :key="k" class="min-w-0">
                <div class="h-14 rounded border border-outline-variant bg-surface-container-lowest overflow-hidden flex items-center justify-center">
                  <img v-if="assetUrl(k)" :src="assetUrl(k)!" class="w-full h-full object-cover" :alt="k" />
                  <span v-else class="material-symbols-outlined text-on-surface-variant/50 text-[18px]">image</span>
                </div>
                <button
                  class="w-full mt-1 py-1 rounded border border-outline-variant text-on-surface hover:bg-surface-variant text-[10px] font-label-mono capitalize"
                  @click="pickAsset(k)"
                >
                  {{ k }}…
                </button>
              </div>
            </div>

            <!-- AI generation (Gemini/Vertex only) -->
            <div v-if="canGenerateArt" class="mb-3">
              <input
                v-model="artPrompt"
                placeholder="Describe the artwork…"
                class="w-full bg-surface-container-lowest border border-outline-variant rounded p-2 text-on-surface text-[12px] focus:border-secondary outline-none mb-1"
              />
              <div class="flex gap-1">
                <select v-model="artKind" class="flex-1 bg-surface-container-lowest border border-outline-variant rounded p-1 text-on-surface text-[11px]">
                  <option value="banner">Banner</option>
                  <option value="background">Background</option>
                </select>
                <select v-model="artMedia" class="flex-1 bg-surface-container-lowest border border-outline-variant rounded p-1 text-on-surface text-[11px]">
                  <option value="image">Image</option>
                  <option value="video">Video</option>
                </select>
                <button
                  class="px-2 rounded bg-primary-container text-on-primary-container hover:bg-primary text-[11px] font-label-mono disabled:opacity-40"
                  :disabled="!artPrompt.trim() || artJobRunning"
                  @click="generateArt"
                >
                  <span class="material-symbols-outlined text-[14px]" :class="{ spinner: artJobRunning }">
                    {{ artJobRunning ? "progress_activity" : "auto_awesome" }}
                  </span>
                </button>
              </div>
              <p class="text-[10px] text-on-surface-variant mt-1">
                Uses assetforge (Imagen / Veo) with your Vertex or Gemini credentials.
              </p>
            </div>

            <div class="flex gap-2">
              <button
                class="flex-1 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant text-[12px] font-label-mono flex items-center justify-center gap-2 disabled:opacity-40"
                :disabled="installing"
                title="Copy this song into the game's Songs folder"
                @click="install()"
              >
                <span class="material-symbols-outlined text-[16px]" :class="{ spinner: installing }">
                  {{ installing ? "progress_activity" : "download" }}
                </span>
                Install song
              </button>
              <button
                v-if="savedPack"
                class="flex-1 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant text-[12px] font-label-mono flex items-center justify-center gap-2 disabled:opacity-40"
                :disabled="installing"
                :title="`Install all ${packSongCount} song(s) in “${savedPack}”`"
                @click="install(savedPack)"
              >
                <span class="material-symbols-outlined text-[16px]">library_music</span>
                Install pack
              </button>
            </div>
            <p v-if="packInfo && !packInfo.songsDirExists" class="text-[10px] text-tertiary-fixed-dim mt-1">
              Songs folder not found — set it in Settings.
            </p>
          </div>

          <!-- Metadata editor: registers TRACK_META so exports aren't titled with the slug -->
          <div>
            <span class="text-on-surface-variant text-[10px] font-label-mono uppercase tracking-widest mb-2 block">Metadata</span>
            <input
              v-model="editTitle"
              placeholder="Title"
              class="w-full mb-2 bg-surface-container-lowest border border-outline-variant rounded p-2 text-on-surface text-[12px] focus:border-secondary outline-none"
            />
            <input
              v-model="editArtist"
              placeholder="Artist"
              class="w-full mb-2 bg-surface-container-lowest border border-outline-variant rounded p-2 text-on-surface text-[12px] focus:border-secondary outline-none"
            />
            <button
              class="w-full py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-variant text-[12px] font-label-mono disabled:opacity-50"
              :disabled="savingMeta || !editTitle.trim()"
              @click="saveMeta"
            >
              {{ savingMeta ? "Saving…" : lib.selected.registered ? "Update registration" : "Register for export" }}
            </button>
            <p class="text-[10px] text-on-surface-variant mt-1">
              Writes TRACK_META in config.py so simfiles carry a real title/artist.
            </p>
          </div>
        </div>

        <div class="pt-4 mt-auto">
          <button
            v-if="lib.selected.status === 'unconverted'"
            class="w-full bg-primary-container text-on-primary-container py-2 rounded-lg font-label-mono hover:bg-primary transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
            :disabled="converting === lib.selected.base"
            @click="convert(lib.selected)"
          >
            <span class="material-symbols-outlined text-[18px]">sync_alt</span> Convert to .ogg
          </button>
          <button
            v-else
            class="w-full bg-primary-container text-on-primary-container py-2 rounded-lg font-label-mono hover:bg-primary transition-colors flex items-center justify-center gap-2"
            @click="
              router.push({
                name: lib.selected!.status === 'charted' ? 'results' : 'generate',
                params: { base: lib.selected!.base },
              })
            "
          >
            <span class="material-symbols-outlined text-[18px]">arrow_forward</span>
            {{ lib.selected.status === "charted" ? "Open Results" : "Generate Charts" }}
          </button>
        </div>
      </aside>
    </div>
  </div>
</template>
