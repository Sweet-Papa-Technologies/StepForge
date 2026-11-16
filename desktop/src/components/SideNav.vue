<script setup lang="ts">
import { getVersion } from "@tauri-apps/api/app";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { useLibrary } from "../stores/library";
import { useEngine } from "../stores/engine";

const lib = useLibrary();
const engine = useEngine();
const router = useRouter();

// Jobs live in the store and the engine process is detached, so work continues
// across navigation. Surface it so that isn't invisible.
const running = computed(() => engine.runningJobs);

const KIND_LABEL: Record<string, string> = {
  analyze: "Analyzing",
  generate: "Generating",
  inpaint: "Re-rolling",
  cost: "Cost report",
};

/* ---- global "current song" indicator + quick switcher ---------------- */
const switcherOpen = ref(false);
const filter = ref("");

const switchable = computed(() => {
  const q = filter.value.trim().toLowerCase();
  const list = q
    ? lib.tracks.filter((t) => t.title.toLowerCase().includes(q) || t.base.includes(q))
    : lib.tracks;
  return list.slice(0, 40);
});

/** Switching keeps you on the same kind of screen where that makes sense. */
function pick(base: string) {
  lib.select(base);
  switcherOpen.value = false;
  filter.value = "";
  const name = router.currentRoute.value.name;
  if (name === "analyze" || name === "generate" || name === "results") {
    router.push({ name: name as string, params: { base } });
  }
}

function closeSwitcher() {
  switcherOpen.value = false;
}
onMounted(() => window.addEventListener("click", closeSwitcher));
onBeforeUnmount(() => window.removeEventListener("click", closeSwitcher));

const STATUS_DOT: Record<string, string> = {
  unconverted: "bg-error",
  draft: "bg-tertiary-fixed-dim",
  analyzed: "bg-green-400",
  charted: "bg-secondary-container",
};

function goToJob(kind: string, trackId?: string) {
  const base = lib.tracks.find((t) => t.id === trackId)?.base;
  if (kind === "cost") router.push({ name: "cost" });
  // A section re-roll is started from, and reports to, the Results screen.
  else if (base && kind === "inpaint") router.push({ name: "results", params: { base } });
  else if (base) router.push({ name: kind === "analyze" ? "analyze" : "generate", params: { base } });
}

// Route each nav item to the currently-selected track where relevant.
const nav = [
  { name: "Library", icon: "library_music", to: () => ({ name: "library" }) },
  { name: "Generate", icon: "auto_awesome", to: () => ({ name: "generate", params: { base: lib.selectedBase ?? undefined } }) },
  { name: "Results", icon: "analytics", to: () => ({ name: "results", params: { base: lib.selectedBase ?? undefined } }) },
];

/** The app's own version, from the bundle (falls back while it loads). */
const appVersion = ref("0.1");
getVersion().then((v) => (appVersion.value = v)).catch(() => {});
/** "0.1.0-beta.1" already says beta; a plain "0.1.0" gets the word added. */
const versionLabel = computed(() =>
  /-/.test(appVersion.value) ? `v${appVersion.value}` : `v${appVersion.value} Beta`);
</script>

<template>
  <nav
    class="bg-surface-container-low fixed left-0 top-0 h-full w-sidebar-width border-r border-outline-variant flex flex-col py-margin-desktop z-50"
  >
    <div class="px-gutter mb-10">
      <h1 class="font-headline-md text-headline-md font-black text-primary flex items-center gap-2">
        <span class="material-symbols-outlined icon-fill text-[28px]">graphic_eq</span>
        StepForge
      </h1>
      <p class="font-label-mono text-label-mono text-on-surface-variant mt-1 uppercase tracking-widest">
        {{ versionLabel }}
      </p>
    </div>

    <!-- Current song: what every other screen acts on. Click to switch. -->
    <div class="px-4 mb-4 relative">
      <p class="font-label-mono text-[10px] uppercase tracking-widest text-on-surface-variant mb-1">
        Current song
      </p>
      <button
        class="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-surface-container border border-outline-variant hover:border-outline transition-colors text-left"
        :title="lib.selected ? `${lib.selected.title} (${lib.selected.base})` : 'No song selected'"
        @click.stop="switcherOpen = !switcherOpen"
      >
        <span
          v-if="lib.selected"
          class="w-2 h-2 rounded-full shrink-0"
          :class="STATUS_DOT[lib.selected.status] ?? 'bg-outline'"
        ></span>
        <span class="flex-1 min-w-0 truncate font-body-md text-[13px]" :class="lib.selected ? 'text-on-surface' : 'text-on-surface-variant'">
          {{ lib.selected?.title ?? "No song selected" }}
        </span>
        <span class="material-symbols-outlined text-[18px] text-on-surface-variant shrink-0">
          {{ switcherOpen ? "expand_less" : "unfold_more" }}
        </span>
      </button>

      <div
        v-if="switcherOpen"
        class="absolute left-4 right-4 top-full mt-1 z-50 rounded-lg bg-surface-container-high border border-outline-variant shadow-xl overflow-hidden"
        @click.stop
      >
        <input
          v-model="filter"
          class="w-full bg-surface-container-lowest border-b border-outline-variant px-3 py-2 text-on-surface text-[12px] outline-none"
          placeholder="Filter songs…"
          autofocus
        />
        <div class="max-h-64 overflow-y-auto">
          <button
            v-for="t in switchable"
            :key="t.base"
            class="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-surface-variant transition-colors"
            :class="t.base === lib.selectedBase ? 'bg-surface-variant/60' : ''"
            @click="pick(t.base)"
          >
            <span class="w-2 h-2 rounded-full shrink-0" :class="STATUS_DOT[t.status] ?? 'bg-outline'"></span>
            <span class="flex-1 min-w-0 truncate text-[12px] text-on-surface">{{ t.title }}</span>
            <span v-if="t.bpm" class="font-label-mono text-[10px] text-on-surface-variant shrink-0">
              {{ t.bpm.toFixed(0) }}
            </span>
          </button>
          <p v-if="!switchable.length" class="px-3 py-3 text-[12px] text-on-surface-variant">No matches.</p>
        </div>
      </div>
    </div>

    <ul class="flex flex-col gap-2 px-4 w-full font-label-mono text-label-mono flex-1">
      <li v-for="item in nav" :key="item.name">
        <router-link
          :to="item.to()"
          v-slot="{ isActive, href, navigate }"
          custom
        >
          <a
            :href="href"
            @click="navigate"
            class="flex items-center gap-3 px-4 py-3 rounded-lg transition-colors"
            :class="
              isActive
                ? 'text-primary border-l-2 border-primary bg-surface-variant/50'
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-variant'
            "
          >
            <span class="material-symbols-outlined" :class="{ 'icon-fill': isActive }">{{ item.icon }}</span>
            {{ item.name }}
          </a>
        </router-link>
      </li>
    </ul>

    <!-- Work in flight — keeps running regardless of which screen you're on. -->
    <div v-if="running.length" class="px-4 mb-3">
      <p class="font-label-mono text-[10px] uppercase tracking-widest text-on-surface-variant mb-2">
        Running
      </p>
      <button
        v-for="j in running"
        :key="j.id"
        class="w-full flex items-center gap-2 px-3 py-2 mb-1 rounded-lg bg-secondary-container/10 border border-secondary-container/30 text-secondary-container hover:bg-secondary-container/20 transition-colors text-left"
        :title="`${KIND_LABEL[j.kind] ?? j.kind} — click to view`"
        @click="goToJob(j.kind, j.trackId)"
      >
        <span class="material-symbols-outlined text-[16px] spinner shrink-0">progress_activity</span>
        <span class="font-label-mono text-[11px] truncate">
          {{ KIND_LABEL[j.kind] ?? j.kind }}<span v-if="j.trackId"> · {{ j.trackId }}</span>
        </span>
      </button>
    </div>

    <ul class="px-4 w-full font-label-mono text-label-mono">
      <li>
        <router-link
          :to="{ name: 'settings' }"
          v-slot="{ isActive, href, navigate }"
          custom
        >
          <a
            :href="href"
            @click="navigate"
            class="flex items-center gap-3 px-4 py-3 rounded-lg transition-colors"
            :class="
              isActive
                ? 'text-primary border-l-2 border-primary bg-surface-variant/50'
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-variant'
            "
          >
            <span class="material-symbols-outlined" :class="{ 'icon-fill': isActive }">settings</span>
            Settings
          </a>
        </router-link>
      </li>
    </ul>
  </nav>
</template>
