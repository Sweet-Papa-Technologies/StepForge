<script setup lang="ts">
import { onMounted } from "vue";
import SideNav from "./components/SideNav.vue";
import ActivityBar from "./components/ActivityBar.vue";
import { useEngine } from "./stores/engine";
import { useLibrary } from "./stores/library";
import { useSettings } from "./stores/settings";

const engine = useEngine();
const lib = useLibrary();
const settings = useSettings();

onMounted(async () => {
  await engine.init();
  await settings.load();
  await Promise.all([lib.refresh(), lib.loadStatus()]);
});
</script>

<template>
  <div class="h-screen w-screen bg-surface text-on-surface overflow-hidden">
    <ActivityBar />
    <SideNav />
    <div class="ml-sidebar-width h-full">
      <router-view v-slot="{ Component }">
        <component :is="Component" />
      </router-view>
    </div>
  </div>
</template>
