<script setup lang="ts">
import { onMounted } from "vue";
import SideNav from "./components/SideNav.vue";
import ActivityBar from "./components/ActivityBar.vue";
import UpdateNotice from "./components/UpdateNotice.vue";
import { useEngine } from "./stores/engine";
import { useLibrary } from "./stores/library";
import { useSettings } from "./stores/settings";
import { useUpdate } from "./stores/update";

const engine = useEngine();
const lib = useLibrary();
const settings = useSettings();
const update = useUpdate();

onMounted(async () => {
  await engine.init();
  await settings.load();
  // In the background: starting up never waits on the network.
  void update.checkAtLaunch();
  await Promise.all([lib.refresh(), lib.loadStatus()]);
});
</script>

<template>
  <div class="h-screen w-screen bg-surface text-on-surface overflow-hidden">
    <ActivityBar />
    <SideNav />
    <div class="ml-sidebar-width h-full flex flex-col">
      <UpdateNotice />
      <div class="flex-1 min-h-0">
        <router-view v-slot="{ Component }">
          <component :is="Component" />
        </router-view>
      </div>
    </div>
  </div>
</template>
