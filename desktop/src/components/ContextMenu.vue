<script setup lang="ts">
/** Reusable right-click menu. Call `open(event, items)` from a `@contextmenu`
 *  handler; it positions itself and closes on outside click, Escape or scroll. */
import { onBeforeUnmount, onMounted, ref } from "vue";

export interface MenuItem {
  label: string;
  icon?: string;
  disabled?: boolean;
  danger?: boolean;
  /** Renders a divider above this entry. */
  separatorBefore?: boolean;
  action?: () => void | Promise<void>;
}

const visible = ref(false);
const x = ref(0);
const y = ref(0);
const items = ref<MenuItem[]>([]);
const menu = ref<HTMLElement | null>(null);

const WIDTH = 220;

function open(e: MouseEvent, list: MenuItem[]) {
  e.preventDefault();
  e.stopPropagation();
  if (!list.length) return;
  items.value = list;
  // Keep the menu inside the viewport.
  const h = list.length * 34 + 12;
  x.value = Math.min(e.clientX, window.innerWidth - WIDTH - 8);
  y.value = Math.min(e.clientY, Math.max(8, window.innerHeight - h - 8));
  visible.value = true;
}
function close() {
  visible.value = false;
}
async function run(item: MenuItem) {
  if (item.disabled) return;
  close();
  await item.action?.();
}
function onKey(e: KeyboardEvent) {
  if (e.key === "Escape") close();
}

onMounted(() => {
  window.addEventListener("click", close);
  window.addEventListener("contextmenu", close);
  window.addEventListener("keydown", onKey);
  window.addEventListener("resize", close);
  window.addEventListener("scroll", close, true);
});
onBeforeUnmount(() => {
  window.removeEventListener("click", close);
  window.removeEventListener("contextmenu", close);
  window.removeEventListener("keydown", onKey);
  window.removeEventListener("resize", close);
  window.removeEventListener("scroll", close, true);
});

defineExpose({ open, close });
</script>

<template>
  <Teleport to="body">
    <div
      v-if="visible"
      ref="menu"
      class="fixed z-[999] py-1.5 rounded-lg bg-surface-container-high border border-outline-variant shadow-xl"
      :style="{ left: x + 'px', top: y + 'px', width: WIDTH + 'px' }"
      @click.stop
      @contextmenu.stop.prevent
    >
      <template v-for="(item, i) in items" :key="i">
        <div v-if="item.separatorBefore" class="my-1 border-t border-outline-variant/60"></div>
        <button
          class="w-full flex items-center gap-3 px-3 py-1.5 text-left font-body-md text-body-md transition-colors"
          :class="[
            item.disabled
              ? 'text-on-surface-variant/40 cursor-not-allowed'
              : item.danger
                ? 'text-error hover:bg-error/10'
                : 'text-on-surface hover:bg-surface-variant',
          ]"
          :disabled="item.disabled"
          @click="run(item)"
        >
          <span v-if="item.icon" class="material-symbols-outlined text-[18px] shrink-0">{{ item.icon }}</span>
          <span class="truncate">{{ item.label }}</span>
        </button>
      </template>
    </div>
  </Teleport>
</template>
