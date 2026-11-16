<script setup lang="ts">
import { computed, ref } from "vue";

/** A text field with a filtered list under it: pick an existing pack or type
 *  a new name. (A native <datalist> does this in a browser, but renders
 *  inconsistently in the platform webviews Tauri runs on.) */
const props = defineProps<{
  modelValue: string;
  /** Names to offer, with an optional note shown beside each. */
  options: { name: string; note?: string }[];
  placeholder?: string;
}>();
const emit = defineEmits<{ "update:modelValue": [value: string]; commit: [] }>();

const open = ref(false);
const cursor = ref(-1);

const matches = computed(() => {
  const q = props.modelValue.trim().toLowerCase();
  const all = props.options;
  if (!q) return all;
  // An exact match means the user is looking at a chosen value: show everything
  // so the list still works as a dropdown.
  if (all.some((o) => o.name.toLowerCase() === q)) return all;
  return all.filter((o) => o.name.toLowerCase().includes(q));
});

function choose(name: string) {
  emit("update:modelValue", name);
  open.value = false;
  cursor.value = -1;
  emit("commit");
}
function onInput(e: Event) {
  emit("update:modelValue", (e.target as HTMLInputElement).value);
  open.value = true;
  cursor.value = -1;
}
function move(step: number) {
  if (!open.value) {
    open.value = true;
    return;
  }
  const n = matches.value.length;
  if (n) cursor.value = (cursor.value + step + n) % n;
}
function onEnter() {
  const pick = open.value ? matches.value[cursor.value] : undefined;
  if (pick) choose(pick.name);
  else {
    open.value = false;
    emit("commit");
  }
}
/** Delayed so a click on an option lands before the list closes. */
function onBlur() {
  window.setTimeout(() => (open.value = false), 120);
}
</script>

<template>
  <div class="relative flex-1 min-w-0">
    <input
      :value="modelValue"
      :placeholder="placeholder"
      class="w-full bg-surface-container-lowest border border-outline-variant rounded p-2 pr-7 text-on-surface text-[12px] focus:border-secondary outline-none"
      role="combobox"
      aria-autocomplete="list"
      :aria-expanded="open"
      autocomplete="off"
      spellcheck="false"
      @input="onInput"
      @focus="open = true"
      @blur="onBlur"
      @keydown.down.prevent="move(1)"
      @keydown.up.prevent="move(-1)"
      @keydown.enter.prevent="onEnter"
      @keydown.esc="open = false"
    />
    <button
      type="button"
      tabindex="-1"
      class="absolute right-1 top-1/2 -translate-y-1/2 text-on-surface-variant hover:text-on-surface"
      title="Show packs"
      @mousedown.prevent="open = !open"
    >
      <span class="material-symbols-outlined text-[16px]">expand_more</span>
    </button>
    <ul
      v-if="open && matches.length"
      class="absolute z-20 left-0 right-0 mt-1 max-h-48 overflow-y-auto rounded border border-outline-variant bg-surface-container-high shadow-lg"
      role="listbox"
    >
      <li
        v-for="(o, i) in matches"
        :key="o.name"
        role="option"
        :aria-selected="i === cursor"
        class="px-2 py-1.5 text-[12px] cursor-pointer flex justify-between gap-2"
        :class="i === cursor ? 'bg-surface-variant text-on-surface' : 'text-on-surface hover:bg-surface-variant'"
        @mousedown.prevent="choose(o.name)"
      >
        <span class="truncate">{{ o.name }}</span>
        <span v-if="o.note" class="shrink-0 text-[10px] text-on-surface-variant">{{ o.note }}</span>
      </li>
    </ul>
  </div>
</template>
