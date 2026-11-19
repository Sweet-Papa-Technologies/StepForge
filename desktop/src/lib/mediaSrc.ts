import { convertFileSrc } from "@tauri-apps/api/core";
import { onBeforeUnmount, ref, watch, type Ref } from "vue";

/**
 * Address of a local audio file for an `<audio>` element.
 *
 * Everywhere but Linux this is the file's `asset:` address, which the web view
 * streams. On Linux the web view (WebKitGTK) plays media through GStreamer,
 * and an `<audio>` given an `asset:` address fails to load there (a format
 * error, before any data is read) although `fetch` of the same address works.
 * So on Linux the file is fetched and handed to the element as a `blob:`
 * address, which WebKitGTK does play.
 */
const NEEDS_BLOB = /\bLinux\b/.test(navigator.userAgent) && !/Android/.test(navigator.userAgent);

/** How long a replaced blob address stays valid, so an element still playing
 *  from it is not cut off before it has switched to the new one. */
const REVOKE_AFTER_MS = 5000;

function assetUrl(path: string, rev?: number): string {
  return convertFileSrc(path) + (rev ? `?v=${rev}` : "");
}

/**
 * A playable address for the file at `path()`, kept up to date. `rev()` is a
 * number that changes when the file was rewritten in place. The value is ""
 * while there is no file, and on Linux while a newly chosen file is loading.
 */
export function useMediaSrc(
  path: () => string | null | undefined,
  rev: () => number | undefined = () => undefined,
): Ref<string> {
  const url = ref("");
  if (!NEEDS_BLOB) {
    watch([path, rev], ([p, r]) => (url.value = p ? assetUrl(p, r) : ""), { immediate: true });
    return url;
  }

  let request = 0;
  let shownPath: string | null = null;
  const retire = (old: string) => {
    if (old.startsWith("blob:")) setTimeout(() => URL.revokeObjectURL(old), REVOKE_AFTER_MS);
  };
  const show = (next: string, p: string | null) => {
    const old = url.value;
    url.value = next;
    shownPath = p;
    retire(old);
  };

  watch(
    [path, rev],
    async ([p, r]) => {
      const mine = ++request;
      if (!p) return show("", null);
      // Another file: stop offering the old one. The same file rewritten: keep
      // it until the new copy is ready, so the player does not blink.
      if (p !== shownPath) show("", null);
      let next: string;
      try {
        const res = await fetch(assetUrl(p, r));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        next = URL.createObjectURL(await res.blob());
      } catch {
        next = assetUrl(p, r); // no worse than before
      }
      if (mine !== request) return retire(next);
      show(next, p);
    },
    { immediate: true },
  );
  onBeforeUnmount(() => {
    request++;
    if (url.value.startsWith("blob:")) URL.revokeObjectURL(url.value);
  });
  return url;
}
