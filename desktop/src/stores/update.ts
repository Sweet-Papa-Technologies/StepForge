import { defineStore } from "pinia";
import { checkForUpdate, type UpdateNotice } from "../lib/updateCheck";
import { useSettings } from "./settings";

/**
 * The update check: once per launch, in the background, and only when the
 * setting is on. It finds out whether a newer release exists and nothing more —
 * the app never downloads or installs one (see lib/updateCheck.ts for the one
 * request it makes).
 */
export const useUpdate = defineStore("update", {
  state: () => ({
    /** A newer release the user has not dismissed, once the check found one. */
    found: null as UpdateNotice | null,
    /** The check has been started this launch (it is never repeated). */
    started: false,
  }),
  getters: {
    /** What the notice shows. Turning the setting off hides it at once. */
    notice(): UpdateNotice | null {
      return useSettings().value.checkForUpdates ? this.found : null;
    },
  },
  actions: {
    /** Call after the settings have loaded. Never throws and never blocks:
     *  offline, timed out or rate-limited all end as "no notice". */
    async checkAtLaunch(): Promise<void> {
      if (this.started) return;
      this.started = true;
      const settings = useSettings();
      if (!settings.loaded || !settings.value.checkForUpdates) return;
      this.found = await checkForUpdate(settings.value.dismissedUpdateVersion);
    },
    /** Hide this version for good. A later release shows the notice again. */
    async dismiss(): Promise<void> {
      if (!this.found) return;
      const settings = useSettings();
      settings.value.dismissedUpdateVersion = this.found.version;
      this.found = null;
      try {
        await settings.save();
      } catch {
        /* Not saved: the notice comes back next launch, which is the safe way to fail. */
      }
    },
  },
});
