import { defineStore } from "pinia";
import { api } from "../lib/api";
import type { EngineStatus, Track } from "../lib/types";

export const useLibrary = defineStore("library", {
  state: () => ({
    tracks: [] as Track[],
    /** Selected track's `base` (the filename stem — unique per file). */
    selectedBase: null as string | null,
    loading: false,
    /** The engine probe is running (it starts Python, so the first one after
     *  launch can take a few seconds, longer on Windows). */
    statusLoading: false,
    error: null as string | null,
    status: null as EngineStatus | null,
  }),
  getters: {
    selected: (s): Track | undefined => s.tracks.find((t) => t.base === s.selectedBase),
    byBase: (s) => (base: string) => s.tracks.find((t) => t.base === base),
    /** The engine can only run when Python + the beatforge package resolve. */
    engineReady: (s): boolean => !!s.status?.pythonOk && !!s.status?.beatforgeOk,
    /** Importing transcodes to .ogg: with ffmpeg, or through the engine
     *  (libsndfile: WAV, MP3, FLAC, OGG) when there is none. */
    canImport: (s): boolean =>
      !!s.status?.ffmpegOk || (!!s.status?.pythonOk && !!s.status?.beatforgeOk),
  },
  actions: {
    async refresh() {
      this.loading = true;
      this.error = null;
      try {
        this.tracks = await api.listTracks();
        if (!this.tracks.some((t) => t.base === this.selectedBase)) {
          this.selectedBase = this.tracks[0]?.base ?? null;
        }
      } catch (e) {
        this.error = String(e);
        this.tracks = [];
      } finally {
        this.loading = false;
      }
    },
    async loadStatus(force = false) {
      this.statusLoading = true;
      try {
        this.status = await api.engineStatus(force);
      } catch {
        this.status = null;
      } finally {
        this.statusLoading = false;
      }
    },
    select(base: string) {
      this.selectedBase = base;
    },
    async importFrom(path: string) {
      const t = await api.importTrack(path);
      await this.refresh();
      this.selectedBase = t.base;
      return t;
    },
    async convert(path: string) {
      const t = await api.convertTrack(path);
      await this.refresh();
      this.selectedBase = t.base;
      return t;
    },
  },
});
