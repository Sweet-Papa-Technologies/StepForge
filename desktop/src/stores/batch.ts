import { defineStore } from "pinia";
import { api } from "../lib/api";
import { useEngine } from "./engine";
import type { ChartEngine, Track } from "../lib/types";

/** One "generate every track" run. A store rather than component state so it
 *  keeps going, and stays visible, when the user leaves the Library. */
export const useBatch = defineStore("batch", {
  state: () => ({
    running: false,
    cancelled: false,
    total: 0,
    index: 0,
    current: "" as string,
    done: [] as string[],
    failed: [] as { title: string; error: string }[],
    installed: 0,
    pack: "",
  }),
  getters: {
    /** A finished run that has not been dismissed. */
    finished: (s): boolean => !s.running && s.total > 0,
  },
  actions: {
    dismiss() {
      if (!this.running) this.total = 0;
    },
    cancel() {
      this.cancelled = true;
    },
    /** Chart `tracks` one after another with a LOCAL engine, all difficulties,
     *  and — when `pack` is given — file each under it and copy it into the
     *  game. Sequential on purpose: runs for one workspace share its cache. */
    async start(opts: {
      tracks: Track[];
      engine: ChartEngine;
      difficulties: string[];
      stylePreset?: string | null;
      styleJson?: string | null;
      pack?: string;
      install?: boolean;
    }) {
      if (this.running) return;
      const engine = useEngine();
      Object.assign(this, {
        running: true, cancelled: false, total: opts.tracks.length, index: 0,
        current: "", done: [], failed: [], installed: 0, pack: opts.pack ?? "",
      });
      try {
        for (const t of opts.tracks) {
          if (this.cancelled) break;
          this.index += 1;
          this.current = t.title || t.base;
          try {
            const job = await engine.startGenerate({
              trackId: t.id,
              target: "stepforge",
              difficulties: opts.difficulties,
              deterministic: opts.engine === "deterministic",
              iHaveRights: false,
              backend: "local",
              engine: opts.engine,
              stylePreset: opts.engine === "neural" ? opts.stylePreset : undefined,
              styleJson: opts.engine === "neural" ? opts.styleJson : undefined,
            });
            await new Promise<void>((resolve) => {
              const tick = window.setInterval(() => {
                if (this.cancelled && engine.job(job.id)?.running) void engine.cancel(job.id);
                if (!engine.job(job.id)?.running) {
                  window.clearInterval(tick);
                  resolve();
                }
              }, 150);
            });
            if (!engine.job(job.id)?.success) {
              if (!this.cancelled)
                this.failed.push({ title: this.current, error: engine.failureOf(job.id) ?? "generation failed" });
              continue;
            }
            this.done.push(t.base);
            if (opts.pack) await api.setSongPack(t.base, opts.pack);
            if (opts.install) {
              await api.installToGame({ base: t.base });
              this.installed += 1;
            }
          } catch (e) {
            this.failed.push({ title: this.current, error: String(e) });
          }
        }
      } finally {
        this.running = false;
        this.current = "";
      }
    },
  },
});
