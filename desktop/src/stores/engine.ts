import { defineStore } from "pinia";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { api } from "../lib/api";
import { newJobId } from "../lib/format";
import { ensureNotificationPermission, notify } from "../lib/notify";
import type { ExitEvent, LogEvent, Target } from "../lib/types";

export interface LogLine {
  stream: "stdout" | "stderr";
  line: string;
}

export interface Job {
  id: string;
  kind: "analyze" | "generate" | "inpaint" | "cost" | "asset";
  trackId?: string;
  lines: LogLine[];
  running: boolean;
  code: number | null;
  success: boolean | null;
  stage: number;
  costEst: number | null;
  startedAt: number;
}

/** Mirrors the stages the engine actually prints while charting. */
export const GEN_STAGES = [
  "Analyze audio (tempo, beats, onsets)",
  "Place the steps (RhythmNet)",
  "Choose the arrows (ChoreoNet)",
  "Check every step & repair",
  "Previews & save",
] as const;

const COST_RE = /\$([0-9]+\.[0-9]+)/;

/** Human labels used in desktop notifications. */
const JOB_LABEL: Record<string, string> = {
  analyze: "Analysis",
  generate: "Chart generation",
  inpaint: "Section re-roll",
  cost: "Cost report",
  asset: "Artwork generation",
};

/** Map a real engine log line onto a pipeline stage. Stages only move forward. */
function inferStage(line: string, current: number): number {
  const l = line.toLowerCase();
  if (l.includes("rendering previews") || l.includes("done —") || l.includes("[stepforge]") || l.includes("[saberforge]") || l.includes(" -> "))
    return Math.max(current, 4);
  if (l.includes("repair") || l.includes("validate") || l.includes("gates")) return Math.max(current, 3);
  // The neural engine's own steps: `[hard] neural: rhythm…` places the steps,
  // `neural: choreography…` chooses the arrows. A step added later under the
  // `neural:` prefix counts as placement until it is given its own stage.
  if (l.includes("neural:") && l.includes("choreograph")) return Math.max(current, 2);
  if (l.includes("neural:")) return Math.max(current, 1);
  if (l.includes("designing") || l.includes("design ok") || l.includes("[chart]")) return Math.max(current, 1);
  if (l.includes("[analyze]") || l.includes("bpm=") || l.includes("onset")) return Math.max(current, 0);
  return current;
}

const unlisteners: UnlistenFn[] = [];

/** Max lines retained per job (rendered views slice a smaller window still). */
const MAX_LINES = 400;

/**
 * The engine can emit output faster than Vue can re-render a log list. Buffer
 * incoming lines outside reactivity and flush on a short timer, so a chatty run
 * costs a few re-renders per second instead of one per line.
 */
const pending = new Map<string, LogLine[]>();
let flushTimer: number | undefined;

export const useEngine = defineStore("engine", {
  state: () => ({
    jobs: {} as Record<string, Job>,
    initialized: false,
  }),
  getters: {
    job: (s) => (id: string | null | undefined) => (id ? s.jobs[id] : undefined),

    /** Every job still executing. Jobs live in the store (not in a component),
     *  so navigating away never interrupts them — the view re-attaches on mount. */
    runningJobs: (s): Job[] => Object.values(s.jobs).filter((j) => j.running),
    anyRunning: (s): boolean => Object.values(s.jobs).some((j) => j.running),

    /** Most recent job of a kind for a track — used to re-attach after navigation. */
    jobFor:
      (s) =>
      (kind: Job["kind"], trackId?: string): Job | undefined => {
        let best: Job | undefined;
        for (const j of Object.values(s.jobs)) {
          if (j.kind !== kind) continue;
          if (trackId && j.trackId !== trackId) continue;
          if (!best || j.startedAt > best.startedAt) best = j;
        }
        return best;
      },

    /** A running job of any kind for this track — used to serialise actions. */
    busyFor:
      (s) =>
      (trackId: string | undefined): Job | undefined =>
        trackId ? Object.values(s.jobs).find((j) => j.running && j.trackId === trackId) : undefined,
    /** Best-effort failure reason for a finished job. */
    failureOf: (s) => (id: string | null | undefined) => {
      const j = id ? s.jobs[id] : undefined;
      if (!j || j.running || j.success !== false) return null;
      const errs = j.lines.filter((l) => l.stream === "stderr" && l.line.trim());
      const tail = errs.length ? errs : j.lines.filter((l) => l.line.trim());
      return tail.slice(-4).map((l) => l.line).join("\n") || "The engine exited with an error.";
    },
  },
  actions: {
    async init() {
      if (this.initialized) return;
      this.initialized = true;
      unlisteners.push(
        await listen<LogEvent>("engine:log", (e) => {
          const { jobId, stream, line } = e.payload;
          if (!this.jobs[jobId]) return;
          const buf = pending.get(jobId);
          if (buf) buf.push({ stream, line });
          else pending.set(jobId, [{ stream, line }]);
          if (flushTimer === undefined) {
            flushTimer = window.setTimeout(() => {
              flushTimer = undefined;
              this.flushLogs();
            }, 80);
          }
        }),
      );
      unlisteners.push(
        await listen<ExitEvent>("engine:exit", (e) => {
          this.flushLogs();
          const j = this.jobs[e.payload.jobId];
          if (!j) return;
          j.running = false;
          j.code = e.payload.code;
          j.success = e.payload.success;
          if (e.payload.success && j.kind === "generate") j.stage = GEN_STAGES.length;

          const what = JOB_LABEL[j.kind] ?? j.kind;
          const who = j.trackId ? ` — ${j.trackId}` : "";
          const secs = Math.round((Date.now() - j.startedAt) / 1000);
          void notify(
            e.payload.success ? `${what} complete` : `${what} failed`,
            e.payload.success
              ? `${j.trackId ?? "Job"} finished in ${secs}s`
              : `${what}${who} exited with code ${e.payload.code}`,
          );
        }),
      );
      // Ask up-front so the first completion isn't swallowed by a prompt.
      void ensureNotificationPermission();
    },
    /** Apply buffered log lines to their jobs in one reactive update. */
    flushLogs() {
      if (!pending.size) return;
      for (const [jobId, buf] of pending) {
        const j = this.jobs[jobId];
        if (!j || !buf.length) continue;
        let stage = j.stage;
        let cost = j.costEst;
        for (const l of buf) {
          if (j.kind === "generate") stage = inferStage(l.line, stage);
          const m = l.line.match(COST_RE);
          if (m) cost = parseFloat(m[1]);
        }
        j.lines.push(...buf);
        if (j.lines.length > MAX_LINES) j.lines.splice(0, j.lines.length - MAX_LINES);
        if (stage !== j.stage) j.stage = stage;
        if (cost !== j.costEst) j.costEst = cost;
      }
      pending.clear();
    },
    _make(kind: Job["kind"], trackId?: string): Job {
      const id = newJobId();
      const job: Job = {
        id, kind, trackId, lines: [], running: true,
        code: null, success: null, stage: 0, costEst: null, startedAt: Date.now(),
      };
      this.jobs[id] = job;
      void notify(
        `${JOB_LABEL[kind] ?? kind} started`,
        trackId ? `Working on ${trackId}` : "Job started",
      );
      return job;
    },
    _fail(job: Job, err: unknown) {
      job.running = false;
      job.success = false;
      job.lines.push({ stream: "stderr", line: String(err) });
    },
    async startAnalyze(trackId: string, backend: string, force = false): Promise<Job> {
      const job = this._make("analyze", trackId);
      try {
        await api.analyzeTrack(job.id, trackId, backend, force);
      } catch (err) {
        this._fail(job, err);
      }
      return job;
    },
    async startGenerate(opts: {
      trackId: string;
      target: Target;
      difficulties: string[];
      deterministic: boolean;
      iHaveRights: boolean;
      backend: string;
      engine?: string;
      stylePreset?: string | null;
      styleJson?: string | null;
    }): Promise<Job> {
      const job = this._make("generate", opts.trackId);
      try {
        await api.generateCharts({ jobId: job.id, ...opts });
      } catch (err) {
        this._fail(job, err);
      }
      return job;
    },
    /** Re-roll a bar range of an existing neural chart. Its own job kind: it is
     *  not a generation (no pipeline stages), but it does hold the track — it
     *  rewrites the song folder — so `busyFor` serialises it with everything else. */
    async startInpaint(opts: {
      trackId: string;
      difficulty: string;
      startBar: number;
      endBar: number;
      seed?: number | null;
    }): Promise<Job> {
      const job = this._make("inpaint", opts.trackId);
      try {
        await api.inpaintChart({ jobId: job.id, ...opts });
      } catch (err) {
        this._fail(job, err);
      }
      return job;
    },
    /** AI artwork generation via assetforge (Imagen / Veo). */
    async startAssetGen(opts: {
      base: string;
      kind: "banner" | "background";
      media: "image" | "video";
      prompt: string;
      trackId?: string;
    }): Promise<Job> {
      const job = this._make("asset", opts.trackId);
      try {
        await api.generateSongAsset({ jobId: job.id, ...opts });
      } catch (err) {
        this._fail(job, err);
      }
      return job;
    },
    async startCostRefresh(): Promise<Job> {
      const job = this._make("cost");
      try {
        await api.refreshCostReport(job.id);
      } catch (err) {
        this._fail(job, err);
      }
      return job;
    },
    async cancel(id: string) {
      await api.cancelJob(id);
      const j = this.jobs[id];
      if (j) {
        j.running = false;
        j.success = false;
        j.lines.push({ stream: "stderr", line: "> Job cancelled by user." });
      }
    },
  },
});
