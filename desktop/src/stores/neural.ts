import { defineStore } from "pinia";
import { api } from "../lib/api";
import type { NeuralStatus, NeuralStyleField } from "../lib/types";

/** A status the views can render without null-checking every field. Missing
 *  keys become "not there" rather than `undefined`: an engine that answers with
 *  less than the full contract must read as not installed, not crash a v-for. */
function normalize(raw: Partial<NeuralStatus> | null | undefined, error?: string): NeuralStatus {
  const r = raw ?? {};
  const fields = Array.isArray(r.style_fields) ? r.style_fields : [];
  return {
    installed: r.installed === true,
    model_dir: typeof r.model_dir === "string" ? r.model_dir : "",
    default_model_dir:
      typeof r.default_model_dir === "string" && r.default_model_dir ? r.default_model_dir : null,
    quantization: typeof r.quantization === "string" ? r.quantization : "",
    style_rag_available: r.style_rag_available === true,
    presets: Array.isArray(r.presets) ? r.presets.filter((p) => typeof p === "string") : [],
    codec_version: typeof r.codec_version === "string" ? r.codec_version : null,
    version: typeof r.version === "string" && r.version ? r.version : null,
    display_name: typeof r.display_name === "string" && r.display_name ? r.display_name : null,
    trained_with_stems: r.trained_with_stems === true,
    params:
      r.params && typeof r.params.stage_a === "number" && typeof r.params.stage_b === "number"
        ? r.params
        : null,
    metrics: r.metrics && typeof r.metrics === "object" ? r.metrics : {},
    style_fields: fields.filter(
      (f): f is NeuralStyleField =>
        !!f && typeof f.key === "string" && [f.min, f.max, f.step].every((n) => Number.isFinite(n)),
    ),
    onnxruntime: r.onnxruntime === true,
    error: error ?? (typeof r.error === "string" && r.error ? r.error : null),
  };
}

/** Concurrent callers (Generate and Settings mounting together) share one probe. */
let inflight: Promise<void> | null = null;

/**
 * The local neural chart engine's status (`beatforge neural-status`).
 *
 * Each probe starts a Python process, so the answer is cached here and only
 * re-fetched on request — Settings does that after a save or on "Re-check".
 */
export const useNeural = defineStore("neural", {
  state: () => ({
    status: null as NeuralStatus | null,
    loading: false,
    /** The probe got no answer at all: the installed engine predates the
     *  neural chart engine (an app update does not replace the engine code of a
     *  workspace that already exists). Rust reads the same probe the same way
     *  and then speaks to that engine in its old argv, so the AI Designer and
     *  Deterministic engines keep working; only neural is out of reach. */
    legacy: false,
  }),
  getters: {
    installed: (s): boolean => !!s.status?.installed,
    /** Weights present AND onnxruntime importable: a neural run can start. */
    ready: (s): boolean => !!s.status?.installed && !!s.status.onnxruntime,
    fields: (s): NeuralStyleField[] => s.status?.style_fields ?? [],
    presets: (s): string[] => s.status?.presets ?? [],
  },
  actions: {
    /** Fetch once; pass `force` to probe again. Never throws — a probe that
     *  cannot run at all becomes a not-installed status carrying the reason. */
    async load(force = false): Promise<void> {
      if (this.status && !force) return;
      if (inflight) return inflight;
      this.loading = true;
      inflight = (async () => {
        try {
          this.status = normalize(await api.neuralStatus());
          this.legacy = false;
        } catch (e) {
          this.status = normalize(null, String(e));
          this.legacy = true;
        } finally {
          this.loading = false;
          inflight = null;
        }
      })();
      return inflight;
    },
    refresh(): Promise<void> {
      return this.load(true);
    },
  },
});
