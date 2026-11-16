import { defineStore } from "pinia";
import { api } from "../lib/api";
import type { Settings } from "../lib/types";

const DEFAULTS: Settings = {
  engineRoot: "",
  pythonPath: "python3",
  ffmpegPath: "",
  llmBackend: "gemini",
  llmMode: "single",
  hybridHeavyModel: "gemini-3.8-flash",
  hybridLiteModel: "gemini-3.5-flash-lite",
  hybridCascade: true,
  geminiModel: "gemini-3.8-flash",
  thinkingLevel: "high",
  geminiApiKey: "",
  vertexProject: "",
  vertexKey: "",
  openaiBaseUrl: "",
  openaiModel: "",
  openaiApiKey: "",
  openaiHeaders: [],
  openaiSendAudio: true,
  inklingEndpointUrl: "",
  inklingModel: "",
  modalTokenId: "",
  modalTokenSecret: "",
  inklingEffortDesigner: "high",
  inklingEffortCritic: "low",
  inklingEffortRetry: "medium",
  argonTransport: "auto",
  argonServiceTier: "standard",
  argonCriticModel: "",
  argonThinkingDesigner: "high",
  argonThinkingRetry: "medium",
  argonThinkingCritic: "low",
  // Chunked processing is the DEFAULT (REQ-CHUNK-08). The two tuning knobs stay
  // null until the user touches them, so an untouched install follows whatever
  // the engine's numbers are rather than being pinned by the GUI.
  chunkMode: "auto",
  maxChunkSeconds: null,
  chunkOverlapBars: null,
  outputDir: "",
  songsDir: "",
  assetforgePath: "",
  theme: "dark",
  // Blank = never picked: Generate defaults to neural when the model is
  // installed. No sliders are set until the user sets one.
  chartEngine: "",
  neuralStylePreset: "",
  neuralStyle: {},
  neuralUserPresets: {},
  lastPack: "",
};

/** A fresh copy — DEFAULTS holds an array (openaiHeaders) and an object
 *  (neuralStyle), so a shallow spread would share them between every reset and
 *  the edits would leak backwards. */
function defaults(): Settings {
  return { ...DEFAULTS, openaiHeaders: [], neuralStyle: {}, neuralUserPresets: {} };
}

/** Saved presets, keeping only well-formed ones. */
function cleanPresets(raw: unknown): Settings["neuralUserPresets"] {
  const out: Settings["neuralUserPresets"] = {};
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    for (const [name, v] of Object.entries(raw as Record<string, any>)) {
      if (!name.trim() || !v || typeof v !== "object") continue;
      out[name] = { preset: typeof v.preset === "string" ? v.preset : "", style: cleanStyle(v.style) };
    }
  }
  return out;
}

/** Keep only real numbers: a NaN would serialise as `null`, which is not a
 *  slider value and would be rejected when the settings are saved. */
function cleanStyle(raw: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    for (const [k, v] of Object.entries(raw)) {
      if (typeof v === "number" && Number.isFinite(v)) out[k] = v;
    }
  }
  return out;
}

export const useSettings = defineStore("settings", {
  state: () => ({
    value: defaults(),
    loaded: false,
    saving: false,
  }),
  actions: {
    async load() {
      try {
        const s = { ...defaults(), ...(await api.getSettings()) };
        // Older settings.json files predate this field, and a non-array would
        // break the editor's v-for.
        if (!Array.isArray(s.openaiHeaders)) s.openaiHeaders = [];
        // Rust sends null for "never set" (see openai_send_audio) — that has to
        // read as the default ON, not as a falsy "user turned audio off".
        s.openaiSendAudio = s.openaiSendAudio !== false;
        // Chunked mode is the default, and a settings.json written before this
        // field existed is a fresh install as far as it is concerned. Rust fills
        // the blank in too; this is the same guarantee on the frontend so the
        // dropdown never renders empty.
        if (!s.chunkMode) s.chunkMode = "auto";
        s.neuralStyle = cleanStyle(s.neuralStyle);
        s.neuralUserPresets = cleanPresets(s.neuralUserPresets);
        if (typeof s.lastPack !== "string") s.lastPack = "";
        this.value = s;
      } catch {
        this.value = defaults();
      }
      this.loaded = true;
    },
    async save() {
      this.saving = true;
      try {
        await api.saveSettings(this.value);
      } finally {
        this.saving = false;
      }
    },
  },
});
