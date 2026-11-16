import { invoke } from "@tauri-apps/api/core";
import type {
  Analysis,
  ArgonCheck,
  Chart,
  EngineStatus,
  GeminiModel,
  InklingCheck,
  InklingProbe,
  NeuralStatus,
  PackInfo,
  ResultDetail,
  ResultSet,
  Settings,
  SongAssets,
  SongCost,
  Target,
  Track,
} from "./types";

export const api = {
  listTracks: () => invoke<Track[]>("list_tracks"),
  importTrack: (source: string, title?: string) => invoke<Track>("import_track", { source, title }),
  convertTrack: (path: string) => invoke<Track>("convert_track", { path }),

  getAnalysis: (base: string) => invoke<Analysis | null>("get_analysis", { base }),
  trackAudioPath: (base: string) => invoke<string | null>("track_audio_path", { base }),

  difficultiesFor: (target: Target) => invoke<string[]>("difficulties_for", { target }),
  defaultDifficultiesFor: (target: Target) => invoke<string[]>("default_difficulties_for", { target }),

  analyzeTrack: (jobId: string, trackId: string, backend: string, force: boolean) =>
    invoke<void>("analyze_track", { jobId, trackId, backend, force }),
  generateCharts: (opts: {
    jobId: string;
    trackId: string;
    target: Target;
    difficulties: string[];
    deterministic: boolean;
    iHaveRights: boolean;
    backend: string;
    /** StepMania target only: neural | llm | deterministic. Omitted, the
     *  engine is derived from `deterministic`. */
    engine?: string;
    /** Neural engine only. */
    stylePreset?: string | null;
    /** Neural engine only: a JSON object of just the sliders the user set. */
    styleJson?: string | null;
  }) => invoke<void>("generate_charts", opts),
  /** Re-roll bars `startBar..endBar` (0-based, end exclusive) of an existing
   *  neural chart in place, in the style the chart was generated with (the
   *  engine reads it back from report.json). */
  inpaintChart: (opts: {
    jobId: string;
    trackId: string;
    difficulty: string;
    startBar: number;
    endBar: number;
    seed?: number | null;
  }) => invoke<void>("inpaint_chart", opts),
  cancelJob: (jobId: string) => invoke<boolean>("cancel_job", { jobId }),

  getChartNotes: (base: string, difficulty: string) =>
    invoke<Chart>("get_chart_notes", { base, difficulty }),
  geminiModels: () => invoke<GeminiModel[]>("gemini_models"),
  llmModes: () => invoke<GeminiModel[]>("llm_modes"),
  chunkModes: () => invoke<GeminiModel[]>("chunk_modes"),

  listResults: (base: string) => invoke<ResultSet[]>("list_results", { base }),
  getResult: (base: string, target: Target, difficulty: string) =>
    invoke<ResultDetail>("get_result", { base, target, difficulty }),

  costReport: () =>
    invoke<{ rollup: any; markdown: string | null; hasLedger: boolean }>("cost_report"),
  costBySong: () => invoke<SongCost[]>("cost_by_song"),

  // Packs & game install
  listPacks: () => invoke<PackInfo>("list_packs"),
  setSongPack: (base: string, pack: string) => invoke<void>("set_song_pack", { base, pack }),
  installToGame: (opts: { base?: string; pack?: string }) =>
    invoke<{ installed: string[] }>("install_to_game", opts),

  // Song artwork
  getSongAssets: (base: string) => invoke<SongAssets>("get_song_assets", { base }),
  setSongAsset: (base: string, kind: "banner" | "background", source: string) =>
    invoke<{ file: string; dir: string; isVideo: boolean }>("set_song_asset", { base, kind, source }),
  generateSongAsset: (opts: {
    jobId: string;
    base: string;
    kind: "banner" | "background";
    media: "image" | "video";
    prompt: string;
  }) => invoke<void>("generate_song_asset", opts),
  attachGeneratedAsset: (base: string, kind: "banner" | "background") =>
    invoke<{ file: string }>("attach_generated_asset", { base, kind }),
  refreshCostReport: (jobId: string) => invoke<void>("refresh_cost_report", { jobId }),

  /** Memoized in Rust (~0.7s to probe); pass force to re-run the checks. */
  engineStatus: (force = false) => invoke<EngineStatus>("engine_status", { force }),
  getSettings: () => invoke<Settings>("get_settings"),
  saveSettings: (settings: Settings) => invoke<void>("save_settings", { settings }),

  /** Resolve Modal proxy-token auth + the served model string. Uses the SAVED
   *  settings, so save before calling. */
  inklingCheck: () => invoke<InklingCheck>("inkling_check"),
  /** Differential audio probe: two clips, one prompt, temperature 0. Slow on a
   *  cold Managed Endpoint (they scale to zero). */
  inklingProbe: () => invoke<InklingProbe>("inkling_probe"),
  /** Gemini 4 Argon: resolve transport + credentials, then the two-tone audio
   *  probe at low thinking (about a cent). Uses the SAVED settings. */
  argonCheck: () => invoke<ArgonCheck>("argon_check"),
  /** The local neural engine: weights installed?, where, presets and sliders.
   *  Free and offline. Uses the SAVED settings (the model folder). */
  neuralStatus: () => invoke<Partial<NeuralStatus>>("neural_status"),
  registerTrackMeta: (trackId: string, title: string, artist: string) =>
    invoke<void>("register_track_meta", { trackId, title, artist }),
  exportSongFolder: (base: string, target: Target, destDir: string) =>
    invoke<string>("export_song_folder", { base, target, destDir }),
};
