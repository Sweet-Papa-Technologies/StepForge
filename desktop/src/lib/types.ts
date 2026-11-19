export type TrackStatus = "unconverted" | "draft" | "analyzed" | "charted";
export type Target = "stepforge" | "saberforge" | "native";

export interface Track {
  /** Id passed to the CLI via --track. */
  id: string;
  /** Filename stem — names every build artifact. Used as the route key. */
  base: string;
  title: string;
  artist: string;
  file: string;
  path: string;
  isOgg: boolean;
  registered: boolean;
  bpm: number | null;
  offset: number | null;
  onsets: number | null;
  durationSec: number | null;
  sizeBytes: number;
  sustainCount: number | null;
  status: TrackStatus;
  targets: Target[];
}

/** One arrow in a parsed simfile chart. */
export interface ChartNote {
  time: number;
  beat: number;
  column: number;
  kind: "tap" | "hold" | "roll" | "mine";
  endTime: number | null;
}

export interface Chart {
  difficulty: string;
  meter: number;
  stepstype: string;
  columns: number;
  beat0Time: number;
  bpm: number;
  notes: ChartNote[];
}

/** A pickable option with the measured trade-off behind it — used for both the
 *  Gemini model list and the model-routing modes. */
export interface GeminiModel {
  id: string;
  label: string;
  note: string;
  recommended: boolean;
}

/** A user-defined request header row (OpenAI-endpoint auth escape hatch). */
export interface HttpHeader {
  name: string;
  value: string;
}

export interface Settings {
  engineRoot: string;
  pythonPath: string;
  ffmpegPath: string;
  llmBackend: string;
  /** hybrid | single — see the BEATFORGE_LLM_MODE block in config.py. */
  llmMode: string;
  hybridHeavyModel: string;
  hybridLiteModel: string;
  hybridCascade: boolean;
  /** Used in `single` mode only. */
  geminiModel: string;
  thinkingLevel: string;
  geminiApiKey: string;
  vertexProject: string;
  vertexKey: string;
  openaiBaseUrl: string;
  openaiModel: string;
  openaiApiKey: string;
  /** Extra headers sent with every request to the OpenAI endpoint. A header set
   *  here overrides the client's default of the same name (case-insensitively);
   *  a blank value drops that default header entirely. */
  openaiHeaders: HttpHeader[];
  /** Attach the track as an `input_audio` part. Off = text-only requests, for
   *  the many endpoints that reject audio; charting then runs on the locally
   *  measured DSP analysis alone. */
  openaiSendAudio: boolean;
  /** Inkling on a Modal Managed Endpoint — the URL from the Modal dashboard. */
  inklingEndpointUrl: string;
  /** Served model id. Blank resolves it from `/v1/models`, which several Modal
   *  endpoints refuse with a 401 even though chat completions work — set it
   *  explicitly there. */
  inklingModel: string;
  /** Modal proxy tokens. Two headers, not a Bearer token; the secret is shown
   *  once at creation and cannot be retrieved afterwards. */
  modalTokenId: string;
  modalTokenSecret: string;
  /** Reasoning effort per call type — reasoning bills at the full output rate. */
  inklingEffortDesigner: string;
  inklingEffortCritic: string;
  inklingEffortRetry: string;
  /** Gemini 4 Argon. Rides an existing Gemini credential (Vertex project or
   *  Gemini API key): auto | vertex | aistudio. */
  argonTransport: string;
  /** standard | flex. Flex is half price with variable latency, and exists for
   *  Gemini 4 on AI Studio only — Vertex Flex PayGo excludes it. */
  argonServiceTier: string;
  /** Blank = Argon judges its own charts. A Flash id here makes the critic score
   *  comparable with Flash-designed runs. */
  argonCriticModel: string;
  /** Thinking per call type (low | medium | high). Thinking bills as output. */
  argonThinkingDesigner: string;
  argonThinkingRetry: string;
  argonThinkingCritic: string;
  /** Chunked processing: auto (default) | always | never. `auto` designs a track
   *  longer than `maxChunkSeconds` one section at a time — that section's audio,
   *  onsets and density budget only. `never` is the whole-song path. */
  chunkMode: string;
  /** Longest a chunk may be, and the `auto` threshold. This is the VRAM lever:
   *  a chunk's audio and inventory are what has to fit in the model's KV cache.
   *  `null` = unset, use the engine default (30). */
  maxChunkSeconds: number | null;
  /** Bars of lead-in/lead-out each chunk hears as context but must not chart.
   *  `null` = unset, use the engine default (1). */
  chunkOverlapBars: number | null;
  outputDir: string;
  songsDir: string;
  assetforgePath: string;
  theme: string;
  /** Ask GitHub at startup whether a newer release exists (lib/updateCheck).
   *  On unless the user turned it off. */
  checkForUpdates: boolean;
  /** The release the update notice was dismissed for ("0.1.0-beta.7"); blank =
   *  none. That one is not offered again; a later one is. */
  dismissedUpdateVersion: string;
  /** Chart engine for the StepMania target: neural | llm | deterministic.
   *  Blank = never picked; Generate then defaults to neural when the model is
   *  installed, else the LLM designer. Not the same thing as `llmBackend`, which
   *  only says who serves the model when the engine is `llm`. */
  chartEngine: string;
  /** Neural style preset (one of `NeuralStatus.presets`); blank = none. */
  neuralStylePreset: string;
  /** Style sliders the user actually set, keyed by `NeuralStyleField.key`. A
   *  key that is ABSENT means "let the model infer it from the audio" — never
   *  store a placeholder number here, it would be sent. */
  neuralStyle: Record<string, number>;
  /** Style presets saved on the Generate screen, by name. */
  neuralUserPresets: Record<string, NeuralUserPreset>;
  /** The pack last filed under in the Library; offered for the next song. */
  lastPack: string;
}

/** The engines the Generate screen offers for the StepMania target. */
export type ChartEngine = "neural" | "llm" | "deterministic";

/** One style slider the neural engine accepts (`neural-status` → style_fields). */
export interface NeuralStyleField {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
}

/** Result of the engine's `neural-status --json`: the local neural chart engine
 *  (StepForge Flow). Keys are the engine's own, as with `Analysis`. `installed`
 *  is false until the weights exist, and `error` then says what to do — that is
 *  a normal state, not a failure. */
export interface NeuralStatus {
  installed: boolean;
  /** Where the engine looks for the weights. The app always uses the model it
   *  ships, so this is the engine's default folder. */
  model_dir: string;
  /** Where it looks when no folder is set. Null from an engine that does not
   *  report it. */
  default_model_dir: string | null;
  quantization: string;
  style_rag_available: boolean;
  presets: string[];
  codec_version: string | null;
  /** The model's release, from model_meta.json (e.g. "1.4"); null if unstated. */
  version: string | null;
  /** Its name for display (e.g. "StepForge Flow v1.4"); null if unstated. */
  display_name: string | null;
  trained_with_stems: boolean;
  params: { stage_a: number; stage_b: number } | null;
  metrics: Record<string, unknown>;
  style_fields: NeuralStyleField[];
  /** False when the onnxruntime package does not import in the engine's Python. */
  onnxruntime: boolean;
  /** Why it did not import: the real exception ("ImportError: DLL load failed
   *  …"). Null when it imported, or from an engine that does not report it. */
  onnxruntime_error: string | null;
  /** The model's files are in place. `installed` also needs onnxruntime, so a
   *  complete model beside a broken onnxruntime is `installed: false` with this
   *  true — which is not "no model". */
  weights_present: boolean;
  /** The engine ran and does not know `neural-status`: it predates the model.
   *  Nothing else sets this (a broken or missing engine is `probe_failed`). */
  legacy: boolean;
  /** The status check itself got no answer; `error` says how it failed. */
  probe_failed: boolean;
  error: string | null;
}

/** Result of Settings → Test connection (engine `inkling-check`). */
export interface InklingCheck {
  ok: boolean;
  model: string | null;
  endpoint?: string;
  credentialSource: string | null;
  error: string | null;
}

/** Result of the differential audio probe (engine `inkling-probe`). A failure
 *  here means the endpoint answers without listening — charts from it would be
 *  designed by a model that never heard the song. */
export interface InklingProbe {
  passed: boolean;
  similarity: number | null;
  reason: string | null;
  responses: string[];
  model: string | null;
  error: string | null;
}

/** Result of Settings → Test Argon (engine `argon-check`): the resolved setup,
 *  plus the two-tone audio probe that proves credentials, model access and
 *  audio input in two tiny calls. */
export interface ArgonCheck {
  ok: boolean;
  error: string | null;
  transport?: string;
  model?: string;
  criticModel?: string;
  serviceTier?: string;
  /** The tier the server confirmed serving — "standard" unless it said Flex. */
  servedTier?: string;
  costUsd?: number;
  endpoint?: string;
  probe: {
    passed: boolean;
    reason: string | null;
    expected?: string[];
    observed?: string[];
    responses?: string[];
  } | null;
}

export interface PackInfo {
  assignments: Record<string, string>;
  packs: string[];
  /** Pack folders that already exist in the game's Songs directory. */
  installedPacks?: string[];
  songsDir: string | null;
  songsDirExists: boolean;
  detected: string[];
}

export interface SongAssets {
  dir: string | null;
  banner: string | null;
  background: string | null;
}

export interface SongCost {
  song: string;
  base: string;
  title: string | null;
  usd: number;
  llmUsd: number;
  computeUsd: number;
  modelCalls: number;
  cacheHitAnalysis: boolean;
  charts: Array<{
    difficulty: string;
    notes: number;
    score: number | null;
    meter: number | null;
    gatesPass: boolean | null;
  }>;
  bestScore: number | null;
  totalNotes: number;
}

export interface EngineStatus {
  engineRoot: string | null;
  python: string;
  pythonOk: boolean;
  /** Why the interpreter did not start ("" when it did). */
  pythonError: string;
  beatforgeOk: boolean;
  /** The first required module that failed and its real exception, or how
   *  Python died while importing ("" when the engine is fine). */
  importError: string;
  /** module -> exception, for every module that did not import. */
  moduleErrors: Record<string, string>;
  /** Why onnxruntime did not import ("" when it did). */
  onnxruntimeError: string;
  /** The engine runs on the Python inside the app (release builds). */
  bundled: boolean;
  ffmpeg: string;
  ffmpegOk: boolean;
  /** Needed by the neural engine only — never part of "is the engine ready". */
  onnxruntimeOk: boolean;
}

/** One DSP onset candidate (analysis.onsets[]). */
export interface Onset {
  id: string;
  time: number;
  nearest_beat: number;
  snap_error_ms: number;
  strength: number;
  sustain?: boolean;
  sustain_beats?: number;
}

/** Bar-indexed structural section (analysis.sections[]). */
export interface Section {
  name?: string;
  start_bar?: number;
  end_bar?: number;
  tier?: string;
  [k: string]: unknown;
}

export interface Analysis {
  bpm?: number;
  offset?: number;
  duration_s?: number;
  onsets?: Onset[];
  sections?: Section[];
  sustain_available?: number;
  beat_backend?: string;
  stem_source?: string;
  track_file?: string;
  sanity_warnings?: string[];
  [k: string]: unknown;
}

export interface ResultSet {
  target: Target;
  difficulties: string[];
  outDir: string;
  mode: string | null;
}

export interface Gate {
  name: string;
  pass: boolean;
}

export interface ResultDetail {
  target: Target;
  difficulty: string;
  /** Critic score on the engine's 0–10 scale (ship threshold 7). */
  score: number | null;
  criticIssues: Array<{ where?: string; problem?: string; [k: string]: unknown }>;
  criticSummary: string | null;
  humanReview: boolean;
  gates: Gate[];
  metrics: Record<string, any> | null;
  notes: number | null;
  designNotes: string | null;
  designSections: unknown[];
  previewPng: string | null;
  clickOgg: string | null;
  mode?: string | null;
  outDir: string;
  extra: Record<string, any> | null;
  attempts?: number;
}

export interface LogEvent {
  jobId: string;
  stream: "stdout" | "stderr";
  line: string;
}

export interface ExitEvent {
  jobId: string;
  code: number;
  success: boolean;
}

/** A style the user saved: an optional built-in preset to start from, plus the
 *  sliders that were set (an absent slider stays on auto). */
export interface NeuralUserPreset {
  preset: string;
  style: Record<string, number>;
}
