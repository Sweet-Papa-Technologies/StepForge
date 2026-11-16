//! Engine integration: locate the StepForge Python engine, run its CLI
//! (`python -m beatforge …`) as a child process, and stream stdout/stderr back
//! to the frontend as Tauri events. Pure filesystem reads (library, results,
//! cost) do not need Python and are handled here too.

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager};

/// One user-defined request header. A list (not a map) because the UI edits it
/// as ordered rows and a half-typed row must be allowed to have a blank name.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct HttpHeader {
    pub name: String,
    pub value: String,
}

/// Persisted user settings (credentials, engine location, tool paths, theme).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    /// Absolute path to the engine root (the folder containing `tools/`).
    pub engine_root: String,
    /// Python interpreter to launch the engine with (default: `python3`).
    pub python_path: String,
    /// ffmpeg path exported to the engine's environment (optional).
    pub ffmpeg_path: String,
    /// LLM backend selector: gemini | google-genai | openai.
    pub llm_backend: String,
    /// Model routing: `single` (default — `gemini_model` for every call) or
    /// `hybrid` (two tiers, one per kind of work). BEATFORGE_LLM_MODE.
    pub llm_mode: String,
    /// Hybrid: the model that authors retries and judges every chart.
    /// BEATFORGE_HYBRID_HEAVY_MODEL.
    pub hybrid_heavy_model: String,
    /// Hybrid: the cheap model that writes the first draft, which then has to
    /// clear the deterministic gates. BEATFORGE_HYBRID_LITE_MODEL.
    pub hybrid_lite_model: String,
    /// Hybrid: let the cheap model draft at all. Off pins drafting to the heavy
    /// model while keeping the rest of the routing. BEATFORGE_HYBRID_CASCADE.
    ///
    /// `Option` on purpose: a plain `bool` would `Default` to `false`, so every
    /// settings.json written before this field existed would silently turn the
    /// cascade OFF — i.e. the feature would ship disabled for every upgrading
    /// user. `None` means "unset, use the engine default" (on).
    pub hybrid_cascade: Option<bool>,
    /// Gemini model id used in `single` mode (BEATFORGE_GEMINI_MODEL).
    pub gemini_model: String,
    /// Reasoning budget: minimal | low | medium | high (BEATFORGE_THINKING_LEVEL).
    pub thinking_level: String,
    pub gemini_api_key: String,
    pub vertex_project: String,
    pub vertex_key: String,
    pub openai_base_url: String,
    pub openai_model: String,
    pub openai_api_key: String,
    /// Arbitrary extra HTTP headers for the OpenAI-compatible backend, so a
    /// gateway that wants more than a bearer token (an API-key header, a tenant
    /// id, a required version pin) can be used without a code change.
    /// Exported as JSON in BEATFORGE_OPENAI_HEADERS.
    pub openai_headers: Vec<HttpHeader>,
    /// Inkling on a Modal Managed Endpoint. The URL from the Modal dashboard;
    /// `/v1` is appended by the engine if it is missing.
    pub inkling_endpoint_url: String,
    /// Served model id. Blank means "resolve it from `/v1/models`" — which several
    /// Modal endpoints refuse with a 401 while accepting chat completions with the
    /// same proxy token, so it has to be settable by hand.
    pub inkling_model: String,
    /// Modal proxy tokens (`wk-…` / `ws-…`). NOT a Bearer token — Modal takes two
    /// custom headers, and the secret is shown only once at creation.
    pub modal_token_id: String,
    pub modal_token_secret: String,
    /// Reasoning effort per call type. Reasoning bills at the full completion
    /// rate, so the designer gets the budget and the cheaper roles do not.
    pub inkling_effort_designer: String,
    pub inkling_effort_critic: String,
    pub inkling_effort_retry: String,
    /// Gemini 4 Argon. Rides the existing Gemini credentials above (Vertex
    /// project / key, or the Gemini API key): auto | vertex | aistudio.
    /// BEATFORGE_ARGON_TRANSPORT.
    pub argon_transport: String,
    /// standard | flex. Flex is half price for variable latency, and is offered
    /// for Gemini 4 on AI Studio only. BEATFORGE_ARGON_SERVICE_TIER.
    pub argon_service_tier: String,
    /// Optional separate judge (blank = Argon). BEATFORGE_ARGON_CRITIC_MODEL.
    pub argon_critic_model: String,
    /// Thinking level per call type (low | medium | high). Thinking bills at the
    /// output rate, so the designer gets the budget and the rest do not.
    pub argon_thinking_designer: String,
    pub argon_thinking_retry: String,
    pub argon_thinking_critic: String,
    /// Attach the track as an `input_audio` part (BEATFORGE_OPENAI_SEND_AUDIO).
    /// Text-only endpoints — most OpenAI-compatible servers — 400 on it.
    ///
    /// `Option` for the same reason as `hybrid_cascade`: the default is ON, so a
    /// plain `bool` would `Default` to `false` and silently stop sending audio
    /// for every user upgrading from a settings.json written before this field.
    pub openai_send_audio: Option<bool>,
    /// CHUNKED PROCESSING MODE: `auto` (default — chunk a track longer than
    /// `max_chunk_seconds`), `always`, or `never` (whole-song, the pre-chunking
    /// behaviour, kept for backend comparison). BEATFORGE_CHUNK_MODE.
    ///
    /// A plain `String` rather than an `Option` because `load_settings` fills a
    /// blank in with the engine default — there is no "unset means off" trap
    /// here the way there is for the booleans below, since an empty string is
    /// not a valid mode.
    pub chunk_mode: String,
    /// Longest a single chunk may be, in seconds, and the `auto` threshold.
    /// BEATFORGE_MAX_CHUNK_SECONDS. This is the VRAM lever: a chunk's audio and
    /// inventory are what has to fit in the model's KV cache, so a local model
    /// on a small card wants this smaller.
    ///
    /// `Option` on purpose, and for the same reason as `hybrid_cascade`: a plain
    /// `f64` would `Default` to 0.0, and a settings.json written before this
    /// field existed would then pin every install to a zero-second ceiling.
    /// `None` means "unset, use the engine default".
    pub max_chunk_seconds: Option<f64>,
    /// Bars of lead-in/lead-out each chunk sees as context but must not chart.
    /// BEATFORGE_CHUNK_OVERLAP_BARS. `Option` for the same reason.
    pub chunk_overlap_bars: Option<u32>,
    /// Directory charts are exported to (default: repo `build/`).
    pub output_dir: String,
    /// The game's `Songs` directory for installs (blank = OS-aware default).
    pub songs_dir: String,
    /// assetforge CLI for AI artwork (blank = `~/.assetforge/assetforge.py`).
    pub assetforge_path: String,
    /// UI theme: dark | light.
    pub theme: String,
    /// Chart engine for the StepMania target: neural | llm | deterministic,
    /// passed as `stepforge --engine`. Blank means the user never picked one;
    /// the Generate screen then defaults to neural when the model is installed
    /// and to the LLM designer otherwise. Deliberately NOT filled in by
    /// `load_settings`, so an install that gains the weights later starts
    /// defaulting to them. Separate from `llm_backend`, which only says who
    /// serves the model when the engine is `llm`.
    pub chart_engine: String,
    /// Built-in style preset for the neural engine (`--style-preset`); blank =
    /// let the model decide.
    pub neural_style_preset: String,
    /// Style sliders the user actually set, keyed by the `style_fields` keys of
    /// `neural-status` (`--style-json`). A map rather than eight numbers on
    /// purpose: an ABSENT key means "let the model infer this from the audio",
    /// and eight `f64`s would `Default` to 0.0 and pin every style to zero.
    #[serde(deserialize_with = "lenient_style_map")]
    pub neural_style: BTreeMap<String, f64>,
    /// Style presets the user saved on the Generate screen: name ->
    /// `{ "preset": <built-in base or "">, "style": { slider: value } }`. Held
    /// as JSON because only the frontend reads it; the engine is always sent
    /// the resolved preset and sliders.
    #[serde(deserialize_with = "lenient_object_map")]
    pub neural_user_presets: BTreeMap<String, Value>,
    /// The pack name last filed under in the Library, offered again for the
    /// next song and used by a batch run.
    pub last_pack: String,
}

/// Read a map of JSON objects without letting a malformed one blank the file
/// (see `lenient_style_map`): a non-object reads as empty.
fn lenient_object_map<'de, D>(d: D) -> Result<BTreeMap<String, Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Value::deserialize(d)?;
    Ok(v.as_object()
        .map(|m| {
            m.iter()
                .filter(|(_, v)| v.is_object())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .unwrap_or_default())
}

/// Read `neural_style` without letting one bad entry take the whole file down.
///
/// `load_settings` falls back to `Settings::default()` when settings.json does
/// not parse — which would blank every credential over a `null` slider. So a
/// non-object reads as empty and a non-numeric value is dropped instead.
fn lenient_style_map<'de, D>(d: D) -> Result<BTreeMap<String, f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Value::deserialize(d)?;
    Ok(v.as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
                .collect()
        })
        .unwrap_or_default())
}

/// A running engine job, tracked so the UI can cancel it.
#[derive(Default)]
pub struct JobRegistry(pub Mutex<HashMap<String, Child>>);

/// One log line streamed from the engine.
///
/// NOTE: the camelCase rename is load-bearing. The frontend keys these events by
/// `jobId`; without it serde emits `job_id`, every event is filed under
/// `undefined`, and jobs stream no output and never appear to finish.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LogEvent {
    pub job_id: String,
    pub stream: String, // "stdout" | "stderr"
    pub line: String,
}

/// Terminal event for a job.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExitEvent {
    pub job_id: String,
    pub code: i32,
    pub success: bool,
}

fn settings_file(app: &AppHandle) -> PathBuf {
    let dir = app
        .path()
        .app_config_dir()
        .unwrap_or_else(|_| std::env::temp_dir());
    let _ = std::fs::create_dir_all(&dir);
    dir.join("settings.json")
}

pub fn load_settings(app: &AppHandle) -> Settings {
    let mut s: Settings = std::fs::read_to_string(settings_file(app))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    // NB: python_path is deliberately left blank when unset so `python_bin`
    // can fall through to the installer's venv before plain `python3`.
    if s.llm_backend.trim().is_empty() {
        s.llm_backend = "gemini".into();
    }
    if s.llm_mode.trim().is_empty() {
        s.llm_mode = "single".into(); // engine default (config.LLM_MODE)
    }
    if s.hybrid_heavy_model.trim().is_empty() {
        s.hybrid_heavy_model = "gemini-3.8-flash".into();
    }
    if s.hybrid_lite_model.trim().is_empty() {
        s.hybrid_lite_model = "gemini-3.5-flash-lite".into();
    }
    if s.gemini_model.trim().is_empty() {
        s.gemini_model = "gemini-3.8-flash".into(); // engine default
    }
    if s.thinking_level.trim().is_empty() {
        s.thinking_level = "high".into();
    }
    // CHUNKED MODE IS THE DEFAULT (REQ-CHUNK-08). Filling the blank in here is
    // what makes a fresh install — and every settings.json written before this
    // field existed — generate in chunked mode with no configuration.
    if s.chunk_mode.trim().is_empty() {
        s.chunk_mode = "auto".into(); // engine default (config.CHUNK_MODE)
    }
    if s.theme.trim().is_empty() {
        s.theme = "dark".into();
    }
    if s.engine_root.trim().is_empty() {
        if let Some(root) = detect_repo_root() {
            s.engine_root = root.to_string_lossy().into();
        }
    }
    s
}

pub fn save_settings(app: &AppHandle, s: &Settings) -> Result<(), String> {
    let txt = serde_json::to_string_pretty(s).map_err(|e| e.to_string())?;
    std::fs::write(settings_file(app), txt).map_err(|e| e.to_string())
}

/// Walk up from the current exe and cwd looking for the engine entrypoint.
///
/// A release build carries its own engine, so it never adopts a checkout just
/// because it was started from inside one; only a development build (no
/// bundled runtime) looks at the cwd. `BEATFORGE_REPO` still overrides both.
fn detect_repo_root() -> Option<PathBuf> {
    if let Ok(env) = std::env::var("BEATFORGE_REPO") {
        let p = PathBuf::from(env);
        if is_repo_root(&p) {
            return Some(p);
        }
    }
    let mut starts: Vec<PathBuf> = Vec::new();
    if crate::install::bundled_runtime().is_none() {
        if let Ok(cwd) = std::env::current_dir() {
            starts.push(cwd);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        starts.push(exe);
    }
    for start in starts {
        let mut cur = Some(start.as_path());
        while let Some(dir) = cur {
            if is_repo_root(dir) {
                return Some(dir.to_path_buf());
            }
            cur = dir.parent();
        }
    }
    // Installed app: no repo beside the binary. Use (and on first run create)
    // the per-user workspace from the installer's engine payload.
    crate::install::ensure_workspace()
}

fn is_repo_root(p: &Path) -> bool {
    p.join("tools/beatforge/__main__.py").is_file()
}

/// Interpreter to run the engine with: the explicit setting wins, then the
/// Python bundled with the app, then an installer-created venv, then whatever
/// `python3` is on PATH.
pub fn python_bin(app: &AppHandle) -> String {
    let s = load_settings(app);
    let configured = s.python_path.trim();
    // A bare "python3"/"python" means "whatever the default interpreter is" —
    // and earlier versions *auto-wrote* that value into settings.json, so
    // honouring it literally would permanently shadow the installer's venv for
    // every upgrading user. Only an explicit path overrides.
    let is_auto = configured.is_empty() || configured == "python3" || configured == "python";
    if !is_auto {
        return configured.to_string();
    }
    if let Some(p) = crate::install::bundled_python() {
        return p.to_string_lossy().into();
    }
    if let Some(p) = crate::install::venv_python() {
        return p.to_string_lossy().into();
    }
    "python3".into()
}

/// The ffmpeg executable to use (setting, else `ffmpeg` from PATH). Import
/// transcoding and the engine's assist-track render both depend on it.
pub fn ffmpeg_bin(app: &AppHandle) -> String {
    let s = load_settings(app);
    if s.ffmpeg_path.trim().is_empty() {
        "ffmpeg".into()
    } else {
        s.ffmpeg_path
    }
}

/// The resolved repo root, honouring the saved setting first.
pub fn repo_root(app: &AppHandle) -> Result<PathBuf, String> {
    let s = load_settings(app);
    if !s.engine_root.trim().is_empty() {
        let p = PathBuf::from(&s.engine_root);
        if is_repo_root(&p) {
            return Ok(p);
        }
    }
    detect_repo_root().ok_or_else(|| {
        "Could not locate the StepForge engine. Set the engine root in Settings.".into()
    })
}

/// A `Command` for the engine's interpreter. When that is the Python bundled
/// with the app, keep it from writing into the (signed, read-only) app bundle
/// and from picking up the user's own site-packages.
pub fn python_command(app: &AppHandle) -> Command {
    let py = python_bin(app);
    let mut cmd = Command::new(&py);
    for (k, v) in UTF8_ENV {
        cmd.env(k, v);
    }
    let bundled = crate::install::bundled_python()
        .map(|p| p.to_string_lossy() == py.as_str())
        .unwrap_or(false);
    if bundled {
        let cache = crate::install::workspace_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("build/pycache");
        for (k, v) in bundled_python_env(cache, |k| std::env::var_os(k).is_some()) {
            cmd.env(k, v);
        }
    }
    cmd
}

/// Every engine process talks UTF-8, whatever the platform. The app decodes
/// the engine's output as UTF-8; on Windows, Python otherwise writes piped
/// output in the ANSI code page (cp1252), where the "…" in the progress lines
/// is the byte 0x85 — not UTF-8. UTF-8 mode also makes the engine's file I/O
/// default to UTF-8 there, as it already is on macOS and Linux.
pub const UTF8_ENV: [(&str, &str); 2] = [("PYTHONUTF8", "1"), ("PYTHONIOENCODING", "utf-8")];

/// Call `f` with each line read from `r`, without the line ending.
///
/// Bytes that are not valid UTF-8 are decoded lossily (as U+FFFD) instead of
/// ending the loop. `BufRead::lines()` stops at the first such line, and the
/// reader is then dropped, which closes the pipe: the engine's next write
/// fails ("OSError: [Errno 22] Invalid argument" on Windows) and the run dies.
pub fn for_each_line_lossy<R: Read>(r: R, mut f: impl FnMut(String)) {
    let mut reader = BufReader::new(r);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Ok(_) => {
                while matches!(buf.last(), Some(b'\n' | b'\r')) {
                    buf.pop();
                }
                f(String::from_utf8_lossy(&buf).into_owned());
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
}

/// Environment for the bundled interpreter. `is_set` says whether the user's
/// own environment already sets a variable (theirs wins for tuning knobs).
///
/// OpenBLAS threads: the bundled numpy/scipy use OpenBLAS, whose worker
/// threads do not pay off on the engine's small matrices and spin-wait badly
/// when the CPU is busy. Measured on Apple silicon (all five difficulties of a
/// song): 4.6 s with the default threads, 4.1 s single-threaded, and far worse
/// under load. onnxruntime keeps its own thread pool, so inference is unaffected.
fn bundled_python_env(
    pycache: PathBuf,
    is_set: impl Fn(&str) -> bool,
) -> Vec<(&'static str, std::ffi::OsString)> {
    let mut env = vec![
        ("PYTHONNOUSERSITE", "1".into()),
        ("PYTHONPYCACHEPREFIX", pycache.into_os_string()),
    ];
    if !is_set("OPENBLAS_NUM_THREADS") {
        env.push(("OPENBLAS_NUM_THREADS", "1".into()));
    }
    env
}

/// Build a `Command` for `python -m beatforge <args>` with PYTHONPATH + creds.
pub fn engine_command(app: &AppHandle, args: &[String]) -> Result<Command, String> {
    let root = repo_root(app)?;
    let s = load_settings(app);
    let mut cmd = python_command(app);
    cmd.current_dir(&root);
    cmd.arg("-m").arg("beatforge");
    cmd.args(args);
    cmd.env("PYTHONPATH", root.join("tools"));
    cmd.env("PYTHONUNBUFFERED", "1");
    apply_credentials(&mut cmd, &s);
    Ok(cmd)
}

fn apply_credentials(cmd: &mut Command, s: &Settings) {
    // The app uses only the neural model it ships (`<root>/models/stepforge_flow`).
    // A model folder inherited from the user's shell must not redirect it.
    cmd.env_remove("BEATFORGE_NEURAL_MODEL_DIR");
    if !s.llm_backend.trim().is_empty() {
        cmd.env("BEATFORGE_LLM_BACKEND", &s.llm_backend);
    }
    // Model routing. `hybrid` uses the two tiers below; `single` uses
    // gemini_model for every call. Both Gemini paths (Vertex and AI Studio) honour
    // these — the router wraps whichever transport the backend selected.
    if !s.llm_mode.trim().is_empty() {
        cmd.env("BEATFORGE_LLM_MODE", &s.llm_mode);
    }
    if !s.hybrid_heavy_model.trim().is_empty() {
        cmd.env("BEATFORGE_HYBRID_HEAVY_MODEL", &s.hybrid_heavy_model);
    }
    if !s.hybrid_lite_model.trim().is_empty() {
        cmd.env("BEATFORGE_HYBRID_LITE_MODEL", &s.hybrid_lite_model);
    }
    // Only exported when the user actually set it, so an untouched install keeps
    // whatever the engine's default is rather than being pinned by the GUI.
    if let Some(on) = s.hybrid_cascade {
        cmd.env("BEATFORGE_HYBRID_CASCADE", if on { "1" } else { "0" });
    }
    // Model choice applies to both Gemini paths (Vertex and AI Studio).
    if !s.gemini_model.trim().is_empty() {
        cmd.env("BEATFORGE_GEMINI_MODEL", &s.gemini_model);
    }
    if !s.thinking_level.trim().is_empty() {
        cmd.env("BEATFORGE_THINKING_LEVEL", &s.thinking_level);
    }
    if !s.gemini_api_key.trim().is_empty() {
        cmd.env("GEMINI_API_KEY", &s.gemini_api_key);
    }
    if !s.vertex_project.trim().is_empty() {
        cmd.env("VERTEX_PROJECT", &s.vertex_project);
    }
    if !s.vertex_key.trim().is_empty() {
        cmd.env("VERTEX_KEY", &s.vertex_key);
    }
    if !s.openai_base_url.trim().is_empty() {
        cmd.env("BEATFORGE_OPENAI_BASE_URL", &s.openai_base_url);
    }
    if !s.openai_model.trim().is_empty() {
        cmd.env("BEATFORGE_OPENAI_MODEL", &s.openai_model);
    }
    if !s.openai_api_key.trim().is_empty() {
        cmd.env("BEATFORGE_OPENAI_API_KEY", &s.openai_api_key);
    }
    // Inkling / Modal. The token pair is NOT a Bearer credential — the engine
    // sends it as the Modal-Key / Modal-Secret header pair (see inkling.py).
    if !s.inkling_endpoint_url.trim().is_empty() {
        cmd.env(
            "BEATFORGE_INKLING_ENDPOINT_URL",
            s.inkling_endpoint_url.trim(),
        );
    }
    if !s.inkling_model.trim().is_empty() {
        cmd.env("BEATFORGE_INKLING_MODEL", s.inkling_model.trim());
    }
    if !s.modal_token_id.trim().is_empty() {
        cmd.env("MODAL_TOKEN_ID", s.modal_token_id.trim());
    }
    if !s.modal_token_secret.trim().is_empty() {
        cmd.env("MODAL_TOKEN_SECRET", s.modal_token_secret.trim());
    }
    for (var, val) in [
        (
            "BEATFORGE_INKLING_EFFORT_DESIGNER",
            &s.inkling_effort_designer,
        ),
        ("BEATFORGE_INKLING_EFFORT_CRITIC", &s.inkling_effort_critic),
        ("BEATFORGE_INKLING_EFFORT_RETRY", &s.inkling_effort_retry),
        // Gemini 4 Argon. No credential of its own — it rides GEMINI_API_KEY /
        // VERTEX_PROJECT above. Blank leaves the engine default in force.
        ("BEATFORGE_ARGON_TRANSPORT", &s.argon_transport),
        ("BEATFORGE_ARGON_SERVICE_TIER", &s.argon_service_tier),
        ("BEATFORGE_ARGON_CRITIC_MODEL", &s.argon_critic_model),
        (
            "BEATFORGE_ARGON_THINKING_DESIGNER",
            &s.argon_thinking_designer,
        ),
        ("BEATFORGE_ARGON_THINKING_RETRY", &s.argon_thinking_retry),
        ("BEATFORGE_ARGON_THINKING_CRITIC", &s.argon_thinking_critic),
    ] {
        if !val.trim().is_empty() {
            cmd.env(var, val.trim());
        }
    }
    // Only exported when the user actually set it, so an untouched install keeps
    // the engine's default (on) rather than being pinned by the GUI.
    if let Some(on) = s.openai_send_audio {
        cmd.env("BEATFORGE_OPENAI_SEND_AUDIO", if on { "1" } else { "0" });
    }
    // CHUNKED MODE (REQ-CHUNK-08). Exported for every backend, not just the
    // local ones: the mode is a property of how a song is designed, not of who
    // serves the model, and REQ-CHUNK-10 compares the two modes on the SAME
    // designer model — which is only possible if the switch reaches Gemini too.
    if !s.chunk_mode.trim().is_empty() {
        cmd.env("BEATFORGE_CHUNK_MODE", s.chunk_mode.trim());
    }
    // The two tuning knobs are only exported when the user actually set them,
    // so an untouched install keeps the engine's defaults rather than being
    // pinned by whatever the GUI last rendered.
    if let Some(secs) = s.max_chunk_seconds {
        if secs > 0.0 {
            cmd.env("BEATFORGE_MAX_CHUNK_SECONDS", format!("{secs}"));
        }
    }
    if let Some(bars) = s.chunk_overlap_bars {
        cmd.env("BEATFORGE_CHUNK_OVERLAP_BARS", bars.to_string());
    }
    // Custom headers travel as a JSON object; config.py parses it back to a dict.
    // Rows with a blank name are half-finished UI edits and are dropped, but a
    // blank *value* is meaningful — it tells the client to omit that header.
    let headers: serde_json::Map<String, Value> = s
        .openai_headers
        .iter()
        .filter(|h| !h.name.trim().is_empty())
        .map(|h| (h.name.trim().to_string(), Value::String(h.value.clone())))
        .collect();
    if !headers.is_empty() {
        cmd.env(
            "BEATFORGE_OPENAI_HEADERS",
            Value::Object(headers).to_string(),
        );
    }
    if !s.ffmpeg_path.trim().is_empty() {
        cmd.env("FFMPEG", &s.ffmpeg_path);
        // Also prepend its directory to PATH so the engine finds ffmpeg.
        if let Some(dir) = Path::new(&s.ffmpeg_path).parent() {
            let path = std::env::var("PATH").unwrap_or_default();
            cmd.env("PATH", format!("{}:{}", dir.display(), path));
        }
    }
}

/// Spawn an engine job that streams its output as `engine:log` events and a
/// final `engine:exit`. Returns once the child is spawned; the reader runs on a
/// background thread. Registers the child under `job_id` for cancellation.
pub fn spawn_streaming(app: &AppHandle, job_id: String, args: Vec<String>) -> Result<(), String> {
    let cmd = engine_command(app, &args)?;
    spawn_command(app, job_id, cmd)
}

/// Stream any child process as a job (used by the engine and by assetforge).
pub fn spawn_command(app: &AppHandle, job_id: String, mut cmd: Command) -> Result<(), String> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to launch engine: {e}"))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    // Register for cancel.
    {
        let reg = app.state::<JobRegistry>();
        reg.0.lock().unwrap().insert(job_id.clone(), child);
    }

    // stdout reader (joined by the exit thread before it reports completion)
    let stdout_handle = stdout.map(|out| {
        let app = app.clone();
        let jid = job_id.clone();
        std::thread::spawn(move || {
            for_each_line_lossy(out, |line| {
                let _ = app.emit(
                    "engine:log",
                    LogEvent {
                        job_id: jid.clone(),
                        stream: "stdout".into(),
                        line,
                    },
                );
            });
        })
    });
    // stderr reader + wait-for-exit
    let app2 = app.clone();
    let jid2 = job_id.clone();
    std::thread::spawn(move || {
        if let Some(err) = stderr {
            for_each_line_lossy(err, |line| {
                let _ = app2.emit(
                    "engine:log",
                    LogEvent {
                        job_id: jid2.clone(),
                        stream: "stderr".into(),
                        line,
                    },
                );
            });
        }
        // Drain stdout fully before reporting exit, so the UI never shows a
        // finished job whose last lines are still in flight (the failure banner
        // is built from the log tail).
        if let Some(h) = stdout_handle {
            let _ = h.join();
        }
        // Reclaim the child from the registry and wait on it.
        let reg = app2.state::<JobRegistry>();
        let child = reg.0.lock().unwrap().remove(&jid2);
        let code = if let Some(mut c) = child {
            c.wait().ok().and_then(|s| s.code()).unwrap_or(-1)
        } else {
            // Already removed (cancelled).
            -1
        };
        let _ = app2.emit(
            "engine:exit",
            ExitEvent {
                job_id: jid2.clone(),
                code,
                success: code == 0,
            },
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Windows Python writes "…" as cp1252 0x85 when its output is piped. The
    /// reader must carry on past it (and past CRLF endings), not stop and close
    /// the pipe under the engine.
    #[test]
    fn engine_output_that_is_not_utf8_does_not_stop_the_reader() {
        let bytes: &[u8] = b"grid\x85\r\nnext line\r\nlast, no newline";
        let mut got = Vec::new();
        for_each_line_lossy(bytes, |l| got.push(l));
        assert_eq!(got, vec!["grid\u{FFFD}", "next line", "last, no newline"]);

        let mut utf8 = Vec::new();
        for_each_line_lossy("beat-warping\u{2026}\n消えてゆく\n".as_bytes(), |l| utf8.push(l));
        assert_eq!(utf8, vec!["beat-warping\u{2026}", "消えてゆく"]);
    }

    #[test]
    fn every_engine_process_is_told_to_speak_utf8() {
        assert!(UTF8_ENV.contains(&("PYTHONUTF8", "1")));
        assert!(UTF8_ENV.contains(&("PYTHONIOENCODING", "utf-8")));
    }

    #[test]
    fn the_bundled_python_runs_openblas_single_threaded_unless_the_user_says_otherwise() {
        let env = bundled_python_env(PathBuf::from("/ws/build/pycache"), |_| false);
        let get = |k: &str| env.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone());
        assert_eq!(get("OPENBLAS_NUM_THREADS"), Some("1".into()));
        assert_eq!(get("PYTHONNOUSERSITE"), Some("1".into()));
        assert_eq!(get("PYTHONPYCACHEPREFIX"), Some("/ws/build/pycache".into()));
        let theirs = bundled_python_env(PathBuf::from("/ws"), |k| k == "OPENBLAS_NUM_THREADS");
        assert!(theirs.iter().all(|(n, _)| *n != "OPENBLAS_NUM_THREADS"));
    }

    /// The frontend keys log/exit events by `jobId`. A snake_case payload lands
    /// under `undefined`, which silently breaks streaming AND job completion —
    /// jobs then appear to run forever. Pin the wire names.
    #[test]
    fn events_serialize_as_camel_case() {
        let log = serde_json::to_value(LogEvent {
            job_id: "BF-1".into(),
            stream: "stdout".into(),
            line: "hi".into(),
        })
        .unwrap();
        assert_eq!(log["jobId"], "BF-1", "LogEvent must expose jobId");
        assert!(log.get("job_id").is_none());

        let exit = serde_json::to_value(ExitEvent {
            job_id: "BF-1".into(),
            code: 0,
            success: true,
        })
        .unwrap();
        assert_eq!(exit["jobId"], "BF-1", "ExitEvent must expose jobId");
        assert!(exit.get("job_id").is_none());
    }

    /// A settings.json written before hybrid routing existed must not come back
    /// with the cascade disabled. `hybrid_cascade` is `Option<bool>` precisely so
    /// "absent" stays distinguishable from "the user turned it off" — with a
    /// plain bool, `Default` would read every old file as `false` and the feature
    /// would ship off for every upgrading user.
    #[test]
    fn an_old_settings_file_leaves_the_cascade_to_the_engine() {
        let old = r#"{"llmBackend":"gemini","geminiModel":"gemini-3.8-flash"}"#;
        let s: Settings = serde_json::from_str(old).unwrap();
        assert_eq!(s.hybrid_cascade, None, "absent must not read as `false`");
        assert_eq!(s.llm_mode, "", "absent mode is filled in by load_settings");

        let off: Settings = serde_json::from_str(r#"{"hybridCascade":false}"#).unwrap();
        assert_eq!(
            off.hybrid_cascade,
            Some(false),
            "an explicit opt-out must survive"
        );
    }

    /// Hybrid routing is opt-in, and the app default must agree with the engine's
    /// (`config.LLM_MODE`). If these drift, the GUI silently runs a different mode
    /// from the CLI on the same machine.
    #[test]
    fn the_default_routing_mode_is_single() {
        let s: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(
            s.llm_mode, "",
            "absent stays absent until load_settings fills it"
        );
        // Mirrors the fill-in branch of load_settings, which needs an AppHandle.
        let filled = if s.llm_mode.trim().is_empty() {
            "single"
        } else {
            &s.llm_mode
        };
        assert_eq!(filled, "single");
    }

    /// The frontend sends camelCase; the engine env vars are the contract with
    /// `tools/beatforge/config.py`. Pin both ends of the hybrid settings.
    #[test]
    fn hybrid_settings_round_trip_as_camel_case() {
        let s = Settings {
            llm_mode: "hybrid".into(),
            hybrid_heavy_model: "gemini-3.8-flash".into(),
            hybrid_lite_model: "gemini-3.5-flash-lite".into(),
            hybrid_cascade: Some(true),
            ..Default::default()
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["llmMode"], "hybrid");
        assert_eq!(v["hybridHeavyModel"], "gemini-3.8-flash");
        assert_eq!(v["hybridLiteModel"], "gemini-3.5-flash-lite");
        assert_eq!(v["hybridCascade"], true);
    }

    /// The Argon env vars are the contract with config.py's ARGON_* block. Set
    /// fields are exported trimmed; blank ones are not exported at all, so an
    /// untouched install follows the engine defaults instead of being pinned.
    #[test]
    fn argon_settings_reach_the_engine_and_blanks_do_not() {
        let s = Settings {
            llm_backend: "argon".into(),
            argon_transport: "aistudio".into(),
            argon_service_tier: " flex ".into(),
            argon_thinking_designer: "high".into(),
            argon_thinking_critic: "low".into(),
            ..Default::default()
        };
        let mut cmd = Command::new("true");
        apply_credentials(&mut cmd, &s);
        let envs: HashMap<String, String> = cmd
            .get_envs()
            .filter_map(|(k, v)| {
                v.map(|v| {
                    (
                        k.to_string_lossy().into_owned(),
                        v.to_string_lossy().into_owned(),
                    )
                })
            })
            .collect();
        assert_eq!(envs["BEATFORGE_LLM_BACKEND"], "argon");
        assert_eq!(envs["BEATFORGE_ARGON_TRANSPORT"], "aistudio");
        assert_eq!(envs["BEATFORGE_ARGON_SERVICE_TIER"], "flex");
        assert_eq!(envs["BEATFORGE_ARGON_THINKING_DESIGNER"], "high");
        assert_eq!(envs["BEATFORGE_ARGON_THINKING_CRITIC"], "low");
        assert!(!envs.contains_key("BEATFORGE_ARGON_THINKING_RETRY"));
        assert!(!envs.contains_key("BEATFORGE_ARGON_CRITIC_MODEL"));
    }

    /// A settings.json written before Argon existed must still load, with the
    /// Argon fields blank (= engine defaults), not fail to parse.
    #[test]
    fn settings_from_before_argon_still_load() {
        let old: Settings = serde_json::from_str(r#"{"llmBackend":"inkling"}"#).unwrap();
        assert_eq!(old.argon_service_tier, "");
        assert_eq!(old.argon_transport, "");
        let v = serde_json::to_value(&old).unwrap();
        assert!(v.get("argonThinkingDesigner").is_some(), "camelCase on the wire");
    }

    /// The BEATFORGE_OPENAI_HEADERS value `apply_credentials` would export for a
    /// given set of editor rows (None = the var is not set at all).
    fn header_env(pairs: &[(&str, &str)]) -> Option<String> {
        let s = Settings {
            openai_headers: pairs
                .iter()
                .map(|(n, v)| HttpHeader {
                    name: (*n).into(),
                    value: (*v).into(),
                })
                .collect(),
            ..Default::default()
        };
        let mut cmd = Command::new("true");
        apply_credentials(&mut cmd, &s);
        cmd.get_envs()
            .find(|(k, _)| *k == "BEATFORGE_OPENAI_HEADERS")
            .and_then(|(_, v)| v.map(|v| v.to_string_lossy().into_owned()))
    }

    /// The env var is the contract with `config._parse_headers`, which expects a
    /// JSON object of name -> value.
    #[test]
    fn custom_headers_are_exported_as_json() {
        let raw = header_env(&[("X-Api-Key", "abc"), (" X-Tenant ", "acme")]).unwrap();
        let obj: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(obj["X-Api-Key"], "abc");
        assert_eq!(obj["X-Tenant"], "acme"); // name whitespace trimmed
    }

    /// A blank value means "drop this default header" and must reach the engine;
    /// a blank name is just an unfinished row in the editor and must not.
    #[test]
    fn blank_value_survives_but_blank_name_does_not() {
        let raw = header_env(&[("Authorization", ""), ("", "orphan")]).unwrap();
        let obj: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(obj["Authorization"], "");
        assert_eq!(obj.as_object().unwrap().len(), 1);
    }

    /// An untouched install must not pin anything the engine would then parse.
    #[test]
    fn no_headers_means_no_env_var() {
        assert!(header_env(&[]).is_none());
        assert!(header_env(&[("  ", "")]).is_none());
    }

    /// Values are user-typed, so they contain whatever a paste brings with it.
    /// JSON encoding has to carry them through byte-exact — a value mangled here
    /// becomes a wrong credential on the wire, which reads as an auth failure at
    /// the far end with nothing pointing back to this function.
    #[test]
    fn awkward_values_survive_the_json_round_trip() {
        let raw = header_env(&[
            ("X-Quote", "a\"b\\c"),
            ("X-Tab", "a\tb"),
            ("X-Unicode", "café ☕"),
            ("X-Spaces", "  padded  "),
            ("X-Empty", ""),
        ])
        .unwrap();
        let obj: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(obj["X-Quote"], "a\"b\\c");
        assert_eq!(obj["X-Tab"], "a\tb");
        assert_eq!(obj["X-Unicode"], "café ☕");
        assert_eq!(
            obj["X-Spaces"], "  padded  ",
            "value whitespace is NOT trimmed"
        );
        assert_eq!(
            obj["X-Empty"], "",
            "blank value is meaningful: drop the header"
        );
    }

    /// The engine parses this with `json.loads`, so it must be a flat
    /// string -> string object. Any other shape (nested, numeric, array) would
    /// come back as a non-string header value and fail deep in the HTTP client.
    #[test]
    fn the_exported_shape_is_a_flat_string_map() {
        let raw = header_env(&[("X-A", "1"), ("X-B", "two")]).unwrap();
        let obj: Value = serde_json::from_str(&raw).unwrap();
        let map = obj.as_object().expect("must be a JSON object");
        assert!(
            map.values().all(|v| v.is_string()),
            "every value must be a string"
        );
        assert_eq!(map["X-A"], "1", "a numeric-looking value stays a string");
    }

    /// The UI cannot stop someone typing the same header twice. Whatever the
    /// resolution, exactly ONE entry may reach the engine — two would be
    /// ambiguous, and the GUI's duplicate warning promises last-row-wins.
    #[test]
    fn a_duplicated_name_collapses_to_the_last_row() {
        let raw = header_env(&[("X-Api-Key", "first"), ("X-Api-Key", "second")]).unwrap();
        let obj: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(obj.as_object().unwrap().len(), 1);
        assert_eq!(obj["X-Api-Key"], "second");
    }

    /// Names differing only in case stay separate here; the engine's merge is
    /// case-insensitive and resolves them (last wins). Pinned so the two halves
    /// of that contract are not changed independently.
    #[test]
    fn case_differing_names_are_left_for_the_engine_to_resolve() {
        let raw = header_env(&[("X-Api-Key", "upper"), ("x-api-key", "lower")]).unwrap();
        let obj: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(obj.as_object().unwrap().len(), 2);
    }

    /// Credentials live in these values, so the export must be exactly the
    /// header table — never the surrounding settings.
    #[test]
    fn only_the_header_rows_are_exported() {
        let s = Settings {
            openai_headers: vec![HttpHeader {
                name: "X-Api-Key".into(),
                value: "secret".into(),
            }],
            openai_api_key: "sk-should-not-appear-here".into(),
            ..Default::default()
        };
        let mut cmd = Command::new("true");
        apply_credentials(&mut cmd, &s);
        let raw = cmd
            .get_envs()
            .find(|(k, _)| *k == "BEATFORGE_OPENAI_HEADERS")
            .and_then(|(_, v)| v)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert!(raw.contains("X-Api-Key"));
        assert!(!raw.contains("sk-should-not-appear-here"));
    }

    /// Every settings.json written before custom headers existed lacks the key;
    /// `serde(default)` must give an empty list, not fail the whole load and
    /// reset the user's other settings.
    #[test]
    fn settings_without_the_headers_field_still_load() {
        let s: Settings = serde_json::from_str(r#"{"openaiBaseUrl":"http://x/v1"}"#).unwrap();
        assert!(s.openai_headers.is_empty());
        assert_eq!(s.openai_base_url, "http://x/v1");
    }

    /// Audio-in defaults to ON, so an absent field must stay absent rather than
    /// reading as `false` — otherwise every upgrading user silently stops
    /// sending the track to an endpoint that was happily receiving it.
    #[test]
    fn an_old_settings_file_leaves_audio_to_the_engine() {
        let old: Settings = serde_json::from_str(r#"{"openaiBaseUrl":"http://x/v1"}"#).unwrap();
        assert_eq!(old.openai_send_audio, None);
        let mut cmd = Command::new("true");
        apply_credentials(&mut cmd, &old);
        assert!(
            !cmd.get_envs()
                .any(|(k, _)| k == "BEATFORGE_OPENAI_SEND_AUDIO"),
            "unset must not pin the engine default"
        );

        let off: Settings = serde_json::from_str(r#"{"openaiSendAudio":false}"#).unwrap();
        assert_eq!(
            off.openai_send_audio,
            Some(false),
            "an explicit opt-out must survive"
        );
    }

    /// CHUNKED MODE IS THE DEFAULT (REQ-CHUNK-08): "a fresh install generates in
    /// chunked mode with no configuration". A settings.json written before the
    /// field existed is a fresh install as far as this setting is concerned, so
    /// it has to come out as `auto` too — not as the empty string, which would
    /// export nothing and leave the mode to whatever env the app inherited.
    #[test]
    fn the_default_chunk_mode_is_auto() {
        let old: Settings = serde_json::from_str(r#"{"llmBackend":"gemini"}"#).unwrap();
        assert_eq!(
            old.chunk_mode, "",
            "absent stays absent until load_settings fills it"
        );
        // Mirrors the fill-in branch of load_settings, which needs an AppHandle.
        let filled = if old.chunk_mode.trim().is_empty() {
            "auto"
        } else {
            &old.chunk_mode
        };
        assert_eq!(filled, "auto", "must agree with config.CHUNK_MODE");
    }

    /// The env var is the contract with `tools/beatforge/config.py::CHUNK_MODE`.
    /// It is exported for EVERY backend — the mode is a property of how a song is
    /// designed, not of who serves the model, and the two-mode benchmark
    /// (REQ-CHUNK-10) needs the switch to reach Gemini as well as a local server.
    #[test]
    fn chunk_mode_is_exported_for_every_backend() {
        for backend in ["gemini", "google-genai", "openai", "inkling", "argon"] {
            let s = Settings {
                llm_backend: backend.into(),
                chunk_mode: "never".into(),
                ..Default::default()
            };
            let mut cmd = Command::new("true");
            apply_credentials(&mut cmd, &s);
            let got = cmd
                .get_envs()
                .find(|(k, _)| *k == "BEATFORGE_CHUNK_MODE")
                .and_then(|(_, v)| v)
                .map(|v| v.to_string_lossy().into_owned());
            assert_eq!(got.as_deref(), Some("never"), "backend {backend}");
        }
    }

    /// The two tuning knobs are `Option` so an untouched install keeps the
    /// engine's numbers. Pinning them from the GUI regardless would mean a
    /// change to the engine default could never reach an existing user.
    #[test]
    fn unset_chunk_tuning_does_not_pin_the_engine() {
        let old: Settings = serde_json::from_str(r#"{"chunkMode":"auto"}"#).unwrap();
        assert_eq!(old.max_chunk_seconds, None);
        assert_eq!(old.chunk_overlap_bars, None);
        let mut cmd = Command::new("true");
        apply_credentials(&mut cmd, &old);
        assert!(!cmd
            .get_envs()
            .any(|(k, _)| k == "BEATFORGE_MAX_CHUNK_SECONDS"));
        assert!(!cmd
            .get_envs()
            .any(|(k, _)| k == "BEATFORGE_CHUNK_OVERLAP_BARS"));

        let tuned = Settings {
            max_chunk_seconds: Some(20.0),
            chunk_overlap_bars: Some(2),
            ..Default::default()
        };
        let mut cmd = Command::new("true");
        apply_credentials(&mut cmd, &tuned);
        let envs: std::collections::HashMap<_, _> = cmd
            .get_envs()
            .filter_map(|(k, v)| {
                Some((
                    k.to_string_lossy().into_owned(),
                    v?.to_string_lossy().into_owned(),
                ))
            })
            .collect();
        assert_eq!(envs["BEATFORGE_MAX_CHUNK_SECONDS"], "20");
        assert_eq!(envs["BEATFORGE_CHUNK_OVERLAP_BARS"], "2");
    }

    /// A zero-second ceiling would chunk every track into single bars, so it is
    /// treated as "not set" rather than passed through. This is the shape a
    /// half-cleared number input sends.
    #[test]
    fn a_zero_chunk_ceiling_is_ignored_rather_than_exported() {
        let s = Settings {
            max_chunk_seconds: Some(0.0),
            ..Default::default()
        };
        let mut cmd = Command::new("true");
        apply_credentials(&mut cmd, &s);
        assert!(!cmd
            .get_envs()
            .any(|(k, _)| k == "BEATFORGE_MAX_CHUNK_SECONDS"));
    }

    /// The engine reads BEATFORGE_OPENAI_SEND_AUDIO as "0"/"1" (config.py treats
    /// "0" as off); anything else there would read as ON and defeat the toggle.
    #[test]
    fn audio_opt_out_exports_zero() {
        let s = Settings {
            openai_send_audio: Some(false),
            ..Default::default()
        };
        let mut cmd = Command::new("true");
        apply_credentials(&mut cmd, &s);
        let v = cmd
            .get_envs()
            .find(|(k, _)| *k == "BEATFORGE_OPENAI_SEND_AUDIO")
            .and_then(|(_, v)| v)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert_eq!(v, "0");
    }

    /// The frontend edits these rows as `{name, value}` — pin the wire shape, as
    /// a rename would silently blank every saved header on the next load.
    #[test]
    fn headers_round_trip_through_the_frontend_shape() {
        let s: Settings =
            serde_json::from_str(r#"{"openaiHeaders":[{"name":"X-Api-Key","value":"abc"}]}"#)
                .unwrap();
        assert_eq!(s.openai_headers.len(), 1);
        assert_eq!(s.openai_headers[0].name, "X-Api-Key");
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["openaiHeaders"][0]["name"], "X-Api-Key");
        assert_eq!(v["openaiHeaders"][0]["value"], "abc");
    }

    /// Every env var `apply_credentials` would export for these settings.
    fn exported_env(s: &Settings) -> HashMap<String, String> {
        let mut cmd = Command::new("true");
        apply_credentials(&mut cmd, s);
        cmd.get_envs()
            .filter_map(|(k, v)| {
                Some((
                    k.to_string_lossy().into_owned(),
                    v?.to_string_lossy().into_owned(),
                ))
            })
            .collect()
    }

    /// The app charts only with the model it ships: it never points the engine
    /// at another model folder, and an old settings file that still names one
    /// is read without error and ignored.
    #[test]
    fn the_app_never_points_the_engine_at_another_model() {
        let old: Settings = serde_json::from_str(
            r#"{"chartEngine":"neural","neuralModelDir":"/opt/models/flow v2"}"#,
        )
        .unwrap();
        assert_eq!(old.chart_engine, "neural");
        assert!(!exported_env(&old).contains_key("BEATFORGE_NEURAL_MODEL_DIR"));
        // ...and one set in the user's shell is removed from the engine's env.
        let mut cmd = Command::new("true");
        apply_credentials(&mut cmd, &old);
        assert!(cmd
            .get_envs()
            .any(|(k, v)| k == "BEATFORGE_NEURAL_MODEL_DIR" && v.is_none()));
        let v = serde_json::to_value(&old).unwrap();
        assert!(v.get("neuralModelDir").is_none(), "not written back on save");
    }

    /// The frontend sends camelCase and a field that exists only on one side is
    /// silently dropped on save. Pin the wire names of the neural settings, both
    /// directions.
    #[test]
    fn neural_settings_round_trip_as_camel_case() {
        let from_ui: Settings = serde_json::from_str(
            r#"{"chartEngine":"neural","neuralStylePreset":"itg_stamina",
                "neuralStyle":{"jump_share":0.12,"nps":6.5}}"#,
        )
        .unwrap();
        assert_eq!(from_ui.chart_engine, "neural");
        assert_eq!(from_ui.neural_style_preset, "itg_stamina");
        assert_eq!(from_ui.neural_style.get("jump_share"), Some(&0.12));
        assert_eq!(from_ui.neural_style.get("nps"), Some(&6.5));

        let v = serde_json::to_value(&from_ui).unwrap();
        assert_eq!(v["chartEngine"], "neural");
        assert_eq!(v["neuralStylePreset"], "itg_stamina");
        assert_eq!(v["neuralStyle"]["jump_share"], 0.12);
        assert!(v.get("chart_engine").is_none(), "camelCase on the wire");

        let back: Settings = serde_json::from_value(v).unwrap();
        assert_eq!(back.neural_style, from_ui.neural_style);
    }

    /// A settings.json written before the neural engine existed must load with
    /// the engine UNSET (so the UI can default it from `neural-status`) and no
    /// sliders — an absent slider means "let the model decide", never 0.0.
    #[test]
    fn settings_from_before_the_neural_engine_still_load() {
        let old: Settings = serde_json::from_str(r#"{"llmBackend":"gemini"}"#).unwrap();
        assert_eq!(old.chart_engine, "");
        assert_eq!(old.neural_style_preset, "");
        assert!(old.neural_style.is_empty());
        assert_eq!(old.llm_backend, "gemini", "the engine choice is not the backend");
    }

    /// One malformed slider must not fail the whole load: `load_settings` would
    /// then fall back to defaults and the user's credentials would vanish.
    #[test]
    fn a_malformed_style_map_does_not_take_the_settings_down() {
        let s: Settings = serde_json::from_str(
            r#"{"vertexProject":"keep-me",
                "neuralStyle":{"jump_share":0.1,"hold_share":null,"nps":"fast"}}"#,
        )
        .unwrap();
        assert_eq!(s.vertex_project, "keep-me");
        assert_eq!(s.neural_style.len(), 1, "non-numeric entries are dropped");
        assert_eq!(s.neural_style.get("jump_share"), Some(&0.1));

        for odd in [r#"{"neuralStyle":null}"#, r#"{"neuralStyle":[1,2]}"#] {
            let s: Settings = serde_json::from_str(odd).unwrap();
            assert!(s.neural_style.is_empty(), "{odd}");
        }
    }

    #[test]
    fn saved_presets_and_last_pack_round_trip_and_tolerate_junk() {
        use serde_json::json;
        let from_ui: Settings = serde_json::from_value(json!({
            "lastPack": "SPT-V4",
            "neuralUserPresets": {
                "My tech": {"preset": "tech_footswitch", "style": {"jump_share": 0.1}},
                "broken": 7,
            },
        }))
        .unwrap();
        assert_eq!(from_ui.last_pack, "SPT-V4");
        assert_eq!(from_ui.neural_user_presets.len(), 1, "a non-object preset is dropped");
        let v = serde_json::to_value(&from_ui).unwrap();
        assert_eq!(v["neuralUserPresets"]["My tech"]["style"]["jump_share"], json!(0.1));
        assert_eq!(v["lastPack"], json!("SPT-V4"));
        // an older settings.json has neither field
        let old: Settings = serde_json::from_value(json!({"pythonPath": "python3"})).unwrap();
        assert!(old.neural_user_presets.is_empty() && old.last_pack.is_empty());
        // and a wrong-typed presets field must not blank the rest of the file
        let odd: Settings =
            serde_json::from_value(json!({"neuralUserPresets": [], "pythonPath": "py"})).unwrap();
        assert!(odd.neural_user_presets.is_empty());
        assert_eq!(odd.python_path, "py");
    }
}

/// The assetforge CLI used for AI artwork (Imagen stills / Veo video). It is a
/// sibling tool, not part of the engine, so a missing install is a clean error.
pub fn assetforge_script(app: &AppHandle) -> PathBuf {
    let s = load_settings(app);
    if !s.assetforge_path.trim().is_empty() {
        return PathBuf::from(s.assetforge_path);
    }
    if let Ok(env) = std::env::var("BEATFORGE_ASSETFORGE") {
        if !env.trim().is_empty() {
            return PathBuf::from(env);
        }
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    PathBuf::from(home).join(".assetforge/assetforge.py")
}

/// Spawn an assetforge generation, streaming like any other job.
/// `media` is "image" (Imagen) or "video" (Veo).
pub fn spawn_assetforge(
    app: &AppHandle,
    job_id: String,
    media: String,
    prompt: String,
    out_path: String,
    aspect: String,
) -> Result<(), String> {
    let script = assetforge_script(app);
    if !script.is_file() {
        return Err(format!(
            "assetforge CLI not found at {}. Set its path in Settings.",
            script.display()
        ));
    }
    let s = load_settings(app);
    let py = python_bin(app);
    let sub = if media == "video" { "video" } else { "image" };

    let mut cmd = Command::new(py);
    cmd.arg(&script).arg(sub).arg(&prompt);
    cmd.arg("-o").arg(&out_path);
    if sub == "image" {
        cmd.arg("--aspect").arg(&aspect);
    }
    // assetforge talks to Vertex/Gemini; pass through whatever creds are set.
    if !s.vertex_project.trim().is_empty() {
        cmd.env("VERTEX_PROJECT", &s.vertex_project);
    }
    if !s.vertex_key.trim().is_empty() {
        cmd.env("VERTEX_KEY", &s.vertex_key);
        cmd.env("GOOGLE_APPLICATION_CREDENTIALS", &s.vertex_key);
    }
    if !s.gemini_api_key.trim().is_empty() {
        cmd.env("GEMINI_API_KEY", &s.gemini_api_key);
    }
    cmd.env("PYTHONUNBUFFERED", "1");
    spawn_command(app, job_id, cmd)
}

pub fn cancel(app: &AppHandle, job_id: &str) -> bool {
    let reg = app.state::<JobRegistry>();
    let mut map = reg.0.lock().unwrap();
    if let Some(mut child) = map.remove(job_id) {
        let _ = child.kill();
        true
    } else {
        false
    }
}
