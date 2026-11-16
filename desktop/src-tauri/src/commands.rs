//! Tauri commands invoked from the Vue frontend.
//!
//! Every path and JSON key here mirrors the real `tools/beatforge` contract:
//!   * a track is addressed by an id whose audio MUST exist as `<base>.ogg`
//!     under `public/assets/tracks/` or `assets/tracks/` (see `analyze._audio_for`);
//!   * DSP truth lands in `build/analysis/<base>.analysis.json`
//!     (keys: bpm, offset, duration_s, onsets[], sustain_available, sections[]);
//!   * the native `chart` path writes `build/analysis/<base>.<diff>.qa.json`
//!     (critic + design_notes + previews live INSIDE it);
//!   * STEPFORGE / SABERFORGE write a song folder
//!     `build/{stepmania,saberforge}/<base>/` containing `report.json`,
//!     `<diff>.design.json`, `<diff>.critic.json` and preview artifacts.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};
use tauri::AppHandle;

use crate::engine::{self, Settings};

/// Difficulty sets, straight from the engine (config.DIFFICULTIES and each
/// adapter's grammar.DIFFICULTIES). `chart --difficulty` is argparse-restricted
/// to these, so getting them wrong makes the CLI reject the run outright.
pub const NATIVE_DIFFS: [&str; 3] = ["casual", "standard", "overdrive"];
pub const STEPFORGE_DIFFS: [&str; 5] = ["beginner", "easy", "medium", "hard", "challenge"];
/// Selected by default on the Generate screen: the whole ladder. A pack wants
/// all five, and with the local engine a chart costs about a second.
pub const STEPFORGE_DEFAULT: [&str; 5] = STEPFORGE_DIFFS;
pub const SABERFORGE_DIFFS: [&str; 5] = ["easy", "normal", "hard", "expert", "expertplus"];

/// Targets the app currently exposes. SABERFORGE (Beat Saber) exists in the
/// engine but is **not supported by the app yet**, so it is hidden end-to-end:
/// it is never offered for generation, never reported as emitted output, and a
/// direct request for it is rejected. Re-enable by adding it back here.
pub const ENABLED_TARGETS: [&str; 2] = ["stepforge", "native"];

pub fn target_enabled(t: &str) -> bool {
    ENABLED_TARGETS.contains(&t)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Track {
    /// Id passed to the CLI (`--track`). Equals `base` unless the file is a
    /// registered catalogue track (e.g. base `overdrive_pulse` -> id `overdrive`).
    pub id: String,
    /// Filename stem — what the engine uses to name every build artifact.
    pub base: String,
    pub title: String,
    pub artist: String,
    pub file: String,
    pub path: String,
    /// False when the source is not .ogg — the engine cannot open it until converted.
    pub is_ogg: bool,
    pub registered: bool,
    pub bpm: Option<f64>,
    pub offset: Option<f64>,
    pub onsets: Option<u64>,
    pub duration_sec: Option<f64>,
    pub size_bytes: u64,
    pub sustain_count: Option<u64>,
    /// "unconverted" | "draft" | "analyzed" | "charted"
    pub status: String,
    /// Targets that already have emitted output: stepforge / saberforge / native.
    pub targets: Vec<String>,
}

const AUDIO_EXTS: [&str; 6] = ["ogg", "wav", "mp3", "flac", "m4a", "aac"];

fn analysis_dir(root: &Path) -> PathBuf {
    root.join("build/analysis")
}
fn target_dir(root: &Path, target: &str) -> PathBuf {
    match target {
        "stepforge" => root.join("build/stepmania"),
        "saberforge" => root.join("build/saberforge"),
        _ => analysis_dir(root),
    }
}

/// Slug matching `batch_stepforge.safe_base`.
fn safe_base(name: &str) -> String {
    let mut out = String::new();
    let mut prev_us = false;
    for ch in name.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_us = false;
        } else if !prev_us {
            out.push('_');
            prev_us = true;
        }
    }
    let s = out.trim_matches('_').to_string();
    if s.is_empty() {
        "song".into()
    } else {
        s
    }
}

/// Parse `TRACK_CATALOGUE` (id -> base) and `TRACK_META` (id -> title, artist)
/// out of config.py so the UI shows real titles for registered tracks.
fn catalogue(root: &Path) -> (HashMap<String, String>, HashMap<String, (String, String)>) {
    let mut cat = HashMap::new();
    let mut meta = HashMap::new();
    let Ok(src) = std::fs::read_to_string(root.join("tools/beatforge/config.py")) else {
        return (cat, meta);
    };

    // Dict-literal blocks.
    let mut in_cat = false;
    let mut in_meta = false;
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with("TRACK_CATALOGUE") && t.contains('{') {
            in_cat = true;
            continue;
        }
        if t.starts_with("TRACK_META") && t.contains('{') {
            in_meta = true;
            continue;
        }
        if (in_cat || in_meta) && t.starts_with('}') {
            in_cat = false;
            in_meta = false;
            continue;
        }
        if in_cat {
            if let Some((k, v)) = parse_pair(t) {
                cat.insert(k, v);
            }
        } else if in_meta {
            if let Some((k, a, b)) = parse_tuple(t) {
                meta.insert(k, (a, b));
            }
        }
    }
    // Appended `TRACK_META['id'] = ('Title', 'Artist')` lines.
    for line in src.lines() {
        let t = line.trim();
        if !t.starts_with("TRACK_META[") {
            continue;
        }
        if let Some(eq) = t.find('=') {
            let key = between(&t[..eq], '[', ']').map(|s| strip_quotes(s.trim()));
            if let (Some(k), Some((a, b))) = (key, parse_tuple_rhs(&t[eq + 1..])) {
                meta.insert(k, (a, b));
            }
        }
        if t.starts_with("TRACK_CATALOGUE[") {
            continue;
        }
    }
    (cat, meta)
}

fn strip_quotes(s: &str) -> String {
    s.trim().trim_matches(|c| c == '\'' || c == '"').to_string()
}
fn between(s: &str, a: char, b: char) -> Option<&str> {
    let i = s.find(a)?;
    let j = s.rfind(b)?;
    if j > i {
        Some(&s[i + 1..j])
    } else {
        None
    }
}
/// `"key": "value",`
fn parse_pair(t: &str) -> Option<(String, String)> {
    let (k, v) = t.split_once(':')?;
    let k = strip_quotes(k);
    let v = strip_quotes(v.trim().trim_end_matches(','));
    if k.is_empty() || v.is_empty() {
        None
    } else {
        Some((k, v))
    }
}
/// `"key": ("Title", "Artist"),`
fn parse_tuple(t: &str) -> Option<(String, String, String)> {
    let (k, v) = t.split_once(':')?;
    let k = strip_quotes(k);
    let (a, b) = parse_tuple_rhs(v)?;
    Some((k, a, b))
}
fn parse_tuple_rhs(v: &str) -> Option<(String, String)> {
    let inner = between(v, '(', ')')?;
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut q: Option<char> = None;
    for ch in inner.chars() {
        match q {
            Some(qc) => {
                if ch == qc {
                    q = None;
                    parts.push(std::mem::take(&mut cur));
                } else {
                    cur.push(ch);
                }
            }
            None => {
                if ch == '\'' || ch == '"' {
                    q = Some(ch);
                }
            }
        }
    }
    if parts.len() >= 2 {
        Some((parts[0].clone(), parts[1].clone()))
    } else {
        None
    }
}

fn scan_dir(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_file() {
                if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                    if AUDIO_EXTS.contains(&ext.to_lowercase().as_str()) {
                        out.push(p);
                    }
                }
            }
        }
    }
}

fn emitted_targets(root: &Path, base: &str) -> Vec<String> {
    let mut out = Vec::new();
    for t in ["stepforge", "saberforge"] {
        if target_enabled(t) && target_dir(root, t).join(base).join("report.json").is_file() {
            out.push(t.to_string());
        }
    }
    let adir = analysis_dir(root);
    if target_enabled("native")
        && NATIVE_DIFFS
            .iter()
            .any(|d| adir.join(format!("{base}.{d}.qa.json")).is_file())
    {
        out.push("native".into());
    }
    out
}

fn track_from_path(
    root: &Path,
    path: &Path,
    cat: &HashMap<String, String>,
    meta: &HashMap<String, (String, String)>,
) -> Track {
    let base = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("track")
        .to_string();
    let file = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let is_ogg = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|e| e.eq_ignore_ascii_case("ogg"))
        .unwrap_or(false);
    let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    // A catalogue entry maps id -> base; prefer that id so TRACK_META applies.
    let id = cat
        .iter()
        .find(|(_, v)| **v == base)
        .map(|(k, _)| k.clone())
        .unwrap_or_else(|| base.clone());
    let registered = meta.contains_key(&id);
    let (mut title, mut artist) = meta
        .get(&id)
        .cloned()
        .unwrap_or_else(|| (base.replace('_', " "), String::new()));
    if title.trim().is_empty() {
        title = base.replace('_', " ");
    }
    if !registered {
        // Title-case the slug for display.
        title = title
            .split(' ')
            .filter(|w| !w.is_empty())
            .map(|w| {
                let mut c = w.chars();
                match c.next() {
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        artist = String::new();
    }

    let mut bpm = None;
    let mut offset = None;
    let mut onsets = None;
    let mut duration_sec = None;
    let mut sustain_count = None;
    let mut analyzed = false;

    if let Ok(txt) =
        std::fs::read_to_string(analysis_dir(root).join(format!("{base}.analysis.json")))
    {
        if let Ok(v) = serde_json::from_str::<Value>(&txt) {
            bpm = v.get("bpm").and_then(|x| x.as_f64());
            offset = v.get("offset").and_then(|x| x.as_f64());
            onsets = v
                .get("onsets")
                .and_then(|x| x.as_array())
                .map(|a| a.len() as u64);
            duration_sec = v.get("duration_s").and_then(|x| x.as_f64());
            sustain_count = v.get("sustain_available").and_then(|x| x.as_u64());
            analyzed = true;
        }
    }

    let targets = emitted_targets(root, &base);
    let status = if !is_ogg {
        "unconverted"
    } else if !targets.is_empty() {
        "charted"
    } else if analyzed {
        "analyzed"
    } else {
        "draft"
    }
    .to_string();

    Track {
        id,
        base,
        title,
        artist,
        file,
        path: path.to_string_lossy().into(),
        is_ogg,
        registered,
        bpm,
        offset,
        onsets,
        duration_sec,
        size_bytes,
        sustain_count,
        status,
        targets,
    }
}

#[tauri::command(async)]
pub fn list_tracks(app: AppHandle) -> Result<Vec<Track>, String> {
    let root = engine::repo_root(&app)?;
    let (cat, meta) = catalogue(&root);
    let mut paths = Vec::new();
    scan_dir(&root.join("public/assets/tracks"), &mut paths);
    scan_dir(&root.join("assets/tracks"), &mut paths);

    // Dedupe by stem, preferring the .ogg the engine can actually open.
    let mut best: HashMap<String, PathBuf> = HashMap::new();
    for p in paths {
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let is_ogg = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|e| e.eq_ignore_ascii_case("ogg"))
            .unwrap_or(false);
        match best.get(&stem) {
            Some(existing) => {
                let existing_ogg = existing
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|e| e.eq_ignore_ascii_case("ogg"))
                    .unwrap_or(false);
                if is_ogg && !existing_ogg {
                    best.insert(stem, p);
                }
            }
            None => {
                best.insert(stem, p);
            }
        }
    }
    let mut tracks: Vec<Track> = best
        .values()
        .map(|p| track_from_path(&root, p, &cat, &meta))
        .collect();
    tracks.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
    Ok(tracks)
}

/// `python -m beatforge to-ogg`: the engine's libsndfile transcoder.
fn transcode_without_ffmpeg(app: &AppHandle, src: &Path, out: &Path) -> Result<(), String> {
    let args = vec![
        "to-ogg".to_string(),
        "--in".to_string(),
        src.to_string_lossy().into_owned(),
        "--out".to_string(),
        out.to_string_lossy().into_owned(),
    ];
    let res = engine::engine_command(app, &args)?
        .output()
        .map_err(|e| format!("could not run the engine to convert {}: {e}", src.display()))?;
    if res.status.success() {
        return Ok(());
    }
    let said = String::from_utf8_lossy(&res.stdout);
    let detail = said
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str::<Value>(l).ok())
        .and_then(|v| v["error"].as_str().map(str::to_string))
        .unwrap_or_else(|| String::from_utf8_lossy(&res.stderr).trim().to_string());
    Err(format!(
        "could not convert {} without ffmpeg ({detail}). WAV, MP3, FLAC and OGG work as is; \
         for other formats install ffmpeg or set its path in Settings.",
        src.display()
    ))
}

/// Transcode any audio file to `<base>.ogg` in both track roots — mirroring
/// `batch_stepforge.transcode`. `-vn -map 0:a:0` drops embedded album art, which
/// would otherwise become a theora stream that soundfile cannot decode.
fn transcode_to_ogg(
    app: &AppHandle,
    root: &Path,
    src: &Path,
    base: &str,
) -> Result<PathBuf, String> {
    let pub_dir = root.join("public/assets/tracks");
    let src_dir = root.join("assets/tracks");
    std::fs::create_dir_all(&pub_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&src_dir).map_err(|e| e.to_string())?;
    let out = pub_dir.join(format!("{base}.ogg"));

    let ffmpeg = engine::ffmpeg_bin(app);
    match std::process::Command::new(&ffmpeg)
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(src)
        .args(["-vn", "-map", "0:a:0", "-c:a", "libvorbis", "-q:a", "6"])
        .arg(&out)
        .output()
    {
        Ok(res) if res.status.success() => {}
        Ok(res) => {
            return Err(format!(
                "ffmpeg failed to convert {}: {}",
                src.display(),
                String::from_utf8_lossy(&res.stderr).trim()
            ))
        }
        // No ffmpeg: the engine converts what libsndfile can read (WAV, FLAC,
        // OGG/Opus, AIFF, MP3) on its own.
        Err(_) => transcode_without_ffmpeg(app, src, &out)?,
    }
    std::fs::copy(&out, src_dir.join(format!("{base}.ogg"))).map_err(|e| e.to_string())?;
    Ok(out)
}

#[tauri::command(async)]
pub fn import_track(
    app: AppHandle,
    source: String,
    title: Option<String>,
) -> Result<Track, String> {
    let root = engine::repo_root(&app)?;
    let src = PathBuf::from(&source);
    if !src.is_file() {
        return Err(format!("no such file: {source}"));
    }
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("song");
    let base = safe_base(title.as_deref().unwrap_or(stem));
    let out = transcode_to_ogg(&app, &root, &src, &base)?;
    let (cat, meta) = catalogue(&root);
    Ok(track_from_path(&root, &out, &cat, &meta))
}

/// Convert an already-present non-ogg file in the track folders.
#[tauri::command(async)]
pub fn convert_track(app: AppHandle, path: String) -> Result<Track, String> {
    let root = engine::repo_root(&app)?;
    let src = PathBuf::from(&path);
    let base = src
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("song")
        .to_string();
    let out = transcode_to_ogg(&app, &root, &src, &base)?;
    let (cat, meta) = catalogue(&root);
    Ok(track_from_path(&root, &out, &cat, &meta))
}

#[tauri::command(async)]
pub fn get_analysis(app: AppHandle, base: String) -> Result<Option<Value>, String> {
    let root = engine::repo_root(&app)?;
    let p = analysis_dir(&root).join(format!("{base}.analysis.json"));
    match std::fs::read_to_string(&p) {
        Ok(txt) => serde_json::from_str(&txt)
            .map(Some)
            .map_err(|e| e.to_string()),
        Err(_) => Ok(None),
    }
}

/// Audio file path for a track (for waveform rendering in the webview).
#[tauri::command(async)]
pub fn track_audio_path(app: AppHandle, base: String) -> Result<Option<String>, String> {
    let root = engine::repo_root(&app)?;
    for dir in ["public/assets/tracks", "assets/tracks"] {
        let p = root.join(dir).join(format!("{base}.ogg"));
        if p.is_file() {
            return Ok(Some(p.to_string_lossy().into()));
        }
    }
    Ok(None)
}

#[tauri::command(async)]
pub fn difficulties_for(target: String) -> Vec<String> {
    match target.as_str() {
        "stepforge" => STEPFORGE_DIFFS.iter().map(|s| s.to_string()).collect(),
        "saberforge" => SABERFORGE_DIFFS.iter().map(|s| s.to_string()).collect(),
        _ => NATIVE_DIFFS.iter().map(|s| s.to_string()).collect(),
    }
}

#[tauri::command(async)]
pub fn default_difficulties_for(target: String) -> Vec<String> {
    match target.as_str() {
        "stepforge" => STEPFORGE_DEFAULT.iter().map(|s| s.to_string()).collect(),
        "saberforge" => vec!["hard".into(), "expert".into()],
        _ => vec!["standard".into()],
    }
}

#[tauri::command(async)]
pub fn analyze_track(
    app: AppHandle,
    job_id: String,
    track_id: String,
    backend: Option<String>,
    force: bool,
) -> Result<(), String> {
    let mut args = vec!["analyze".into(), "--track".into(), track_id];
    args.push("--backend".into());
    args.push(backend.unwrap_or_else(|| "local".into()));
    if force {
        args.push("--force".into());
    }
    engine::spawn_streaming(&app, job_id, args)
}

/* ------------------------------ chart engines ----------------------------- */

/// Engines `stepforge --engine` accepts. `auto` means neural when the weights
/// are installed, else the LLM designer; the app never sends it itself, but a
/// headless script may.
pub const CHART_ENGINES: [&str; 4] = ["neural", "llm", "deterministic", "auto"];

/// Style-descriptor keys `--style-json` accepts: the `style_fields` keys of
/// `neural-status` (conditioning indices 1..8). An omitted key means "let the
/// model infer it from the audio".
pub const NEURAL_STYLE_KEYS: [&str; 8] = [
    "nps",
    "stream_share",
    "jump_share",
    "hold_share",
    "crossover_share",
    "jack_share",
    "footswitch_share",
    "mine_share",
];

/// The engine a run uses. An explicit `engine` wins; without one the legacy
/// `deterministic` switch decides, so a caller written before `--engine`
/// existed gets exactly the run it always got (and never drifts onto the neural
/// model just because the weights turned up).
pub fn resolve_chart_engine(engine: Option<&str>, deterministic: bool) -> Result<String, String> {
    match engine.map(str::trim).filter(|e| !e.is_empty()) {
        Some(e) if CHART_ENGINES.contains(&e) => Ok(e.to_string()),
        Some(e) => Err(format!(
            "'{e}' is not a chart engine (expected one of {})",
            CHART_ENGINES.join(", ")
        )),
        None => Ok(if deterministic { "deterministic" } else { "llm" }.to_string()),
    }
}

/// A preset name fit to hand to the CLI, or None for "no preset". The NAMES are
/// the engine's to define (`neural-status` lists them), so this only checks the
/// shape: a bare identifier that argparse cannot mistake for another flag.
fn clean_style_preset(preset: Option<&str>) -> Result<Option<String>, String> {
    let Some(p) = preset.map(str::trim).filter(|p| !p.is_empty()) else {
        return Ok(None);
    };
    let ok = !p.starts_with('-')
        && p.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if ok {
        Ok(Some(p.to_string()))
    } else {
        Err(format!("'{p}' is not a valid style preset name"))
    }
}

/// Check a `--style-json` payload before a process is spawned for it: a JSON
/// object, only known style keys, only numbers. Returns it re-serialized (None
/// when there is nothing to send), so what reaches the CLI is canonical JSON
/// rather than whatever the caller typed.
pub fn normalize_style_json(raw: Option<&str>) -> Result<Option<String>, String> {
    let Some(raw) = raw.map(str::trim).filter(|r| !r.is_empty()) else {
        return Ok(None);
    };
    let v: Value =
        serde_json::from_str(raw).map_err(|e| format!("style overrides are not valid JSON: {e}"))?;
    let Some(obj) = v.as_object() else {
        return Err("style overrides must be a JSON object of style key -> number".into());
    };
    for (k, val) in obj {
        if !NEURAL_STYLE_KEYS.contains(&k.as_str()) {
            return Err(format!(
                "'{k}' is not a style key (expected one of {})",
                NEURAL_STYLE_KEYS.join(", ")
            ));
        }
        if !val.is_number() {
            return Err(format!("style key '{k}' must be a number, got {val}"));
        }
    }
    if obj.is_empty() {
        return Ok(None);
    }
    Ok(Some(v.to_string()))
}

/// Whether the installed engine CLI knows the neural-era interface.
///
/// An app update does not refresh the `tools/` of a workspace that already
/// exists, so a current app can be driving an engine from before `--engine`,
/// `neural-status` and `stepforge-inpaint` existed. That engine's argparse
/// rejects `--engine` outright, which would take the LLM and deterministic
/// engines down with the neural one. So it is detected and spoken to in the
/// argv it understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CliSupport {
    /// `neural-status` answered: `--engine` and the style flags are understood.
    #[default]
    Current,
    /// `neural-status` is not a command there: pre-neural argv only.
    Legacy,
}

/// Read the `neural-status` probe as a statement about the CLI itself.
///
/// The command's contract is "always one JSON object, always exit 0" — with
/// the weights installed or not. So ANY JSON object back means a current CLI,
/// and no usable answer (argparse "invalid choice", a process that would not
/// start, output that is not an object) means one that predates the command.
/// `installed` is deliberately not consulted: a current CLI without weights
/// still wants `--engine`.
pub fn cli_support(probe: &Result<Value, String>) -> CliSupport {
    match probe {
        Ok(v) if v.is_object() => CliSupport::Current,
        _ => CliSupport::Legacy,
    }
}

/// One `stepforge` run, as both the GUI and `--headless generate` describe it.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepforgeRun<'a> {
    pub track_id: &'a str,
    pub difficulties: &'a [String],
    /// neural | llm | deterministic | auto. None = derive from `deterministic`.
    pub engine: Option<&'a str>,
    /// The pre-`--engine` switch, still honoured when `engine` is None.
    pub deterministic: bool,
    pub style_preset: Option<&'a str>,
    pub style_json: Option<&'a str>,
    /// Re-run the DSP analysis (`--force`). Headless `--reanalyze` only.
    pub reanalyze: bool,
    pub backend: Option<&'a str>,
    /// What the installed CLI understands (see `CliSupport`).
    pub cli: CliSupport,
}

/// The argv for `python -m beatforge stepforge …` — the ONE place it is built,
/// so the GUI and the headless path cannot drift apart.
///
/// To a current CLI the engine is always sent explicitly (an absent `--engine`
/// means `auto` there, which would move a run onto the neural model the day the
/// weights are installed). The style options are validated for every engine but
/// only sent where they mean something (`neural`, and `auto`, which may resolve
/// to it). With `auto`, `--deterministic` is still forwarded: there it picks the
/// fallback when the weights are missing.
///
/// To a LEGACY CLI the argv is the one it has always taken: no `--engine`, no
/// style flags, `--deterministic` for the DSP path and nothing for the designer.
/// The neural engine cannot run there and is refused before anything is spawned.
pub fn stepforge_args(run: &StepforgeRun) -> Result<Vec<String>, String> {
    if run.difficulties.is_empty() {
        return Err("pick at least one difficulty".into());
    }
    let engine = resolve_chart_engine(run.engine, run.deterministic)?;
    let preset = clean_style_preset(run.style_preset)?;
    let style = normalize_style_json(run.style_json)?;
    let legacy = run.cli == CliSupport::Legacy;
    if legacy && engine == "neural" {
        return Err(
            "the installed engine is too old for the neural model: it does not know \
             `--engine neural`. Update or reinstall the engine, or pick another chart engine."
                .into(),
        );
    }

    let mut args: Vec<String> = vec![
        "stepforge".into(),
        "--track".into(),
        run.track_id.to_string(),
        "--difficulties".into(),
        run.difficulties.join(","),
    ];
    if legacy {
        // `auto` on an engine with no neural model is its own fallback rule:
        // the DSP path when `--deterministic` was asked for, else the designer.
        if engine == "deterministic" || (engine == "auto" && run.deterministic) {
            args.push("--deterministic".into());
        }
    } else {
        args.push("--engine".into());
        args.push(engine.clone());
        if engine == "auto" && run.deterministic {
            args.push("--deterministic".into());
        }
    }
    if !legacy && (engine == "neural" || engine == "auto") {
        if let Some(p) = preset {
            args.push("--style-preset".into());
            args.push(p);
        }
        if let Some(j) = style {
            args.push("--style-json".into());
            args.push(j);
        }
    }
    if run.reanalyze {
        args.push("--force".into());
    }
    args.push("--backend".into());
    args.push(run.backend.unwrap_or("local").to_string());
    Ok(args)
}

/// The argv for `stepforge-inpaint`: re-roll bars `start_bar..end_bar` (0-based,
/// end exclusive) of an existing neural chart in place.
///
/// No style flags, on purpose. The CLI records the preset and sliders a chart
/// was generated with in `report.json` and re-rolls with exactly those when the
/// flags are absent — so the new section matches the chart it is spliced into,
/// rather than whatever the Generate screen happens to be set to now.
pub fn inpaint_args(
    track_id: &str,
    difficulty: &str,
    start_bar: u32,
    end_bar: u32,
    seed: Option<u64>,
) -> Result<Vec<String>, String> {
    if track_id.trim().is_empty() {
        return Err("no track to re-roll".into());
    }
    if !STEPFORGE_DIFFS.contains(&difficulty) {
        return Err(format!("'{difficulty}' is not a stepforge difficulty"));
    }
    if start_bar >= end_bar {
        return Err(format!(
            "bars {start_bar}:{end_bar} is empty: the end bar is exclusive and must be greater than the start bar"
        ));
    }
    let mut args: Vec<String> = vec![
        "stepforge-inpaint".into(),
        "--track".into(),
        track_id.to_string(),
        "--difficulty".into(),
        difficulty.to_string(),
        "--bars".into(),
        format!("{start_bar}:{end_bar}"),
    ];
    if let Some(n) = seed {
        args.push("--seed".into());
        args.push(n.to_string());
    }
    Ok(args)
}

/// Everything `generate_charts` is asked for, minus the app handle — so the
/// argv it turns into can be tested without one.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenerateRequest<'a> {
    pub target: &'a str,
    pub track_id: &'a str,
    pub difficulties: &'a [String],
    pub deterministic: bool,
    pub i_have_rights: bool,
    pub backend: Option<&'a str>,
    pub engine: Option<&'a str>,
    pub style_preset: Option<&'a str>,
    pub style_json: Option<&'a str>,
    /// What the installed CLI understands. Only the stepforge argv depends on it.
    pub cli: CliSupport,
}

/// The engine argv for a Generate request, whichever target it is for. Only the
/// StepMania target has a choice of engine; the native `chart` path always uses
/// the designer and ignores `engine` and the style options entirely.
pub fn generate_args(req: &GenerateRequest) -> Result<Vec<String>, String> {
    let target = req.target;
    let difficulties = req.difficulties;
    if !target_enabled(target) {
        return Err(format!(
            "the '{target}' target is not supported by this app yet"
        ));
    }
    if difficulties.is_empty() {
        return Err("pick at least one difficulty".into());
    }
    let mut args: Vec<String> = Vec::new();
    match target {
        "stepforge" => {
            if let Some(bad) = difficulties
                .iter()
                .find(|d| !STEPFORGE_DIFFS.contains(&d.as_str()))
            {
                return Err(format!("'{bad}' is not a {target} difficulty"));
            }
            return stepforge_args(&StepforgeRun {
                track_id: req.track_id,
                difficulties,
                engine: req.engine,
                deterministic: req.deterministic,
                style_preset: req.style_preset,
                style_json: req.style_json,
                reanalyze: false,
                backend: req.backend,
                cli: req.cli,
            });
        }
        "saberforge" => {
            if let Some(bad) = difficulties
                .iter()
                .find(|d| !SABERFORGE_DIFFS.contains(&d.as_str()))
            {
                return Err(format!("'{bad}' is not a {target} difficulty"));
            }
            args.push(target.to_string());
            args.push("--track".into());
            args.push(req.track_id.to_string());
            args.push("--difficulties".into());
            args.push(difficulties.join(","));
            if req.deterministic {
                args.push("--deterministic".into());
            }
            if req.i_have_rights {
                args.push("--i-have-rights".into());
            }
        }
        _ => {
            // Native beatmap via `chart`; --difficulty is a single argparse choice.
            let d = &difficulties[0];
            if !NATIVE_DIFFS.contains(&d.as_str()) {
                return Err(format!(
                    "'{d}' is not a native difficulty (expected one of {})",
                    NATIVE_DIFFS.join(", ")
                ));
            }
            args.push("chart".into());
            args.push("--track".into());
            args.push(req.track_id.to_string());
            args.push("--difficulty".into());
            args.push(d.clone());
        }
    }
    args.push("--backend".into());
    args.push(req.backend.unwrap_or("local").to_string());
    Ok(args)
}

/// `engine`, `style_preset` and `style_json` are the neural-era additions and
/// are all optional, so a caller that still sends only `deterministic` keeps
/// working: the engine is then derived from it (see `resolve_chart_engine`).
#[tauri::command(async)]
#[allow(clippy::too_many_arguments)]
pub fn generate_charts(
    app: AppHandle,
    job_id: String,
    track_id: String,
    target: String,
    difficulties: Vec<String>,
    deterministic: bool,
    i_have_rights: bool,
    backend: Option<String>,
    engine: Option<String>,
    style_preset: Option<String>,
    style_json: Option<String>,
) -> Result<(), String> {
    // Only the stepforge argv differs between CLI generations, so only that
    // target pays for the probe (once; it is cached).
    let cli = if target == "stepforge" {
        cli_support_cached(&app)
    } else {
        CliSupport::Current
    };
    let args = generate_args(&GenerateRequest {
        target: &target,
        track_id: &track_id,
        difficulties: &difficulties,
        deterministic,
        i_have_rights,
        backend: backend.as_deref(),
        engine: engine.as_deref(),
        style_preset: style_preset.as_deref(),
        style_json: style_json.as_deref(),
        cli,
    })?;
    // `engine` the parameter shadows `engine` the module here.
    crate::engine::spawn_streaming(&app, job_id, args)
}

/// Re-roll bars `start_bar..end_bar` (0-based, end exclusive) of an existing
/// neural chart, keeping the rest of it. Streams like any other job. The style
/// is the one the chart was generated with (see `inpaint_args`).
#[tauri::command(async)]
pub fn inpaint_chart(
    app: AppHandle,
    job_id: String,
    track_id: String,
    difficulty: String,
    start_bar: u32,
    end_bar: u32,
    seed: Option<u64>,
) -> Result<(), String> {
    let args = inpaint_args(&track_id, &difficulty, start_bar, end_bar, seed)?;
    engine::spawn_streaming(&app, job_id, args)
}

#[tauri::command(async)]
pub fn cancel_job(app: AppHandle, job_id: String) -> bool {
    engine::cancel(&app, &job_id)
}

/// Which targets/difficulties have emitted results for this track.
#[tauri::command(async)]
pub fn list_results(app: AppHandle, base: String) -> Result<Vec<Value>, String> {
    let root = engine::repo_root(&app)?;
    let mut out = Vec::new();
    for target in ["stepforge", "saberforge"] {
        if !target_enabled(target) {
            continue;
        }
        let report = target_dir(&root, target).join(&base).join("report.json");
        if let Some(v) = read_json(&report) {
            let diffs: Vec<String> = v
                .get("charts")
                .and_then(|c| c.as_object())
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default();
            if !diffs.is_empty() {
                out.push(json!({"target": target, "difficulties": diffs,
                                "outDir": target_dir(&root, target).join(&base).to_string_lossy(),
                                "mode": v.get("mode")}));
            }
        }
    }
    let adir = analysis_dir(&root);
    let native: Vec<String> = NATIVE_DIFFS
        .iter()
        .filter(|d| adir.join(format!("{base}.{d}.qa.json")).is_file())
        .map(|d| d.to_string())
        .collect();
    if target_enabled("native") && !native.is_empty() {
        out.push(json!({"target": "native", "difficulties": native,
                        "outDir": adir.to_string_lossy(), "mode": Value::Null}));
    }
    Ok(out)
}

fn read_json(p: &Path) -> Option<Value> {
    std::fs::read_to_string(p)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
}

/// Pick a preview artifact out of a `previews` dict, whatever the adapter named it.
fn pick_preview(previews: Option<&Value>, exts: &[&str]) -> Option<String> {
    let obj = previews?.as_object()?;
    for v in obj.values() {
        if let Some(s) = v.as_str() {
            let lower = s.to_lowercase();
            if exts.iter().any(|e| lower.ends_with(e)) && Path::new(s).is_file() {
                return Some(s.to_string());
            }
        }
    }
    None
}

fn gates_from(metrics: &Value) -> Vec<Value> {
    metrics
        .get("gates")
        .and_then(|g| g.as_object())
        .map(|m| {
            m.iter()
                .map(|(k, v)| json!({"name": k, "pass": v.as_bool().unwrap_or(false)}))
                .collect()
        })
        .unwrap_or_default()
}

/// Normalized per-difficulty result, whichever target produced it.
#[tauri::command(async)]
pub fn get_result(
    app: AppHandle,
    base: String,
    target: String,
    difficulty: String,
) -> Result<Value, String> {
    let root = engine::repo_root(&app)?;

    if target == "native" {
        let qa = read_json(&analysis_dir(&root).join(format!("{base}.{difficulty}.qa.json")))
            .ok_or("no native result for that difficulty")?;
        let metrics = qa.get("metrics").cloned().unwrap_or(Value::Null);
        let critic = qa.get("critic").cloned().unwrap_or(Value::Null);
        return Ok(json!({
            "target": target,
            "difficulty": difficulty,
            "score": critic.get("score").and_then(|x| x.as_f64()),
            "criticIssues": critic.get("issues").cloned().unwrap_or(json!([])),
            "criticSummary": critic.get("summary").and_then(|x| x.as_str())
                .or_else(|| critic.get("notes").and_then(|x| x.as_str())),
            "humanReview": critic.get("human_review").and_then(|x| x.as_bool()).unwrap_or(false),
            "gates": gates_from(&metrics),
            "metrics": metrics,
            "notes": metrics.get("event_count").and_then(|x| x.as_u64()),
            "designNotes": qa.get("design_notes").cloned().unwrap_or(Value::Null),
            "designSections": qa.get("design_sections").cloned().unwrap_or(json!([])),
            "previewPng": pick_preview(qa.get("previews"), &[".png"]),
            "clickOgg": pick_preview(qa.get("previews"), &[".ogg"]),
            "attempts": qa.get("attempts"),
            "outDir": analysis_dir(&root).to_string_lossy(),
            "extra": Value::Null,
        }));
    }

    let dir = target_dir(&root, &target).join(&base);
    let report = read_json(&dir.join("report.json")).ok_or("no report.json for that target")?;
    let chart = report
        .get("charts")
        .and_then(|c| c.get(&difficulty))
        .cloned()
        .ok_or("that difficulty is not in the report")?;
    let metrics = chart.get("metrics").cloned().unwrap_or(Value::Null);
    // Adapters write the critic both into the report and as a sidecar file.
    let critic = chart
        .get("critic")
        .filter(|c| !c.is_null())
        .cloned()
        .or_else(|| read_json(&dir.join(format!("{difficulty}.critic.json"))))
        .unwrap_or(Value::Null);
    let design = read_json(&dir.join(format!("{difficulty}.design.json"))).unwrap_or(Value::Null);

    let extra = if target == "saberforge" {
        json!({
            "njs": chart.get("njs"), "bombs": chart.get("bombs"), "arcs": chart.get("arcs"),
            "chains": chart.get("chains"), "simulatorClean": chart.get("simulator_clean"),
            "forcedResets": chart.get("forced_resets"), "difficultyRank": chart.get("difficulty_rank"),
            "jsmapVerified": report.get("jsmap_verified"), "referees": report.get("referees"),
        })
    } else {
        json!({
            "meter": chart.get("meter"), "jumps": chart.get("jumps"), "holds": chart.get("holds"),
            "repaired": chart.get("repaired"), "repairOk": chart.get("repair_ok"),
            "density": chart.get("density"), "meterMonotonic": report.get("meter_monotonic"),
            // Bars the chart spans (neural-era reports): bounds the re-roll range.
            "bars": chart.get("bars"),
        })
    };

    Ok(json!({
        "target": target,
        "difficulty": difficulty,
        "score": critic.get("score").and_then(|x| x.as_f64()),
        "criticIssues": critic.get("issues").cloned().unwrap_or(json!([])),
        "criticSummary": critic.get("summary").and_then(|x| x.as_str())
            .or_else(|| critic.get("notes").and_then(|x| x.as_str())),
        "humanReview": critic.get("human_review").and_then(|x| x.as_bool()).unwrap_or(false),
        "gates": gates_from(&metrics),
        "metrics": metrics,
        "notes": chart.get("notes").and_then(|x| x.as_u64()),
        "designNotes": design.get("design_notes").cloned().unwrap_or(Value::Null),
        "designSections": design.get("sections").cloned().unwrap_or(json!([])),
        "previewPng": pick_preview(chart.get("previews"), &[".png"]),
        "clickOgg": pick_preview(chart.get("previews"), &[".ogg"]),
        "mode": report.get("mode"),
        "outDir": dir.to_string_lossy(),
        "extra": extra,
    }))
}

/// Parsed note data for the in-app chart player. STEPFORGE only — the native
/// target emits beatmap JSON, not a simfile.
#[tauri::command(async)]
pub fn get_chart_notes(
    app: AppHandle,
    base: String,
    difficulty: String,
) -> Result<crate::simfile::Chart, String> {
    let root = engine::repo_root(&app)?;
    let dir = target_dir(&root, "stepforge").join(&base);
    // Prefer .ssc (richer); fall back to .sm.
    let mut candidates: Vec<PathBuf> = Vec::new();
    for ext in ["ssc", "sm"] {
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) == Some(ext) {
                    candidates.push(p);
                }
            }
        }
    }
    let path = candidates
        .first()
        .ok_or_else(|| format!("no simfile in {}", dir.display()))?;
    let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    crate::simfile::parse_chart(&src, &difficulty)
        .ok_or_else(|| format!("difficulty '{difficulty}' not found in {}", path.display()))
}

/// Gemini models the app offers, with the measured trade-offs from
/// docs/model-comparison.md so the choice isn't blind.
#[tauri::command(async)]
pub fn gemini_models() -> Vec<Value> {
    vec![
        json!({"id": "gemini-3.8-flash", "label": "Gemini 3.8 Flash",
               "note": "Successor to 3.6 Flash (best foot flow of the measured models, \
                        ~$0.25/chart) at half 3.6's price until 2027-01-01. Not yet \
                        benchmarked on this pipeline itself.",
               "recommended": true}),
        json!({"id": "gemini-3.6-flash", "label": "Gemini 3.6 Flash",
               "note": "The measured previous default: best foot flow in the \
                        model comparison (~$0.25/chart at its price then).",
               "recommended": false}),
        json!({"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash",
               "note": "Slower and ~38% pricier; kept for comparison (~$0.41/chart)",
               "recommended": false}),
        json!({"id": "gemini-3.5-flash-lite", "label": "Gemini 3.5 Flash Lite",
               "note": "~86% cheaper (~$0.06/chart). Drafts well but over-uses jumps, \
                        so on its own it makes less danceable charts — it is the \
                        drafting tier in Hybrid mode, where the gates catch that.",
               "recommended": false}),
    ]
}

/// Model-routing modes, with the reasoning behind the default. Mirrors the
/// `BEATFORGE_LLM_MODE` block in `tools/beatforge/config.py`.
#[tauri::command(async)]
pub fn llm_modes() -> Vec<Value> {
    vec![
        json!({"id": "single", "label": "Single model (recommended)",
               "note": "The chosen model serves every call — designer, retries and \
                        critic. Predictable, and on the one full-length track measured \
                        so far it was also the cheaper of the two.",
               "recommended": true}),
        json!({"id": "hybrid", "label": "Hybrid — cheap drafts, escalated on failure",
               "note": "Flash Lite writes the first draft and it only ships if it clears \
                        every gate (timing, panel balance, holds, foot flow, dynamics); \
                        failures are re-authored by the quality model, which also always \
                        judges. Same quality floor either way — this is purely a cost \
                        bet, and it can go the wrong way: on a 178s track it came out \
                        ~17% MORE expensive because no cheap draft passed. Worth trying \
                        on short tracks; check Cost for the by-tier split.",
               "recommended": false}),
    ]
}

/// Chunked-processing modes, with the reasoning behind the default. Mirrors the
/// `BEATFORGE_CHUNK_MODE` block in `tools/beatforge/config.py`.
#[tauri::command(async)]
pub fn chunk_modes() -> Vec<Value> {
    vec![
        json!({"id": "auto", "label": "Chunked — section by section (recommended)",
               "note": "Designs one section of the song at a time: that section's \
                        audio, its onsets and its density budget only, plus a short \
                        summary of what the earlier sections decided. Tempo, the beat \
                        grid, the section map and the density plan are still measured \
                        once over the whole track and handed to every section as \
                        numbers, so nothing about the timing changes. Chunks a track \
                        longer than the chunk length below; shorter ones go in one \
                        call.",
               "recommended": true}),
        json!({"id": "always", "label": "Chunked — always, even for short tracks",
               "note": "Same, but never sends a whole track in one call however short \
                        it is. Use this if a local model is still running out of VRAM \
                        on tracks under the chunk length.",
               "recommended": false}),
        json!({"id": "never", "label": "Whole song — one call per chart",
               "note": "The pre-chunking behaviour: the entire track and the entire \
                        onset inventory go in a single request. Fine on Gemini, which \
                        has the context for it; on a local model this is what spills \
                        out of VRAM into system RAM and crawls. Kept so the two modes \
                        can be compared on the same song and the same model. (On the \
                        Gemini 4 Argon backend, the recommended mode already sends a \
                        song of up to 15 minutes whole — that is what lets its prompt \
                        cache pay for itself.)",
               "recommended": false}),
    ]
}

/* ----------------------------- packs & install ---------------------------- */

/// Pack assignments plus the resolved game `Songs` directory.
#[tauri::command(async)]
pub fn list_packs(app: AppHandle) -> Result<Value, String> {
    let root = engine::repo_root(&app)?;
    let packs = crate::library::load_packs(&root);
    let mut names: Vec<String> = packs.assignments.values().cloned().collect();
    names.sort();
    names.dedup();
    let s = engine::load_settings(&app);
    let songs_dir = if s.songs_dir.trim().is_empty() {
        crate::library::default_songs_dir()
    } else {
        Some(PathBuf::from(&s.songs_dir))
    };
    // Pack folders already in the game's Songs directory, so a song can be
    // filed under an existing pack without retyping its name.
    let mut installed: Vec<String> = songs_dir
        .as_ref()
        .and_then(|d| std::fs::read_dir(d).ok())
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| !n.starts_with('.'))
                .collect()
        })
        .unwrap_or_default();
    installed.sort_by_key(|n| n.to_lowercase());
    Ok(json!({
        "assignments": packs.assignments,
        "packs": names,
        "installedPacks": installed,
        "songsDir": songs_dir.as_ref().map(|p| p.to_string_lossy().to_string()),
        "songsDirExists": songs_dir.as_ref().map(|p| p.is_dir()).unwrap_or(false),
        "detected": crate::library::detect_song_dirs()
            .iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
    }))
}

/// Assign a song to a pack (empty string unfiles it).
#[tauri::command(async)]
pub fn set_song_pack(app: AppHandle, base: String, pack: String) -> Result<(), String> {
    let root = engine::repo_root(&app)?;
    let mut packs = crate::library::load_packs(&root);
    if pack.trim().is_empty() {
        packs.assignments.remove(&base);
    } else {
        packs.assignments.insert(base, pack.trim().to_string());
    }
    crate::library::save_packs(&root, &packs)
}

/// Copy emitted song folders into the game's Songs directory as
/// `<Songs>/<Pack>/<Song>/`. With `base` set, installs that one song; otherwise
/// installs every song assigned to `pack`.
#[tauri::command(async)]
pub fn install_to_game(
    app: AppHandle,
    base: Option<String>,
    pack: Option<String>,
) -> Result<Value, String> {
    let root = engine::repo_root(&app)?;
    let s = engine::load_settings(&app);
    let songs_dir = if s.songs_dir.trim().is_empty() {
        crate::library::default_songs_dir()
            .ok_or("could not determine the game's Songs directory")?
    } else {
        PathBuf::from(&s.songs_dir)
    };
    if !songs_dir.is_dir() {
        return Err(format!(
            "Songs directory not found: {}. Set it in Settings.",
            songs_dir.display()
        ));
    }
    let assignments = crate::library::load_packs(&root).assignments;

    let targets: Vec<String> = match (&base, &pack) {
        (Some(b), _) => vec![b.clone()],
        (None, Some(p)) => crate::library::Packs {
            assignments: assignments.clone(),
        }
        .songs_in(p),
        _ => return Err("specify a song or a pack".into()),
    };
    if targets.is_empty() {
        return Err("nothing to install".into());
    }

    let mut installed = Vec::new();
    for b in targets {
        let src = target_dir(&root, "stepforge").join(&b);
        if !src.is_dir() {
            continue; // not charted for StepMania yet
        }
        let pack_name = pack
            .clone()
            .or_else(|| assignments.get(&b).cloned())
            .unwrap_or_else(|| "StepForge".to_string());
        let dest = songs_dir.join(&pack_name).join(&b);
        crate::library::copy_dir_recursive(&src, &dest).map_err(|e| e.to_string())?;
        installed.push(dest.to_string_lossy().to_string());
    }
    if installed.is_empty() {
        return Err("no emitted StepMania folders for the selected song(s)".into());
    }
    Ok(json!({ "installed": installed }))
}

/* ------------------------------- song art -------------------------------- */

/// Copy an image/video into the emitted song folder and point the simfile at it.
/// `kind` is "banner" or "background"; background accepts video too.
#[tauri::command(async)]
pub fn set_song_asset(
    app: AppHandle,
    base: String,
    kind: String,
    source: String,
) -> Result<Value, String> {
    let root = engine::repo_root(&app)?;
    let dir = target_dir(&root, "stepforge").join(&base);
    if !dir.is_dir() {
        return Err("this song has no emitted StepMania folder yet — generate charts first".into());
    }
    let src = PathBuf::from(&source);
    let ext = src
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("png")
        .to_lowercase();
    let is_video = matches!(
        ext.as_str(),
        "mp4" | "avi" | "mpg" | "mpeg" | "mov" | "webm"
    );
    if kind == "banner" && is_video {
        return Err("banners must be images".into());
    }
    // Names the STEPFORGE serializer already recognises on its next run.
    let file = match kind.as_str() {
        "banner" => format!("{base}-banner.{ext}"),
        "background" => format!("{base}-bg.{ext}"),
        _ => return Err("kind must be 'banner' or 'background'".into()),
    };
    std::fs::copy(&src, dir.join(&file)).map_err(|e| format!("copy failed: {e}"))?;

    let tag = if kind == "banner" {
        "BANNER"
    } else {
        "BACKGROUND"
    };
    for sf in crate::library::simfiles_in(&dir) {
        crate::library::set_simfile_tag(&sf, tag, &file)?;
    }
    Ok(json!({ "file": file, "dir": dir.to_string_lossy(), "isVideo": is_video }))
}

/// Current artwork for a song, if any.
#[tauri::command(async)]
pub fn get_song_assets(app: AppHandle, base: String) -> Result<Value, String> {
    let root = engine::repo_root(&app)?;
    let dir = target_dir(&root, "stepforge").join(&base);
    let find = |prefix: String| -> Option<String> {
        std::fs::read_dir(&dir).ok()?.flatten().find_map(|e| {
            let p = e.path();
            let name = p.file_name()?.to_str()?.to_string();
            name.starts_with(&prefix)
                .then(|| p.to_string_lossy().to_string())
        })
    };
    Ok(json!({
        "dir": dir.is_dir().then(|| dir.to_string_lossy().to_string()),
        "banner": find(format!("{base}-banner.")),
        "background": find(format!("{base}-bg.")),
    }))
}

/// Generate artwork with assetforge (Vertex/Gemini: Imagen for stills, Veo for
/// video) and attach it. Streams progress like any other job.
#[tauri::command(async)]
pub fn generate_song_asset(
    app: AppHandle,
    job_id: String,
    base: String,
    kind: String,
    media: String,
    prompt: String,
) -> Result<(), String> {
    let root = engine::repo_root(&app)?;
    let dir = target_dir(&root, "stepforge").join(&base);
    if !dir.is_dir() {
        return Err("this song has no emitted StepMania folder yet — generate charts first".into());
    }
    let ext = if media == "video" { "mp4" } else { "png" };
    let file = match kind.as_str() {
        "banner" => format!("{base}-banner.{ext}"),
        "background" => format!("{base}-bg.{ext}"),
        _ => return Err("kind must be 'banner' or 'background'".into()),
    };
    // ITGmania wheel banners are 2:1; backgrounds are widescreen.
    let aspect = if kind == "banner" { "16:9" } else { "16:9" };
    let out = dir.join(&file);
    engine::spawn_assetforge(
        &app,
        job_id,
        media,
        prompt,
        out.to_string_lossy().to_string(),
        aspect.to_string(),
    )
}

/// Point the simfile at artwork that generation just produced.
#[tauri::command(async)]
pub fn attach_generated_asset(app: AppHandle, base: String, kind: String) -> Result<Value, String> {
    let root = engine::repo_root(&app)?;
    let dir = target_dir(&root, "stepforge").join(&base);
    let prefix = if kind == "banner" {
        format!("{base}-banner.")
    } else {
        format!("{base}-bg.")
    };
    let found = std::fs::read_dir(&dir)
        .map_err(|e| e.to_string())?
        .flatten()
        .find_map(|e| {
            let p = e.path();
            let n = p.file_name()?.to_str()?.to_string();
            n.starts_with(&prefix).then_some(n)
        })
        .ok_or("no generated artwork found")?;
    let tag = if kind == "banner" {
        "BANNER"
    } else {
        "BACKGROUND"
    };
    for sf in crate::library::simfiles_in(&dir) {
        crate::library::set_simfile_tag(&sf, tag, &found)?;
    }
    Ok(json!({ "file": found }))
}

/* --------------------------------- cost ---------------------------------- */

/// Per-song cost joined with what that spend actually bought: emitted charts,
/// critic scores and note counts.
#[tauri::command(async)]
pub fn cost_by_song(app: AppHandle) -> Result<Vec<Value>, String> {
    let root = engine::repo_root(&app)?;
    let rollup = read_json(&root.join("build/cost/cost-report.json"));
    let per_song = rollup
        .as_ref()
        .and_then(|r| r.get("per_song"))
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    // Union of songs that cost money and songs that produced output.
    let mut names: Vec<String> = per_song.keys().cloned().collect();
    for t in ["stepforge", "saberforge"] {
        if !target_enabled(t) {
            continue;
        }
        if let Ok(rd) = std::fs::read_dir(target_dir(&root, t)) {
            for e in rd.flatten() {
                if e.path().join("report.json").is_file() {
                    if let Some(n) = e.file_name().to_str() {
                        names.push(n.to_string());
                    }
                }
            }
        }
    }
    names.sort();
    names.dedup();

    let (cat, meta) = catalogue(&root);
    let mut out = Vec::new();
    for name in names {
        // The ledger keys by track id; artifacts key by base. Try both.
        let base = cat.get(&name).cloned().unwrap_or_else(|| name.clone());
        let cost = per_song
            .get(&name)
            .or_else(|| per_song.get(&base))
            .cloned()
            .unwrap_or(Value::Null);

        let mut charts = Vec::new();
        let mut best: Option<f64> = None;
        let mut total_notes: u64 = 0;
        if let Some(report) = read_json(
            &target_dir(&root, "stepforge")
                .join(&base)
                .join("report.json"),
        ) {
            if let Some(map) = report.get("charts").and_then(|c| c.as_object()) {
                for (diff, info) in map {
                    let score = info
                        .get("critic")
                        .and_then(|c| c.get("score"))
                        .and_then(|s| s.as_f64());
                    let notes = info.get("notes").and_then(|n| n.as_u64()).unwrap_or(0);
                    total_notes += notes;
                    if let Some(s) = score {
                        best = Some(best.map_or(s, |b: f64| b.max(s)));
                    }
                    charts.push(json!({
                        "difficulty": diff, "notes": notes, "score": score,
                        "meter": info.get("meter"),
                        "gatesPass": info.get("metrics").and_then(|m| m.get("gates"))
                            .and_then(|g| g.as_object())
                            .map(|g| g.values().all(|v| v.as_bool().unwrap_or(false))),
                    }));
                }
            }
        }
        let title = meta
            .get(&name)
            .or_else(|| meta.get(&base))
            .map(|(t, _)| t.clone());
        out.push(json!({
            "song": name,
            "base": base,
            "title": title,
            "usd": cost.get("usd").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "llmUsd": cost.get("llm_usd").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "computeUsd": cost.get("compute_usd").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "modelCalls": cost.get("model_calls").and_then(|v| v.as_u64()).unwrap_or(0),
            "cacheHitAnalysis": cost.get("cache_hit_analysis").and_then(|v| v.as_bool()).unwrap_or(false),
            "charts": charts,
            "bestScore": best,
            "totalNotes": total_notes,
        }));
    }
    Ok(out)
}

#[tauri::command(async)]
pub fn cost_report(app: AppHandle) -> Result<Value, String> {
    let root = engine::repo_root(&app)?;
    Ok(json!({
        "rollup": read_json(&root.join("build/cost/cost-report.json")),
        "markdown": std::fs::read_to_string(root.join("build/cost/cost-report.md")).ok(),
        "hasLedger": root.join("build/cost").is_dir(),
    }))
}

#[tauri::command(async)]
pub fn refresh_cost_report(app: AppHandle, job_id: String) -> Result<(), String> {
    engine::spawn_streaming(&app, job_id, vec!["cost-report".into(), "--stdout".into()])
}

/// Probing the environment costs ~0.7s (three subprocess spawns, one of which
/// imports numpy/scipy/soundfile). It was being re-run on every library refresh,
/// so the result is memoized; pass `force` to re-probe (Settings "recheck").
static STATUS_CACHE: Mutex<Option<(Instant, Value)>> = Mutex::new(None);
const STATUS_TTL: Duration = Duration::from_secs(20);

#[tauri::command(async)]
pub fn engine_status(app: AppHandle, force: Option<bool>) -> Value {
    if !force.unwrap_or(false) {
        if let Ok(guard) = STATUS_CACHE.lock() {
            if let Some((at, v)) = guard.as_ref() {
                if at.elapsed() < STATUS_TTL {
                    return v.clone();
                }
            }
        }
    }
    let v = probe_engine_status(app);
    if let Ok(mut guard) = STATUS_CACHE.lock() {
        *guard = Some((Instant::now(), v.clone()));
    }
    v
}

/// The import probe. The first line is the engine's hard requirements and
/// decides `beatforgeOk`. onnxruntime is tried separately and only REPORTED:
/// it is needed by the neural engine alone, so a machine without it must still
/// read as ready for the LLM and deterministic engines.
const STATUS_PROBE: &str = "import beatforge, numpy, scipy, soundfile
try:
    import onnxruntime
    print('onnxruntime=ok')
except Exception:
    print('onnxruntime=missing')
";

fn probe_engine_status(app: AppHandle) -> Value {
    let root = engine::repo_root(&app).ok();
    let py = engine::python_bin(&app);

    let python_ok = engine::python_command(&app)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let mut beatforge_ok = false;
    let mut onnxruntime_ok = false;
    let mut import_error = String::new();
    if let Some(r) = &root {
        match engine::python_command(&app)
            .args(["-c", STATUS_PROBE])
            .env("PYTHONPATH", r.join("tools"))
            .current_dir(r)
            .output()
        {
            Ok(o) => {
                beatforge_ok = o.status.success();
                onnxruntime_ok =
                    beatforge_ok && String::from_utf8_lossy(&o.stdout).contains("onnxruntime=ok");
                if !beatforge_ok {
                    import_error = String::from_utf8_lossy(&o.stderr)
                        .trim()
                        .lines()
                        .last()
                        .unwrap_or("")
                        .to_string();
                }
            }
            Err(e) => import_error = e.to_string(),
        }
    }
    let ffmpeg = engine::ffmpeg_bin(&app);
    let ffmpeg_ok = std::process::Command::new(&ffmpeg)
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    json!({
        "engineRoot": root.map(|r| r.to_string_lossy().to_string()),
        "python": py,
        "pythonOk": python_ok,
        "beatforgeOk": beatforge_ok,
        "importError": import_error,
        "ffmpeg": ffmpeg,
        "ffmpegOk": ffmpeg_ok,
        "onnxruntimeOk": onnxruntime_ok,
    })
}

#[tauri::command(async)]
pub fn get_settings(app: AppHandle) -> Settings {
    engine::load_settings(&app)
}

#[tauri::command(async)]
pub fn save_settings(app: AppHandle, settings: Settings) -> Result<(), String> {
    // Python/ffmpeg/engine-root may have changed — drop the memoized probe.
    if let Ok(mut guard) = STATUS_CACHE.lock() {
        *guard = None;
    }
    // …and with them, possibly, WHICH engine CLI is being driven. Only then:
    // the Generate screen saves on every toggle and slider, and forgetting the
    // answer each time would cost a Python start-up before every run.
    if cli_identity_changed(&engine::load_settings(&app), &settings) {
        if let Ok(mut guard) = CLI_SUPPORT.lock() {
            *guard = None;
        }
    }
    engine::save_settings(&app, &settings)
}

/// Run one of the engine's backend checks (`inkling-check`, `inkling-probe`,
/// `argon-check`) and return its json verdict.
///
/// All are short-lived and answer a question the user is waiting on, so they
/// run to completion here rather than streaming as a job. The engine prints one
/// json object and never raises — a failed check is a verdict with a message,
/// which is the entire reason to offer the button (REQ-INK-09).
fn run_check_cmd(app: AppHandle, sub: &str, timeout_hint: &str) -> Result<Value, String> {
    run_check_args(app, sub, &[], timeout_hint)
}

/// `run_check_cmd` for a subcommand that takes arguments (`neural-status --json`).
fn run_check_args(
    app: AppHandle,
    sub: &str,
    extra: &[&str],
    timeout_hint: &str,
) -> Result<Value, String> {
    let mut argv = vec![sub.to_string()];
    argv.extend(extra.iter().map(|a| a.to_string()));
    let mut cmd = engine::engine_command(&app, &argv)?;
    let out = cmd
        .output()
        .map_err(|e| format!("could not run the engine ({sub}): {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The verdict is the LAST json line: the engine may print warnings first.
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .unwrap_or("");
    serde_json::from_str::<Value>(line).map_err(|_| {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = err.trim().lines().rev().take(6).collect();
        format!(
            "the engine returned no result for `{sub}` ({timeout_hint}).\n{}",
            tail.into_iter().rev().collect::<Vec<_>>().join("\n")
        )
    })
}

/// Resolve Modal proxy-token auth and the served model string (REQ-INK-02).
#[tauri::command(async)]
pub fn inkling_check(app: AppHandle) -> Result<Value, String> {
    run_check_cmd(
        app,
        "inkling-check",
        "a Managed Endpoint scales to zero, so a cold start can take a while",
    )
}

/// The differential audio probe (REQ-INK-05) — two clips, one prompt, temp 0.
#[tauri::command(async)]
pub fn inkling_probe(app: AppHandle) -> Result<Value, String> {
    run_check_cmd(
        app,
        "inkling-probe",
        "this sends two clips and can take a minute on a cold endpoint",
    )
}

/// Gemini 4 Argon: resolve the transport and credentials, then run the two-tone
/// audio probe at `low` thinking. Two tiny paid calls (about a cent) that prove
/// the credentials, model access and audio input before a chart pays for them.
#[tauri::command(async)]
pub fn argon_check(app: AppHandle) -> Result<Value, String> {
    run_check_cmd(
        app,
        "argon-check",
        "Argon thinks before it answers, so even the probe can take a minute",
    )
}

/// The local neural chart engine (StepForge Flow): whether the ONNX weights are
/// installed, where the engine looks for them, and the presets and style
/// sliders it offers. Free and offline — it reads a json file and test-imports
/// onnxruntime; nothing is sent anywhere. "Not installed" is an answer
/// (`installed: false` plus an `error` saying what to do), not a failure.
///
/// The model directory follows the SAVED settings (BEATFORGE_NEURAL_MODEL_DIR),
/// like every other check here.
#[tauri::command(async)]
pub fn neural_status(app: AppHandle) -> Result<Value, String> {
    let probe = probe_neural_status(&app);
    remember_cli_support(cli_support(&probe));
    probe
}

fn probe_neural_status(app: &AppHandle) -> Result<Value, String> {
    // An app update may have shipped newer weights than the workspace holds;
    // make sure that copy has landed (it also runs at startup) before asking.
    let _ = crate::install::refresh_workspace_models();
    run_check_args(
        app.clone(),
        "neural-status",
        &["--json"],
        "an engine from before the neural chart engine does not have this command",
    )
}

/// Could these two settings be driving different engine installs? The CLI is
/// identified by where its code is and which interpreter runs it.
fn cli_identity_changed(before: &Settings, after: &Settings) -> bool {
    before.engine_root.trim() != after.engine_root.trim()
        || before.python_path.trim() != after.python_path.trim()
}

/// What the last `neural-status` probe said about the CLI. Memoized because a
/// probe is a Python start-up, and every stepforge run needs the answer; reset
/// when settings are saved (the engine root or interpreter may have changed)
/// and refreshed whenever the status is re-checked.
static CLI_SUPPORT: Mutex<Option<CliSupport>> = Mutex::new(None);

fn remember_cli_support(support: CliSupport) {
    if let Ok(mut guard) = CLI_SUPPORT.lock() {
        *guard = Some(support);
    }
}

/// Which argv the installed engine takes, probing once if nobody has yet. The
/// GUI normally has (Generate loads the status on mount); headless has not.
pub fn cli_support_cached(app: &AppHandle) -> CliSupport {
    if let Some(known) = CLI_SUPPORT.lock().ok().and_then(|g| *g) {
        return known;
    }
    let support = cli_support(&probe_neural_status(app));
    remember_cli_support(support);
    support
}

/// Register a track's title/artist in config.py's TRACK_META, following the
/// existing generated-block convention. Without this, exported simfiles are
/// titled with the raw slug (see the comment above the Founders Mix block).
#[tauri::command(async)]
pub fn register_track_meta(
    app: AppHandle,
    track_id: String,
    title: String,
    artist: String,
) -> Result<(), String> {
    let root = engine::repo_root(&app)?;
    let cfg = root.join("tools/beatforge/config.py");
    let src = std::fs::read_to_string(&cfg).map_err(|e| e.to_string())?;

    // Escape for a Python single-quoted literal, and flatten newlines/CR so a
    // pasted value can never break config.py.
    let esc = |s: &str| {
        s.replace('\\', "\\\\")
            .replace('\'', "\\'")
            .replace(['\n', '\r'], " ")
            .trim()
            .to_string()
    };
    if track_id.trim().is_empty() || track_id.contains(['\'', '\\', '\n']) {
        return Err("invalid track id".into());
    }
    let line = format!(
        "TRACK_META['{}'] = ('{}', '{}')",
        esc(&track_id),
        esc(&title),
        esc(&artist)
    );
    let prefix = format!("TRACK_META['{}']", track_id);

    let mut lines: Vec<String> = src.lines().map(|s| s.to_string()).collect();
    if let Some(existing) = lines
        .iter_mut()
        .find(|l| l.trim_start().starts_with(&prefix))
    {
        *existing = line;
    } else {
        let marker = "# --- Registered from StepForge Desktop ---";
        if !src.contains(marker) {
            lines.push(String::new());
            lines.push(marker.to_string());
        }
        lines.push(line);
    }
    std::fs::write(&cfg, lines.join("\n") + "\n").map_err(|e| e.to_string())
}

/// Copy an emitted song folder to a destination. Returns the created path.
#[tauri::command(async)]
pub fn export_song_folder(
    app: AppHandle,
    base: String,
    target: String,
    dest_dir: String,
) -> Result<String, String> {
    let root = engine::repo_root(&app)?;
    if target == "native" {
        return Err(
            "the native beatmap target emits JSON into build/analysis, not a song folder".into(),
        );
    }
    let src = target_dir(&root, &target).join(&base);
    if !src.is_dir() {
        return Err(format!("no emitted {target} folder for '{base}'"));
    }
    let dest = PathBuf::from(&dest_dir).join(&base);
    copy_dir_recursive(&src, &dest).map_err(|e| e.to_string())?;
    Ok(dest.to_string_lossy().into())
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let from = e.path();
        let to = dest.join(e.file_name());
        if from.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `<repo>/desktop/src-tauri` -> `<repo>`
    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn safe_base_matches_engine_slug() {
        assert_eq!(safe_base("Midnight City"), "midnight_city");
        assert_eq!(safe_base("Some  Will -- Say!"), "some_will_say");
        assert_eq!(safe_base("---"), "song");
        assert_eq!(safe_base("Track 01"), "track_01");
    }

    #[test]
    fn parses_real_config_catalogue_and_meta() {
        let (cat, meta) = catalogue(&repo());
        // Dict-literal block entries.
        assert_eq!(
            cat.get("overdrive").map(|s| s.as_str()),
            Some("overdrive_pulse")
        );
        assert_eq!(cat.get("neon").map(|s| s.as_str()), Some("neon_nights"));
        assert_eq!(
            meta.get("overdrive").cloned(),
            Some(("Overdrive Pulse".into(), "Sweet Papa & the Tones".into()))
        );
        // Appended `TRACK_META['x'] = (...)` lines.
        assert_eq!(
            meta.get("fm2_banana_banana").cloned(),
            Some(("Banana Banana".into(), "Sweet Papa & the Tones".into()))
        );
        // Non-ASCII title survives.
        assert!(meta.contains_key("fm2_kieteyuku"));
    }

    /// Build a throwaway tree mirroring the engine's real on-disk layout.
    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn fixture(tag: &str) -> Fixture {
        let root = std::env::temp_dir().join(format!("bf-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("assets/tracks")).unwrap();
        std::fs::create_dir_all(root.join("build/analysis")).unwrap();
        Fixture(root)
    }

    fn write(p: &Path, s: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, s).unwrap();
    }

    #[test]
    fn track_status_follows_real_artifacts() {
        let f = fixture("status");
        let root = &f.0;
        let empty_cat = HashMap::new();
        let empty_meta = HashMap::new();

        // A non-ogg source cannot be opened by the engine at all.
        let mp3 = root.join("assets/tracks/song.mp3");
        write(&mp3, "x");
        let t = track_from_path(root, &mp3, &empty_cat, &empty_meta);
        assert_eq!(t.status, "unconverted");
        assert!(!t.is_ogg);

        // .ogg with no analysis -> draft
        let ogg = root.join("assets/tracks/song.ogg");
        write(&ogg, "x");
        let t = track_from_path(root, &ogg, &empty_cat, &empty_meta);
        assert_eq!(t.status, "draft");
        assert_eq!(t.base, "song");

        // analysis.json -> analyzed, with the engine's real key names
        write(
            &root.join("build/analysis/song.analysis.json"),
            r#"{"bpm":119.945,"offset":0.423,"duration_s":30.0,
                "sustain_available":4,
                "onsets":[{"id":"m001","time":0.43},{"id":"m002","time":0.9}]}"#,
        );
        let t = track_from_path(root, &ogg, &empty_cat, &empty_meta);
        assert_eq!(t.status, "analyzed");
        assert_eq!(t.bpm, Some(119.945));
        assert_eq!(t.duration_sec, Some(30.0)); // `duration_s`, not `duration`
        assert_eq!(t.onsets, Some(2));
        assert_eq!(t.sustain_count, Some(4)); // a count, not a bool

        // an emitted song folder -> charted
        write(
            &root.join("build/stepmania/song/report.json"),
            r#"{"charts":{"easy":{}}}"#,
        );
        let t = track_from_path(root, &ogg, &empty_cat, &empty_meta);
        assert_eq!(t.status, "charted");
        assert_eq!(t.targets, vec!["stepforge".to_string()]);
    }

    #[test]
    fn catalogue_id_overrides_base_for_registered_tracks() {
        let f = fixture("cat");
        let root = &f.0;
        let ogg = root.join("assets/tracks/overdrive_pulse.ogg");
        write(&ogg, "x");
        let mut cat = HashMap::new();
        cat.insert("overdrive".to_string(), "overdrive_pulse".to_string());
        let mut meta = HashMap::new();
        meta.insert(
            "overdrive".to_string(),
            ("Overdrive Pulse".to_string(), "SPT".to_string()),
        );

        let t = track_from_path(root, &ogg, &cat, &meta);
        // artifacts are named by base, but the CLI is driven by the catalogue id
        assert_eq!(t.base, "overdrive_pulse");
        assert_eq!(t.id, "overdrive");
        assert_eq!(t.title, "Overdrive Pulse");
        assert_eq!(t.artist, "SPT");
        assert!(t.registered);
    }

    #[test]
    fn native_targets_detected_only_for_native_difficulties() {
        let f = fixture("native");
        let root = &f.0;
        // `hard` is NOT a native difficulty, so it must not register as native output.
        write(&root.join("build/analysis/song.hard.qa.json"), "{}");
        assert!(emitted_targets(root, "song").is_empty());
        write(&root.join("build/analysis/song.standard.qa.json"), "{}");
        assert_eq!(emitted_targets(root, "song"), vec!["native".to_string()]);
    }

    #[test]
    fn saberforge_is_hidden_end_to_end() {
        assert!(!target_enabled("saberforge"), "Beat Saber must stay hidden");
        assert!(target_enabled("stepforge"));
        assert!(target_enabled("native"));

        // Even with an emitted Beat Saber song folder present, it must not
        // surface as available output anywhere in the UI.
        let f = fixture("hidden");
        let root = &f.0;
        write(&root.join("assets/tracks/song.ogg"), "x");
        write(
            &root.join("build/saberforge/song/report.json"),
            r#"{"charts":{"expert":{}}}"#,
        );
        assert!(emitted_targets(root, "song").is_empty());

        let t = track_from_path(
            root,
            &root.join("assets/tracks/song.ogg"),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(t.targets.is_empty());
        assert_eq!(t.status, "draft"); // not "charted" — that output doesn't count
    }

    #[test]
    fn difficulty_sets_match_engine() {
        // config.DIFFICULTIES — argparse restricts `chart --difficulty` to these.
        assert_eq!(NATIVE_DIFFS, ["casual", "standard", "overdrive"]);
        assert_eq!(
            SABERFORGE_DIFFS,
            ["easy", "normal", "hard", "expert", "expertplus"]
        );
        assert!(STEPFORGE_DIFFS.contains(&"challenge"));
        // the Generate screen starts with the whole ladder selected
        assert_eq!(default_difficulties_for("stepforge".into()), difficulties_for("stepforge".into()));
        assert_eq!(default_difficulties_for("stepforge".into()).len(), 5);
    }

    fn diffs(d: &[&str]) -> Vec<String> {
        d.iter().map(|s| s.to_string()).collect()
    }

    /// The whole argv, as one string, for readable assertions.
    fn line(args: &[String]) -> String {
        args.join(" ")
    }

    /// The value that follows `flag`, if the flag is present.
    fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    }

    /// The contract with `beatforge stepforge`: every engine is named
    /// explicitly, in the position the CLI documents.
    #[test]
    fn stepforge_argv_names_the_engine() {
        let d = diffs(&["easy", "medium", "hard"]);
        for engine in ["neural", "llm", "deterministic", "auto"] {
            let args = stepforge_args(&StepforgeRun {
                track_id: "fm2_lucky_lucky",
                difficulties: &d,
                engine: Some(engine),
                ..Default::default()
            })
            .unwrap();
            assert_eq!(
                line(&args),
                format!(
                    "stepforge --track fm2_lucky_lucky --difficulties easy,medium,hard \
                     --engine {engine} --backend local"
                )
            );
        }
    }

    #[test]
    fn an_unknown_engine_is_rejected_before_anything_is_spawned() {
        let d = diffs(&["easy"]);
        for bad in ["neurall", "NEURAL", "--deterministic", "gemini"] {
            let err = stepforge_args(&StepforgeRun {
                track_id: "t",
                difficulties: &d,
                engine: Some(bad),
                ..Default::default()
            })
            .unwrap_err();
            assert!(err.contains("not a chart engine"), "{bad}: {err}");
        }
    }

    /// A caller from before `--engine` sends only `deterministic`. It must get
    /// the run it always got: the DSP path when the switch is on, the LLM
    /// designer when it is off — never `auto`, which would move it onto the
    /// neural model the day the weights are installed.
    #[test]
    fn the_legacy_deterministic_switch_still_picks_the_engine() {
        let d = diffs(&["hard"]);
        let run = |engine: Option<&'static str>, deterministic: bool| {
            stepforge_args(&StepforgeRun {
                track_id: "t",
                difficulties: &d,
                engine,
                deterministic,
                ..Default::default()
            })
            .unwrap()
        };
        assert_eq!(flag(&run(None, true), "--engine"), Some("deterministic"));
        assert_eq!(flag(&run(None, false), "--engine"), Some("llm"));
        // A blank engine is the same as none (an unset setting reaches us as "").
        assert_eq!(flag(&run(Some("  "), true), "--engine"), Some("deterministic"));
        // An explicit engine wins over the old switch…
        let neural = run(Some("neural"), true);
        assert_eq!(flag(&neural, "--engine"), Some("neural"));
        assert!(!neural.contains(&"--deterministic".to_string()));
        // …except under `auto`, where the switch still chooses the fallback.
        let auto = run(Some("auto"), true);
        assert_eq!(flag(&auto, "--engine"), Some("auto"));
        assert!(auto.contains(&"--deterministic".to_string()));
        assert!(!run(Some("auto"), false).contains(&"--deterministic".to_string()));
    }

    #[test]
    fn neural_style_options_reach_the_cli() {
        let d = diffs(&["hard", "challenge"]);
        let args = stepforge_args(&StepforgeRun {
            track_id: "overdrive",
            difficulties: &d,
            engine: Some("neural"),
            style_preset: Some("itg_stamina"),
            style_json: Some(r#"{ "jump_share": 0.12, "nps": 6.5 }"#),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            line(&args),
            "stepforge --track overdrive --difficulties hard,challenge --engine neural \
             --style-preset itg_stamina --style-json {\"jump_share\":0.12,\"nps\":6.5} \
             --backend local"
        );
        // One argv element, however the JSON was spaced: there is no shell here.
        let style: Value = serde_json::from_str(flag(&args, "--style-json").unwrap()).unwrap();
        assert_eq!(style["jump_share"], 0.12);
        assert_eq!(style["nps"], 6.5);
    }

    /// Nothing set means nothing sent — an absent flag is how "let the model
    /// decide" is spelled. A blank preset and an empty object are both "nothing".
    #[test]
    fn unset_style_options_are_omitted() {
        let d = diffs(&["hard"]);
        let args = stepforge_args(&StepforgeRun {
            track_id: "t",
            difficulties: &d,
            engine: Some("neural"),
            style_preset: Some(""),
            style_json: Some("{}"),
            ..Default::default()
        })
        .unwrap();
        assert!(flag(&args, "--style-preset").is_none());
        assert!(flag(&args, "--style-json").is_none());
    }

    /// The style options belong to the neural model. They are still validated
    /// for the other engines, but never passed to them.
    #[test]
    fn style_options_are_only_sent_to_an_engine_that_uses_them() {
        let d = diffs(&["hard"]);
        let run = |engine: &'static str| {
            stepforge_args(&StepforgeRun {
                track_id: "t",
                difficulties: &d,
                engine: Some(engine),
                style_preset: Some("jumpstream"),
                style_json: Some(r#"{"jump_share":0.3}"#),
                ..Default::default()
            })
            .unwrap()
        };
        for engine in ["llm", "deterministic"] {
            let args = run(engine);
            assert!(flag(&args, "--style-preset").is_none(), "{engine}");
            assert!(flag(&args, "--style-json").is_none(), "{engine}");
        }
        // `auto` may resolve to neural, so it gets them.
        assert_eq!(flag(&run("auto"), "--style-preset"), Some("jumpstream"));
        assert!(flag(&run("auto"), "--style-json").is_some());
    }

    /// The payload is checked here because the alternative is a traceback from
    /// the middle of a run. Unknown keys, non-numbers and non-objects all stop
    /// before a process exists.
    #[test]
    fn style_json_is_validated() {
        let ok = normalize_style_json(Some(r#"{"mine_share":0,"stream_share":0.6}"#)).unwrap();
        assert_eq!(ok.as_deref(), Some(r#"{"mine_share":0,"stream_share":0.6}"#));
        assert_eq!(normalize_style_json(None).unwrap(), None);
        assert_eq!(normalize_style_json(Some("  ")).unwrap(), None);
        assert_eq!(normalize_style_json(Some("{}")).unwrap(), None);

        let unknown = normalize_style_json(Some(r#"{"jump_share":0.1,"swagger":1}"#)).unwrap_err();
        assert!(unknown.contains("'swagger' is not a style key"), "{unknown}");
        // `meter` and `has_stems` are conditioning inputs, but not user sliders.
        assert!(normalize_style_json(Some(r#"{"meter_norm":0.5}"#)).is_err());

        for non_numeric in [
            r#"{"jump_share":"0.1"}"#,
            r#"{"jump_share":null}"#,
            r#"{"jump_share":true}"#,
            r#"{"jump_share":[0.1]}"#,
            r#"{"jump_share":{"v":0.1}}"#,
        ] {
            let err = normalize_style_json(Some(non_numeric)).unwrap_err();
            assert!(err.contains("must be a number"), "{non_numeric}: {err}");
        }
        for non_object in ["[]", "0.5", r#""jump_share""#, "null", r#"[{"jump_share":0.1}]"#] {
            let err = normalize_style_json(Some(non_object)).unwrap_err();
            assert!(err.contains("must be a JSON object"), "{non_object}: {err}");
        }
        for garbage in ["{jump_share:0.1}", "{", "jump_share=0.1"] {
            let err = normalize_style_json(Some(garbage)).unwrap_err();
            assert!(err.contains("not valid JSON"), "{garbage}: {err}");
        }
    }

    /// A bad style payload fails the run request itself, for either builder.
    #[test]
    fn a_bad_style_payload_fails_the_request() {
        let d = diffs(&["hard"]);
        assert!(stepforge_args(&StepforgeRun {
            track_id: "t",
            difficulties: &d,
            engine: Some("neural"),
            style_json: Some(r#"{"swagger":1}"#),
            ..Default::default()
        })
        .is_err());
        assert!(stepforge_args(&StepforgeRun {
            track_id: "t",
            difficulties: &d,
            engine: Some("neural"),
            style_preset: Some("--model-dir"),
            ..Default::default()
        })
        .is_err());
    }

    /// The keys the app validates against are the engine's `style_fields`
    /// (spec §5, conditioning indices 1..8).
    #[test]
    fn style_keys_match_the_engine_contract() {
        assert_eq!(
            NEURAL_STYLE_KEYS,
            [
                "nps",
                "stream_share",
                "jump_share",
                "hold_share",
                "crossover_share",
                "jack_share",
                "footswitch_share",
                "mine_share"
            ]
        );
        assert_eq!(CHART_ENGINES, ["neural", "llm", "deterministic", "auto"]);
    }

    /// The probe is read as a statement about the CLI, not about the weights:
    /// any JSON object back is a current CLI — installed or not.
    #[test]
    fn the_probe_decides_which_cli_is_installed() {
        let installed = Ok(json!({"installed": true, "model_dir": "/m", "error": null}));
        assert_eq!(cli_support(&installed), CliSupport::Current);

        let not_installed = Ok(json!({
            "installed": false, "onnxruntime": false,
            "error": "no model in /m (model_meta.json is missing)"
        }));
        assert_eq!(
            cli_support(&not_installed),
            CliSupport::Current,
            "no weights is not an old engine"
        );

        // What `run_check_args` returns when argparse rejects the subcommand.
        let unknown: Result<Value, String> = Err(
            "the engine returned no result for `neural-status` (…).\n\
             beatforge: error: argument cmd: invalid choice: 'neural-status'"
                .into(),
        );
        assert_eq!(cli_support(&unknown), CliSupport::Legacy);
        let no_process: Result<Value, String> =
            Err("could not run the engine (neural-status): No such file or directory".into());
        assert_eq!(cli_support(&no_process), CliSupport::Legacy);
        // A JSON line that is not the status object is not an answer either.
        assert_eq!(cli_support(&Ok(json!([]))), CliSupport::Legacy);
        assert_eq!(cli_support(&Ok(Value::Null)), CliSupport::Legacy);
    }

    /// An engine from before `--engine` gets exactly the argv it always got —
    /// its argparse would reject the new flag and fail every run, including the
    /// LLM and deterministic ones that have nothing to do with the neural model.
    #[test]
    fn a_legacy_cli_gets_the_pre_neural_argv() {
        let d = diffs(&["easy", "hard"]);
        let run = |engine: Option<&'static str>, deterministic: bool, cli: CliSupport| {
            stepforge_args(&StepforgeRun {
                track_id: "neon",
                difficulties: &d,
                engine,
                deterministic,
                // Sent by a caller that does not know better: must be dropped.
                style_preset: Some("jumpstream"),
                style_json: Some(r#"{"jump_share":0.3}"#),
                cli,
                ..Default::default()
            })
        };
        let legacy = CliSupport::Legacy;
        assert_eq!(
            line(&run(Some("llm"), false, legacy).unwrap()),
            "stepforge --track neon --difficulties easy,hard --backend local"
        );
        assert_eq!(
            line(&run(Some("deterministic"), false, legacy).unwrap()),
            "stepforge --track neon --difficulties easy,hard --deterministic --backend local"
        );
        // The old switch alone (headless `--deterministic`, no `--engine`).
        assert_eq!(
            line(&run(None, true, legacy).unwrap()),
            "stepforge --track neon --difficulties easy,hard --deterministic --backend local"
        );
        assert_eq!(
            line(&run(None, false, legacy).unwrap()),
            "stepforge --track neon --difficulties easy,hard --backend local"
        );
        // `auto` has no neural model to resolve to there: its fallback applies.
        assert_eq!(
            line(&run(Some("auto"), true, legacy).unwrap()),
            "stepforge --track neon --difficulties easy,hard --deterministic --backend local"
        );
        assert_eq!(
            line(&run(Some("auto"), false, legacy).unwrap()),
            "stepforge --track neon --difficulties easy,hard --backend local"
        );
        for engine in ["llm", "deterministic", "auto"] {
            let args = run(Some(engine), true, legacy).unwrap();
            for new_flag in ["--engine", "--style-preset", "--style-json"] {
                assert!(!args.contains(&new_flag.to_string()), "{engine}: {new_flag}");
            }
        }

        // Neural cannot run there, and says why instead of spawning a failure.
        let err = run(Some("neural"), false, legacy).unwrap_err();
        assert!(err.contains("too old for the neural model"), "{err}");

        // The same requests to a current CLI name the engine, as always.
        assert_eq!(
            line(&run(Some("llm"), false, CliSupport::Current).unwrap()),
            "stepforge --track neon --difficulties easy,hard --engine llm --backend local"
        );
        assert_eq!(
            flag(&run(Some("neural"), false, CliSupport::Current).unwrap(), "--style-preset"),
            Some("jumpstream")
        );
    }

    /// The remembered answer is dropped when the engine being driven may have
    /// changed, and only then — every slider on the Generate screen saves.
    #[test]
    fn the_cli_answer_is_forgotten_only_when_the_engine_may_have_changed() {
        let base = Settings {
            engine_root: "/ws".into(),
            python_path: "/venv/bin/python3".into(),
            ..Default::default()
        };
        let mut sliders = base.clone();
        sliders.chart_engine = "neural".into();
        sliders.neural_style.insert("jump_share".into(), 0.2);
        sliders.llm_backend = "argon".into();
        assert!(!cli_identity_changed(&base, &sliders));

        let mut moved = base.clone();
        moved.engine_root = "/other/checkout".into();
        assert!(cli_identity_changed(&base, &moved));
        let mut python = base.clone();
        python.python_path = "python3.12".into();
        assert!(cli_identity_changed(&base, &python));
    }

    /// The GUI request carries the same decision through to the builder; the
    /// native target does not depend on it at all.
    #[test]
    fn generate_requests_follow_the_cli_generation() {
        let d = diffs(&["hard"]);
        let legacy = generate_args(&GenerateRequest {
            target: "stepforge",
            track_id: "neon",
            difficulties: &d,
            deterministic: true,
            engine: Some("deterministic"),
            cli: CliSupport::Legacy,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            line(&legacy),
            "stepforge --track neon --difficulties hard --deterministic --backend local"
        );
        assert!(generate_args(&GenerateRequest {
            target: "stepforge",
            track_id: "neon",
            difficulties: &d,
            engine: Some("neural"),
            cli: CliSupport::Legacy,
            ..Default::default()
        })
        .is_err());

        let native = diffs(&["standard"]);
        let native_args = |cli| {
            generate_args(&GenerateRequest {
                target: "native",
                track_id: "neon",
                difficulties: &native,
                cli,
                ..Default::default()
            })
            .unwrap()
        };
        assert_eq!(native_args(CliSupport::Legacy), native_args(CliSupport::Current));
    }

    /// The native target has no engine choice: its argv is what it was before
    /// the neural engine existed, whatever the request carries.
    #[test]
    fn the_native_target_is_unaffected_by_the_engine_choice() {
        let d = diffs(&["standard"]);
        let plain = generate_args(&GenerateRequest {
            target: "native",
            track_id: "neon",
            difficulties: &d,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(line(&plain), "chart --track neon --difficulty standard --backend local");

        let with_neural_opts = generate_args(&GenerateRequest {
            target: "native",
            track_id: "neon",
            difficulties: &d,
            deterministic: true,
            engine: Some("neural"),
            style_preset: Some("jumpstream"),
            style_json: Some(r#"{"jump_share":0.3}"#),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(with_neural_opts, plain);
    }

    /// The GUI request goes through the same stepforge builder as headless, and
    /// keeps the checks it always made before spawning.
    #[test]
    fn generate_requests_build_the_stepforge_argv() {
        let d = diffs(&["easy", "hard"]);
        let args = generate_args(&GenerateRequest {
            target: "stepforge",
            track_id: "neon",
            difficulties: &d,
            engine: Some("neural"),
            style_preset: Some("casual_flow"),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            line(&args),
            "stepforge --track neon --difficulties easy,hard --engine neural \
             --style-preset casual_flow --backend local"
        );

        // A pre-neural frontend: only the old switch.
        let legacy = generate_args(&GenerateRequest {
            target: "stepforge",
            track_id: "neon",
            difficulties: &d,
            deterministic: true,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            line(&legacy),
            "stepforge --track neon --difficulties easy,hard --engine deterministic --backend local"
        );

        let bad_diff = diffs(&["standard"]);
        assert!(generate_args(&GenerateRequest {
            target: "stepforge",
            track_id: "neon",
            difficulties: &bad_diff,
            ..Default::default()
        })
        .unwrap_err()
        .contains("not a stepforge difficulty"));
        assert!(generate_args(&GenerateRequest {
            target: "stepforge",
            track_id: "neon",
            difficulties: &[],
            ..Default::default()
        })
        .is_err());
        assert!(generate_args(&GenerateRequest {
            target: "saberforge",
            track_id: "neon",
            difficulties: &d,
            ..Default::default()
        })
        .unwrap_err()
        .contains("not supported"));
    }

    /// The contract with `beatforge stepforge-inpaint`: bars are `START:END`,
    /// 0-based with END exclusive, and the seed is optional.
    #[test]
    fn inpaint_argv_matches_the_cli() {
        let bare = inpaint_args("overdrive", "hard", 16, 24, None).unwrap();
        assert_eq!(
            line(&bare),
            "stepforge-inpaint --track overdrive --difficulty hard --bars 16:24"
        );
        let seeded = inpaint_args("overdrive", "challenge", 0, 1, Some(42)).unwrap();
        assert_eq!(
            line(&seeded),
            "stepforge-inpaint --track overdrive --difficulty challenge --bars 0:1 --seed 42"
        );
        // Seed 0 is a seed, not "no seed".
        let zero = inpaint_args("t", "easy", 2, 3, Some(0)).unwrap();
        assert_eq!(flag(&zero, "--seed"), Some("0"));
    }

    /// A re-roll never names a style: the CLI then uses the preset and sliders
    /// recorded in report.json when the chart was generated, so the new section
    /// matches the chart around it.
    #[test]
    fn inpaint_leaves_the_style_to_the_chart() {
        for args in [
            inpaint_args("overdrive", "hard", 16, 24, None).unwrap(),
            inpaint_args("overdrive", "hard", 16, 24, Some(7)).unwrap(),
        ] {
            assert!(!args.iter().any(|a| a.starts_with("--style")), "{args:?}");
        }
    }

    #[test]
    fn inpaint_rejects_requests_the_cli_could_not_run() {
        // An empty or backwards range.
        assert!(inpaint_args("t", "hard", 8, 8, None)
            .unwrap_err()
            .contains("end bar is exclusive"));
        assert!(inpaint_args("t", "hard", 9, 8, None).is_err());
        // A difficulty from another target, and no track at all.
        assert!(inpaint_args("t", "expertplus", 0, 4, None)
            .unwrap_err()
            .contains("not a stepforge difficulty"));
        assert!(inpaint_args("  ", "hard", 0, 4, None).is_err());
    }

    /// onnxruntime is the neural engine's dependency only. The probe must report
    /// it without letting its absence fail the line that decides `beatforgeOk`.
    #[test]
    fn a_missing_onnxruntime_cannot_fail_the_engine_probe() {
        let mut lines = STATUS_PROBE.lines();
        let required = lines.next().unwrap();
        assert_eq!(required, "import beatforge, numpy, scipy, soundfile");
        assert!(!required.contains("onnxruntime"));
        let rest: Vec<&str> = lines.collect();
        assert_eq!(rest[0], "try:");
        assert!(rest[1].trim() == "import onnxruntime");
        assert!(rest.iter().any(|l| l.starts_with("except")));
    }
}
