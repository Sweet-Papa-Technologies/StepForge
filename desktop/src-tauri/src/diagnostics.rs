//! What is actually wrong with the engine, in words a bug report can use.
//!
//! Two jobs:
//!
//! 1. Say WHY the engine did not answer. "The interpreter is missing", "Python
//!    crashed while loading a library", "numpy did not import" and "this engine
//!    predates the neural model" are four different problems; before this they
//!    were all reported as the last one.
//! 2. Build the text behind "Copy diagnostics": app version, OS, CPU, the
//!    engine's self-test and the last error — and nothing that identifies the
//!    user or their music. No audio, no song names, no folder names from their
//!    machine (`redact`).

use std::collections::BTreeMap;
use std::process::{ExitStatus, Output};
use std::sync::Mutex;

use serde_json::{json, Value};
use tauri::AppHandle;

use crate::engine;

// --------------------------------------------------------------------------- //
// Why a process did not answer
// --------------------------------------------------------------------------- //

/// How a child process ended, in plain words. `None` for a normal exit 0.
///
/// A crash is not an error message: on Windows the exit code is an NTSTATUS
/// (0xC0000135 = a DLL it needs is not on the machine), on Unix the process is
/// killed by a signal. Both used to surface as "the engine returned no result".
pub fn describe_exit(code: Option<i32>, signal: Option<i32>) -> Option<String> {
    if let Some(sig) = signal {
        let name = match sig {
            4 => "SIGILL: an instruction this CPU does not have",
            6 => "SIGABRT: it aborted itself",
            7 | 10 => "SIGBUS: bad memory access",
            9 => "SIGKILL: it was killed, often by the system running out of memory",
            11 => "SIGSEGV: invalid memory access",
            15 => "SIGTERM: it was stopped",
            _ => "",
        };
        return Some(if name.is_empty() {
            format!("was killed by signal {sig}")
        } else {
            format!("crashed (signal {sig}, {name})")
        });
    }
    match code {
        Some(0) => None,
        Some(c) => {
            // NTSTATUS failures are 0xC0000000 and up; as an i32 they are negative.
            let u = c as u32;
            let nt = match u {
                0xC000_0005 => "access violation",
                0xC000_001D => "illegal instruction: this CPU lacks an instruction it needs",
                0xC000_0135 => "a DLL it needs was not found",
                0xC000_0139 => "a DLL it needs is the wrong version (entry point not found)",
                0xC000_0142 => "a DLL failed to initialise",
                0xC000_007B => "a DLL it needs is for a different CPU architecture",
                0xC000_0409 => "it aborted itself (stack buffer overrun / fast fail)",
                0xC000_00FD => "stack overflow",
                0xC000_013A => "it was interrupted",
                _ => "",
            };
            Some(if !nt.is_empty() {
                format!("crashed (0x{u:08X}, {nt})")
            } else if u >= 0xC000_0000 {
                format!("crashed (0x{u:08X})")
            } else {
                format!("exited with code {c}")
            })
        }
        None => Some("ended without an exit code".into()),
    }
}

#[cfg(unix)]
fn signal_of(status: &ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[cfg(not(unix))]
fn signal_of(_status: &ExitStatus) -> Option<i32> {
    None
}

/// `describe_exit` for a real exit status.
pub fn describe_status(status: &ExitStatus) -> Option<String> {
    describe_exit(status.code(), signal_of(status))
}

/// The last `n` non-empty lines of a process's stderr, joined with " | ".
pub fn stderr_tail(stderr: &[u8], n: usize) -> String {
    let text = String::from_utf8_lossy(stderr);
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join(" | ")
}

/// Why a finished process gave no usable answer: how it ended plus what it
/// said last. Used wherever the engine's output could not be parsed.
pub fn describe_failure(out: &Output) -> String {
    let tail = stderr_tail(&out.stderr, 4);
    match (describe_status(&out.status), tail.is_empty()) {
        (Some(how), false) => format!("it {how}: {tail}"),
        (Some(how), true) => format!("it {how}, with no error message"),
        (None, false) => format!("it printed no result: {tail}"),
        (None, true) => "it printed no result and no error message".into(),
    }
}

// --------------------------------------------------------------------------- //
// The import probe
// --------------------------------------------------------------------------- //

/// The marker the probe prints before its JSON, so a warning some library
/// writes to stdout while importing cannot be mistaken for the answer.
pub const PROBE_MARKER: &str = "STEPFORGE_PROBE ";

/// Modules the engine cannot chart without, in the order they are reported.
/// onnxruntime is listed separately: only the StepForge Flow model needs it.
pub const REQUIRED_MODULES: [&str; 5] = ["beatforge", "numpy", "scipy", "soundfile", "simfile"];

/// Imports every dependency in its own try block and prints, for each one,
/// either null or the real exception. Standard library only, so it can run —
/// and say what is wrong — in an install where the engine itself cannot.
pub const STATUS_PROBE: &str = r#"import json, sys
out = {}
for name in ("beatforge", "numpy", "scipy", "soundfile", "simfile", "onnxruntime"):
    try:
        __import__(name)
        out[name] = None
    except BaseException as e:
        out[name] = (type(e).__name__ + ": " + str(e)).strip().rstrip(":")
out["python"] = sys.version.split()[0]
print("STEPFORGE_PROBE " + json.dumps(out))
"#;

/// What the import probe found.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ImportProbe {
    /// module -> the exception it raised, for every module that did not import.
    pub failed: BTreeMap<String, String>,
    /// The probe did not finish (the interpreter did not start, or it crashed
    /// while importing): how, in words. `failed` is empty then.
    pub died: Option<String>,
    pub python_version: Option<String>,
}

impl ImportProbe {
    /// Python started, and everything the engine needs imported.
    pub fn engine_ok(&self) -> bool {
        self.died.is_none() && REQUIRED_MODULES.iter().all(|m| !self.failed.contains_key(*m))
    }

    pub fn onnxruntime_ok(&self) -> bool {
        self.died.is_none() && !self.failed.contains_key("onnxruntime")
    }

    /// One sentence naming the first required module that failed and its real
    /// error, or how the probe died. Empty when the engine is fine.
    pub fn engine_error(&self) -> String {
        if let Some(d) = &self.died {
            return format!("Python stopped while loading the engine's libraries: {d}");
        }
        REQUIRED_MODULES
            .iter()
            .find_map(|m| self.failed.get(*m).map(|e| format!("{m} did not import: {e}")))
            .unwrap_or_default()
    }
}

/// Read the probe's stdout. `Err` when it holds no probe line.
pub fn parse_probe(stdout: &str) -> Result<ImportProbe, ()> {
    let line = stdout
        .lines()
        .rev()
        .find_map(|l| l.trim().strip_prefix(PROBE_MARKER.trim_end()))
        .ok_or(())?;
    let v: Value = serde_json::from_str(line.trim()).map_err(|_| ())?;
    let obj = v.as_object().ok_or(())?;
    let mut probe = ImportProbe::default();
    for (k, val) in obj {
        if k == "python" {
            probe.python_version = val.as_str().map(str::to_string);
        } else if let Some(err) = val.as_str() {
            probe.failed.insert(k.clone(), err.to_string());
        }
    }
    Ok(probe)
}

/// The probe result for a process that ran to some end.
pub fn probe_from_output(out: &Output) -> ImportProbe {
    parse_probe(&String::from_utf8_lossy(&out.stdout)).unwrap_or_else(|_| ImportProbe {
        died: Some(describe_failure(out)),
        ..Default::default()
    })
}

// --------------------------------------------------------------------------- //
// The last error, for the diagnostics report
// --------------------------------------------------------------------------- //

static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// Remember the most recent engine failure (a probe that died, a job that
/// exited non-zero). Kept in memory only.
pub fn record_error(msg: impl Into<String>) {
    let msg = msg.into();
    if msg.trim().is_empty() {
        return;
    }
    if let Ok(mut g) = LAST_ERROR.lock() {
        *g = Some(msg);
    }
}

pub fn last_error() -> Option<String> {
    LAST_ERROR.lock().ok().and_then(|g| g.clone())
}

// --------------------------------------------------------------------------- //
// Redaction
// --------------------------------------------------------------------------- //

/// What must not leave the machine in a diagnostics report.
#[derive(Debug, Clone, Default)]
pub struct Redaction {
    /// (path, replacement), longest path first: the workspace, the app's
    /// runtime, the home folder.
    pub roots: Vec<(String, String)>,
    pub user: String,
    /// File stems of the songs in the library.
    pub songs: Vec<String>,
}

impl Redaction {
    pub fn new(roots: Vec<(String, String)>, user: String, songs: Vec<String>) -> Self {
        let mut roots: Vec<(String, String)> = roots
            .into_iter()
            .map(|(p, r)| (p.trim_end_matches(['/', '\\']).to_string(), r))
            .filter(|(p, _)| p.len() > 3)
            .collect();
        roots.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let mut songs: Vec<String> = songs.into_iter().filter(|s| s.chars().count() >= 3).collect();
        songs.sort_by(|a, b| b.len().cmp(&a.len()));
        Self { roots, user, songs }
    }

    /// `text` with the user's folders, user name and song names replaced.
    /// Windows paths are matched with either slash and without regard to case.
    pub fn apply(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (root, with) in &self.roots {
            out = replace_path(&out, root, with);
        }
        for song in &self.songs {
            out = out.replace(song.as_str(), "<song>");
        }
        if self.user.chars().count() >= 3 {
            out = replace_word(&out, &self.user, "<user>");
        }
        out
    }
}

/// Replace `root` wherever it appears as a path prefix, accepting `/` or `\`
/// as the separator and, when the root looks like a Windows path, any case.
fn replace_path(text: &str, root: &str, with: &str) -> String {
    let windows = root.contains('\\') || root.chars().nth(1) == Some(':');
    let norm = |s: &str| -> String {
        let s = s.replace('\\', "/");
        if windows {
            s.to_lowercase()
        } else {
            s
        }
    };
    let needle = norm(root);
    if needle.is_empty() {
        return text.to_string();
    }
    // Normalising keeps byte offsets only when lower-casing does not change
    // lengths; fall back to an exact replace when it would.
    let hay = norm(text);
    if hay.len() != text.len() {
        return text.replace(root, with);
    }
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while let Some(pos) = hay[i..].find(&needle) {
        let start = i + pos;
        out.push_str(&text[i..start]);
        out.push_str(with);
        i = start + needle.len();
    }
    out.push_str(&text[i..]);
    out
}

/// Replace `word` where it stands alone (not inside a longer identifier).
fn replace_word(text: &str, word: &str, with: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while let Some(pos) = text[i..].find(word) {
        let start = i + pos;
        let end = start + word.len();
        let before = text[..start].chars().next_back();
        let after = text[end..].chars().next();
        let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric() && c != '_');
        out.push_str(&text[i..start]);
        out.push_str(if boundary(before) && boundary(after) { with } else { word });
        i = end;
    }
    out.push_str(&text[i..]);
    out
}

// --------------------------------------------------------------------------- //
// The machine
// --------------------------------------------------------------------------- //

/// `PRETTY_NAME` from /etc/os-release ("CachyOS Linux", "Ubuntu 24.04.1 LTS").
pub fn os_release_name(text: &str) -> Option<String> {
    let get = |key: &str| {
        text.lines().find_map(|l| {
            l.strip_prefix(key)
                .and_then(|r| r.strip_prefix('='))
                .map(|v| v.trim().trim_matches('"').to_string())
        })
    };
    get("PRETTY_NAME").or_else(|| get("NAME")).filter(|s| !s.is_empty())
}

/// The first `model name` in /proc/cpuinfo.
pub fn cpuinfo_model(text: &str) -> Option<String> {
    text.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        (k.trim() == "model name").then(|| v.trim().to_string())
    })
}

fn run_text(program: &str, args: &[&str]) -> Option<String> {
    let out = engine::command(program).args(args).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !s.is_empty()).then_some(s)
}

/// Operating system and version, without asking Python (it may be what is broken).
pub fn os_description() -> String {
    if cfg!(target_os = "macos") {
        let v = run_text("/usr/bin/sw_vers", &["-productVersion"]).unwrap_or_else(|| "unknown version".into());
        format!("macOS {v}")
    } else if cfg!(target_os = "windows") {
        // "Microsoft Windows [Version 10.0.26100.2314]"
        run_text("cmd", &["/c", "ver"]).unwrap_or_else(|| "Windows (version unknown)".into())
    } else {
        let name = std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|t| os_release_name(&t))
            .unwrap_or_else(|| "Linux".into());
        let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        if kernel.is_empty() {
            name
        } else {
            format!("{name} (kernel {kernel})")
        }
    }
}

/// CPU model plus the instruction sets that numeric libraries care about.
pub fn cpu_description() -> String {
    let model = if cfg!(target_os = "macos") {
        run_text("/usr/sbin/sysctl", &["-n", "machdep.cpu.brand_string"])
    } else if cfg!(target_os = "windows") {
        run_text(
            "reg",
            &[
                "query",
                r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                "/v",
                "ProcessorNameString",
            ],
        )
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.split("REG_SZ").nth(1).map(|v| v.trim().to_string()))
        })
        .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok())
    } else {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|t| cpuinfo_model(&t))
    }
    .unwrap_or_else(|| "unknown".into());
    format!("{model} ({}{})", std::env::consts::ARCH, cpu_features())
}

#[cfg(target_arch = "x86_64")]
fn cpu_features() -> String {
    let yn = |b: bool| if b { "yes" } else { "NO" };
    format!(
        "; SSE4.2 {}, AVX {}, AVX2 {}",
        yn(std::arch::is_x86_feature_detected!("sse4.2")),
        yn(std::arch::is_x86_feature_detected!("avx")),
        yn(std::arch::is_x86_feature_detected!("avx2")),
    )
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_features() -> String {
    String::new()
}

/// How this copy of the app was packaged.
pub fn packaging() -> &'static str {
    if cfg!(target_os = "macos") {
        "macOS app"
    } else if cfg!(target_os = "windows") {
        "Windows installer"
    } else if std::env::var_os("APPIMAGE").is_some() || std::env::var_os("APPDIR").is_some() {
        "AppImage"
    } else if crate::install::bundled_runtime().is_some() {
        ".deb / system package"
    } else {
        "development build"
    }
}

/// Linux desktop session ("wayland, KDE"); empty elsewhere.
pub fn session_description() -> String {
    if !cfg!(target_os = "linux") {
        return String::new();
    }
    let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    let kind = var("XDG_SESSION_TYPE").unwrap_or_else(|| {
        if var("WAYLAND_DISPLAY").is_some() {
            "wayland".into()
        } else if var("DISPLAY").is_some() {
            "x11".into()
        } else {
            "unknown".into()
        }
    });
    match var("XDG_CURRENT_DESKTOP") {
        Some(d) => format!("{kind}, {d}"),
        None => kind,
    }
}

// --------------------------------------------------------------------------- //
// Media (Linux)
// --------------------------------------------------------------------------- //

/// GStreamer elements WebKitGTK needs to decode and play the song on the
/// Results screen. All are in gst-plugins-base / gst-plugins-good.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub const MEDIA_ELEMENTS: [&str; 10] = [
    "appsrc",
    "appsink",
    "decodebin",
    "playbin",
    "giostreamsrc",
    "audioconvert",
    "audioresample",
    "autoaudiosink",
    "oggdemux",
    "vorbisdec",
];

/// Which of `MEDIA_ELEMENTS` this process cannot create. `None` when the
/// question does not apply (not Linux) or GStreamer is not loaded.
///
/// The AppImage carries its own GStreamer. Without its plugins the web view's
/// process dies the moment a page creates an audio element — the window stops
/// taking clicks and goes black — so "can the app play audio here" is part of
/// the self-test, not a nicety.
#[cfg(target_os = "linux")]
pub fn missing_media_elements() -> Option<Vec<String>> {
    use std::ffi::{c_char, c_int, c_void, CString};
    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    type InitCheck = unsafe extern "C" fn(*mut c_int, *mut *mut *mut c_char, *mut *mut c_void) -> c_int;
    type FactoryFind = unsafe extern "C" fn(*const c_char) -> *mut c_void;
    // GStreamer is a dependency of WebKitGTK, so it is already in this process;
    // looking the two functions up by name avoids linking against its headers.
    let sym = |name: &str| -> *mut c_void {
        let c = CString::new(name).unwrap();
        // RTLD_DEFAULT is a null handle on Linux.
        unsafe { dlsym(std::ptr::null_mut(), c.as_ptr()) }
    };
    let (init, find) = (sym("gst_init_check"), sym("gst_element_factory_find"));
    if init.is_null() || find.is_null() {
        return None;
    }
    unsafe {
        let init: InitCheck = std::mem::transmute(init);
        let find: FactoryFind = std::mem::transmute(find);
        if init(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()) == 0 {
            return Some(MEDIA_ELEMENTS.iter().map(|s| s.to_string()).collect());
        }
        Some(
            MEDIA_ELEMENTS
                .iter()
                .filter(|name| {
                    let c = CString::new(**name).unwrap();
                    find(c.as_ptr()).is_null()
                })
                .map(|s| s.to_string())
                .collect(),
        )
    }
}

#[cfg(not(target_os = "linux"))]
pub fn missing_media_elements() -> Option<Vec<String>> {
    None
}

/// `{"checked": bool, "ok": bool, "missing": [...]}` for the self-test.
pub fn media_report() -> Value {
    match missing_media_elements() {
        None => json!({"checked": false, "ok": true, "missing": []}),
        Some(missing) => json!({"checked": true, "ok": missing.is_empty(), "missing": missing}),
    }
}

// --------------------------------------------------------------------------- //
// The engine self-test and the report
// --------------------------------------------------------------------------- //

/// Run `python -m beatforge.selftest --json` with the app's own interpreter and
/// environment. `Err` carries why there is no report (and is recorded as the
/// last error).
pub fn run_selftest(app: &AppHandle, chart: bool, self_contained: bool) -> Result<Value, String> {
    let root = engine::repo_root(app)?;
    let mut cmd = engine::python_command(app);
    cmd.current_dir(&root)
        .env("PYTHONPATH", root.join("tools"))
        .env_remove("BEATFORGE_NEURAL_MODEL_DIR")
        .args(["-m", "beatforge.selftest", "--json"]);
    if chart {
        cmd.arg("--chart");
    }
    if self_contained {
        cmd.arg("--self-contained");
    }
    let out = cmd.output().map_err(|e| {
        let msg = format!("Python could not be started ({}): {e}", engine::python_bin(app));
        record_error(msg.clone());
        msg
    })?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let report = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .and_then(|l| serde_json::from_str::<Value>(l).ok())
        .filter(Value::is_object);
    report.ok_or_else(|| {
        let msg = format!("the engine self-test gave no report: {}", describe_failure(&out));
        record_error(msg.clone());
        msg
    })
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

/// The lines of the self-test section. Pure, so the wording is testable.
pub fn selftest_lines(report: &Result<Value, String>) -> Vec<String> {
    let r = match report {
        Err(e) => return vec![format!("Engine self-test: could not run — {e}")],
        Ok(r) => r,
    };
    let mut out = vec![format!(
        "Engine self-test: {}",
        if r["ok"].as_bool().unwrap_or(false) { "passed" } else { "FAILED" }
    )];
    if let Some(p) = r["platform"].as_str() {
        out.push(format!("  Python {} on {p}", r["python"].as_str().unwrap_or("?")));
    }
    if let Some(mods) = r["modules"].as_object() {
        for (name, m) in mods {
            out.push(if m["ok"].as_bool().unwrap_or(false) {
                format!("  {name}: ok {}", m["version"].as_str().unwrap_or(""))
                    .trim_end()
                    .to_string()
            } else {
                format!("  {name}: FAILED — {}", m["error"].as_str().unwrap_or("unknown error"))
            });
        }
    }
    let model = &r["model"];
    if model.is_object() {
        out.push(if model["ok"].as_bool().unwrap_or(false) {
            format!(
                "  model: ok ({})",
                model["display_name"].as_str().or(model["version"].as_str()).unwrap_or("loaded")
            )
        } else {
            format!("  model: FAILED — {}", model["error"].as_str().unwrap_or("unknown error"))
        });
    }
    let chart = &r["chart"];
    if chart.is_object() {
        out.push(if chart["ok"].as_bool().unwrap_or(false) {
            format!("  test chart: ok ({})", chart["charts"])
        } else {
            format!("  test chart: FAILED — {}", chart["error"].as_str().unwrap_or("unknown error"))
        });
    }
    if let Some(vc) = r["system_vc_runtime"].as_array().filter(|a| !a.is_empty()) {
        let names: Vec<&str> = vc.iter().filter_map(Value::as_str).collect();
        out.push(format!("  VC++ runtime taken from Windows, not the app: {}", names.join(", ")));
    }
    out
}

/// Everything the report needs that is not about the engine.
pub struct Machine {
    pub app_version: String,
    pub packaging: String,
    pub os: String,
    pub session: String,
    pub cpu: String,
    pub engine_version: String,
    pub bundled_runtime: bool,
    pub profile_ascii: bool,
    pub profile_spaces: bool,
    pub ffmpeg: bool,
}

/// The report text. Pure: `redaction` is applied to the engine-derived parts
/// here, and nothing else in it comes from the user's files.
pub fn format_report(
    m: &Machine,
    selftest: &Result<Value, String>,
    media: &Value,
    last_error: Option<&str>,
    redaction: &Redaction,
) -> String {
    let mut lines = vec![
        "StepForge diagnostics".to_string(),
        format!("App: {} ({})", m.app_version, m.packaging),
        format!("OS: {}", m.os),
    ];
    if !m.session.is_empty() {
        lines.push(format!("Session: {}", m.session));
    }
    lines.push(format!("CPU: {}", m.cpu));
    lines.push(format!(
        "Engine: {} ({})",
        if m.engine_version.is_empty() { "version unknown" } else { m.engine_version.as_str() },
        if m.bundled_runtime { "the app's own Python" } else { "a Python outside the app" }
    ));
    lines.push(format!(
        "Profile path: plain ASCII {}, contains spaces {}",
        yes_no(m.profile_ascii),
        yes_no(m.profile_spaces)
    ));
    lines.push(format!("ffmpeg: {}", if m.ffmpeg { "found" } else { "not found (optional)" }));
    lines.extend(selftest_lines(selftest).iter().map(|l| redaction.apply(l)));
    if media["checked"].as_bool().unwrap_or(false) {
        let missing: Vec<&str> = media["missing"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        lines.push(if missing.is_empty() {
            "Audio playback (GStreamer): ok".to_string()
        } else {
            format!("Audio playback (GStreamer): MISSING {}", missing.join(", "))
        });
    }
    lines.push(format!(
        "Last error: {}",
        last_error.map(|e| redaction.apply(e)).unwrap_or_else(|| "none".into())
    ));
    lines.push("(No audio, song names or folder names are included.)".to_string());
    lines.join("\n")
}

fn home_dir() -> Option<String> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .filter(|s| !s.trim().is_empty())
}

/// Gather the machine facts and the redaction rules for this install.
pub fn collect(app: &AppHandle) -> (Machine, Redaction) {
    let home = home_dir().unwrap_or_default();
    let root = engine::repo_root(app).ok();
    let engine_version = root
        .as_ref()
        .and_then(|r| std::fs::read_to_string(r.join("ENGINE_VERSION")).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let ffmpeg = engine::command(engine::ffmpeg_bin(app))
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let machine = Machine {
        app_version: app.package_info().version.to_string(),
        packaging: packaging().to_string(),
        os: os_description(),
        session: session_description(),
        cpu: cpu_description(),
        engine_version,
        bundled_runtime: crate::install::bundled_python()
            .map(|p| p.to_string_lossy() == engine::python_bin(app))
            .unwrap_or(false),
        profile_ascii: home.is_ascii(),
        profile_spaces: home.contains(' '),
        ffmpeg,
    };

    let mut roots = Vec::new();
    if let Some(r) = &root {
        roots.push((r.to_string_lossy().to_string(), "<workspace>".to_string()));
    }
    if let Some(rt) = crate::install::bundled_runtime() {
        roots.push((rt.to_string_lossy().to_string(), "<app>/runtime".to_string()));
        if let Ok(real) = rt.canonicalize() {
            roots.push((real.to_string_lossy().to_string(), "<app>/runtime".to_string()));
        }
    }
    if !home.is_empty() {
        roots.push((home.clone(), "~".to_string()));
    }
    let user = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_default();
    let songs = crate::commands::list_tracks(app.clone())
        .map(|ts| ts.into_iter().flat_map(|t| [t.base, t.id]).collect())
        .unwrap_or_default();
    (machine, Redaction::new(roots, user, songs))
}

/// The text behind "Copy diagnostics". Runs the engine self-test (imports and
/// model load; a few seconds), so call it off the main thread.
pub fn report(app: &AppHandle) -> String {
    let (machine, redaction) = collect(app);
    let selftest = run_selftest(app, false, false);
    format_report(&machine, &selftest, &media_report(), last_error().as_deref(), &redaction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_crash_is_described_as_a_crash_not_as_silence() {
        assert_eq!(describe_exit(Some(0), None), None);
        assert_eq!(describe_exit(Some(1), None).unwrap(), "exited with code 1");
        // Windows: python.exe could not load a DLL / hit an unsupported instruction.
        let dll = describe_exit(Some(0xC000_0135_u32 as i32), None).unwrap();
        assert!(dll.contains("0xC0000135") && dll.contains("DLL it needs was not found"), "{dll}");
        let ill = describe_exit(Some(0xC000_001D_u32 as i32), None).unwrap();
        assert!(ill.contains("illegal instruction"), "{ill}");
        assert!(describe_exit(Some(0xC000_0FFF_u32 as i32), None).unwrap().starts_with("crashed (0xC0000FFF"));
        // Unix: killed by a signal, so there is no exit code at all.
        let sigill = describe_exit(None, Some(4)).unwrap();
        assert!(sigill.contains("SIGILL") && sigill.contains("CPU"), "{sigill}");
        assert!(describe_exit(None, Some(11)).unwrap().contains("SIGSEGV"));
        assert_eq!(describe_exit(None, Some(63)).unwrap(), "was killed by signal 63");
    }

    #[test]
    fn the_probe_reports_each_module_with_its_own_error() {
        let out = format!(
            "a warning some library printed\n{PROBE_MARKER}{}\n",
            json!({
                "beatforge": null, "numpy": null, "scipy": null, "soundfile": null, "simfile": null,
                "onnxruntime": "ImportError: DLL load failed while importing onnxruntime_pybind11_state: The specified module could not be found.",
                "python": "3.12.16"
            })
        );
        let p = parse_probe(&out).unwrap();
        assert!(p.engine_ok(), "onnxruntime is not one of the engine's requirements");
        assert!(!p.onnxruntime_ok());
        assert!(p.failed["onnxruntime"].contains("DLL load failed"));
        assert_eq!(p.python_version.as_deref(), Some("3.12.16"));
        assert_eq!(p.engine_error(), "");

        let broken = format!(
            "{PROBE_MARKER}{}",
            json!({"beatforge": null, "numpy": "ModuleNotFoundError: No module named 'numpy'",
                   "scipy": "ModuleNotFoundError: No module named 'numpy'", "onnxruntime": null})
        );
        let p = parse_probe(&broken).unwrap();
        assert!(!p.engine_ok());
        assert!(p.onnxruntime_ok());
        assert_eq!(
            p.engine_error(),
            "numpy did not import: ModuleNotFoundError: No module named 'numpy'"
        );
        assert!(parse_probe("Traceback (most recent call last):\n  ...").is_err());
    }

    /// The probe itself must keep going past a module that fails, and must not
    /// need anything outside the standard library to say so.
    #[test]
    fn the_probe_script_tries_every_module_separately() {
        assert!(STATUS_PROBE.starts_with("import json, sys\n"));
        for m in REQUIRED_MODULES {
            assert!(STATUS_PROBE.contains(&format!("\"{m}\"")), "{m} is not probed");
        }
        assert!(STATUS_PROBE.contains("\"onnxruntime\""));
        assert!(STATUS_PROBE.contains("except BaseException"));
        assert!(STATUS_PROBE.contains(PROBE_MARKER.trim_end()));
    }

    #[test]
    fn a_probe_that_died_says_how() {
        let p = ImportProbe { died: Some("it crashed (signal 4, SIGILL: …)".into()), ..Default::default() };
        assert!(!p.engine_ok() && !p.onnxruntime_ok());
        assert!(p.engine_error().starts_with("Python stopped while loading the engine's libraries: it crashed"));
    }

    #[test]
    fn diagnostics_carry_no_user_paths_user_name_or_song_names() {
        let r = Redaction::new(
            vec![
                ("/Users/maria/Library/Application Support/StepForge/workspace".into(), "<workspace>".into()),
                ("/Users/maria".into(), "~".into()),
            ],
            "maria".into(),
            vec!["Tokyo Drift (Extended)".into(), "ab".into()],
        );
        let text = "FileNotFoundError: [Errno 2] No such file or directory: \
                    '/Users/maria/Library/Application Support/StepForge/workspace/public/assets/tracks/Tokyo Drift (Extended).ogg' \
                    (also /Users/maria/Music, owner maria; mariachi stays)";
        let out = r.apply(text);
        assert!(!out.contains("maria/") && !out.contains("/Users/"), "{out}");
        assert!(!out.contains("Tokyo Drift"), "{out}");
        assert!(out.contains("'<workspace>/public/assets/tracks/<song>.ogg'"), "{out}");
        assert!(out.contains("~/Music"), "{out}");
        assert!(out.contains("owner <user>;"), "{out}");
        assert!(out.contains("mariachi stays"), "a longer word is not the user name: {out}");
    }

    #[test]
    fn windows_paths_are_redacted_whatever_the_slashes_and_case() {
        let r = Redaction::new(
            vec![(r"C:\Users\Zoë Q".into(), "~".into())],
            "Zoë Q".into(),
            vec![],
        );
        let out = r.apply(r"cannot open c:/users/zoë q/AppData/Roaming/StepForge and C:\Users\Zoë Q\Music");
        assert_eq!(out, r"cannot open ~/AppData/Roaming/StepForge and ~\Music");
    }

    #[test]
    fn os_and_cpu_are_read_from_the_system_files() {
        let os = "NAME=\"CachyOS Linux\"\nPRETTY_NAME=\"CachyOS Linux\"\nID=cachyos\n";
        assert_eq!(os_release_name(os).as_deref(), Some("CachyOS Linux"));
        assert_eq!(os_release_name("NAME=Arch\n").as_deref(), Some("Arch"));
        assert_eq!(os_release_name("ID=x\n"), None);
        let cpu = "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 7 5800X 8-Core Processor\n";
        assert_eq!(cpuinfo_model(cpu).as_deref(), Some("AMD Ryzen 7 5800X 8-Core Processor"));
    }

    fn machine() -> Machine {
        Machine {
            app_version: "0.1.0-beta.7".into(),
            packaging: "AppImage".into(),
            os: "CachyOS Linux (kernel 6.17.1)".into(),
            session: "wayland, KDE".into(),
            cpu: "AMD Ryzen 7 5800X (x86_64; SSE4.2 yes, AVX yes, AVX2 yes)".into(),
            engine_version: "0.1.0-beta.7 abc1234".into(),
            bundled_runtime: true,
            profile_ascii: false,
            profile_spaces: true,
            ffmpeg: false,
        }
    }

    #[test]
    fn the_report_names_each_failure_and_leaves_the_user_out() {
        let selftest = Ok(json!({
            "ok": false, "python": "3.12.16", "platform": "Windows-10-10.0.19045-SP0",
            "modules": {
                "numpy": {"ok": true, "version": "2.5.5"},
                "onnxruntime": {"ok": false, "error": "ImportError: DLL load failed while importing onnxruntime_pybind11_state: The specified module could not be found."}
            },
            "model": {"ok": false, "error": "ModelNotInstalled: The neural engine needs the `onnxruntime` package"},
            "chart": null, "system_vc_runtime": ["vcruntime140.dll"], "errors": ["…"]
        }));
        let red = Redaction::new(vec![("/home/sam".into(), "~".into())], "sam".into(), vec!["My Song".into()]);
        let media = json!({"checked": true, "ok": false, "missing": ["appsrc", "decodebin"]});
        let text = format_report(
            &machine(),
            &selftest,
            &media,
            Some("job exited with code 1: could not read /home/sam/x/My Song.ogg"),
            &red,
        );
        for want in [
            "App: 0.1.0-beta.7 (AppImage)",
            "OS: CachyOS Linux (kernel 6.17.1)",
            "Session: wayland, KDE",
            "CPU: AMD Ryzen 7 5800X",
            "Profile path: plain ASCII no, contains spaces yes",
            "Engine self-test: FAILED",
            "  numpy: ok 2.5.5",
            "  onnxruntime: FAILED — ImportError: DLL load failed",
            "  model: FAILED — ModelNotInstalled",
            "VC++ runtime taken from Windows, not the app: vcruntime140.dll",
            "Audio playback (GStreamer): MISSING appsrc, decodebin",
            "Last error: job exited with code 1: could not read ~/x/<song>.ogg",
        ] {
            assert!(text.contains(want), "missing {want:?} in:\n{text}");
        }
        assert!(!text.contains("/home/sam") && !text.contains("My Song"), "{text}");

        // No self-test at all is its own, accurate line.
        let none: Result<Value, String> =
            Err("Python could not be started (/opt/x/python3): No such file or directory (os error 2)".into());
        let text = format_report(&machine(), &none, &json!({"checked": false}), None, &red);
        assert!(text.contains("Engine self-test: could not run — Python could not be started"), "{text}");
        assert!(text.contains("Last error: none"));
        assert!(!text.contains("Audio playback"), "not checked, not reported: {text}");
    }
}
