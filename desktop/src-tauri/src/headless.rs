//! headless.rs — run the app's own jobs with no window, from a script.
//!
//! The point is UNIFORMITY, not convenience. Anything that drives the engine
//! from outside the GUI has to resolve the interpreter, the repo root and the
//! whole credential/env block exactly as the GUI does, and a re-implementation
//! of that mapping drifts the moment a setting is added. So this does not
//! re-implement it: it builds a real `AppHandle` and calls the same
//! `engine::engine_command` and `commands::*` functions the UI calls. A setting
//! the GUI honours is honoured here by construction.
//!
//! Output contract, so this is scriptable:
//!   * STDOUT is NDJSON — one JSON object per line, nothing else. Machine input.
//!   * STDERR carries the engine's own chatter and progress. Human input.
//!   * exit code 0 = every requested unit succeeded, 1 = at least one failed,
//!     2 = the request itself was bad (unknown flag, missing file).
//!
//!   StepForge --headless tracks
//!   StepForge --headless settings
//!   StepForge --headless generate --track my_song --difficulties easy,hard
//!   StepForge --headless generate --track my_song --difficulties easy,hard \
//!       --style-preset itg_stamina --style-json '{"jump_share":0.12}'
//!   StepForge --headless install  --base my_song --pack "My Mix"
//!   StepForge --headless batch    --manifest mix.json [--force]

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;

use serde_json::{json, Value};
use tauri::{AppHandle, Manager};

/// True when argv asks for headless mode. Checked before the GUI builder runs so
/// a scripted call never creates a window.
pub fn requested() -> bool {
    std::env::args().skip(1).any(|a| a == "--headless" || a == "-H")
}

fn out(v: Value) {
    println!("{v}");
    let _ = std::io::stdout().flush();
}

fn fail(msg: impl std::fmt::Display) -> ! {
    out(json!({"ok": false, "error": msg.to_string()}));
    std::process::exit(2)
}

/// `--key value` pairs plus bare flags, after the subcommand.
fn parse_flags(args: &[String]) -> (HashMap<String, String>, Vec<String>) {
    let mut kv = HashMap::new();
    let mut flags = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(name) = a.strip_prefix("--") {
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                kv.insert(name.to_string(), args[i + 1].clone());
                i += 2;
                continue;
            }
            flags.push(name.to_string());
        }
        i += 1;
    }
    (kv, flags)
}

fn need<'a>(kv: &'a HashMap<String, String>, key: &str) -> &'a String {
    kv.get(key)
        .unwrap_or_else(|| fail(format!("missing required --{key}")))
}

/// Run the engine synchronously through the GUI's own command builder.
///
/// `spawn_streaming` is the GUI path and is deliberately asynchronous — it
/// registers the child for cancellation and pushes lines out as Tauri events,
/// which needs an event loop that is not running here. This waits instead, and
/// relays the child's output to STDERR so STDOUT stays parseable.
fn run_engine(app: &AppHandle, args: Vec<String>) -> Result<i32, String> {
    let mut cmd = crate::engine::engine_command(app, &args)?;
    cmd.stdout(Stdio::piped()).stderr(Stdio::inherit());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to launch engine: {e}"))?;
    if let Some(stdout) = child.stdout.take() {
        crate::engine::for_each_line_lossy(stdout, |line| eprintln!("{line}"));
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    Ok(status.code().unwrap_or(-1))
}

fn generate(app: &AppHandle, kv: &HashMap<String, String>, flags: &[String]) -> bool {
    let track = need(kv, "track").clone();
    let target = kv
        .get("target")
        .cloned()
        .unwrap_or_else(|| "stepforge".into());
    let diffs: Vec<String> = need(kv, "difficulties")
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let deterministic = flags.iter().any(|f| f == "deterministic");
    // The engine's --force re-runs the DSP analysis, which is a property of the
    // audio and not of the model; re-charting needs no flag at all. Kept as an
    // explicit opt-in so a model swap does not silently redo hours of DSP.
    let reanalyze = flags.iter().any(|f| f == "reanalyze");

    let args = if target == "stepforge" {
        // The GUI's own builder, so `--engine`, `--style-preset` and
        // `--style-json` mean here exactly what the Generate screen sends. With
        // no `--engine` it charts with the onboard model (`--deterministic`
        // still picks the DSP path).
        crate::commands::stepforge_args(&crate::commands::StepforgeRun {
            track_id: &track,
            difficulties: &diffs,
            engine: kv.get("engine").map(String::as_str),
            deterministic,
            style_preset: kv.get("style-preset").map(String::as_str),
            style_json: kv.get("style-json").map(String::as_str),
            reanalyze,
            backend: kv.get("backend").map(String::as_str),
            // The same probe-and-decide the GUI makes (cached, so a batch asks
            // once): an engine from before `--engine` gets the argv it knows.
            cli: crate::commands::cli_support_cached(app),
        })
        .unwrap_or_else(|e| fail(e))
    } else {
        let mut args = vec![
            target.clone(),
            "--track".into(),
            track.clone(),
            "--difficulties".into(),
            diffs.join(","),
        ];
        if deterministic {
            args.push("--deterministic".into());
        }
        if reanalyze {
            args.push("--force".into());
        }
        args.push("--backend".into());
        args.push(kv.get("backend").cloned().unwrap_or_else(|| "local".into()));
        args
    };

    match run_engine(app, args) {
        Ok(0) => {
            out(json!({"ok": true, "op": "generate", "track": track,
                       "target": target, "difficulties": diffs}));
            true
        }
        Ok(code) => {
            out(json!({"ok": false, "op": "generate", "track": track, "exit": code}));
            false
        }
        Err(e) => {
            out(json!({"ok": false, "op": "generate", "track": track, "error": e}));
            false
        }
    }
}

fn install(app: &AppHandle, kv: &HashMap<String, String>) -> bool {
    let base = need(kv, "base").clone();
    if let Some(pack) = kv.get("pack") {
        if let Err(e) = crate::commands::set_song_pack(app.clone(), base.clone(), pack.clone()) {
            out(json!({"ok": false, "op": "install", "base": base, "error": e}));
            return false;
        }
    }
    match crate::commands::install_to_game(app.clone(), Some(base.clone()), kv.get("pack").cloned())
    {
        Ok(v) => {
            out(json!({"ok": true, "op": "install", "base": base, "result": v}));
            true
        }
        Err(e) => {
            out(json!({"ok": false, "op": "install", "base": base, "error": e}));
            false
        }
    }
}

/// A whole mix in one call: generate every song, install each as it finishes.
///
/// Installing per song rather than in a final pass is deliberate — this runs for
/// hours, and an interrupted run should leave a pack that is playable as far as
/// it got. Re-running skips songs whose charts already cover every requested
/// difficulty, so resume is just "run it again".
fn batch(app: &AppHandle, kv: &HashMap<String, String>, flags: &[String]) -> bool {
    let path = PathBuf::from(need(kv, "manifest"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| fail(format!("cannot read manifest {}: {e}", path.display())));
    let m: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| fail(format!("manifest is not valid JSON: {e}")));

    let target = m
        .get("target")
        .and_then(|v| v.as_str())
        .unwrap_or("stepforge")
        .to_string();
    let diffs: Vec<String> = m
        .get("difficulties")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_else(|| fail("manifest needs a \"difficulties\" array"));
    let songs: Vec<String> = m
        .get("songs")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_else(|| fail("manifest needs a \"songs\" array"));
    let pack = m.get("pack").and_then(|v| v.as_str()).map(str::to_string);
    let force = flags.iter().any(|f| f == "force");

    let root = crate::engine::repo_root(app).unwrap_or_else(|e| fail(e));
    let mut ok_count = 0usize;
    let mut failed: Vec<String> = Vec::new();

    out(json!({"ok": true, "op": "batch:start", "songs": songs.len(),
               "difficulties": diffs, "target": target, "pack": pack}));

    for (i, base) in songs.iter().enumerate() {
        // Resume: a song already carrying every requested difficulty is skipped.
        let report = root
            .join("build")
            .join(if target == "saberforge" { "saberforge" } else { "stepmania" })
            .join(base)
            .join("report.json");
        let done = !force
            && std::fs::read_to_string(&report)
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                .and_then(|v| v.get("charts").cloned())
                .map(|c| diffs.iter().all(|d| c.get(d).is_some()))
                .unwrap_or(false);
        if done {
            out(json!({"ok": true, "op": "batch:skip", "base": base,
                       "index": i + 1, "reason": "already charted"}));
            ok_count += 1;
            continue;
        }

        eprintln!("=== [{}/{}] {base} ===", i + 1, songs.len());
        let mut kv2 = HashMap::new();
        kv2.insert("track".to_string(), base.clone());
        kv2.insert("difficulties".to_string(), diffs.join(","));
        kv2.insert("target".to_string(), target.clone());
        // The engine choice applies to the whole mix.
        for key in ["engine", "style-preset", "style-json"] {
            if let Some(v) = kv.get(key) {
                kv2.insert(key.to_string(), v.clone());
            }
        }
        if !generate(app, &kv2, flags) {
            failed.push(base.clone());
            continue;
        }
        let mut kv3 = HashMap::new();
        kv3.insert("base".to_string(), base.clone());
        if let Some(p) = &pack {
            kv3.insert("pack".to_string(), p.clone());
        }
        if install(app, &kv3) {
            ok_count += 1;
        } else {
            failed.push(base.clone());
        }
    }

    out(json!({"ok": failed.is_empty(), "op": "batch:done",
               "installed": ok_count, "failed": failed}));
    failed.is_empty()
}

/// Entry point. Builds a real app handle, runs one subcommand, exits.
///
/// Takes the context rather than expanding `generate_context!` itself: that
/// macro embeds Info.plist, and expanding it twice in one binary is a duplicate
/// symbol. One context, two ways to run it.
pub fn main(ctx: tauri::Context) -> ! {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let rest: Vec<String> = argv
        .iter()
        .filter(|a| *a != "--headless" && *a != "-H")
        .cloned()
        .collect();
    let sub = rest.first().cloned().unwrap_or_else(|| {
        fail("usage: --headless <tracks|settings|generate|install|batch> [flags]")
    });
    let (kv, flags) = parse_flags(&rest[1..]);

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .manage(crate::engine::JobRegistry::default())
        .build(ctx)
        .unwrap_or_else(|e| fail(format!("could not initialise app: {e}")));
    let handle = app.handle().clone();
    if let Ok(dir) = handle.path().resource_dir() {
        crate::install::set_resource_dir(dir);
    }
    // The config window is created during build(); a scripted run must not show
    // one. Closing rather than never-creating keeps the GUI config untouched.
    for (_, w) in handle.webview_windows() {
        let _ = w.close();
    }

    // The GUI does this from its setup hook, which does not run here: an app
    // update may ship neural weights the existing workspace does not have yet.
    if let Err(e) = crate::install::refresh_workspace() {
        eprintln!("warning: could not refresh the workspace from the app: {e}");
    }

    let ok = match sub.as_str() {
        "tracks" => match crate::commands::list_tracks(handle.clone()) {
            Ok(t) => {
                out(json!({"ok": true, "op": "tracks",
                           "tracks": serde_json::to_value(&t).unwrap_or(Value::Null)}));
                true
            }
            Err(e) => {
                out(json!({"ok": false, "op": "tracks", "error": e}));
                false
            }
        },
        // The resolved configuration this run WILL use — interpreter, repo root
        // and settings. Cheap to call before a multi-hour job, and it is the
        // honest answer to "is the scripted run the same as the GUI run".
        "settings" => {
            let s = crate::engine::load_settings(&handle);
            out(json!({"ok": true, "op": "settings",
                       "python": crate::engine::python_bin(&handle),
                       "repoRoot": crate::engine::repo_root(&handle).ok()
                                     .map(|p| p.to_string_lossy().to_string()),
                       "settings": serde_json::to_value(&s).unwrap_or(Value::Null)}));
            true
        }
        "generate" => generate(&handle, &kv, &flags),
        "install" => install(&handle, &kv),
        "batch" => batch(&handle, &kv, &flags),
        other => fail(format!("unknown headless command '{other}'")),
    };
    std::process::exit(if ok { 0 } else { 1})
}
