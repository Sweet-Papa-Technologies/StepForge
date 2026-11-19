//! Installed-app layout: where the engine lives and where the user's workspace is.
//!
//! In a checkout the app just walks up to the repo. An *installed* app has no
//! repo beside it: the release builds bundle a read-only runtime as app
//! resources (`runtime/engine/` — `tools/`, `requirements.txt`, `models/` — and
//! `runtime/python/`, a self-contained Python with the engine's packages), and
//! on first run the app provisions a writable per-user workspace from it.
//!
//! The engine writes into `build/`, `assets/tracks/` and `public/` relative to
//! its root, so the root must be writable — which a system install location and
//! a signed `.app` bundle are not. Hence the split.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// The app's resource directory, recorded at startup (GUI and headless).
static RESOURCE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Record where the platform put the app's bundled resources.
pub fn set_resource_dir(dir: PathBuf) {
    let _ = RESOURCE_DIR.set(dir);
}

/// The runtime the release builds bundle (`<resources>/runtime`), if this is
/// one. Falls back to the platform layouts relative to the executable, so it is
/// found even before `set_resource_dir` has run.
pub fn bundled_runtime() -> Option<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Some(r) = RESOURCE_DIR.get() {
        cands.push(r.join("runtime"));
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        cands.push(dir.join("runtime")); // Windows: resources sit beside the exe
        cands.push(dir.join("../Resources/runtime")); // macOS .app
        cands.push(dir.join("../lib/StepForge/runtime")); // Linux .deb / AppImage
    }
    cands
        .into_iter()
        .find(|d| d.join("engine/tools/beatforge/__main__.py").is_file())
}

/// The interpreter of the bundled runtime, if this build has one.
pub fn bundled_python() -> Option<PathBuf> {
    let rt = bundled_runtime()?;
    let p = if cfg!(target_os = "windows") {
        rt.join("python/python.exe")
    } else {
        rt.join("python/bin/python3")
    };
    p.is_file().then_some(p)
}

fn home() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

/// Read-only engine payload locations: an explicit override, then the runtime
/// bundled with the app.
pub fn installed_engine_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(env) = std::env::var("BEATFORGE_ENGINE_DIR") {
        if !env.trim().is_empty() {
            v.push(PathBuf::from(env));
        }
    }
    if let Some(rt) = bundled_runtime() {
        v.push(rt.join("engine"));
    }
    v
}

/// The first installed engine payload that actually contains the engine.
pub fn find_installed_engine() -> Option<PathBuf> {
    installed_engine_dirs()
        .into_iter()
        .find(|d| d.join("tools/beatforge/__main__.py").is_file())
}

/// Writable per-user workspace: the engine's root when running installed.
pub fn workspace_dir() -> Option<PathBuf> {
    if let Ok(env) = std::env::var("BEATFORGE_WORKSPACE") {
        if !env.trim().is_empty() {
            return Some(PathBuf::from(env));
        }
    }
    let h = home()?;
    Some(if cfg!(target_os = "macos") {
        h.join("Library/Application Support/StepForge/workspace")
    } else if cfg!(target_os = "windows") {
        std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| h.join("AppData/Roaming"))
            .join("StepForge")
            .join("workspace")
    } else {
        std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| h.join(".local/share"))
            .join("StepForge")
            .join("workspace")
    })
}

/// The dedicated virtualenv the installers create for the engine's Python deps.
///
/// A venv (rather than `pip install --user`) is deliberate: Homebrew and Debian
/// Python are marked externally-managed (PEP 668) and refuse to install into
/// site-packages, so a shared venv is the only reliable target — and it keeps the
/// app's dependencies off the user's system Python entirely.
pub fn venv_dir() -> Option<PathBuf> {
    if let Ok(env) = std::env::var("BEATFORGE_VENV") {
        if !env.trim().is_empty() {
            return Some(PathBuf::from(env));
        }
    }
    let h = home()?;
    Some(if cfg!(target_os = "macos") {
        h.join("Library/Application Support/StepForge/venv")
    } else if cfg!(target_os = "windows") {
        std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| h.join("AppData/Roaming"))
            .join("StepForge")
            .join("venv")
    } else {
        std::env::var("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| h.join(".local/share"))
            .join("StepForge")
            .join("venv")
    })
}

/// The venv's interpreter, if the venv exists.
pub fn venv_python() -> Option<PathBuf> {
    let v = venv_dir()?;
    let p = if cfg!(target_os = "windows") {
        v.join("Scripts/python.exe")
    } else {
        v.join("bin/python3")
    };
    p.is_file().then_some(p)
}

/// Directories the engine expects to exist under its root.
const WORKSPACE_DIRS: [&str; 5] = [
    "assets/tracks",
    "public/assets/tracks",
    "public/maps",
    "build/analysis",
    "build/cost",
];

/// Copy the engine payload into `dest` and create the writable tree.
/// Idempotent: refreshes `tools/` so an app update ships a newer engine.
pub fn provision_workspace(engine_src: &Path, dest: &Path) -> Result<(), String> {
    if !engine_src.join("tools/beatforge/__main__.py").is_file() {
        return Err(format!("no engine payload at {}", engine_src.display()));
    }
    std::fs::create_dir_all(dest).map_err(|e| e.to_string())?;

    // Refresh the engine code (remove first so deleted modules don't linger).
    let tools_dest = dest.join("tools");
    if tools_dest.exists() {
        let _ = std::fs::remove_dir_all(&tools_dest);
    }
    copy_dir(&engine_src.join("tools"), &tools_dest).map_err(|e| e.to_string())?;

    for extra in ["requirements.txt", "package.json", ENGINE_VERSION] {
        let from = engine_src.join(extra);
        if from.is_file() {
            let _ = std::fs::copy(&from, dest.join(extra));
        }
    }
    // The neural weights, when the payload carries them. Best-effort like the
    // extras above: the LLM and deterministic engines do not need them, and the
    // startup refresh retries a copy that did not land.
    let _ = refresh_models(engine_src, dest);
    copy_other_models(engine_src, dest);
    for d in WORKSPACE_DIRS {
        std::fs::create_dir_all(dest.join(d)).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn copy_dir(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let from = e.path();
        // Skip caches and test fixtures — they bloat the copy and aren't used.
        if let Some(name) = from.file_name().and_then(|n| n.to_str()) {
            if name == "__pycache__" || name == ".pytest_cache" {
                continue;
            }
        }
        let to = dest.join(e.file_name());
        if from.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// The neural chart engine's weights, relative to an engine root or payload.
/// This is the engine's own default model directory (`models/stepforge_flow`).
const MODEL_DIR: &str = "models/stepforge_flow";
/// The file whose presence means "a model is installed" (the engine's rule),
/// and whose bytes identify WHICH model: the export writes it per build.
const MODEL_META: &str = "model_meta.json";

/// Serialises model refreshes: startup, first-run provisioning and the status
/// check can all get here at once, and they share one staging directory.
static MODELS_LOCK: Mutex<()> = Mutex::new(());

/// Should the model in `bundled` replace the one in `installed`?
///
/// Yes when the payload ships a model and the workspace either has none or has
/// a different one. `model_meta.json` is compared byte for byte — it is tiny,
/// and hashing ~100 MB of weights on every launch to learn the same thing is
/// not worth it. No bundled model means there is nothing to copy, whatever the
/// workspace holds.
pub fn models_refresh_needed(bundled: &Path, installed: &Path) -> bool {
    let Ok(new) = std::fs::read(bundled.join(MODEL_META)) else {
        return false;
    };
    match std::fs::read(installed.join(MODEL_META)) {
        Ok(current) => current != new,
        Err(_) => true,
    }
}

/// Bring `dest`'s neural model up to date with the one in the engine payload.
/// Returns whether anything was copied.
///
/// The copy is staged beside the target and swapped in by rename, so a run that
/// starts mid-copy sees the old model or the new one — never `model_meta.json`
/// next to half an `.onnx` file. Replacing the directory (not merging into it)
/// also drops weight files the new build no longer ships.
pub fn refresh_models(engine_src: &Path, dest: &Path) -> Result<bool, String> {
    let from = engine_src.join(MODEL_DIR);
    let to = dest.join(MODEL_DIR);
    let _guard = MODELS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    if !models_refresh_needed(&from, &to) {
        return Ok(false);
    }
    let parent = to.parent().ok_or("model directory has no parent")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let staging = parent.join(".stepforge_flow.incoming");
    let retired = parent.join(".stepforge_flow.old");
    // Leftovers from a run that was killed mid-refresh.
    let _ = std::fs::remove_dir_all(&staging);
    let _ = std::fs::remove_dir_all(&retired);

    if let Err(e) = copy_dir(&from, &staging) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(format!("could not copy the bundled neural model: {e}"));
    }
    if to.exists() {
        std::fs::rename(&to, &retired).map_err(|e| e.to_string())?;
    }
    if let Err(e) = std::fs::rename(&staging, &to) {
        // Put the previous model back rather than leave the workspace with none.
        let _ = std::fs::rename(&retired, &to);
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e.to_string());
    }
    let _ = std::fs::remove_dir_all(&retired);
    Ok(true)
}

/// Anything else the payload ships under `models/`. `stepforge_flow` is left to
/// `refresh_models`, which replaces it atomically.
fn copy_other_models(engine_src: &Path, dest: &Path) {
    let Ok(rd) = std::fs::read_dir(engine_src.join("models")) else {
        return;
    };
    for e in rd.flatten() {
        let from = e.path();
        if from.file_name().and_then(|n| n.to_str()) == Some("stepforge_flow") {
            continue;
        }
        let to = dest.join("models").join(e.file_name());
        if from.is_dir() {
            let _ = copy_dir(&from, &to);
        } else if std::fs::create_dir_all(dest.join("models")).is_ok() {
            let _ = std::fs::copy(&from, &to);
        }
    }
}

/// Written into the bundled engine payload at build time (app version + commit);
/// copied into the workspace with the engine, so a mismatch means the app was
/// updated and the workspace's engine code is stale.
const ENGINE_VERSION: &str = "ENGINE_VERSION";

/// Does the workspace's engine differ from the payload's? Only when the payload
/// carries a version stamp: an unstamped payload (a dev build) never forces a
/// refresh.
pub fn engine_refresh_needed(payload: &Path, ws: &Path) -> bool {
    let Ok(new) = std::fs::read(payload.join(ENGINE_VERSION)) else {
        return false;
    };
    std::fs::read(ws.join(ENGINE_VERSION)).map_or(true, |cur| cur != new)
}

/// Bring an existing workspace up to date with the installed app: the whole
/// engine (code, requirements, model) after an app update, otherwise just the
/// model when it differs. User data (`assets/`, `public/`, `build/`) is never
/// touched. Returns whether anything was copied.
pub fn refresh_workspace() -> Result<bool, String> {
    let Some(ws) = workspace_dir() else {
        return Ok(false);
    };
    if !ws.join("tools/beatforge/__main__.py").is_file() {
        return Ok(false);
    }
    let Some(engine) = find_installed_engine() else {
        return Ok(false);
    };
    if engine_refresh_needed(&engine, &ws) {
        provision_workspace(&engine, &ws)?;
        return Ok(true);
    }
    refresh_models(&engine, &ws)
}

/// "Has the startup refresh finished?" — `false` only between
/// `begin_startup_refresh` and the end of the refresh it starts.
pub struct RefreshGate {
    running: Mutex<bool>,
    done: std::sync::Condvar,
}

impl RefreshGate {
    pub const fn new() -> Self {
        Self { running: Mutex::new(false), done: std::sync::Condvar::new() }
    }

    pub fn begin(&self) {
        *self.running.lock().unwrap_or_else(|p| p.into_inner()) = true;
    }

    pub fn finish(&self) {
        *self.running.lock().unwrap_or_else(|p| p.into_inner()) = false;
        self.done.notify_all();
    }

    /// Block while a refresh is running. Returns at once when none is (or none
    /// was ever started: tests, the headless path, a development build).
    pub fn wait(&self) {
        let mut running = self.running.lock().unwrap_or_else(|p| p.into_inner());
        while *running {
            running = self.done.wait(running).unwrap_or_else(|p| p.into_inner());
        }
    }
}

static STARTUP_REFRESH: RefreshGate = RefreshGate::new();

/// Bring the workspace up to date with the installed app, off the calling
/// thread (after an update it can be ~150 MB to copy). Until it finishes,
/// `wait_for_startup_refresh` holds back everything that would run the engine.
///
/// Without that, the first status checks of a freshly updated app raced the
/// copy: `tools/` is removed and rewritten, and a Python started in that window
/// failed with "No module named beatforge" — which the app then reported as
/// "the installed engine is too old", until the next restart.
pub fn begin_startup_refresh() {
    STARTUP_REFRESH.begin();
    std::thread::spawn(|| {
        let _ = refresh_workspace();
        STARTUP_REFRESH.finish();
    });
}

pub fn wait_for_startup_refresh() {
    STARTUP_REFRESH.wait();
}

/// Give an EXISTING workspace the neural model the installed app ships.
///
/// `ensure_workspace` provisions a workspace once and then leaves it alone, so
/// without this an app update could never deliver weights (or newer weights) to
/// anyone who had already run the app. Called at startup and before the neural
/// status check; cheap when there is nothing to do (two small file reads).
///
/// Only the per-user workspace is touched, and only its model directory: a
/// checkout, or an engine root the user pointed the app at, is theirs.
pub fn refresh_workspace_models() -> Result<bool, String> {
    let Some(ws) = workspace_dir() else {
        return Ok(false);
    };
    // Not provisioned yet: first-run provisioning copies the model itself.
    if !ws.join("tools/beatforge/__main__.py").is_file() {
        return Ok(false);
    }
    let Some(engine) = find_installed_engine() else {
        return Ok(false);
    };
    refresh_models(&engine, &ws)
}

/// Resolve the engine root for an installed app, provisioning on first run.
pub fn ensure_workspace() -> Option<PathBuf> {
    let ws = workspace_dir()?;
    if ws.join("tools/beatforge/__main__.py").is_file() {
        return Some(ws);
    }
    let engine = find_installed_engine()?;
    match provision_workspace(&engine, &ws) {
        Ok(()) => Some(ws),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bf-inst-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn fake_engine(root: &Path) {
        std::fs::create_dir_all(root.join("tools/beatforge/adapters")).unwrap();
        std::fs::create_dir_all(root.join("tools/beatforge/__pycache__")).unwrap();
        std::fs::write(root.join("tools/beatforge/__main__.py"), "# engine").unwrap();
        std::fs::write(root.join("tools/beatforge/adapters/base.py"), "# adapter").unwrap();
        std::fs::write(root.join("tools/beatforge/__pycache__/x.pyc"), "junk").unwrap();
        std::fs::write(root.join("requirements.txt"), "numpy\n").unwrap();
    }

    #[test]
    fn provisions_a_writable_workspace_from_the_payload() {
        let d = tmp("prov");
        let engine = d.join("engine");
        let ws = d.join("workspace");
        fake_engine(&engine);

        provision_workspace(&engine, &ws).unwrap();

        assert!(ws.join("tools/beatforge/__main__.py").is_file(), "engine copied");
        assert!(ws.join("tools/beatforge/adapters/base.py").is_file(), "recursive");
        assert!(ws.join("requirements.txt").is_file(), "requirements copied");
        assert!(!ws.join("tools/beatforge/__pycache__").exists(), "caches skipped");
        for sub in WORKSPACE_DIRS {
            assert!(ws.join(sub).is_dir(), "missing writable dir {sub}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn reprovision_refreshes_engine_but_keeps_user_data() {
        let d = tmp("refresh");
        let engine = d.join("engine");
        let ws = d.join("workspace");
        fake_engine(&engine);
        provision_workspace(&engine, &ws).unwrap();

        // User data lands in the writable tree…
        std::fs::write(ws.join("assets/tracks/song.ogg"), "audio").unwrap();
        // …and a stale engine module exists from the old version.
        std::fs::write(ws.join("tools/beatforge/removed.py"), "old").unwrap();

        provision_workspace(&engine, &ws).unwrap();

        assert!(ws.join("assets/tracks/song.ogg").is_file(), "user data preserved");
        assert!(!ws.join("tools/beatforge/removed.py").exists(), "stale module dropped");
        assert!(ws.join("tools/beatforge/__main__.py").is_file());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The engine must not be started while the workspace's code is being
    /// replaced: callers wait for the refresh that began before them.
    #[test]
    fn engine_calls_wait_for_a_refresh_in_progress() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let gate = Arc::new(RefreshGate::new());
        gate.wait(); // nothing running: returns at once

        gate.begin();
        let refreshed = Arc::new(AtomicBool::new(false));
        let (g, r) = (gate.clone(), refreshed.clone());
        let caller = std::thread::spawn(move || {
            g.wait();
            r.load(Ordering::SeqCst) // what the engine call would have seen
        });
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert!(!caller.is_finished(), "the caller ran before the refresh finished");
        refreshed.store(true, Ordering::SeqCst);
        gate.finish();
        assert!(caller.join().unwrap(), "the caller saw the refreshed workspace");
        gate.wait(); // and nothing is held afterwards
    }

    #[test]
    fn rejects_a_payload_without_the_engine() {
        let d = tmp("bad");
        let ws = d.join("ws");
        assert!(provision_workspace(&d.join("nothing"), &ws).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn workspace_and_engine_paths_are_platform_shaped() {
        std::env::set_var("BEATFORGE_WORKSPACE", "/tmp/ws-override");
        assert_eq!(workspace_dir().unwrap(), PathBuf::from("/tmp/ws-override"));
        std::env::remove_var("BEATFORGE_WORKSPACE");

        assert!(workspace_dir().unwrap().ends_with("StepForge/workspace"));
        // An explicit override comes first; the bundled runtime (if any) after.
        std::env::set_var("BEATFORGE_ENGINE_DIR", "/opt/sf/engine");
        assert_eq!(installed_engine_dirs()[0], PathBuf::from("/opt/sf/engine"));
        assert!(installed_engine_dirs().iter().all(|p| p.ends_with("engine")));
        std::env::remove_var("BEATFORGE_ENGINE_DIR");
    }

    #[test]
    fn the_bundled_runtime_is_found_from_the_resource_dir() {
        let res = tmp("resources");
        let rt = res.join("runtime");
        fake_engine(&rt.join("engine"));
        let py = if cfg!(target_os = "windows") { "python/python.exe" } else { "python/bin/python3" };
        std::fs::create_dir_all(rt.join(py).parent().unwrap()).unwrap();
        std::fs::write(rt.join(py), "").unwrap();
        set_resource_dir(res.clone());
        assert_eq!(bundled_runtime(), Some(rt.clone()));
        assert_eq!(bundled_python(), Some(rt.join(py)));
        assert_eq!(find_installed_engine(), Some(rt.join("engine")));
    }

    /// A fake exported model: the meta file plus weights of a given content.
    fn fake_model(root: &Path, meta: &str, weights: &str) {
        let dir = root.join(MODEL_DIR);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(MODEL_META), meta).unwrap();
        std::fs::write(dir.join("stage_a_rhythm.onnx"), weights).unwrap();
        std::fs::write(dir.join("stage_b_choreo.onnx"), weights).unwrap();
    }

    /// The decision the startup refresh turns on: copy when the workspace has
    /// no model or a different one, and only when there is one to copy.
    #[test]
    fn a_bundled_model_replaces_a_missing_or_different_one() {
        let d = tmp("refresh-decision");
        let engine = d.join("engine");
        let ws = d.join("workspace");
        let bundled = engine.join(MODEL_DIR);
        let installed = ws.join(MODEL_DIR);

        // No bundled model: nothing to do, whatever the workspace holds.
        assert!(!models_refresh_needed(&bundled, &installed));
        fake_model(&ws, r#"{"v":1}"#, "user");
        assert!(!models_refresh_needed(&bundled, &installed), "nothing bundled");
        std::fs::remove_dir_all(ws.join("models")).unwrap();

        // Bundled, workspace has none -> copy.
        fake_model(&engine, r#"{"v":2}"#, "new");
        assert!(models_refresh_needed(&bundled, &installed), "missing");

        // A model directory without the meta file is not an installed model.
        std::fs::create_dir_all(&installed).unwrap();
        std::fs::write(installed.join("stage_a_rhythm.onnx"), "orphan").unwrap();
        assert!(models_refresh_needed(&bundled, &installed), "no meta");

        // Identical meta -> leave it alone.
        fake_model(&ws, r#"{"v":2}"#, "new");
        assert!(!models_refresh_needed(&bundled, &installed), "identical");

        // Different meta (even by whitespace: it is a byte comparison) -> copy.
        fake_model(&ws, r#"{"v":1}"#, "old");
        assert!(models_refresh_needed(&bundled, &installed), "different");
        fake_model(&ws, "{\"v\":2}\n", "new");
        assert!(models_refresh_needed(&bundled, &installed), "bytes, not json");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// First run: the weights arrive with the engine.
    #[test]
    fn provisioning_carries_the_bundled_model() {
        let d = tmp("prov-models");
        let engine = d.join("engine");
        let ws = d.join("workspace");
        fake_engine(&engine);
        fake_model(&engine, r#"{"v":1}"#, "weights");
        std::fs::create_dir_all(engine.join("models/other")).unwrap();
        std::fs::write(engine.join("models/other/x.bin"), "x").unwrap();

        provision_workspace(&engine, &ws).unwrap();

        let m = ws.join(MODEL_DIR);
        assert_eq!(std::fs::read_to_string(m.join(MODEL_META)).unwrap(), r#"{"v":1}"#);
        assert!(m.join("stage_a_rhythm.onnx").is_file());
        assert!(m.join("stage_b_choreo.onnx").is_file());
        assert!(ws.join("models/other/x.bin").is_file(), "the rest of models/ too");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A payload without weights (a build made before they existed) provisions
    /// exactly as it always did.
    #[test]
    fn provisioning_without_a_model_still_works() {
        let d = tmp("prov-nomodels");
        let engine = d.join("engine");
        let ws = d.join("workspace");
        fake_engine(&engine);
        provision_workspace(&engine, &ws).unwrap();
        assert!(ws.join("tools/beatforge/__main__.py").is_file());
        assert!(!ws.join("models").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// An existing workspace picks up a newer bundled model — and ONLY the
    /// model: the user's engine code and data are not touched by this path.
    #[test]
    fn an_existing_workspace_gets_the_updated_model_and_nothing_else() {
        let d = tmp("refresh-models");
        let engine = d.join("engine");
        let ws = d.join("workspace");
        fake_engine(&engine);
        provision_workspace(&engine, &ws).unwrap(); // an install from before the weights
        std::fs::write(ws.join("tools/beatforge/__main__.py"), "# user-patched").unwrap();
        std::fs::write(ws.join("assets/tracks/song.ogg"), "audio").unwrap();

        // The app updates and now ships a model.
        fake_model(&engine, r#"{"v":1}"#, "v1");
        assert_eq!(refresh_models(&engine, &ws), Ok(true));
        let m = ws.join(MODEL_DIR);
        assert_eq!(std::fs::read_to_string(m.join("stage_a_rhythm.onnx")).unwrap(), "v1");
        assert_eq!(refresh_models(&engine, &ws), Ok(false), "second launch: nothing to do");

        // A later update ships different weights and drops the int8 variant.
        std::fs::write(m.join("stage_a_rhythm.int8.onnx"), "stale").unwrap();
        fake_model(&engine, r#"{"v":2}"#, "v2");
        assert_eq!(refresh_models(&engine, &ws), Ok(true));
        assert_eq!(std::fs::read_to_string(m.join(MODEL_META)).unwrap(), r#"{"v":2}"#);
        assert_eq!(std::fs::read_to_string(m.join("stage_b_choreo.onnx")).unwrap(), "v2");
        assert!(!m.join("stage_a_rhythm.int8.onnx").exists(), "replaced, not merged");
        assert!(!ws.join("models/.stepforge_flow.incoming").exists(), "staging cleaned up");
        assert!(!ws.join("models/.stepforge_flow.old").exists());

        assert_eq!(
            std::fs::read_to_string(ws.join("tools/beatforge/__main__.py")).unwrap(),
            "# user-patched",
            "tools/ is not refreshed by the model path"
        );
        assert!(ws.join("assets/tracks/song.ogg").is_file());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A refresh that was killed mid-copy leaves a staging directory behind;
    /// the next one must clear it rather than trip over it or ship its contents.
    #[test]
    fn a_stale_staging_directory_does_not_block_or_leak_into_the_refresh() {
        let d = tmp("refresh-stale");
        let engine = d.join("engine");
        let ws = d.join("workspace");
        fake_model(&engine, r#"{"v":3}"#, "v3");
        let staging = ws.join("models/.stepforge_flow.incoming");
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("half.onnx"), "truncated").unwrap();

        assert_eq!(refresh_models(&engine, &ws), Ok(true));
        let m = ws.join(MODEL_DIR);
        assert!(m.join(MODEL_META).is_file());
        assert!(!m.join("half.onnx").exists());
        assert!(!staging.exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn an_app_update_refreshes_the_workspace_engine_but_not_user_data() {
        let engine = tmp("upd-engine");
        let ws = tmp("upd-ws");
        fake_engine(&engine);
        std::fs::write(engine.join(ENGINE_VERSION), "0.1.0 abc").unwrap();
        assert!(!engine_refresh_needed(&tmp("no-stamp"), &ws), "unstamped payload: no refresh");
        provision_workspace(&engine, &ws).unwrap();
        assert!(!engine_refresh_needed(&engine, &ws), "same stamp after provisioning");
        std::fs::write(ws.join("assets/tracks/mine.ogg"), "user data").unwrap();

        std::fs::write(engine.join(ENGINE_VERSION), "0.2.0 def").unwrap();
        std::fs::write(engine.join("tools/beatforge/__main__.py"), "# engine v2").unwrap();
        assert!(engine_refresh_needed(&engine, &ws));
        provision_workspace(&engine, &ws).unwrap();
        assert_eq!(std::fs::read_to_string(ws.join("tools/beatforge/__main__.py")).unwrap(), "# engine v2");
        assert_eq!(std::fs::read_to_string(ws.join(ENGINE_VERSION)).unwrap(), "0.2.0 def");
        assert!(ws.join("assets/tracks/mine.ogg").is_file(), "user data kept");
    }

}
