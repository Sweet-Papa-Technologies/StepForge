mod commands;
mod engine;
mod headless;
mod install;
mod library;
mod simfile;

use engine::JobRegistry;
use tauri::Manager;

/// WebKitGTK settings applied unless the user's environment already sets them.
///
/// WebKitGTK's DMABUF renderer leaves the window blank or black on many Linux
/// setups (NVIDIA's driver especially, and AppImages running on newer distros
/// than they were built on). Turning it off costs the faster compositing path,
/// which this UI does not need: it draws with 2D canvas, never WebGL. A user can
/// still opt back in with WEBKIT_DISABLE_DMABUF_RENDERER=0.
#[cfg(any(test, target_os = "linux"))]
fn webkit_env_defaults(is_set: impl Fn(&str) -> bool) -> Vec<(&'static str, &'static str)> {
    [("WEBKIT_DISABLE_DMABUF_RENDERER", "1")]
        .into_iter()
        .filter(|(k, _)| !is_set(k))
        .collect()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Before anything starts a thread or initialises GTK.
    #[cfg(target_os = "linux")]
    for (k, v) in webkit_env_defaults(|k| std::env::var_os(k).is_some()) {
        std::env::set_var(k, v);
    }
    // Scripted mode is decided BEFORE the builder runs, so a headless call never
    // creates a window and never needs an event loop. It builds its own handle
    // from the same context, so it resolves settings exactly as the GUI does.
    let ctx = tauri::generate_context!();
    if headless::requested() {
        headless::main(ctx);
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .manage(JobRegistry::default())
        .setup(|app| {
            if let Ok(dir) = app.path().resource_dir() {
                install::set_resource_dir(dir);
            }
            // An update may ship a newer engine or model than the existing
            // workspace has. Off the main thread: it can be ~150 MB to copy.
            std::thread::spawn(|| {
                let _ = install::refresh_workspace();
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_tracks,
            commands::import_track,
            commands::convert_track,
            commands::get_analysis,
            commands::track_audio_path,
            commands::difficulties_for,
            commands::default_difficulties_for,
            commands::analyze_track,
            commands::generate_charts,
            commands::inpaint_chart,
            commands::cancel_job,
            commands::list_results,
            commands::get_result,
            commands::get_chart_notes,
            commands::gemini_models,
            commands::llm_modes,
            commands::chunk_modes,
            commands::cost_report,
            commands::cost_by_song,
            commands::refresh_cost_report,
            commands::list_packs,
            commands::set_song_pack,
            commands::install_to_game,
            commands::set_song_asset,
            commands::get_song_assets,
            commands::generate_song_asset,
            commands::attach_generated_asset,
            commands::engine_status,
            commands::get_settings,
            commands::save_settings,
            commands::inkling_check,
            commands::inkling_probe,
            commands::argon_check,
            commands::neural_status,
            commands::register_track_meta,
            commands::export_song_folder,
        ])
        .run(ctx)
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::webkit_env_defaults;

    #[test]
    fn the_dmabuf_renderer_is_off_unless_the_user_chose() {
        assert_eq!(
            webkit_env_defaults(|_| false),
            vec![("WEBKIT_DISABLE_DMABUF_RENDERER", "1")]
        );
        assert!(webkit_env_defaults(|k| k == "WEBKIT_DISABLE_DMABUF_RENDERER").is_empty());
    }
}
