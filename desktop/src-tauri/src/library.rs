//! Song packs, game installation and per-song artwork.
//!
//! A StepMania "pack" is just a folder of song folders under the game's `Songs`
//! directory, so installing is a recursive copy to `<Songs>/<Pack>/<Song>/`.
//! Pack membership is app-side metadata (the engine has no concept of it) and
//! lives in `build/packs.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// `base` -> pack name. Songs with no entry are unfiled.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Packs {
    #[serde(default)]
    pub assignments: BTreeMap<String, String>,
}

impl Packs {
    /// Songs filed under `pack`, sorted for a stable install order.
    pub fn songs_in(&self, pack: &str) -> Vec<String> {
        let mut v: Vec<String> = self
            .assignments
            .iter()
            .filter(|(_, p)| p.as_str() == pack)
            .map(|(s, _)| s.clone())
            .collect();
        v.sort();
        v
    }
}

pub fn packs_path(root: &Path) -> PathBuf {
    root.join("build/packs.json")
}

pub fn load_packs(root: &Path) -> Packs {
    std::fs::read_to_string(packs_path(root))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

pub fn save_packs(root: &Path, p: &Packs) -> Result<(), String> {
    let path = packs_path(root);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, serde_json::to_string_pretty(p).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

/// The game's `Songs` directory. Mirrors `tools/chartbench/itg_paths.py` so the
/// app and the harness agree, and honours the same env override.
///
/// macOS:   ~/Library/Application Support/ITGmania/Songs
/// Windows: %APPDATA%\ITGmania\Songs
/// Linux:   $XDG_DATA_HOME/Songs, else ~/.itgmania/Songs
pub fn default_songs_dir() -> Option<PathBuf> {
    if let Ok(env) = std::env::var("BEATFORGE_ITG_SONGS") {
        if !env.trim().is_empty() {
            return Some(PathBuf::from(env));
        }
    }
    let home = home_dir()?;
    if cfg!(target_os = "macos") {
        return Some(home.join("Library/Application Support/ITGmania/Songs"));
    }
    if cfg!(target_os = "windows") {
        let base = std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join("AppData/Roaming"));
        return Some(base.join("ITGmania").join("Songs"));
    }
    let xdg = std::env::var("XDG_DATA_HOME").ok().map(PathBuf::from);
    Some(xdg.unwrap_or_else(|| home.join(".itgmania")).join("Songs"))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

/// Candidate install roots we know how to detect, for the Settings hint.
pub fn detect_song_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Some(home) = home_dir() else { return out };
    let candidates: Vec<PathBuf> = if cfg!(target_os = "macos") {
        vec![
            home.join("Library/Application Support/ITGmania/Songs"),
            home.join("Library/Application Support/StepMania 5/Songs"),
            home.join("Library/Application Support/StepMania 5.1/Songs"),
        ]
    } else if cfg!(target_os = "windows") {
        let base = std::env::var("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join("AppData/Roaming"));
        vec![
            base.join("ITGmania/Songs"),
            base.join("StepMania 5/Songs"),
            base.join("StepMania 5.1/Songs"),
        ]
    } else {
        let xdg = std::env::var("XDG_DATA_HOME").ok().map(PathBuf::from);
        let root = xdg.unwrap_or_else(|| home.clone());
        vec![
            home.join(".itgmania/Songs"),
            home.join(".stepmania-5.1/Songs"),
            root.join("itgmania/Songs"),
        ]
    };
    for c in candidates {
        if c.is_dir() {
            out.push(c);
        }
    }
    out
}

pub fn copy_dir_recursive(src: &Path, dest: &Path) -> std::io::Result<()> {
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

/// Rewrite (or insert) a `#TAG:value;` in a simfile.
///
/// STEPFORGE's serializer picks up `<base>-banner.png` / `<base>-bg.png` on its
/// next run, but patching the emitted file directly means artwork applies to the
/// charts you already have without paying for a regeneration.
pub fn set_simfile_tag(path: &Path, tag: &str, value: &str) -> Result<(), String> {
    let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let needle = format!("#{tag}:");
    let out = if let Some(i) = src.find(&needle) {
        let rest = &src[i + needle.len()..];
        let end = rest.find(';').ok_or("malformed simfile tag")?;
        format!("{}{}{}{}", &src[..i], needle, value, &rest[end..])
    } else {
        // Insert after the title line, or at the top if there isn't one.
        let anchor = src.find("#TITLE:").map(|i| src[i..].find(';').map(|j| i + j + 1).unwrap_or(0));
        let at = anchor.unwrap_or(0);
        format!("{}\n#{tag}:{value};{}", &src[..at], &src[at..])
    };
    std::fs::write(path, out).map_err(|e| e.to_string())
}

/// Every simfile in a song folder.
pub fn simfiles_in(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            match p.extension().and_then(|s| s.to_str()) {
                Some("ssc") | Some("sm") => out.push(p),
                _ => {}
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bf-lib-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn replaces_existing_tag() {
        let d = tmp("tag");
        let f = d.join("a.ssc");
        std::fs::write(&f, "#TITLE:x;\n#BANNER:old.png;\n#BPMS:0=120;\n").unwrap();
        set_simfile_tag(&f, "BANNER", "new.png").unwrap();
        let s = std::fs::read_to_string(&f).unwrap();
        assert!(s.contains("#BANNER:new.png;"));
        assert!(!s.contains("old.png"));
        assert!(s.contains("#BPMS:0=120;"), "other tags must survive");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn inserts_missing_tag_after_title() {
        let d = tmp("ins");
        let f = d.join("a.ssc");
        std::fs::write(&f, "#TITLE:x;\n#BPMS:0=120;\n").unwrap();
        set_simfile_tag(&f, "BACKGROUND", "bg.mp4").unwrap();
        let s = std::fs::read_to_string(&f).unwrap();
        assert!(s.contains("#BACKGROUND:bg.mp4;"));
        assert!(s.find("#TITLE:").unwrap() < s.find("#BACKGROUND:").unwrap());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn installs_into_pack_song_layout() {
        let d = tmp("install");
        // An emitted song folder…
        let src = d.join("build/stepmania/my_song");
        std::fs::create_dir_all(src.join("nested")).unwrap();
        std::fs::write(src.join("my_song.ssc"), "#TITLE:x;").unwrap();
        std::fs::write(src.join("my_song.ogg"), "audio").unwrap();
        std::fs::write(src.join("nested/easy.notefield.png"), "png").unwrap();

        // …copied to <Songs>/<Pack>/<Song>/ the way the game expects.
        let songs = d.join("Songs");
        let dest = songs.join("My Pack").join("my_song");
        copy_dir_recursive(&src, &dest).unwrap();

        assert!(dest.join("my_song.ssc").is_file());
        assert!(dest.join("my_song.ogg").is_file());
        assert!(dest.join("nested/easy.notefield.png").is_file(), "recursive copy");
        assert_eq!(
            std::fs::read_to_string(dest.join("my_song.ssc")).unwrap(),
            "#TITLE:x;"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn selects_only_songs_in_the_requested_pack() {
        let mut p = Packs::default();
        p.assignments.insert("zeta".into(), "Mix A".into());
        p.assignments.insert("alpha".into(), "Mix A".into());
        p.assignments.insert("beta".into(), "Mix B".into());

        // Sorted, and scoped to the requested pack only.
        assert_eq!(p.songs_in("Mix A"), vec!["alpha".to_string(), "zeta".to_string()]);
        assert_eq!(p.songs_in("Mix B"), vec!["beta".to_string()]);
        assert!(p.songs_in("Nonexistent").is_empty());
    }

    #[test]
    fn pack_assignments_round_trip() {
        let d = tmp("packs");
        let mut p = Packs::default();
        p.assignments.insert("song_a".into(), "My Pack".into());
        save_packs(&d, &p).unwrap();
        let back = load_packs(&d);
        assert_eq!(back.assignments.get("song_a").map(String::as_str), Some("My Pack"));
        // Missing file is an empty set, not an error.
        assert!(load_packs(&d.join("nope")).assignments.is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn songs_dir_is_platform_shaped() {
        // The env override always wins, on every platform.
        std::env::set_var("BEATFORGE_ITG_SONGS", "/tmp/custom-songs");
        assert_eq!(default_songs_dir().unwrap(), PathBuf::from("/tmp/custom-songs"));
        std::env::remove_var("BEATFORGE_ITG_SONGS");

        let d = default_songs_dir().expect("a default path on every platform");
        assert!(d.ends_with("Songs"), "should point at a Songs dir, got {d:?}");
    }
}
