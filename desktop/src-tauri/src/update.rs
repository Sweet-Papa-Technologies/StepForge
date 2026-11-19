//! The update check: is there a newer StepForge than the one running?
//!
//! The request itself is made by the web view (`src/lib/updateCheck.ts`): one
//! anonymous GET of the repository's public release list. It carries nothing
//! about the user, not even the version they have. The answer is handed to
//! this module, which decides — on this computer — whether any release in it is
//! newer than the running app. Nothing is downloaded or installed: the notice
//! only links to the release page.
//!
//! Everything here is pure, so it is tested without a network.

use std::cmp::Ordering;

use semver::Version;
use serde::Serialize;
use serde_json::Value;
use tauri::AppHandle;

/// The public repository releases are published in.
pub const REPO: &str = "Sweet-Papa-Technologies/StepForge";

/// The one URL the update check requests. `/releases/latest` would be shorter,
/// but it skips pre-releases, and every beta is one. `per_page` only keeps the
/// answer small; it says nothing about the user.
///
/// The web view makes the request, so this constant is the reference a test
/// holds `updateCheck.ts` to.
#[cfg_attr(not(test), allow(dead_code))]
pub const RELEASES_API_URL: &str =
    "https://api.github.com/repos/Sweet-Papa-Technologies/StepForge/releases?per_page=10";

/// Release notes are shown as plain text in a small box; a novel is cut.
const NOTES_MAX_CHARS: usize = 6000;
const NAME_MAX_CHARS: usize = 120;

/// A published release, reduced to what the notice shows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Release {
    /// "0.1.0-beta.7": the tag without its leading `v`.
    pub version: String,
    pub tag: String,
    /// The release's title, or the tag when it has none.
    pub name: String,
    /// The release notes as written (Markdown source). Shown as plain text.
    pub notes: String,
    /// The release page. Built here from the tag, never taken from the answer.
    pub url: String,
    pub prerelease: bool,
}

/// What the notice needs: the newer release and the version it replaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateNotice {
    #[serde(flatten)]
    pub release: Release,
    /// The running app's version.
    pub current: String,
}

/// A version from a tag or a version string: "v0.1.0-beta.6", "0.1.0".
/// `None` for anything that is not semver ("nightly", "", "v1").
pub fn parse_version(text: &str) -> Option<Version> {
    let t = text.trim();
    let t = t.strip_prefix(['v', 'V']).unwrap_or(t);
    Version::parse(t).ok()
}

/// Semver precedence: `0.1.0-beta.9 < 0.1.0-beta.10 < 0.1.0-rc.1 < 0.1.0`.
/// Build metadata (`+…`) does not count.
pub fn compare(a: &Version, b: &Version) -> Ordering {
    a.cmp_precedence(b)
}

/// `candidate` is a later version than `current`. False when either is not a
/// version at all: an unreadable answer must never produce a notice.
pub fn is_newer(candidate: &str, current: &str) -> bool {
    match (parse_version(candidate), parse_version(current)) {
        (Some(c), Some(cur)) => compare(&c, &cur) == Ordering::Greater,
        _ => false,
    }
}

/// Text from the network, made safe to show: no control characters, bounded.
fn clean_text(raw: &str, max_chars: usize) -> String {
    let text: String = raw
        .replace("\r\n", "\n")
        .chars()
        .map(|c| if c == '\r' { '\n' } else { c })
        .filter(|c| *c == '\n' || *c == '\t' || !c.is_control())
        .collect();
    let text = text.trim();
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let cut: String = text.chars().take(max_chars).collect();
    format!("{}…", cut.trim_end())
}

/// The release page for a tag. The tag has already parsed as a version, so it
/// holds only letters, digits, `.`, `-` and `+`; `+` is escaped for the path.
fn release_url(tag: &str) -> String {
    format!("https://github.com/{REPO}/releases/tag/{}", tag.replace('+', "%2B"))
}

/// One entry of the release list, if it is a published release with a version
/// for a tag. Drafts and anything malformed are `None`.
fn release_from(entry: &Value) -> Option<(Version, Release)> {
    let obj = entry.as_object()?;
    if obj.get("draft").and_then(Value::as_bool).unwrap_or(false) {
        return None;
    }
    let tag = obj.get("tag_name")?.as_str()?.trim();
    let version = parse_version(tag)?;
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .map(|n| clean_text(n, NAME_MAX_CHARS).replace('\n', " "))
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| tag.to_string());
    let notes = obj
        .get("body")
        .and_then(Value::as_str)
        .map(|b| clean_text(b, NOTES_MAX_CHARS))
        .unwrap_or_default();
    let release = Release {
        version: version.to_string(),
        tag: tag.to_string(),
        name,
        notes,
        url: release_url(tag),
        prerelease: obj.get("prerelease").and_then(Value::as_bool).unwrap_or(false),
    };
    Some((version, release))
}

/// The highest-versioned published release in a GitHub "list releases" answer.
///
/// Pre-releases count (every beta is one); drafts do not. Entries that are not
/// releases, or whose tag is not a version, are skipped. `None` when the text
/// is not a release list at all — an error page, a rate-limit message.
pub fn newest_release(body: &str) -> Option<Release> {
    let parsed: Value = serde_json::from_str(body).ok()?;
    let mut best: Option<(Version, Release)> = None;
    for found in parsed.as_array()?.iter().filter_map(release_from) {
        // On a tie the first one listed stays (the list is newest first).
        if best.as_ref().is_none_or(|b| compare(&found.0, &b.0) == Ordering::Greater) {
            best = Some(found);
        }
    }
    best.map(|(_, release)| release)
}

/// The release to tell the user about, if any: the newest one in `body`, when
/// it is later than `current` and later than the version they dismissed.
pub fn available_update(body: &str, current: &str, dismissed: &str) -> Option<Release> {
    let release = newest_release(body)?;
    let offered = is_newer(&release.version, current)
        // A dismissal that is not a version (blank: none yet) hides nothing.
        && (parse_version(dismissed).is_none() || is_newer(&release.version, dismissed));
    offered.then_some(release)
}

/// Read the release list the web view fetched. `None` means "nothing to show":
/// up to date, already dismissed, or an answer that could not be read.
#[tauri::command(async)]
pub fn update_from_releases(app: AppHandle, body: String, dismissed: Option<String>) -> Option<UpdateNotice> {
    let current = app.package_info().version.to_string();
    available_update(&body, &current, dismissed.as_deref().unwrap_or(""))
        .map(|release| UpdateNotice { release, current })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(tag: &str) -> Value {
        json!({
            "tag_name": tag, "name": format!("StepForge {tag}"), "draft": false,
            "prerelease": tag.contains('-'), "body": format!("Notes for {tag}"),
            "html_url": format!("https://github.com/{REPO}/releases/tag/{tag}"),
            "published_at": "2026-01-01T00:00:00Z",
        })
    }

    fn list(entries: &[Value]) -> String {
        Value::Array(entries.to_vec()).to_string()
    }

    #[test]
    fn versions_are_ordered_as_semver_orders_them() {
        // Each is older than the next. A plain string compare gets the first
        // pair wrong ("beta.9" > "beta.10").
        let chain = [
            "0.1.0-beta.1",
            "0.1.0-beta.9",
            "0.1.0-beta.10",
            "0.1.0-rc.1",
            "0.1.0",
            "0.1.1-beta.1",
            "0.1.1",
            "0.2.0",
            "0.10.0",
            "1.0.0",
        ];
        for (i, older) in chain.iter().enumerate() {
            for newer in &chain[i + 1..] {
                assert!(is_newer(newer, older), "{newer} should be newer than {older}");
                assert!(!is_newer(older, newer), "{older} should not be newer than {newer}");
            }
            assert!(!is_newer(older, older), "{older} is not newer than itself");
        }
    }

    #[test]
    fn tags_and_versions_compare_alike() {
        assert!(is_newer("v0.1.0-beta.7", "0.1.0-beta.6"));
        assert!(is_newer("V0.1.0", "v0.1.0-rc.1"));
        assert!(!is_newer("v0.1.0-beta.6", "0.1.0-beta.6"));
        // Build metadata is not a newer version.
        assert!(!is_newer("0.1.0+build.5", "0.1.0+build.4"));
    }

    #[test]
    fn something_that_is_not_a_version_is_never_newer() {
        for bad in ["", "nightly", "v1", "1.2", "latest", "0.1.0-", "v0.1.0 beta"] {
            assert_eq!(parse_version(bad), None, "{bad:?}");
            assert!(!is_newer(bad, "0.1.0"));
            assert!(!is_newer("9.9.9", bad));
        }
    }

    #[test]
    fn the_newest_release_is_the_highest_version_whatever_the_order() {
        let body = list(&[
            entry("v0.1.0-beta.5"),
            entry("v0.1.0-beta.10"),
            entry("v0.1.0-beta.9"),
        ]);
        let r = newest_release(&body).unwrap();
        assert_eq!(r.version, "0.1.0-beta.10");
        assert_eq!(r.tag, "v0.1.0-beta.10");
        assert_eq!(r.name, "StepForge v0.1.0-beta.10");
        assert_eq!(r.notes, "Notes for v0.1.0-beta.10");
        assert!(r.prerelease, "pre-releases are offered: every beta is one");
    }

    #[test]
    fn drafts_are_ignored() {
        let mut draft = entry("v0.2.0");
        draft["draft"] = json!(true);
        let body = list(&[draft, entry("v0.1.0-beta.6")]);
        assert_eq!(newest_release(&body).unwrap().version, "0.1.0-beta.6");
    }

    #[test]
    fn a_full_release_beats_its_own_release_candidates() {
        let body = list(&[entry("v0.1.0-rc.2"), entry("v0.1.0"), entry("v0.1.0-beta.12")]);
        let r = newest_release(&body).unwrap();
        assert_eq!(r.version, "0.1.0");
        assert!(!r.prerelease);
    }

    #[test]
    fn malformed_entries_are_skipped_not_fatal() {
        let body = list(&[
            json!(null),
            json!(42),
            json!("v9.9.9"),
            json!([]),
            json!({}),
            json!({"tag_name": null}),
            json!({"tag_name": 7}),
            json!({"tag_name": "nightly"}),
            json!({"tag_name": "v9.9", "draft": false}),
            // No name, notes or flags at all: still a release.
            json!({"tag_name": "v0.1.0-beta.6"}),
            json!({"tag_name": "v0.1.0-beta.4", "name": null, "body": null, "draft": null}),
        ]);
        let r = newest_release(&body).unwrap();
        assert_eq!(r.version, "0.1.0-beta.6");
        assert_eq!(r.name, "v0.1.0-beta.6", "the tag stands in for a missing title");
        assert_eq!(r.notes, "");
    }

    #[test]
    fn an_answer_that_is_not_a_release_list_is_no_release() {
        for body in [
            "",
            "not json",
            "<html><body>502 Bad Gateway</body></html>",
            "[]",
            "null",
            r#"{"message":"API rate limit exceeded","documentation_url":"https://docs.github.com/"}"#,
            r#"{"tag_name":"v9.9.9"}"#,
            r#"[{"tag_name":"nightly"}]"#,
        ] {
            assert_eq!(newest_release(body), None, "{body:?}");
            assert_eq!(available_update(body, "0.1.0-beta.1", ""), None, "{body:?}");
        }
    }

    #[test]
    fn the_release_page_is_built_from_the_tag_not_taken_from_the_answer() {
        let mut e = entry("v0.1.0-beta.7");
        e["html_url"] = json!("https://example.invalid/not-the-release-page");
        let r = newest_release(&list(&[e])).unwrap();
        assert_eq!(
            r.url,
            "https://github.com/Sweet-Papa-Technologies/StepForge/releases/tag/v0.1.0-beta.7"
        );
        let r = newest_release(&list(&[entry("v0.2.0+build.3")])).unwrap();
        assert!(r.url.ends_with("/releases/tag/v0.2.0%2Bbuild.3"), "{}", r.url);
    }

    #[test]
    fn release_notes_are_plain_bounded_text() {
        let mut e = entry("v0.1.0-beta.7");
        e["body"] = json!("## What's new\r\n\r\n- Faster\u{0007}\u{001b}[31m start\r\n- <script>alert(1)</script>\r\n");
        e["name"] = json!("Beta 7\nsecond line");
        let r = newest_release(&list(&[e.clone()])).unwrap();
        // Markup is kept as the text it is (the notice never renders HTML);
        // control characters are dropped.
        assert_eq!(r.notes, "## What's new\n\n- Faster[31m start\n- <script>alert(1)</script>");
        assert_eq!(r.name, "Beta 7 second line");

        e["body"] = json!("x".repeat(NOTES_MAX_CHARS * 3));
        let r = newest_release(&list(&[e])).unwrap();
        assert_eq!(r.notes.chars().count(), NOTES_MAX_CHARS + 1);
        assert!(r.notes.ends_with('…'));
    }

    #[test]
    fn an_update_is_offered_only_when_it_is_newer_than_the_running_app() {
        let body = list(&[entry("v0.1.0-beta.6"), entry("v0.1.0-beta.5")]);
        assert_eq!(available_update(&body, "0.1.0-beta.5", "").unwrap().version, "0.1.0-beta.6");
        assert_eq!(available_update(&body, "0.1.0-beta.6", ""), None);
        assert_eq!(available_update(&body, "0.1.0-beta.7", ""), None);
        // An app that does not know its own version does not guess.
        assert_eq!(available_update(&body, "", ""), None);
        assert_eq!(available_update(&body, "dev", ""), None);
    }

    #[test]
    fn a_development_build_is_not_nagged_about_the_betas() {
        // Builds from source are plain 0.1.0, which is later than 0.1.0-beta.N.
        let body = list(&[entry("v0.1.0-beta.6"), entry("v0.1.0-beta.10"), entry("v0.1.0-rc.1")]);
        assert_eq!(available_update(&body, "0.1.0", ""), None);
    }

    #[test]
    fn a_dismissed_version_stays_hidden_and_a_later_one_shows_again() {
        let beta7 = list(&[entry("v0.1.0-beta.7"), entry("v0.1.0-beta.6")]);
        assert_eq!(available_update(&beta7, "0.1.0-beta.6", "0.1.0-beta.7"), None);
        // The dismissal may have been stored as the tag.
        assert_eq!(available_update(&beta7, "0.1.0-beta.6", "v0.1.0-beta.7"), None);

        let beta8 = list(&[entry("v0.1.0-beta.8"), entry("v0.1.0-beta.7")]);
        let again = available_update(&beta8, "0.1.0-beta.6", "0.1.0-beta.7").unwrap();
        assert_eq!(again.version, "0.1.0-beta.8");

        // A dismissal that is not a version hides nothing.
        assert!(available_update(&beta7, "0.1.0-beta.6", "garbage").is_some());
        // Dismissing a release that was later withdrawn does not bring back an
        // older one.
        assert_eq!(available_update(&beta7, "0.1.0-beta.6", "0.1.0-beta.9"), None);
    }

    #[test]
    fn the_notice_serializes_in_the_shape_the_frontend_reads() {
        let release = newest_release(&list(&[entry("v0.1.0-beta.7")])).unwrap();
        let v = serde_json::to_value(UpdateNotice { release, current: "0.1.0-beta.6".into() }).unwrap();
        assert_eq!(v["version"], "0.1.0-beta.7");
        assert_eq!(v["current"], "0.1.0-beta.6");
        assert_eq!(v["prerelease"], true);
        for key in ["tag", "name", "notes", "url"] {
            assert!(v[key].is_string(), "{key}");
        }
    }

    #[test]
    fn the_check_is_on_until_the_user_turns_it_off() {
        use crate::engine::Settings;
        // A settings.json from before the setting existed: unset, which means on.
        let old: Settings = serde_json::from_str(r#"{"theme":"dark"}"#).unwrap();
        assert_eq!(old.check_for_updates, None);
        assert_eq!(old.dismissed_update_version, "");
        let off: Settings =
            serde_json::from_str(r#"{"checkForUpdates":false,"dismissedUpdateVersion":"0.1.0-beta.7"}"#).unwrap();
        assert_eq!(off.check_for_updates, Some(false));
        assert_eq!(off.dismissed_update_version, "0.1.0-beta.7");
        // Saved under the names the frontend reads, and read back unchanged.
        let saved = serde_json::to_value(&off).unwrap();
        assert_eq!(saved["checkForUpdates"], false);
        assert_eq!(saved["dismissedUpdateVersion"], "0.1.0-beta.7");
        // The frontend's own half of "unset means on".
        let store = include_str!("../../src/stores/settings.ts");
        assert!(store.contains("checkForUpdates: true,"));
        assert!(store.contains("s.checkForUpdates = s.checkForUpdates !== false;"));
    }

    /// The request is made by the web view, so these hold its source to what
    /// Settings tells the user: one fixed URL, no credentials, no referrer.
    #[test]
    fn the_web_view_requests_exactly_the_documented_url_and_sends_no_credentials() {
        let src = include_str!("../../src/lib/updateCheck.ts");
        assert!(src.contains(&format!("\"{RELEASES_API_URL}\"")), "the request URL changed");
        assert_eq!(src.matches("fetch(").count(), 1, "exactly one request");
        assert!(src.contains("credentials: \"omit\""));
        assert!(src.contains("referrerPolicy: \"no-referrer\""));
        // Only paging in the query: nothing that could describe the user.
        let query = RELEASES_API_URL.split_once('?').map(|(_, q)| q).unwrap_or("");
        assert_eq!(query, "per_page=10");
        assert!(RELEASES_API_URL.contains(REPO));
    }
}
