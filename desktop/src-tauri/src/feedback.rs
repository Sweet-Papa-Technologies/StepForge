//! "Send feedback" and "Report this chart": the text of a GitHub issue, and the
//! link that opens GitHub's new-issue form with that text filled in.
//!
//! The app sends nothing. It builds a link; the user's browser opens GitHub's
//! form, and the user reads, edits and submits it there (or closes the tab).
//!
//! What goes in: the app version, the OS, the model, the diagnostics report
//! (`diagnostics.rs`, already stripped of folders, the user name and song
//! names) and, for a chart report, the difficulty, meter, note count and the
//! generation settings recorded for that chart in `report.json`.
//!
//! What never goes in: audio, file paths, and — unless the user ticks the box —
//! the song's title. Only named fields are read from `report.json`, and every
//! value from it passes through `safe_value`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::AppHandle;

use crate::update::REPO;
use crate::{commands, diagnostics, engine};

/// Where the form lives. Nothing but `title`, `body` and `labels` is added.
fn new_issue_url() -> String {
    format!("https://github.com/{REPO}/issues/new")
}

/// GitHub answers a very long link with an error page, and an operating system
/// only hands the browser a link of limited length. The whole link stays under
/// this.
pub const MAX_URL_CHARS: usize = 7000;

/// Put in place of the diagnostics that did not fit. Never cut silently.
pub const TRUNCATED_NOTE: &str = "(truncated — use Copy diagnostics for the full text)";

/// The label `.github/ISSUE_TEMPLATE/chart_feedback.md` files chart reports under.
const CHART_LABEL: &str = "chart-quality";

const VALUE_MAX_ENCODED: usize = 120;
const KEY_MAX_ENCODED: usize = 60;
const TITLE_MAX_ENCODED: usize = 300;
const MAX_STYLE_KEYS: usize = 24;
const MAX_OPTION_KEYS: usize = 24;

// --------------------------------------------------------------------------- //
// Encoding and scrubbing
// --------------------------------------------------------------------------- //

/// Percent-encode for a query value: everything but RFC 3986's unreserved
/// characters, so the text survives whatever opens the link.
pub fn percent_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 3);
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn encoded_len(c: char) -> usize {
    if c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | '~') {
        1
    } else {
        c.len_utf8() * 3
    }
}

/// `text`, cut so that it takes at most `max` characters of a link.
fn cap_encoded(text: &str, max: usize) -> String {
    let mut used = 0;
    let mut out = String::new();
    for c in text.chars() {
        used += encoded_len(c);
        if used > max {
            out.push('…');
            break;
        }
        out.push(c);
    }
    out
}

/// One line, no control characters.
fn one_line(text: &str) -> String {
    let flat: String = text.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    flat.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Replace `word` wherever it appears, whatever its case.
fn replace_ignoring_case(text: &str, word: &str, with: &str) -> String {
    let (hay, needle) = (text.to_lowercase(), word.to_lowercase());
    // Lower-casing can change byte lengths (rare); then only exact matches go.
    if needle.is_empty() || hay.len() != text.len() {
        return text.replace(word, with);
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

/// A value read from `report.json` (or handed over by the frontend), made safe
/// to publish: one line, bounded, never a path, and without the song's names.
///
/// Anything holding a slash is treated as a path — a setting is a number, a
/// flag or a short word, so nothing real is lost by refusing them all.
pub fn safe_value(raw: &str, hide: &[String], max_encoded: usize) -> String {
    let mut text = one_line(raw);
    if text.contains('/') || text.contains('\\') {
        return "<path>".into();
    }
    for word in hide {
        let word = word.trim();
        if word.chars().count() >= 3 {
            text = replace_ignoring_case(&text, word, "<song>");
        }
    }
    cap_encoded(&text, max_encoded)
}

/// A JSON value as a setting is written: `8`, `true`, `0.35`, `auto`.
fn scalar_text(v: &Value, hide: &[String]) -> String {
    match v {
        Value::Null => "none".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => safe_value(s, hide, VALUE_MAX_ENCODED),
        // Not expected (options are plain key -> scalar), but shown rather than
        // dropped, through the same scrubbing.
        other => safe_value(&other.to_string(), hide, VALUE_MAX_ENCODED),
    }
}

// --------------------------------------------------------------------------- //
// The chart's recorded facts
// --------------------------------------------------------------------------- //

/// What `report.json` says about one difficulty. Every field is optional: older
/// reports, and charts not made by the model, record less.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChartFacts {
    pub difficulty: String,
    /// `report.mode`: "neural" for a StepForge Flow chart.
    pub mode: Option<String>,
    pub meter: Option<Value>,
    pub notes: Option<u64>,
    /// `charts[difficulty].engine`: `preset`, `style`, `seed`, `options`, and
    /// statistics this module does not read.
    pub engine: Option<Value>,
}

impl ChartFacts {
    /// Pick one difficulty's facts out of a parsed `report.json`.
    pub fn from_report(report: &Value, difficulty: &str) -> Self {
        let chart = &report["charts"][difficulty];
        Self {
            difficulty: difficulty.to_string(),
            mode: report["mode"].as_str().map(str::to_string),
            meter: Some(chart["meter"].clone()).filter(|m| m.is_number() || m.is_string()),
            notes: chart["notes"].as_u64(),
            engine: Some(chart["engine"].clone()).filter(Value::is_object),
        }
    }
}

/// `key=value, key=value` for up to `max` entries of an object.
fn pairs(entries: &[(String, String)], max: usize) -> String {
    let mut shown: Vec<String> = entries.iter().take(max).map(|(k, v)| format!("{k}={v}")).collect();
    if entries.len() > max {
        shown.push(format!("(+{} more)", entries.len() - max));
    }
    shown.join(", ")
}

/// The generation settings recorded for a chart, one Markdown bullet each.
///
/// `style` maps a slider to a number, or to null for "left to the model".
/// `options` is listed key by key, whatever the keys are; a report written
/// before options were recorded simply has no such line.
pub fn settings_lines(engine: Option<&Value>, hide: &[String]) -> Vec<String> {
    let Some(engine) = engine.filter(|e| e.is_object()) else {
        return vec!["- Generation settings: not recorded for this chart".into()];
    };
    let key = |k: &str| safe_value(k, hide, KEY_MAX_ENCODED);
    let mut out = Vec::new();

    out.push(match &engine["preset"] {
        Value::String(p) if !p.trim().is_empty() => {
            format!("- Preset: {}", safe_value(p, hide, VALUE_MAX_ENCODED))
        }
        _ => "- Preset: none".into(),
    });

    match engine["style"].as_object() {
        Some(style) => {
            let set: Vec<(String, String)> = style
                .iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (key(k), scalar_text(v, hide)))
                .collect();
            let auto: Vec<String> = style.iter().filter(|(_, v)| v.is_null()).map(|(k, _)| key(k)).collect();
            if set.is_empty() {
                out.push("- Sliders: all left to the model".into());
            } else {
                out.push(format!("- Sliders set: {}", pairs(&set, MAX_STYLE_KEYS)));
                if !auto.is_empty() {
                    let mut names: Vec<String> = auto.iter().take(MAX_STYLE_KEYS).cloned().collect();
                    if auto.len() > MAX_STYLE_KEYS {
                        names.push(format!("(+{} more)", auto.len() - MAX_STYLE_KEYS));
                    }
                    out.push(format!("- Sliders left to the model: {}", names.join(", ")));
                }
            }
        }
        None => out.push("- Sliders: not recorded".into()),
    }

    out.push(match &engine["seed"] {
        Value::Null => "- Seed: not recorded".into(),
        seed => format!("- Seed: {}", scalar_text(seed, hide)),
    });

    if let Some(options) = engine["options"].as_object() {
        let all: Vec<(String, String)> = options.iter().map(|(k, v)| (key(k), scalar_text(v, hide))).collect();
        out.push(if all.is_empty() {
            "- Options: none".into()
        } else {
            format!("- Options: {}", pairs(&all, MAX_OPTION_KEYS))
        });
    }
    out
}

// --------------------------------------------------------------------------- //
// The issue
// --------------------------------------------------------------------------- //

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// "Send feedback": a bug, a wish, anything.
    General,
    /// "Report this chart": about one generated difficulty.
    Chart,
}

/// Everything an issue is built from.
#[derive(Debug, Clone)]
pub struct IssueInput {
    pub kind: Kind,
    pub app_version: String,
    pub os: String,
    /// "StepForge Flow v1.4", when the app knows it.
    pub model: Option<String>,
    /// The song's display title. Used only when `include_title` is set.
    pub song_title: Option<String>,
    /// The song's other names (its file and folder name). Never written; taken
    /// out of anything read from `report.json`.
    pub song_names: Vec<String>,
    /// The user ticked "Include the song title".
    pub include_title: bool,
    pub chart: Option<ChartFacts>,
    /// The Copy diagnostics text.
    pub diagnostics: String,
}

/// The issue as the user will see it on GitHub, and the link that opens it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Issue {
    pub title: String,
    pub body: String,
    pub url: String,
    /// The diagnostics did not fit in a link and were cut (the body says so).
    pub truncated: bool,
}

fn or_unknown(text: &str) -> String {
    let t = safe_value(text, &[], VALUE_MAX_ENCODED);
    if t.is_empty() {
        "unknown".into()
    } else {
        t
    }
}

impl IssueInput {
    /// Names to take out of the values read from `report.json`: the song's
    /// file names always, and its title unless the user chose to share it.
    fn hidden(&self) -> Vec<String> {
        let title = self.song_title.iter().filter(|_| !self.include_title);
        self.song_names.iter().chain(title).cloned().collect()
    }

    /// The title to publish: only when the box is ticked, and never a path.
    fn shared_title(&self) -> Option<String> {
        if !self.include_title {
            return None;
        }
        let t = safe_value(self.song_title.as_deref()?, &[], TITLE_MAX_ENCODED);
        (!t.is_empty()).then_some(t)
    }

    fn model_text(&self) -> Option<String> {
        self.model
            .as_deref()
            .map(|m| safe_value(m, &[], VALUE_MAX_ENCODED))
            .filter(|m| !m.is_empty())
    }

    /// "Model: …" for a chart the model made; otherwise what made it, without
    /// repeating a name from the report.
    fn made_by_line(&self, chart: &ChartFacts) -> String {
        match chart.mode.as_deref() {
            Some("neural") | None => {
                format!("- Model: {}", self.model_text().unwrap_or_else(|| "unknown".into()))
            }
            Some("deterministic") => "- Chart engine: deterministic (not the StepForge Flow model)".into(),
            Some(_) => "- Chart engine: not the StepForge Flow model".into(),
        }
    }

    fn difficulty_text(&self, chart: &ChartFacts) -> String {
        // The app's own word for the tier (a key of `report.charts`), so the
        // song's names are not looked for in it: a song called "Hard" would
        // otherwise blank the difficulty.
        let d = safe_value(&chart.difficulty, &[], KEY_MAX_ENCODED);
        if d.is_empty() {
            "unknown".into()
        } else {
            d
        }
    }

    fn issue_title(&self) -> String {
        match (&self.kind, &self.chart) {
            (Kind::Chart, Some(chart)) => {
                let mut t = format!("Chart report: {}", self.difficulty_text(chart));
                if let Some(m) = &chart.meter {
                    t.push_str(&format!(", meter {}", scalar_text(m, &[])));
                }
                if let Some(song) = self.shared_title() {
                    t.push_str(&format!(" — {song}"));
                }
                t
            }
            _ => format!("Feedback from StepForge {}", or_unknown(&self.app_version)),
        }
    }

    /// The part of the body before the diagnostics. Headings follow
    /// `.github/ISSUE_TEMPLATE/{chart_feedback,bug_report}.md`.
    fn body_head(&self, with_settings: bool) -> String {
        let version = or_unknown(&self.app_version);
        let os = or_unknown(&self.os);
        let mut l: Vec<String> = Vec::new();
        match (&self.kind, &self.chart) {
            (Kind::Chart, Some(chart)) => {
                let difficulty = self.difficulty_text(chart);
                l.push("**Song** (title / artist, and a link if it is freely available)".into());
                l.push(self.shared_title().unwrap_or_else(|| {
                    "_Not included. Add it here if you are happy to share it._".into()
                }));
                l.push(String::new());
                l.push(format!("**Difficulty**: {difficulty}"));
                l.push(String::new());
                l.push("**What felt wrong (or right)**".into());
                l.push("- [ ] Off the beat / out of sync".into());
                l.push("- [ ] Awkward to stand on".into());
                l.push("- [ ] Too dense / too sparse".into());
                l.push("- [ ] Doesn't follow the music".into());
                l.push("- [ ] Other:".into());
                l.push(String::new());
                l.push("**Details** (timestamps help: \"0:42–0:50, three crossovers in a row\")".into());
                l.push(String::new());
                l.push(String::new());
                l.push("**Chart and setup** (filled in by StepForge)".into());
                l.push(format!("- StepForge version: {version}"));
                l.push(format!("- OS and version: {os}"));
                l.push(self.made_by_line(chart));
                let mut chart_line = format!("- Difficulty: {difficulty}");
                if let Some(m) = &chart.meter {
                    chart_line.push_str(&format!(", meter {}", scalar_text(m, &[])));
                }
                if let Some(n) = chart.notes {
                    chart_line.push_str(&format!(", {n} notes"));
                }
                l.push(chart_line);
                if with_settings {
                    l.extend(settings_lines(chart.engine.as_ref(), &self.hidden()));
                } else {
                    l.push("- Generation settings: too long to include here".into());
                }
            }
            _ => {
                l.push("**What happened**".into());
                l.push(String::new());
                l.push(String::new());
                l.push("**What you expected**".into());
                l.push(String::new());
                l.push(String::new());
                l.push("**Steps to reproduce**".into());
                l.push("1.".into());
                l.push("2.".into());
                l.push(String::new());
                l.push("**Your setup** (filled in by StepForge)".into());
                l.push(format!("- StepForge version: {version}"));
                l.push(format!("- OS and version: {os}"));
                l.push(format!("- Model: {}", self.model_text().unwrap_or_else(|| "unknown".into())));
                l.push("- Game (ITGmania / StepMania version):".into());
                l.push(String::new());
                l.push("**Logs or screenshots**".into());
            }
        }
        l.join("\n")
    }

    fn labels(&self) -> Option<&'static str> {
        (self.kind == Kind::Chart && self.chart.is_some()).then_some(CHART_LABEL)
    }
}

/// The diagnostics as a collapsed block. A code fence inside the text would
/// end the block early, so it is defused.
fn diagnostics_block(text: &str, truncated: bool) -> String {
    let mut body = text.trim().replace("```", "'''");
    if truncated {
        if !body.is_empty() {
            body.push('\n');
        }
        body.push_str(TRUNCATED_NOTE);
    } else if body.is_empty() {
        body = "(no diagnostics were collected)".into();
    }
    format!(
        "<details><summary>Diagnostics (no audio, song names or folder names)</summary>\n\n```text\n{body}\n```\n\n</details>"
    )
}

fn link(title: &str, body: &str, labels: Option<&str>) -> String {
    let mut url = format!(
        "{}?title={}&body={}",
        new_issue_url(),
        percent_encode(title),
        percent_encode(body)
    );
    if let Some(l) = labels {
        url.push_str("&labels=");
        url.push_str(&percent_encode(l));
    }
    url
}

/// Build the issue. The link is at most `MAX_URL_CHARS` long: when the full
/// text does not fit, the diagnostics are cut back to whole lines and
/// `TRUNCATED_NOTE` is written where they stop.
pub fn build_issue(input: &IssueInput) -> Issue {
    let title = input.issue_title();
    let labels = input.labels();
    let diagnostics = input.diagnostics.replace("\r\n", "\n");
    let diagnostics = diagnostics.trim();
    let assemble = |head: &str, diag: &str, truncated: bool| -> Issue {
        let body = format!("{head}\n\n{}", diagnostics_block(diag, truncated));
        let url = link(&title, &body, labels);
        Issue { title: title.clone(), body, url, truncated }
    };

    let head = input.body_head(true);
    let whole = assemble(&head, diagnostics, false);
    if whole.url.len() <= MAX_URL_CHARS {
        return whole;
    }

    // Settings so long that nothing else fits are not settings (a damaged
    // report); say so rather than produce a link GitHub will refuse.
    let head = if assemble(&head, "", true).url.len() <= MAX_URL_CHARS {
        head
    } else {
        let short = input.body_head(false);
        let whole = assemble(&short, diagnostics, false);
        if whole.url.len() <= MAX_URL_CHARS {
            return whole;
        }
        short
    };

    // The longest prefix of the diagnostics that fits. A longer prefix never
    // makes a shorter link, so a binary search finds it. `ends[k]` is where
    // the first `k` characters end.
    let ends: Vec<usize> = diagnostics
        .char_indices()
        .map(|(i, _)| i)
        .chain([diagnostics.len()])
        .collect();
    let (mut lo, mut hi) = (0usize, ends.len() - 1);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if assemble(&head, &diagnostics[..ends[mid]], true).url.len() <= MAX_URL_CHARS {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let (mut kept, rest) = diagnostics.split_at(ends[lo]);
    // Stop at the end of a line rather than in the middle of one.
    if !rest.is_empty() && !rest.starts_with('\n') {
        if let Some(end) = kept.rfind('\n') {
            kept = &kept[..end];
        }
    }
    assemble(&head, kept, true)
}

// --------------------------------------------------------------------------- //
// The command
// --------------------------------------------------------------------------- //

/// What the dialog asks for. `diagnostics` is the Copy diagnostics text the
/// frontend already collected (the self-test takes seconds; the checkbox must
/// not re-run it).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct FeedbackRequest {
    /// "chart" or "general".
    pub kind: String,
    pub base: Option<String>,
    pub target: Option<String>,
    pub difficulty: Option<String>,
    pub song_title: Option<String>,
    pub include_title: bool,
    pub model: Option<String>,
    pub diagnostics: String,
}

/// A song's folder name, as a single path component.
fn plain_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', '\0'])
}

/// The issue text and link for the feedback dialog. Reads `report.json` for a
/// chart report; makes no network request and opens nothing.
#[tauri::command(async)]
pub fn feedback_issue(app: AppHandle, request: FeedbackRequest) -> Result<Issue, String> {
    let mut input = IssueInput {
        kind: Kind::General,
        app_version: app.package_info().version.to_string(),
        os: diagnostics::os_description(),
        model: request.model,
        song_title: request.song_title,
        song_names: Vec::new(),
        include_title: request.include_title,
        chart: None,
        diagnostics: request.diagnostics,
    };
    if request.kind == "chart" {
        let difficulty = request.difficulty.unwrap_or_default();
        let base = request.base.unwrap_or_default();
        let target = request.target.unwrap_or_default();
        let report = (plain_name(&base) && matches!(target.as_str(), "stepforge" | "saberforge"))
            .then(|| engine::repo_root(&app).ok())
            .flatten()
            .and_then(|root| commands::read_json(&commands::target_dir(&root, &target).join(&base).join("report.json")));
        input.kind = Kind::Chart;
        input.chart = Some(match &report {
            Some(r) => ChartFacts::from_report(r, &difficulty),
            None => ChartFacts { difficulty, ..Default::default() },
        });
        // The catalogue id can differ from the file name; neither is published.
        input.song_names = vec![base];
        if let Some(id) = report.as_ref().and_then(|r| r["track_id"].as_str()) {
            input.song_names.push(id.to_string());
        }
    }
    Ok(build_issue(&input))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SONG: &str = "Neon Harbor Lights";
    const BASE: &str = "neon_harbor_lights";

    /// A report as the engine writes it, including the parts that must never
    /// be published (folders, the track's file name).
    fn report(options: Option<Value>) -> Value {
        let mut engine = json!({
            "name": "neural",
            "preset": "tech",
            "style": {
                "stream_share": 0.35, "jump_share": null, "hold_share": 0.1,
                "mine_share": null, "footswitch_share": 0.0,
            },
            "seed": 4242,
            "rhythm": {"theta": 0.41, "notes": 512},
            "difficulty": "hard",
            "tau": 0.9,
        });
        if let Some(o) = options {
            engine["options"] = o;
        }
        json!({
            "track_id": BASE,
            "mode": "neural",
            "out": format!("/data/workspace/build/stepmania/{BASE}"),
            "files": {"ssc": format!("/data/workspace/build/stepmania/{BASE}/{BASE}.ssc")},
            "engine": {"name": "neural", "model_dir": "/data/workspace/models/stepforge_flow"},
            "charts": {"hard": {
                "notes": 512, "meter": 9, "bars": 96,
                "previews": {"png": format!("/data/workspace/build/stepmania/{BASE}/hard.png")},
                "engine": engine,
            }},
        })
    }

    const DIAGNOSTICS: &str = "StepForge diagnostics\nApp: 0.1.0-beta.6 (macOS app)\nOS: macOS 15.5\n\
        Engine self-test: passed\n  model: ok (StepForge Flow v1.4)\nLast error: none\n\
        (No audio, song names or folder names are included.)";

    fn chart_input(options: Option<Value>, include_title: bool) -> IssueInput {
        IssueInput {
            kind: Kind::Chart,
            app_version: "0.1.0-beta.6".into(),
            os: "macOS 15.5".into(),
            model: Some("StepForge Flow v1.4".into()),
            song_title: Some(SONG.into()),
            song_names: vec![BASE.into()],
            include_title,
            chart: Some(ChartFacts::from_report(&report(options), "hard")),
            diagnostics: DIAGNOSTICS.into(),
        }
    }

    fn general_input() -> IssueInput {
        IssueInput {
            kind: Kind::General,
            app_version: "0.1.0-beta.6".into(),
            os: "Ubuntu 24.04.1 LTS (kernel 6.8.0)".into(),
            model: Some("StepForge Flow v1.4".into()),
            song_title: None,
            song_names: Vec::new(),
            include_title: false,
            chart: None,
            diagnostics: DIAGNOSTICS.into(),
        }
    }

    /// Undo `percent_encode`, to read a link the way GitHub will.
    fn decode(text: &str) -> String {
        let b = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'%' {
                out.push(u8::from_str_radix(&text[i + 1..i + 3], 16).unwrap());
                i += 3;
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        String::from_utf8(out).unwrap()
    }

    /// (name, decoded value) for each query parameter of a link.
    fn query(url: &str) -> Vec<(String, String)> {
        let (_, q) = url.split_once('?').unwrap();
        q.split('&')
            .map(|kv| {
                let (k, v) = kv.split_once('=').unwrap();
                (k.to_string(), decode(v))
            })
            .collect()
    }

    /// Everything in the issue that is not the fixed template wording, where a
    /// stray path or name would show.
    fn assert_no_paths(issue: &Issue) {
        for text in [&issue.title, &issue.body] {
            for needle in ["/data", "workspace", "/build/", "stepmania/", ".ssc", ".png", "model_dir", "\\"] {
                assert!(!text.contains(needle), "found {needle:?} in:\n{text}");
            }
        }
    }

    #[test]
    fn percent_encoding_leaves_only_unreserved_characters() {
        assert_eq!(percent_encode("AZaz09-._~"), "AZaz09-._~");
        assert_eq!(percent_encode("a b&c=d#e?f/g+h%i"), "a%20b%26c%3Dd%23e%3Ff%2Fg%2Bh%25i");
        assert_eq!(percent_encode("line\nbreak"), "line%0Abreak");
        assert_eq!(percent_encode("— é 🎵"), "%E2%80%94%20%C3%A9%20%F0%9F%8E%B5");
        let text = "**Details** (\"0:42–0:50\")\n- [ ] Doesn't & won't <b>\n```text";
        assert_eq!(decode(&percent_encode(text)), text);
    }

    #[test]
    fn a_chart_report_carries_the_chart_facts_and_the_recorded_settings() {
        let issue = build_issue(&chart_input(Some(json!({"max_subdivision": 8, "stream": true})), false));
        let b = &issue.body;
        for line in [
            "**Difficulty**: hard",
            "- StepForge version: 0.1.0-beta.6",
            "- OS and version: macOS 15.5",
            "- Model: StepForge Flow v1.4",
            "- Difficulty: hard, meter 9, 512 notes",
            "- Preset: tech",
            "- Sliders set: footswitch_share=0.0, hold_share=0.1, stream_share=0.35",
            "- Sliders left to the model: jump_share, mine_share",
            "- Seed: 4242",
            "- Options: max_subdivision=8, stream=true",
            "Engine self-test: passed",
            "Last error: none",
        ] {
            assert!(b.lines().any(|l| l == line), "missing line {line:?} in:\n{b}");
        }
        assert_eq!(issue.title, "Chart report: hard, meter 9");
        assert!(!issue.truncated);
        assert!(!b.contains(TRUNCATED_NOTE));
        // Statistics that are not settings stay out.
        assert!(!b.contains("theta") && !b.contains("tau"));
    }

    #[test]
    fn the_chart_report_follows_the_issue_template() {
        let template = include_str!("../../../.github/ISSUE_TEMPLATE/chart_feedback.md");
        let issue = build_issue(&chart_input(None, false));
        for heading in [
            "**Song**",
            "**Difficulty**",
            "**What felt wrong (or right)**",
            "**Details**",
            "**Chart and setup**",
        ] {
            assert!(template.contains(heading), "the template no longer has {heading}");
            assert!(issue.body.contains(heading), "the report is missing {heading}");
        }
        for line in template.lines().filter(|l| l.starts_with("- [ ]")) {
            assert!(issue.body.lines().any(|l| l == line), "missing checklist line {line:?}");
        }
        assert!(template.contains(&format!("labels: {CHART_LABEL}")));
        assert!(query(&issue.url).contains(&("labels".to_string(), CHART_LABEL.to_string())));
    }

    #[test]
    fn general_feedback_follows_the_bug_report_template() {
        let template = include_str!("../../../.github/ISSUE_TEMPLATE/bug_report.md");
        let issue = build_issue(&general_input());
        for heading in [
            "**What happened**",
            "**What you expected**",
            "**Steps to reproduce**",
            "**Your setup**",
            "**Logs or screenshots**",
        ] {
            assert!(template.contains(heading), "the template no longer has {heading}");
            assert!(issue.body.contains(heading), "the report is missing {heading}");
        }
        assert_eq!(issue.title, "Feedback from StepForge 0.1.0-beta.6");
        for line in [
            "- StepForge version: 0.1.0-beta.6",
            "- OS and version: Ubuntu 24.04.1 LTS (kernel 6.8.0)",
            "- Model: StepForge Flow v1.4",
            "Engine self-test: passed",
        ] {
            assert!(issue.body.lines().any(|l| l == line), "missing line {line:?}");
        }
        // Not a chart report: no chart facts, no label of its own.
        assert!(!issue.body.contains("Difficulty") && !issue.body.contains("Preset"));
        assert!(query(&issue.url).iter().all(|(k, _)| k != "labels"));
    }

    #[test]
    fn the_song_title_is_left_out_unless_the_box_is_ticked() {
        let without = build_issue(&chart_input(Some(json!({"stream": true})), false));
        for text in [&without.title, &without.body, &decode(&without.url)] {
            let lower = text.to_lowercase();
            assert!(!lower.contains(&SONG.to_lowercase()), "the title leaked:\n{text}");
            assert!(!lower.contains(BASE), "the file name leaked:\n{text}");
            assert!(!lower.contains("neon") && !lower.contains("harbor"), "part of the title leaked:\n{text}");
        }
        assert!(without.body.contains("_Not included. Add it here if you are happy to share it._"));

        let with = build_issue(&chart_input(Some(json!({"stream": true})), true));
        assert_eq!(with.title, format!("Chart report: hard, meter 9 — {SONG}"));
        assert!(with.body.lines().any(|l| l == SONG));
        // Ticking the box shares the title, not the file name.
        assert!(!with.body.contains(BASE) && !with.title.contains(BASE));
    }

    #[test]
    fn a_song_named_like_a_setting_does_not_blank_the_report() {
        // "Hard" is a difficulty and "tech" a preset; a song can be called either.
        let mut input = chart_input(None, false);
        input.song_title = Some("Hard".into());
        input.song_names = vec!["tech".into()];
        let issue = build_issue(&input);
        // The difficulty is the app's own word, so it is never scrubbed…
        assert_eq!(issue.title, "Chart report: hard, meter 9");
        assert!(issue.body.contains("**Difficulty**: hard\n"));
        // …while a recorded value that equals the song's name is hidden,
        assert!(issue.body.contains("- Preset: <song>"));
        // and the title stops being hidden once the user shares it.
        input.song_names = vec![BASE.into()];
        input.song_title = Some("tech".into());
        assert!(build_issue(&input).body.contains("- Preset: <song>"));
        input.include_title = true;
        let shared = build_issue(&input);
        assert!(shared.body.contains("- Preset: tech"));
        assert_eq!(shared.title, "Chart report: hard, meter 9 — tech");
    }

    #[test]
    fn general_feedback_never_names_a_song() {
        let mut input = general_input();
        input.song_title = Some(SONG.into());
        input.include_title = true; // not offered for general feedback; ignored if sent
        let issue = build_issue(&input);
        assert!(!decode(&issue.url).contains(SONG));
    }

    #[test]
    fn the_link_is_the_title_and_body_shown_and_nothing_else() {
        for input in [chart_input(Some(json!({"max_subdivision": 8})), true), general_input()] {
            let issue = build_issue(&input);
            let (base, _) = issue.url.split_once('?').unwrap();
            assert_eq!(base, "https://github.com/Sweet-Papa-Technologies/StepForge/issues/new");
            let q = query(&issue.url);
            let names: Vec<&str> = q.iter().map(|(k, _)| k.as_str()).collect();
            assert!(names == ["title", "body", "labels"] || names == ["title", "body"], "{names:?}");
            assert_eq!(q[0].1, issue.title);
            assert_eq!(q[1].1, issue.body, "the link carries exactly the text the dialog shows");
            // The whole link is plain ASCII with nothing a shell or a browser
            // would read as structure.
            assert!(issue.url.bytes().all(|b| b.is_ascii_graphic()));
            let (_, raw) = issue.url.split_once('?').unwrap();
            assert!(!raw.contains(['#', ' ', '"', '\'', '<', '>', '+']));
        }
    }

    #[test]
    fn nothing_path_like_from_the_report_is_published() {
        let options = json!({
            "max_subdivision": 8,
            "weights": "/data/workspace/models/stepforge_flow",
            "cache": "C:\\Users\\someone\\AppData\\cache",
            "source": "https://example.invalid/a/b",
            "note": format!("made for {SONG} ({BASE})"),
            "multi\nline": "a\nb\tc",
            "nested": {"file": "/data/workspace/x.ogg"},
        });
        let mut input = chart_input(Some(options), false);
        // A report that holds a path where a preset name belongs.
        input.chart.as_mut().unwrap().engine.as_mut().unwrap()["preset"] =
            json!(format!("/data/workspace/presets/{BASE}.json"));
        let issue = build_issue(&input);
        assert_no_paths(&issue);
        let b = &issue.body;
        assert!(b.lines().any(|l| l == "- Preset: <path>"), "{b}");
        let options_line = b.lines().find(|l| l.starts_with("- Options: ")).unwrap();
        for part in [
            "max_subdivision=8",
            "weights=<path>",
            "cache=<path>",
            "source=<path>",
            "nested=<path>",
            "note=made for <song> (<song>)",
            "multi line=a b c",
        ] {
            assert!(options_line.contains(part), "missing {part:?} in {options_line:?}");
        }
        assert!(!b.to_lowercase().contains("neon"));
        // The report's own paths (`out`, `files`, `engine.model_dir`,
        // `previews`) are never read at all.
        let plain = build_issue(&chart_input(None, true));
        assert_no_paths(&plain);
    }

    #[test]
    fn a_title_that_is_a_path_is_not_published_even_when_ticked() {
        let mut input = chart_input(None, true);
        input.song_title = Some("/home/someone/Music/Neon Harbor Lights.ogg".into());
        let issue = build_issue(&input);
        assert!(!issue.body.contains("someone") && !issue.body.contains("Music"));
        assert!(issue.body.lines().any(|l| l == "<path>"));
    }

    #[test]
    fn a_report_without_options_has_no_options_line() {
        let issue = build_issue(&chart_input(None, false));
        assert!(!issue.body.contains("Options"));
        assert!(issue.body.contains("- Seed: 4242"));
        let empty = build_issue(&chart_input(Some(json!({})), false));
        assert!(empty.body.lines().any(|l| l == "- Options: none"));
        // `options` that is not an object is ignored, not printed.
        let odd = build_issue(&chart_input(Some(json!("fast")), false));
        assert!(!odd.body.contains("Options"));
    }

    #[test]
    fn every_option_key_is_listed_whatever_it_is_called() {
        let lines = settings_lines(
            Some(&json!({"preset": null, "style": {}, "seed": null, "options": {
                "zeta": 1.5, "alpha": false, "mode": "auto", "limit": null, "count": 16,
            }})),
            &[],
        );
        assert_eq!(
            lines,
            vec![
                "- Preset: none",
                "- Sliders: all left to the model",
                "- Seed: not recorded",
                "- Options: alpha=false, count=16, limit=none, mode=auto, zeta=1.5",
            ]
        );
    }

    #[test]
    fn older_and_non_model_charts_say_what_is_missing() {
        assert_eq!(settings_lines(None, &[]), vec!["- Generation settings: not recorded for this chart"]);
        assert_eq!(
            settings_lines(Some(&json!("neural")), &[]),
            vec!["- Generation settings: not recorded for this chart"]
        );
        assert_eq!(
            settings_lines(Some(&json!({"name": "neural"})), &[]),
            vec!["- Preset: none", "- Sliders: not recorded", "- Seed: not recorded"]
        );

        // A chart the model did not make is not credited to it, and a backend
        // name in the report is not repeated.
        let mut input = chart_input(None, false);
        input.chart = Some(ChartFacts::from_report(
            &json!({"mode": "deterministic", "charts": {"easy": {"notes": 120, "meter": 3}}}),
            "easy",
        ));
        let b = build_issue(&input).body;
        assert!(b.contains("- Chart engine: deterministic (not the StepForge Flow model)"));
        assert!(!b.contains("- Model:"));
        input.chart.as_mut().unwrap().mode = Some("some-backend".into());
        let b = build_issue(&input).body;
        assert!(b.contains("- Chart engine: not the StepForge Flow model"));
        assert!(!b.contains("some-backend"));

        // A difficulty the report does not have: the report still builds.
        input.chart = Some(ChartFacts::from_report(&json!({"charts": {}}), "challenge"));
        let issue = build_issue(&input);
        assert_eq!(issue.title, "Chart report: challenge");
        assert!(issue.body.contains("- Difficulty: challenge\n"));
        assert!(issue.body.contains("- Generation settings: not recorded for this chart"));
    }

    #[test]
    fn long_diagnostics_are_cut_with_a_visible_note_and_the_link_stays_short() {
        let mut input = chart_input(Some(json!({"max_subdivision": 8})), true);
        let long: String = (0..600)
            .map(|i| format!("  module_{i}: FAILED — ImportError: could not load part {i}\n"))
            .collect();
        input.diagnostics = format!("StepForge diagnostics\nApp: 0.1.0-beta.6\n{long}Last error: none");
        let issue = build_issue(&input);

        assert!(issue.truncated);
        assert!(issue.url.len() <= MAX_URL_CHARS, "{} characters", issue.url.len());
        // Not cut needlessly short either: most of the room is used.
        assert!(issue.url.len() > MAX_URL_CHARS - 300, "{} characters", issue.url.len());
        assert_eq!(issue.body.matches(TRUNCATED_NOTE).count(), 1);
        assert_eq!(TRUNCATED_NOTE, "(truncated — use Copy diagnostics for the full text)");

        // The start of the report is kept, whole lines only, and the note is
        // the last thing in the block.
        let block = issue.body.split("```text\n").nth(1).unwrap().split("\n```").next().unwrap();
        let lines: Vec<&str> = block.lines().collect();
        assert_eq!(lines[0], "StepForge diagnostics");
        assert_eq!(*lines.last().unwrap(), TRUNCATED_NOTE);
        for l in &lines[2..lines.len() - 1] {
            assert!(l.ends_with(char::is_numeric), "a line was cut mid-way: {l:?}");
        }
        // Everything that is not diagnostics is intact.
        assert!(issue.body.contains("- Options: max_subdivision=8"));
        assert!(issue.body.ends_with("</details>"));
        assert_eq!(query(&issue.url)[1].1, issue.body);
    }

    #[test]
    fn diagnostics_that_fit_are_not_touched() {
        let issue = build_issue(&general_input());
        assert!(!issue.truncated);
        assert!(issue.body.contains(DIAGNOSTICS));
        assert!(issue.url.len() < MAX_URL_CHARS);
    }

    #[test]
    fn the_link_stays_short_whatever_the_report_holds() {
        // Characters that take nine or twelve characters of a link each.
        let wide = "譜面🎵".repeat(400);
        let mut options = serde_json::Map::new();
        let mut style = serde_json::Map::new();
        for i in 0..200 {
            options.insert(format!("{wide}{i}"), json!(wide));
            style.insert(format!("{wide}{i}"), if i % 2 == 0 { json!(0.5) } else { json!(null) });
        }
        let mut input = chart_input(Some(Value::Object(options)), true);
        input.chart.as_mut().unwrap().engine.as_mut().unwrap()["style"] = Value::Object(style);
        input.chart.as_mut().unwrap().engine.as_mut().unwrap()["preset"] = json!(wide);
        input.chart.as_mut().unwrap().difficulty = wide.clone();
        input.song_title = Some(wide.clone());
        input.model = Some(wide.clone());
        input.app_version = wide.clone();
        input.os = wide.clone();
        input.diagnostics = wide.repeat(20);
        let issue = build_issue(&input);
        assert!(issue.url.len() <= MAX_URL_CHARS, "{} characters", issue.url.len());
        assert!(issue.truncated);
        assert!(issue.body.contains(TRUNCATED_NOTE));
        assert_eq!(query(&issue.url)[1].1, issue.body);

        // And with one enormous line, which cannot be cut at a line break.
        let mut input = general_input();
        input.diagnostics = "x".repeat(50_000);
        let issue = build_issue(&input);
        assert!(issue.url.len() <= MAX_URL_CHARS);
        assert!(issue.truncated && issue.body.contains(TRUNCATED_NOTE));
    }

    #[test]
    fn a_code_fence_in_the_diagnostics_cannot_end_the_block() {
        let mut input = general_input();
        input.diagnostics = "Last error: ```\n</details>\n# heading".into();
        let body = build_issue(&input).body;
        assert_eq!(body.matches("```").count(), 2, "only the block's own fences");
        assert!(body.ends_with("\n```\n\n</details>"));
    }

    #[test]
    fn empty_diagnostics_are_said_to_be_empty() {
        let mut input = general_input();
        input.diagnostics = "  \n".into();
        let issue = build_issue(&input);
        assert!(issue.body.contains("(no diagnostics were collected)"));
        assert!(!issue.truncated);
    }

    #[test]
    fn values_are_scrubbed_one_way_for_every_source() {
        let hide = vec![SONG.to_string(), BASE.to_string(), "ab".to_string()];
        assert_eq!(safe_value("tech", &hide, 120), "tech");
        assert_eq!(safe_value("  two\r\nlines\there ", &hide, 120), "two lines here");
        assert_eq!(safe_value("~/Music/x.ogg", &hide, 120), "<path>");
        assert_eq!(safe_value("..\\x", &hide, 120), "<path>");
        assert_eq!(safe_value("NEON HARBOR LIGHTS remix", &hide, 120), "<song> remix");
        // A two-letter name is too short to be a name: it would eat real words.
        assert_eq!(safe_value("stable", &hide, 120), "stable");
        assert_eq!(safe_value(&"a".repeat(500), &hide, 120).chars().count(), 121);
        assert!(percent_encode(&safe_value(&"é".repeat(500), &hide, 120)).len() <= 120 + 9);
    }

    #[test]
    fn only_a_plain_folder_name_is_looked_up() {
        assert!(plain_name(BASE));
        assert!(plain_name("Neon Harbor Lights (Extended Mix)"));
        for bad in ["", ".", "..", "../other", "a/b", "a\\b", "/abs", "nul\0"] {
            assert!(!plain_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn the_request_reads_the_frontends_field_names() {
        let r: FeedbackRequest = serde_json::from_value(json!({
            "kind": "chart", "base": BASE, "target": "stepforge", "difficulty": "hard",
            "songTitle": SONG, "includeTitle": true, "model": "StepForge Flow v1.4",
            "diagnostics": DIAGNOSTICS,
        }))
        .unwrap();
        assert_eq!(r.song_title.as_deref(), Some(SONG));
        assert!(r.include_title);
        // Everything is optional: a general request sends almost nothing.
        let r: FeedbackRequest = serde_json::from_value(json!({"kind": "general"})).unwrap();
        assert!(!r.include_title && r.base.is_none() && r.diagnostics.is_empty());
        let issue = serde_json::to_value(build_issue(&general_input())).unwrap();
        for key in ["title", "body", "url"] {
            assert!(issue[key].is_string(), "{key}");
        }
        assert_eq!(issue["truncated"], false);
    }
}
