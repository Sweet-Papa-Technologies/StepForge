//! Minimal StepMania simfile reader — enough to drive the in-app chart player.
//!
//! We parse the `.ssc` (falling back to `.sm`) that STEPFORGE actually emits, so
//! the player shows exactly what the game would: real columns, real timing.
//! `design.json` is not usable here — it only carries onset refs, no panels.
//!
//! Note rows are 4 chars for dance-single: `0` none, `1` tap, `2` hold head,
//! `3` hold/roll tail, `4` roll head, `M` mine. Measures are separated by `,`
//! and each measure spans 4 beats regardless of how many rows it contains.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Note {
    /// Seconds from the start of the audio file.
    pub time: f64,
    pub beat: f64,
    pub column: usize,
    /// "tap" | "hold" | "roll" | "mine"
    pub kind: String,
    /// End time for holds/rolls, else None.
    pub end_time: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Chart {
    pub difficulty: String,
    pub meter: i64,
    pub stepstype: String,
    pub columns: usize,
    /// Seconds at which beat 0 occurs (StepMania `#OFFSET` is its negation).
    pub beat0_time: f64,
    pub bpm: f64,
    pub notes: Vec<Note>,
}

/// `#TAG:value;` — returns the first value for `tag`.
fn tag(src: &str, name: &str) -> Option<String> {
    let needle = format!("#{name}:");
    let start = src.find(&needle)? + needle.len();
    let rest = &src[start..];
    let end = rest.find(';')?;
    Some(rest[..end].trim().to_string())
}

/// `#BPMS:0.000=119.945,16.000=140.0;` -> [(beat, bpm)]
fn parse_bpms(src: &str) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    if let Some(raw) = tag(src, "BPMS") {
        for pair in raw.split(',') {
            let mut it = pair.split('=');
            if let (Some(b), Some(v)) = (it.next(), it.next()) {
                if let (Ok(b), Ok(v)) = (b.trim().parse::<f64>(), v.trim().parse::<f64>()) {
                    out.push((b, v));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if out.is_empty() {
        out.push((0.0, 120.0));
    }
    out
}

/// Seconds for a beat, walking BPM segments.
fn beat_to_time(beat: f64, beat0: f64, bpms: &[(f64, f64)]) -> f64 {
    let mut t = beat0;
    for i in 0..bpms.len() {
        let (seg_start, bpm) = bpms[i];
        if bpm <= 0.0 {
            continue;
        }
        let seg_end = bpms.get(i + 1).map(|x| x.0).unwrap_or(f64::INFINITY);
        if beat <= seg_start {
            break;
        }
        let span = beat.min(seg_end) - seg_start;
        if span > 0.0 {
            t += span * 60.0 / bpm;
        }
        if beat <= seg_end {
            break;
        }
    }
    t
}

/// Strip `//` comments, which simfiles may carry inside note data.
fn strip_comments(s: &str) -> String {
    s.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_notedata(raw: &str, beat0: f64, bpms: &[(f64, f64)]) -> (Vec<Note>, usize) {
    let cleaned = strip_comments(raw);
    let mut notes = Vec::new();
    let mut columns = 0usize;
    // column -> (start_beat, kind) for an open hold/roll
    let mut open: std::collections::HashMap<usize, (f64, String)> = std::collections::HashMap::new();

    for (m_idx, measure) in cleaned.split(',').enumerate() {
        let rows: Vec<&str> = measure
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        if rows.is_empty() {
            continue;
        }
        let rows_n = rows.len() as f64;
        for (r_idx, row) in rows.iter().enumerate() {
            let beat = 4.0 * (m_idx as f64 + r_idx as f64 / rows_n);
            let time = beat_to_time(beat, beat0, bpms);
            columns = columns.max(row.chars().count());
            for (c, ch) in row.chars().enumerate() {
                match ch {
                    '1' => notes.push(Note { time, beat, column: c, kind: "tap".into(), end_time: None }),
                    'M' | 'm' => notes.push(Note { time, beat, column: c, kind: "mine".into(), end_time: None }),
                    '2' => { open.insert(c, (beat, "hold".into())); }
                    '4' => { open.insert(c, (beat, "roll".into())); }
                    '3' => {
                        if let Some((start_beat, kind)) = open.remove(&c) {
                            notes.push(Note {
                                time: beat_to_time(start_beat, beat0, bpms),
                                beat: start_beat,
                                column: c,
                                kind,
                                end_time: Some(time),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    notes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    (notes, if columns == 0 { 4 } else { columns })
}

/// Parse one difficulty out of a simfile. `difficulty` matches case-insensitively.
pub fn parse_chart(src: &str, difficulty: &str) -> Option<Chart> {
    let beat0 = -tag(src, "OFFSET").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    let bpms = parse_bpms(src);
    let want = difficulty.to_lowercase();

    // .ssc: one `#NOTEDATA:` block per chart.
    if src.contains("#NOTEDATA:") {
        for section in src.split("#NOTEDATA:").skip(1) {
            let diff = tag(section, "DIFFICULTY").unwrap_or_default();
            if diff.to_lowercase() != want {
                continue;
            }
            let start = section.find("#NOTES:")? + "#NOTES:".len();
            let body = &section[start..];
            let end = body.find(';').unwrap_or(body.len());
            let (notes, columns) = parse_notedata(&body[..end], beat0, &bpms);
            return Some(Chart {
                difficulty: diff,
                meter: tag(section, "METER").and_then(|v| v.parse().ok()).unwrap_or(0),
                stepstype: tag(section, "STEPSTYPE").unwrap_or_else(|| "dance-single".into()),
                columns,
                beat0_time: beat0,
                bpm: bpms[0].1,
                notes,
            });
        }
        return None;
    }

    // .sm: `#NOTES: type: desc: difficulty: meter: radar: <data>;`
    for section in src.split("#NOTES:").skip(1) {
        let end = section.find(';').unwrap_or(section.len());
        let block = &section[..end];
        let parts: Vec<&str> = block.splitn(6, ':').collect();
        if parts.len() < 6 {
            continue;
        }
        let diff = parts[2].trim().to_string();
        if diff.to_lowercase() != want {
            continue;
        }
        let (notes, columns) = parse_notedata(parts[5], beat0, &bpms);
        return Some(Chart {
            difficulty: diff,
            meter: parts[3].trim().parse().unwrap_or(0),
            stepstype: parts[0].trim().to_string(),
            columns,
            beat0_time: beat0,
            bpm: bpms[0].1,
            notes,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const SSC: &str = "\
#OFFSET:-0.423;
#BPMS:0.000=120.000;
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Easy;
#METER:3;
#NOTES:
0000
0000
0000
0000
,
1000
0000
0010
0000
;
";

    #[test]
    fn parses_beats_columns_and_timing() {
        let c = parse_chart(SSC, "easy").expect("chart");
        assert_eq!(c.difficulty, "Easy");
        assert_eq!(c.meter, 3);
        assert_eq!(c.columns, 4);
        // #OFFSET is negated: beat 0 lands 0.423s into the audio.
        assert!((c.beat0_time - 0.423).abs() < 1e-9);

        assert_eq!(c.notes.len(), 2);
        // Measure 1 row 0 -> beat 4; at 120bpm that's 2s after beat 0.
        assert_eq!(c.notes[0].column, 0);
        assert!((c.notes[0].beat - 4.0).abs() < 1e-9);
        assert!((c.notes[0].time - (0.423 + 2.0)).abs() < 1e-9);
        // Row 2 of 4 -> beat 6.
        assert_eq!(c.notes[1].column, 2);
        assert!((c.notes[1].beat - 6.0).abs() < 1e-9);
    }

    #[test]
    fn pairs_hold_head_with_tail() {
        let ssc = SSC.replace("1000\n0000\n0010", "2000\n0000\n3000");
        let c = parse_chart(&ssc, "easy").expect("chart");
        let hold = c.notes.iter().find(|n| n.kind == "hold").expect("hold");
        assert!((hold.beat - 4.0).abs() < 1e-9);
        assert!(hold.end_time.unwrap() > hold.time);
    }

    #[test]
    fn unknown_difficulty_is_none() {
        assert!(parse_chart(SSC, "challenge").is_none());
    }

    /// Validate against a real STEPFORGE emission when one is present. Skipped
    /// (not failed) on a clean checkout so the suite stays hermetic.
    #[test]
    fn matches_real_stepforge_output() {
        let repo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap().to_path_buf();
        let dir = repo.join("build/stepmania");
        let Ok(entries) = std::fs::read_dir(&dir) else { return };
        for song in entries.flatten() {
            let Some(ssc) = std::fs::read_dir(song.path()).ok().and_then(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .find(|p| p.extension().and_then(|s| s.to_str()) == Some("ssc"))
            }) else { continue };
            let src = std::fs::read_to_string(&ssc).unwrap();
            let report: serde_json::Value = match std::fs::read_to_string(song.path().join("report.json")) {
                Ok(t) => serde_json::from_str(&t).unwrap(),
                Err(_) => continue,
            };
            let charts = report["charts"].as_object().unwrap();
            for (diff, info) in charts {
                // `report.notes` counts *placements* (a jump is one placement on
                // two panels), so arrows == placements + jumps.
                let placements = info["notes"].as_u64().unwrap_or(0);
                let jumps = info["jumps"].as_u64().unwrap_or(0);
                let c = parse_chart(&src, diff)
                    .unwrap_or_else(|| panic!("failed to parse '{diff}' from {}", ssc.display()));
                let arrows = c.notes.iter().filter(|n| n.kind != "mine").count() as u64;
                assert_eq!(
                    arrows,
                    placements + jumps,
                    "{}/{diff}: parsed {arrows} arrows, report says {placements} placements + {jumps} jumps",
                    song.file_name().to_string_lossy()
                );
                // Distinct onset times should equal the placement count.
                let mut times: Vec<f64> = c.notes.iter().filter(|n| n.kind != "mine").map(|n| n.beat).collect();
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                times.dedup();
                assert_eq!(times.len() as u64, placements, "distinct rows should equal placements");
                assert!(c.columns == 4, "dance-single should have 4 columns");
                assert!(c.notes.iter().all(|n| n.time >= 0.0), "negative note time");
            }
            return; // one real song is enough
        }
    }
}
