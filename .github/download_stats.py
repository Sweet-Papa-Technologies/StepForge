#!/usr/bin/env python3
"""download_stats.py — take a daily snapshot of the release download counts.

    GH_TOKEN=... GITHUB_REPOSITORY=owner/repo python3 .github/download_stats.py <stats dir>

GitHub keeps one running total per release file and nothing else: no dates, no
history. The download-stats workflow runs this once a day and commits the result
to the `stats` branch, so the history builds up there:

  downloads.csv  one row per file per day: date, tag, asset, platform, downloads
                 (the running total GitHub reported that day). A re-run on the
                 same day replaces that day's rows.
  README.md      totals by platform and by release, and new downloads per day.

Standard library only.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

FIELDS = ["date", "tag", "asset", "platform", "downloads"]
PLATFORMS = ["Windows", "macOS (Apple silicon)", "macOS (Intel)", "Linux AppImage", "Linux .deb"]


def platform(asset: str) -> str | None:
    n = asset.lower()
    if n.startswith("sha256"):
        return None
    if n.endswith(".exe") or "windows" in n:
        return "Windows"
    if "macos-arm64" in n or "aarch64" in n:
        return "macOS (Apple silicon)"
    if "macos-x64" in n:
        return "macOS (Intel)"
    if n.endswith(".appimage"):
        return "Linux AppImage"
    if n.endswith(".deb"):
        return "Linux .deb"
    return "Other"


def releases(repo: str, token: str) -> list[dict]:
    out, page = [], 1
    while True:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28"})
        with urllib.request.urlopen(req, timeout=60) as r:
            batch = json.load(r)
        out += batch
        if len(batch) < 100:
            return out
        page += 1


def main() -> int:
    stats = Path(sys.argv[1])
    stats.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    repo, token = os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"]

    rels = [r for r in releases(repo, token) if not r.get("draft")]
    published = {r["tag_name"]: r["published_at"] or "" for r in rels}
    now = [{"date": today, "tag": r["tag_name"], "asset": a["name"], "platform": p,
            "downloads": a["download_count"]}
           for r in rels for a in r["assets"] if (p := platform(a["name"]))]

    csv_path = stats / "downloads.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["date"] != today]
    rows += now
    rows.sort(key=lambda r: (r["date"], published.get(r["tag"], ""), r["asset"]))
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    (stats / "README.md").write_text(report(rows, today, published, repo), encoding="utf-8")
    print(f"{today}: {sum(int(r['downloads']) for r in now)} downloads across "
          f"{len({r['tag'] for r in now})} releases")
    return 0


def report(rows: list[dict], today: str, published: dict, repo: str) -> str:
    latest = [r for r in rows if r["date"] == today]
    by_platform: dict[str, int] = defaultdict(int)
    by_release: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in latest:
        by_platform[r["platform"]] += int(r["downloads"])
        by_release[r["tag"]][r["platform"]] += int(r["downloads"])
    plats = [p for p in PLATFORMS if p in by_platform] + sorted(set(by_platform) - set(PLATFORMS))
    total = sum(by_platform.values())

    per_day: dict[str, int] = defaultdict(int)
    for r in rows:
        per_day[r["date"]] += int(r["downloads"])
    days = sorted(per_day)

    out = [
        "# StepForge downloads",
        "",
        f"Updated {today} (UTC) by the `download-stats` workflow from GitHub's release "
        "download counters. Raw data: [downloads.csv](downloads.csv).",
        "",
        f"**{total} downloads** of installers across {len(by_release)} releases.",
        "",
        "| Platform | Downloads |",
        "|---|---:|",
        *[f"| {p} | {by_platform[p]} |" for p in plats],
        "",
        "## By release",
        "",
        "| Release | " + " | ".join(plats) + " | Total |",
        "|---|" + "---:|" * (len(plats) + 1),
    ]
    for tag in sorted(by_release, key=lambda t: published.get(t, ""), reverse=True):
        cells = [str(by_release[tag].get(p, 0)) for p in plats]
        out.append(f"| [{tag}](https://github.com/{repo}/releases/tag/{tag}) | "
                   + " | ".join(cells) + f" | {sum(by_release[tag].values())} |")
    out += ["", "## New downloads per day", "",
            "Each day's increase in the running total (all releases and platforms).", "",
            "| Date | New | Running total |", "|---|---:|---:|"]
    for prev, day in reversed(list(zip(days, days[1:]))[-60:]):
        out.append(f"| {day} | {per_day[day] - per_day[prev]} | {per_day[day]} |")
    if days:
        out.append(f"| {days[0]} | (first snapshot) | {per_day[days[0]]} |")
    out += [
        "",
        "## Reading these numbers",
        "",
        "- GitHub counts every download of a file, including repeats by the same person.",
        "- The first snapshot (2026-11-17, 58 downloads) includes about 16 test downloads by the "
        "maintainers and what looks like one automated download of every file in every release.",
        "  Downloads by other people up to then were probably around ten.",
        "- Visits to step.fofo.dev and clicks on its download buttons are counted separately, "
        "without cookies: <https://stepforge.goatcounter.com>.",
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
