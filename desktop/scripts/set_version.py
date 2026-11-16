#!/usr/bin/env python3
"""set_version.py — stamp a release version into the app's manifests.

    python3 desktop/scripts/set_version.py 0.1.0-beta.1

Writes `version` in src-tauri/tauri.conf.json (what the bundles and the app's
About/SideNav show), src-tauri/Cargo.toml and package.json. The release
workflow runs it from the tag, so the files in git keep a development version.
"""
import json
import re
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$")


def main() -> int:
    if len(sys.argv) != 2 or not SEMVER.match(sys.argv[1]):
        print("usage: set_version.py X.Y.Z[-pre]", file=sys.stderr)
        return 2
    v = sys.argv[1]
    for name in ("src-tauri/tauri.conf.json", "package.json"):
        p = DESKTOP / name
        d = json.loads(p.read_text())
        d["version"] = v
        p.write_text(json.dumps(d, indent=2) + "\n")
    cargo = DESKTOP / "src-tauri" / "Cargo.toml"
    cargo.write_text(re.sub(r'(?m)^version = "[^"]*"', f'version = "{v}"', cargo.read_text(), count=1))
    print(f"version set to {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
