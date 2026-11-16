#!/usr/bin/env python3
"""Release notes for the GitHub release. Usage (the workflow pipes the commit
subjects since the previous tag on stdin):

    release_notes.py VERSION PRERELEASE(true|false) PREVIOUS_TAG < changes
"""
import sys

version, pre, prev = sys.argv[1], sys.argv[2] == "true", sys.argv[3]
changes = sys.stdin.read().strip() or "- First release."
beta = """
> **This is a beta.** Expect rough edges, and please tell us about them:
> [open an issue](https://github.com/Sweet-Papa-Technologies/StepForge/issues),
> or find us on [Discord](https://discord.gg/kFHgCKr5E3).
""" if pre else ""

print(f"""## StepForge {version}

Turn any song into playable StepMania / ITGmania dance charts, on your own
computer. Everything the app needs, including its charting model, is inside
the download.
{beta}
### Download

| Platform | File |
|---|---|
| macOS 11+ (Apple silicon) | `StepForge-{version}-macos-arm64.dmg` |
| Windows 10/11 (x64) | `StepForge-{version}-windows-x64-setup.exe` |
| Linux (x64) | `StepForge-{version}-linux-x86_64.AppImage` or `StepForge-{version}-linux-amd64.deb` |

Verify a download with `SHA256SUMS.txt`. The macOS app is signed and notarized
by Apple; the Windows installer is signed by Sweet Papa Technologies.

### Changes{f" since {prev}" if prev else ""}

{changes}
""")
