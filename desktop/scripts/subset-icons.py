#!/usr/bin/env python3
"""Subset Material Symbols to just the icons this app uses.

The upstream variable font is ~3.3 MB — the whole icon set — and the webview
pays to decompress and build glyph tables for it on every cold start. We use
about two dozen icons.

Icon names are rendered as ligatures (`<span class="material-symbols-outlined">
search</span>`), so the subset must keep the letter glyphs and the GSUB ligature
rules as well as the icon glyphs. `--no-layout-closure` is the key: without it,
including a-z would pull in every ligature in the font (i.e. the whole set).

The icon list is derived from the source rather than hand-maintained: every
lowercase token in src/ that is also a real glyph name is kept. That
over-includes a few words (`error`, `person`, …) which is harmless, and
guarantees we never drop an icon that is only referenced dynamically.

Run:  python3 scripts/subset-icons.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
SRC = APP / "src"
FONT = APP / "node_modules/material-symbols/material-symbols-outlined.woff2"
OUT_DIR = APP / "src/assets"
OUT = OUT_DIR / "material-symbols-subset.woff2"

TOKEN = re.compile(r"[a-z][a-z0-9_]{2,}")
# Icon names reach the DOM as the text of a `material-symbols-outlined` span,
# either literally or from an inline ternary (`{{ busy ? "progress_activity" :
# "save" }}`), so both spellings are pulled out of the markup for --check.
ICON_SPAN = re.compile(r'class="material-symbols-outlined[^"]*"[^>]*>([^<]*)<')
BARE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
QUOTED_NAME = re.compile(r'"([a-z][a-z0-9_]*)"')
# In the font, an underscore component is the glyph named "underscore".
GLYPH_TO_CHAR = {"underscore": "_"}


def referenced_icons() -> set[str]:
    """Icon names the markup actually asks for."""
    names: set[str] = set()
    for p in SRC.rglob("*.vue"):
        for text in ICON_SPAN.findall(p.read_text(encoding="utf-8", errors="ignore")):
            text = text.strip()
            if BARE_NAME.match(text):
                names.add(text)
            elif "?" in text:
                # A ternary. Only the BRANCHES are icon names — a string in the
                # condition is being compared, not rendered, and counting it
                # reports a missing icon that no one ever asked for
                # (`m === "player" ? "play_arrow" : "image"`). Splitting on the
                # first `?` is safe even when that `?` belongs to optional
                # chaining, because both branches still follow it.
                names.update(QUOTED_NAME.findall(text.split("?", 1)[1]))
            else:
                names.update(QUOTED_NAME.findall(text))
    return names


def ligatures_in(path: Path) -> set[str]:
    """Every ligature the font can actually form.

    Material Symbols renders an icon by substituting the LETTERS of its name, so
    "does this font have icon X" means "is there a ligature spelling X" — not "is
    there a glyph named X". Subsetting renames glyphs, so checking glyph names
    finds nothing and looks like total failure. The rules also sit inside
    ExtensionSubst lookups, which have to be unwrapped or every font reads as
    having zero ligatures.
    """
    from fontTools.ttLib import TTFont
    out: set[str] = set()
    for lookup in TTFont(path)["GSUB"].table.LookupList.Lookup:
        for sub in lookup.SubTable:
            sub = getattr(sub, "ExtSubTable", sub)
            for first, ligs in getattr(sub, "ligatures", {}).items():
                for lig in ligs:
                    out.add(GLYPH_TO_CHAR.get(first, first)
                            + "".join(GLYPH_TO_CHAR.get(c, c) for c in lig.Component))
    return out


def check() -> int:
    """Fail if the committed subset cannot render an icon the app uses.

    This exists because the failure it catches is silent and shipped: the subset
    is a build artefact that lives in git, nothing regenerates it, and an icon
    added to a .vue file renders as blank until someone remembers this script.
    That is exactly how `hearing` and `wifi_tethering` — the Inkling "Run audio
    probe" and "Test connection" buttons — went out with no icon on them.
    """
    if not OUT.is_file():
        print(f"error: {OUT} missing — run `npm run icons`", file=sys.stderr)
        return 1
    missing = sorted(referenced_icons() - ligatures_in(OUT))
    if missing:
        print(f"error: {len(missing)} icon(s) used in src/ are not in "
              f"{OUT.name} and will render as blank:\n  " + "\n  ".join(missing)
              + "\n\nRegenerate it with:  npm run icons", file=sys.stderr)
        return 1
    print(f"icons ok: every referenced icon resolves in {OUT.name}")
    return 0


def main() -> int:
    if "--check" in sys.argv:
        try:
            return check()
        except ImportError:
            # A contributor without fontTools should not be blocked from
            # building; the check is a guard, not a dependency.
            print("warning: fontTools not installed — skipping the icon check "
                  "(pip install fonttools brotli)", file=sys.stderr)
            return 0
    if not FONT.is_file():
        print(f"error: {FONT} missing — run `pnpm install` first", file=sys.stderr)
        return 1
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        print("error: fontTools not installed (pip install fonttools brotli)", file=sys.stderr)
        return 1

    glyphs = set(TTFont(FONT).getGlyphOrder())

    tokens: set[str] = set()
    for p in SRC.rglob("*"):
        if p.suffix in {".vue", ".ts"} and p.is_file():
            tokens.update(TOKEN.findall(p.read_text(encoding="utf-8", errors="ignore")))

    icons = sorted(tokens & glyphs)
    if not icons:
        print("error: no icon names found", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "fontTools.subset", str(FONT),
        f"--output-file={OUT}",
        "--flavor=woff2",
        # Material Symbols drives icon ligatures through rlig + rclt (not liga).
        "--layout-features+=liga,dlig,rlig,rclt,calt,ccmp",
        "--no-layout-closure",          # else a-z drags in every ligature
        f"--glyphs={','.join(icons)}",
        "--text=abcdefghijklmnopqrstuvwxyz_0123456789",
    ]
    subprocess.run(cmd, check=True)

    before = FONT.stat().st_size
    after = OUT.stat().st_size
    print(f"{len(icons)} icons: {before/1e6:.2f} MB -> {after/1024:.1f} KB "
          f"({100 - after / before * 100:.1f}% smaller)")
    print("icons:", " ".join(icons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
