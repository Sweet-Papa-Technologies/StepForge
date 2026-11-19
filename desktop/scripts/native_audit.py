#!/usr/bin/env python3
"""native_audit.py — does the bundled runtime carry every native library it loads?

    python3 desktop/scripts/native_audit.py desktop/src-tauri/runtime/python [windows|linux|macos]

Reads the dependency list of every native file under a folder (PE imports, ELF
DT_NEEDED, Mach-O load commands) and reports each library that is neither
inside that folder nor part of the operating system itself.

Why it exists: beta 6 for Windows shipped onnxruntime without the Microsoft
Visual C++ runtime it imports (msvcp140.dll, msvcp140_1.dll). numpy and scipy
bring private copies, so they imported; only onnxruntime did not — and only on
PCs without the VC++ redistributable, which the build machines and our test
machines all have. A smoke test on such a machine cannot see that; reading the
import tables can, on any machine.

Pure standard library; the PE and ELF readers need no platform tools, so the
Windows and Linux bundles can be audited from anywhere. Mach-O uses `otool`.
"""
from __future__ import annotations

import re
import struct
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #


def pe_imports(data: bytes) -> list[str] | None:
    """DLL names a PE file imports (import + delay-import tables), or None if
    `data` is not a PE file."""
    if len(data) < 0x40 or data[:2] != b"MZ":
        return None
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return None
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    optsz = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    dirs = opt + (112 if struct.unpack_from("<H", data, opt)[0] == 0x20B else 96)
    sections = []
    for i in range(nsec):
        vsize, vaddr, rsize, raw = struct.unpack_from("<IIII", data, opt + optsz + 40 * i + 8)
        sections.append((vaddr, max(vsize, rsize), raw))

    def offset(rva: int) -> int | None:
        for vaddr, size, raw in sections:
            if vaddr <= rva < vaddr + size:
                return rva - vaddr + raw
        return None

    def name_at(rva: int) -> str | None:
        o = offset(rva)
        if o is None:
            return None
        return data[o:data.index(b"\0", o)].decode("latin-1")

    names: list[str] = []
    # (data directory index, descriptor size, offset of the name RVA in it)
    for index, size, name_field in ((1, 20, 12), (13, 32, 4)):
        if dirs + 8 * index + 4 > opt + optsz:
            continue
        table = offset(struct.unpack_from("<I", data, dirs + 8 * index)[0] or 0)
        while table is not None:
            rva = struct.unpack_from("<I", data, table + name_field)[0]
            if not rva:
                break
            n = name_at(rva)
            if n:
                names.append(n)
            table += size
    return names


def elf_needed(data: bytes) -> list[str] | None:
    """DT_NEEDED entries of a 64-bit little-endian ELF file, or None if `data`
    is not one (static executables and non-ELF files give None or [])."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return None
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    dynamic = strtab = None
    sections = []
    for i in range(shnum):
        base = shoff + i * shentsize
        sh_type, = struct.unpack_from("<I", data, base + 4)
        sh_offset, sh_size, sh_link = struct.unpack_from("<QQI", data, base + 0x18)
        sections.append((sh_type, sh_offset, sh_size, sh_link))
    for sh_type, sh_offset, sh_size, sh_link in sections:
        if sh_type == 6:                                   # SHT_DYNAMIC
            dynamic = (sh_offset, sh_size)
            strtab = sections[sh_link][1] if sh_link < len(sections) else None
    if dynamic is None or strtab is None:
        return []
    out = []
    for o in range(dynamic[0], dynamic[0] + dynamic[1], 16):
        tag, val = struct.unpack_from("<qQ", data, o)
        if tag == 0:
            break
        if tag == 1:                                       # DT_NEEDED
            s = strtab + val
            out.append(data[s:data.index(b"\0", s)].decode("latin-1"))
    return out


def macho_deps(path: Path) -> list[str] | None:
    """Install names a Mach-O file links against (`otool -L`), or None if it is
    not Mach-O. The file's own id line is dropped."""
    with open(path, "rb") as f:
        if f.read(4) not in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe",
                             b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce", b"\xbf\xba\xfe\xca"):
            return None
    def otool(flag: str) -> list[str]:
        r = subprocess.run(["otool", "-arch", "all", flag, str(path)], capture_output=True, text=True)
        return r.stdout.splitlines()

    # A library's own install name (LC_ID_DYLIB) is listed among its "links".
    own = {line.strip() for line in otool("-D") if not line.rstrip().endswith(":")}
    deps = []
    for line in otool("-L"):
        if not line.startswith("\t"):
            continue
        name = line.strip().rsplit(" (", 1)[0]
        if name not in own:
            deps.append(name)
    return deps


# --------------------------------------------------------------------------- #
# What the operating system provides
# --------------------------------------------------------------------------- #

#: The Microsoft Visual C++ runtime. NOT part of Windows: it arrives with the
#: "VC++ redistributable", which a fresh PC may not have. Anything that imports
#: one of these needs a copy inside the bundle.
VC_RUNTIME = re.compile(r"^(msvcp|vcruntime|concrt|vcomp|vccorlib)\d+\w*\.dll$", re.I)

#: DLLs every supported Windows (10 and later) has. Deliberately a list and not
#: "whatever is in System32 on the build machine": that folder also holds the
#: VC++ redistributable there, which is the mistake this audit exists to catch.
WINDOWS_SYSTEM = frozenset("""
advapi32 authz avrt bcrypt bcryptprimitives cabinet cfgmgr32 combase comctl32 comdlg32
crypt32 cryptbase cryptsp d2d1 d3d10 d3d11 d3d12 d3d9 dbgcore dbghelp dnsapi dwmapi
dwrite dxcore dxgi gdi32 gdiplus imm32 iphlpapi kernel32 kernelbase ksuser mf mfplat
mfreadwrite mpr msacm32 mscoree msi msimg32 msvcp_win msvcrt mswsock ncrypt netapi32
normaliz ntdll ole32 oleacc oleaut32 opengl32 pathcch pdh powrprof profapi propsys psapi
rpcrt4 secur32 setupapi shcore shell32 shlwapi sspicli ucrtbase user32 userenv uxtheme
version win32u winhttp wininet winmm winspool.drv wintrust wldap32 ws2_32 wsock32
wtsapi32
""".split())

#: glibc (libmvec is its vector-math part), the dynamic loader, the GCC runtime
#: and zlib: on every Linux desktop, and what the manylinux wheel policy lets
#: wheels assume. Notably NOT here: libcrypt.so.1, which Arch-based
#: distributions do not install by default.
LINUX_SYSTEM = frozenset("""
libc.so.6 libm.so.6 libmvec.so.1 libdl.so.2 librt.so.1 libpthread.so.0 libutil.so.1
libresolv.so.2 libnsl.so.1 libanl.so.1 ld-linux-x86-64.so.2 ld-linux-aarch64.so.1
libgcc_s.so.1 libstdc++.so.6 linux-vdso.so.1 libz.so.1
""".split())

MACOS_SYSTEM_PREFIXES = ("/usr/lib/", "/System/Library/")


def is_windows_system(name: str) -> bool:
    n = name.lower()
    if n.startswith(("api-ms-win-", "ext-ms-")):
        return True
    return (n[:-4] if n.endswith(".dll") else n) in WINDOWS_SYSTEM


def unresolved(deps: dict[str, list[str]], bundled: set[str], platform: str) -> dict[str, list[str]]:
    """{library: [files that need it]} for every dependency that is neither in
    the bundle (`bundled`: lower-cased file names) nor part of the OS.

    `deps` maps each native file (path relative to the bundle) to what it loads.
    """
    missing: dict[str, list[str]] = {}
    for user, names in deps.items():
        for name in names:
            if platform == "windows":
                ok = name.lower() in bundled or (is_windows_system(name) and not VC_RUNTIME.match(name))
                key = name.lower()
            elif platform == "linux":
                ok = name in LINUX_SYSTEM or Path(name).name.lower() in bundled
                key = name
            else:
                relative = name.startswith("@")            # @rpath/ @loader_path/ @executable_path/
                ok = (Path(name).name.lower() in bundled if relative
                      else name.startswith(MACOS_SYSTEM_PREFIXES))
                key = name
            if not ok:
                missing.setdefault(key, []).append(user)
    return {k: sorted(set(v)) for k, v in sorted(missing.items())}


def scan(root: Path, platform: str) -> tuple[dict[str, list[str]], set[str]]:
    """Read every native file under `root`: ({relative path: dependencies},
    {lower-cased names of all files in the bundle})."""
    deps: dict[str, list[str]] = {}
    bundled: set[str] = set()
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        bundled.add(f.name.lower())
        if f.is_symlink():
            continue
        rel = str(f.relative_to(root))
        if platform == "windows":
            if f.suffix.lower() not in (".dll", ".pyd", ".exe"):
                continue
            found = pe_imports(f.read_bytes())
        elif platform == "linux":
            if ".so" not in f.name and f.parent.name != "bin":
                continue
            found = elf_needed(f.read_bytes())
        else:
            if f.suffix not in (".so", ".dylib") and f.parent.name != "bin":
                continue
            found = macho_deps(f)
        if found:
            deps[rel] = found
    return deps, bundled


def audit(root: Path, platform: str) -> dict[str, list[str]]:
    """Libraries the bundle at `root` loads but does not contain."""
    deps, bundled = scan(root, platform)
    return unresolved(deps, bundled, platform)


def describe(missing: dict[str, list[str]]) -> str:
    return "\n".join(f"  {lib}  <- {', '.join(users[:3])}{' …' if len(users) > 3 else ''}"
                     for lib, users in missing.items())


def host_platform() -> str:
    return "windows" if sys.platform.startswith("win") else "macos" if sys.platform == "darwin" else "linux"


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    platform = sys.argv[2] if len(sys.argv) == 3 else host_platform()
    deps, bundled = scan(root, platform)
    missing = unresolved(deps, bundled, platform)
    print(f"[native-audit] {len(deps)} native files under {root} ({platform})")
    if missing:
        print(f"[native-audit] {len(missing)} libraries are loaded but not bundled:\n{describe(missing)}")
        return 1
    print("[native-audit] every library they load is in the bundle or part of the OS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
