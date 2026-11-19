#!/usr/bin/env python3
"""Tests for the packaging scripts (native_audit.py, stage_runtime.py).

    python3 -m unittest discover -s desktop/scripts -p "test_*.py"

Standard library only, and every binary is synthesized here, so the Windows
rules are tested on any machine.
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_audit  # noqa: E402
import stage_runtime  # noqa: E402


def fake_pe(imports: list[str], delayed: list[str] = ()) -> bytes:
    """A minimal PE32+ file whose import tables name `imports` / `delayed`."""
    rva, raw = 0x1000, 0x200
    body = bytearray()
    names = bytearray()
    imp = bytearray((len(imports) + 1) * 20)
    dly = bytearray((len(delayed) + 1) * 32)
    names_at = len(imp) + len(dly)
    for i, n in enumerate(imports):
        struct.pack_into("<I", imp, i * 20 + 12, rva + names_at + len(names))
        names += n.encode() + b"\0"
    for i, n in enumerate(delayed):
        struct.pack_into("<I", dly, i * 32 + 4, rva + names_at + len(names))
        names += n.encode() + b"\0"
    body += imp + dly + names
    opt = bytearray(240)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<I", opt, 108, 16)
    struct.pack_into("<II", opt, 112 + 8 * 1, rva, len(imp))
    if delayed:
        struct.pack_into("<II", opt, 112 + 8 * 13, rva + len(imp), len(dly))
    section = bytearray(40)
    section[:6] = b".rdata"
    struct.pack_into("<IIII", section, 8, len(body), rva, len(body), raw)
    head = bytearray(0x40)
    head[:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, 0x40)
    coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, len(opt), 0x2022)
    out = head + b"PE\0\0" + coff + opt + section
    return bytes(out + bytes(raw - len(out)) + body)


def fake_elf(needed: list[str]) -> bytes:
    """A minimal 64-bit little-endian ELF with DT_NEEDED entries."""
    strtab = bytearray(b"\0")
    dyn = bytearray()
    for n in needed:
        dyn += struct.pack("<qQ", 1, len(strtab))
        strtab += n.encode() + b"\0"
    dyn += struct.pack("<qQ", 0, 0)
    str_off = 64
    dyn_off = str_off + len(strtab)
    shoff = dyn_off + len(dyn)

    def sh(sh_type=0, offset=0, size=0, link=0) -> bytes:
        return struct.pack("<IIQQQQIIQQ", 0, sh_type, 0, 0, offset, size, link, 0, 0, 0)

    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x28, shoff)
    struct.pack_into("<HH", header, 0x3A, 64, 3)
    return bytes(header + strtab + dyn + sh() + sh(3, str_off, len(strtab)) + sh(6, dyn_off, len(dyn), 1))


class Readers(unittest.TestCase):
    def test_pe_imports_include_delay_loaded_dlls(self):
        pe = fake_pe(["KERNEL32.dll", "MSVCP140.dll"], delayed=["dbghelp.dll"])
        self.assertEqual(native_audit.pe_imports(pe), ["KERNEL32.dll", "MSVCP140.dll", "dbghelp.dll"])
        self.assertIsNone(native_audit.pe_imports(b"not a PE file at all" * 8))

    def test_elf_needed(self):
        self.assertEqual(native_audit.elf_needed(fake_elf(["libcrypt.so.1", "libc.so.6"])),
                         ["libcrypt.so.1", "libc.so.6"])
        self.assertIsNone(native_audit.elf_needed(b"#!/bin/sh\n"))

    @unittest.skipUnless(sys.platform.startswith("linux"), "reads this interpreter's ELF")
    def test_elf_reader_on_a_real_binary(self):
        exe = Path(sys.executable).resolve().read_bytes()
        self.assertIn("libc.so.6", native_audit.elf_needed(exe) or [])


def windows_runtime(root: Path) -> Path:
    """The Windows runtime as beta 6 shipped it: python-build-standalone brings
    vcruntime140*.dll, numpy a private msvcp140, and onnxruntime imports the
    system's msvcp140.dll and msvcp140_1.dll."""
    py = root / "python"
    capi = py / "Lib/site-packages/onnxruntime/capi"
    libs = py / "Lib/site-packages/numpy.libs"
    core = py / "Lib/site-packages/numpy/_core"
    for d in (capi, libs, core):
        d.mkdir(parents=True)
    (py / "python.exe").write_bytes(fake_pe(["python312.dll", "KERNEL32.dll", "VCRUNTIME140.dll"]))
    (py / "python312.dll").write_bytes(fake_pe(["KERNEL32.dll", "VCRUNTIME140.dll",
                                                "api-ms-win-crt-runtime-l1-1-0.dll"]))
    (py / "vcruntime140.dll").write_bytes(fake_pe(["KERNEL32.dll"]))
    (py / "vcruntime140_1.dll").write_bytes(fake_pe(["VCRUNTIME140.dll"]))
    (libs / "msvcp140-a4c2229b.dll").write_bytes(fake_pe(["VCRUNTIME140.dll", "VCRUNTIME140_1.dll"]))
    (core / "_multiarray_umath.cp312-win_amd64.pyd").write_bytes(
        fake_pe(["python312.dll", "msvcp140-a4c2229b.dll", "VCRUNTIME140.dll"]))
    ort = ["KERNEL32.dll", "dxgi.dll", "dbghelp.dll", "SETUPAPI.dll", "MSVCP140.dll",
           "MSVCP140_1.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll"]
    (capi / "onnxruntime.dll").write_bytes(fake_pe(ort))
    (capi / "onnxruntime_pybind11_state.pyd").write_bytes(fake_pe(ort + ["python312.dll", "onnxruntime.dll"]))
    return py


class WindowsRuntime(unittest.TestCase):
    def test_the_beta6_bundle_is_caught_missing_the_vc_runtime_onnxruntime_needs(self):
        with tempfile.TemporaryDirectory() as t:
            py = windows_runtime(Path(t))
            missing = native_audit.audit(py, "windows")
            self.assertEqual(sorted(missing), ["msvcp140.dll", "msvcp140_1.dll"])
            self.assertTrue(all("onnxruntime" in u for users in missing.values() for u in users))

    def test_a_vc_runtime_dll_is_never_counted_as_part_of_windows(self):
        for name in ("msvcp140.dll", "MSVCP140_1.dll", "vcruntime140.dll", "concrt140.dll",
                     "vcomp140.dll", "msvcp140_atomic_wait.dll"):
            self.assertTrue(native_audit.VC_RUNTIME.match(name), name)
            self.assertEqual(list(native_audit.unresolved({"x.pyd": [name]}, set(), "windows")), [name.lower()])
        for name in ("KERNEL32.dll", "api-ms-win-core-synch-l1-2-0.dll", "ucrtbase.dll", "dxgi.dll"):
            self.assertEqual(native_audit.unresolved({"x.pyd": [name]}, set(), "windows"), {})
        self.assertIn("mystery.dll", native_audit.unresolved({"x.pyd": ["mystery.dll"]}, set(), "windows"))

    def test_the_build_copies_the_missing_vc_runtime_next_to_python(self):
        with tempfile.TemporaryDirectory() as t:
            py = windows_runtime(Path(t))
            redist = Path(t) / "redist"
            redist.mkdir()
            (redist / "msvcp140.dll").write_bytes(fake_pe(["VCRUNTIME140.dll", "KERNEL32.dll"]))
            # A copied DLL may need another one: msvcp140_1 imports msvcp140_2 here.
            (redist / "msvcp140_1.dll").write_bytes(fake_pe(["MSVCP140.dll", "MSVCP140_2.dll"]))
            (redist / "msvcp140_2.dll").write_bytes(fake_pe(["MSVCP140.dll"]))
            (redist / "vcruntime140.dll").write_bytes(b"a different build: must not replace ours")
            ours = (py / "vcruntime140.dll").read_bytes()

            copied = stage_runtime.bundle_vc_runtime(py, sources=[redist])

            self.assertEqual(sorted(copied), ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll"])
            for name in copied:
                self.assertTrue((py / name).is_file(), f"{name} is not beside python.exe")
            self.assertEqual((py / "vcruntime140.dll").read_bytes(), ours)
            self.assertEqual(native_audit.audit(py, "windows"), {})
            stage_runtime.audit_native(py, "x86_64-pc-windows-msvc")     # does not raise

    def test_the_build_fails_rather_than_ship_without_a_needed_dll(self):
        with tempfile.TemporaryDirectory() as t:
            py = windows_runtime(Path(t))
            empty = Path(t) / "nothing"
            empty.mkdir()
            with self.assertRaises(SystemExit) as e:
                stage_runtime.bundle_vc_runtime(py, sources=[empty])
            self.assertIn("msvcp140", str(e.exception))
            with self.assertRaises(SystemExit) as e:
                stage_runtime.audit_native(py, "x86_64-pc-windows-msvc")
            self.assertIn("msvcp140.dll", str(e.exception))


class OtherPlatforms(unittest.TestCase):
    def test_linux_allows_glibc_and_the_bundle_but_not_libcrypt(self):
        with tempfile.TemporaryDirectory() as t:
            lib = Path(t) / "lib/python3.12"
            (lib / "lib-dynload").mkdir(parents=True)
            (lib / "site-packages/numpy.libs").mkdir(parents=True)
            (lib / "lib-dynload/_crypt.cpython-312-x86_64-linux-gnu.so").write_bytes(
                fake_elf(["libcrypt.so.1", "libc.so.6"]))
            (lib / "site-packages/numpy.libs/libscipy_openblas64_-abc.so").write_bytes(
                fake_elf(["libm.so.6", "libc.so.6"]))
            (lib / "site-packages/_umath.cpython-312-x86_64-linux-gnu.so").write_bytes(
                fake_elf(["libscipy_openblas64_-abc.so", "libstdc++.so.6", "libgcc_s.so.1", "libz.so.1"]))
            missing = native_audit.audit(Path(t), "linux")
            self.assertEqual(list(missing), ["libcrypt.so.1"])

    def test_macos_allows_system_libraries_and_bundled_rpath_libraries(self):
        deps = {"a.so": ["/usr/lib/libSystem.B.dylib", "@rpath/libonnxruntime.1.dylib",
                         "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"],
                "b.so": ["/opt/homebrew/opt/libomp/lib/libomp.dylib", "@loader_path/libgone.dylib"]}
        missing = native_audit.unresolved(deps, {"libonnxruntime.1.dylib"}, "macos")
        self.assertEqual(sorted(missing), ["/opt/homebrew/opt/libomp/lib/libomp.dylib",
                                           "@loader_path/libgone.dylib"])

    def test_platform_of_a_triple(self):
        self.assertEqual(stage_runtime.platform_of("x86_64-pc-windows-msvc"), "windows")
        self.assertEqual(stage_runtime.platform_of("aarch64-apple-darwin"), "macos")
        self.assertEqual(stage_runtime.platform_of("x86_64-unknown-linux-gnu"), "linux")


if __name__ == "__main__":
    unittest.main()
