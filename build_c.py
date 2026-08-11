"""
build_c.py — Compile c_src/radish_core.c → radish_core.{dll|so}

Usage:
    python build_c.py            # autodetect compiler, build optimized
    python build_c.py --debug    # -O0 -g for diagnostics

Output is placed alongside this script so methods_c.py can find it via
ctypes.CDLL("./radish_core.dll").
"""
from __future__ import annotations
import argparse, os, subprocess, sys, shutil, tempfile, platform
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
SRC    = ROOT / "c_src" / "radish_core.c"
LIBNAM = "radish_core.dll" if os.name == "nt" else "radish_core.so"
OUT    = ROOT / LIBNAM

# Known MSVC locations (update if your install path differs)
VCVARS_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
]


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def build_msvc(debug: bool):
    vcvars = next((c for c in VCVARS_CANDIDATES if Path(c).exists()), None)
    if not vcvars:
        return False
    flags = "/Od /Zi /MDd" if debug else "/O2 /Oi /GL /fp:fast /MD"
    # Wrap cl.exe inside vcvars64 environment via a temporary batch file.
    bat = tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False)
    bat.write(
        f'@echo off\r\n'
        f'call "{vcvars}" >nul\r\n'
        f'cl.exe {flags} /LD "{SRC}" /Fe"{OUT}" /Fo"{ROOT}\\radish_core.obj"\r\n'
    )
    bat.close()
    try:
        r = subprocess.run(["cmd", "/c", bat.name], cwd=str(ROOT))
    finally:
        os.unlink(bat.name)
    # Cleanup intermediate files MSVC drops in CWD
    for ext in (".obj", ".exp", ".lib", ".pdb", ".ilk"):
        p = ROOT / f"radish_core{ext}"
        if p.exists():
            try: p.unlink()
            except OSError: pass
    return r.returncode == 0 and OUT.exists()


def _find_gnu_compiler():
    """Find clang/gcc on PATH or in common Conda locations."""
    configured = os.environ.get("CC")
    candidates = [
        configured,
        shutil.which("clang"),
        shutil.which("gcc"),
        str(Path(sys.prefix) / "Library" / "mingw-w64" / "bin" / "gcc.exe"),
        str(Path(sys.prefix) / "Library" / "bin" / "gcc.exe"),
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)


def build_unix(debug: bool):
    cc = _find_gnu_compiler()
    if not cc:
        return False
    # Avoid -march=native: older Conda MinGW assemblers can fail on newer
    # host CPUs, and a portable shared library is preferable for this repo.
    flags = ["-O0", "-g"] if debug else ["-O3", "-ffast-math"]
    platform_flags = ["-static-libgcc"] if os.name == "nt" else ["-fPIC"]
    # Use paths relative to ROOT so older MinGW builds do not have to parse
    # non-ASCII characters that may occur in the repository's parent path.
    cmd = [cc, "-shared", *platform_flags, *flags,
           "-o", OUT.name, str(SRC.relative_to(ROOT)), "-lm"]
    env = os.environ.copy()
    env["PATH"] = str(Path(cc).parent) + os.pathsep + env.get("PATH", "")
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--mingw", action="store_true",
                    help="Use MinGW gcc on Windows")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"ERROR: source not found at {SRC}"); sys.exit(1)

    print(f"Compiling {SRC} -> {OUT}")

    ok = False
    if os.name == "nt" and not args.mingw:
        ok = build_msvc(args.debug)
        if not ok:
            print("MSVC failed or not found, trying gcc/clang...")
            ok = build_unix(args.debug)
    else:
        ok = build_unix(args.debug)

    if not ok:
        print("BUILD FAILED. Verify a C compiler is on PATH.")
        sys.exit(2)

    print(f"OK -> {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
