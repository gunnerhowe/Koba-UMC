#!/usr/bin/env python
"""Build the UMC C extension kernels.

Usage:
    python -m umc.cext.build          # Build in-place
    python umc/cext/build.py          # Same thing

This compiles kernels.c into a shared library that the Python wrapper
loads via ctypes. No special dependencies needed — just a C compiler.
"""

import os
import platform
import shutil
import subprocess
import sys


def build():
    """Build the C extension."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(base_dir, "kernels.c")

    if not os.path.isfile(src):
        print(f"Error: {src} not found")
        return False

    system = platform.system()

    if system == "Windows":
        output = os.path.join(base_dir, "_umc_kernels.dll")
        # Try MSVC first, then GCC (MinGW)
        if _try_msvc(src, output):
            return True
        if _try_gcc(src, output, system):
            return True
        print("Error: No C compiler found. Install MSVC or MinGW-w64.")
        return False
    elif system == "Darwin":
        output = os.path.join(base_dir, "_umc_kernels.dylib")
        return _try_gcc(src, output, system)
    else:
        output = os.path.join(base_dir, "_umc_kernels.so")
        return _try_gcc(src, output, system)


def _try_msvc(src, output):
    """Try to build with MSVC."""
    cl = shutil.which("cl")
    if cl is None:
        return False

    try:
        cmd = ["cl", "/O2", "/LD", src, f"/Fe{output}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.isfile(output):
            print(f"Built {output} (MSVC)")
            # Clean up .obj and .lib files
            for ext in [".obj", ".lib", ".exp"]:
                cleanup = output.replace(".dll", ext)
                if os.path.isfile(cleanup):
                    os.remove(cleanup)
            return True
        print(f"MSVC failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"MSVC error: {e}")
    return False


def _try_gcc(src, output, system):
    """Try to build with GCC/Clang."""
    for compiler in ["gcc", "cc", "clang"]:
        cc = shutil.which(compiler)
        if cc is None:
            continue

        try:
            cmd = [cc, "-O2", "-shared"]
            if system != "Windows":
                cmd.append("-fPIC")
            cmd += ["-o", output, src]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.isfile(output):
                print(f"Built {output} ({compiler})")
                return True
            print(f"{compiler} failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"{compiler} error: {e}")

    return False


if __name__ == "__main__":
    success = build()
    sys.exit(0 if success else 1)
