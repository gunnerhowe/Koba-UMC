"""Custom setup.py for UMC — compiles the C extension during wheel build.

The C extension (umc/cext/kernels.c) is compiled into a shared library
(_umc_kernels.so/.dylib/.dll) that lives in umc/cext/ and is loaded via
ctypes at runtime.  If compilation fails (no C compiler, cross-compiling,
etc.) the build still succeeds — the pure-Python fallback handles everything.
"""

import os
import platform
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithCExt(build_py):
    """Extend the default build_py to compile the C extension first."""

    def run(self):
        self._compile_c_extension()
        super().run()

    def _compile_c_extension(self):
        """Attempt to compile the C extension.  Failure is non-fatal."""
        # Determine where the source lives relative to setup.py
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "umc", "cext")
        src_file = os.path.join(src_dir, "kernels.c")

        if not os.path.isfile(src_file):
            print("UMC: C source not found, skipping C extension build")
            return

        print("UMC: Compiling C extension ...")
        try:
            # Run the existing build script as a module so it resolves
            # paths correctly regardless of the current working directory.
            result = subprocess.run(
                [sys.executable, "-m", "umc.cext.build"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                print("UMC: C extension compiled successfully")
                if result.stdout.strip():
                    print(result.stdout.strip())
            else:
                print("UMC: C extension compilation failed (non-fatal)")
                if result.stderr.strip():
                    print(result.stderr.strip()[:500])
                if result.stdout.strip():
                    print(result.stdout.strip()[:500])
        except subprocess.TimeoutExpired:
            print("UMC: C extension compilation timed out (non-fatal)")
        except Exception as exc:
            print(f"UMC: C extension compilation error (non-fatal): {exc}")


def _shared_lib_globs():
    """Return glob patterns for the compiled shared library."""
    system = platform.system()
    if system == "Windows":
        return ["*.dll"]
    elif system == "Darwin":
        return ["*.dylib", "*.so"]
    else:
        return ["*.so"]


setup(
    cmdclass={
        "build_py": BuildPyWithCExt,
    },
    package_data={
        # Include the C source so sdists can recompile, plus any
        # pre-compiled shared libraries that the build step produced.
        "umc.cext": ["*.c", "*.h"] + _shared_lib_globs(),
    },
)
