#!/usr/bin/env python3
"""Run all v15 probe experiments sequentially.

Usage: python scripts/run_v15_probes.py [a|b|c|all]
  a   - Run v15a only (big decoder)
  b   - Run v15b only (wide VQ)
  c   - Run v15c only (combined)
  all - Run all three sequentially (default)
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = {
    'a': ('experiment_v15a_big_decoder.py', 'Big Decoder (d=256, 6 layers)'),
    'b': ('experiment_v15b_wide_vq.py', 'Wide VQ (dim=32, cb_dim=16)'),
    'c': ('experiment_v15c_combined.py', 'Combined (d=192, vq_dim=24, 6 layers)'),
}

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    scripts_dir = Path(__file__).parent

    if which == 'all':
        keys = ['a', 'b', 'c']
    elif which in SCRIPTS:
        keys = [which]
    else:
        print(f"Unknown argument: {which}")
        print("Usage: python scripts/run_v15_probes.py [a|b|c|all]")
        sys.exit(1)

    print("=" * 60)
    print("  UMC v15 Probe Experiments")
    print("  Testing three hypotheses to break 3% RMSE wall")
    print("=" * 60)
    print(f"\n  v14b baseline: 2.95% RMSE, 46.9x entropy compression\n")

    results = {}
    total_start = time.perf_counter()

    for key in keys:
        script_name, label = SCRIPTS[key]
        script_path = scripts_dir / script_name

        print(f"\n{'='*60}")
        print(f"  Starting v15{key}: {label}")
        print(f"{'='*60}\n")

        t0 = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(scripts_dir.parent),
        )
        elapsed = time.perf_counter() - t0

        results[key] = {
            'label': label,
            'returncode': result.returncode,
            'time_min': elapsed / 60,
        }

        if result.returncode != 0:
            print(f"\n  WARNING: v15{key} exited with code {result.returncode}")
        print(f"\n  v15{key} completed in {elapsed/60:.1f} minutes")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n\n{'='*60}")
    print(f"  ALL PROBES COMPLETE ({total_elapsed/60:.1f} min total)")
    print(f"{'='*60}")
    for key, info in results.items():
        status = "OK" if info['returncode'] == 0 else f"FAILED ({info['returncode']})"
        print(f"  v15{key}: {info['label']} - {status} ({info['time_min']:.1f}m)")
    print()
    print("  Check output above for RMSE results from each experiment.")
    print("  Compare against v14b baseline: 2.95% RMSE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
