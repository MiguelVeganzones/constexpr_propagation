#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import sys
import os
import pandas as pd
from io import StringIO
from config import BENCHMARK_MIN_RUNTIME


BENCHMARK_GROUPS = [
    "t1_c1",
    "t2_c1",
    "t2_c2",
    "t3_c1",
    "t3_c2",
    "t3_c3",
]

def store_results(csv_text: str, out_dir: Path):
    df = pd.read_csv(StringIO(csv_text))
    out_file = out_dir / f"benchmark_results.csv"
    df.to_csv(out_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preset")
    args = parser.parse_args()

    bin_dir = Path("build") / args.preset / "bin/benchmarks"

    out_dir = Path("results") / "benchmarking"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bin_dir.exists():
        print(f"Benchmark directory not found: {bin_dir}")
        return 1

    # ---------------------------------------------------------
    # collect executables
    # ---------------------------------------------------------
    executables = []

    for path in bin_dir.iterdir():
        if path.is_file():
            if os.name == "nt":
                if path.suffix == ".exe":
                    executables.append(path)
            else:
                if os.access(path, os.X_OK):
                    executables.append(path)

    if not executables:
        print(f"No executables found in {bin_dir}")
        return 1

    print(f"Found {len(executables)} executables")

    # ---------------------------------------------------------
    # run all benchmarks and concatenate CSV output
    # ---------------------------------------------------------
    # all_csv = ["name,iterations,real_time,time_unit"]
    all_csv = []

    executables.sort()
    try:
        for exe in executables:
            cmd = [
                str(exe),
                f"--benchmark_min_time={BENCHMARK_MIN_RUNTIME}s",
                "--benchmark_format=csv",
                "--benchmark_report_aggregates_only=false",
            ]

            # cmd = [
            #     str(exe)
            # ]

            if sys.platform.startswith("linux"):
                cmd = ["taskset", "-c", "0"] + cmd

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
            all_csv.append(result.stdout)
    except:
        print("Benchmark run terminated unexpectedly")
        all_csv = all_csv[:-1]
        pass

    csv_text = "\n".join(all_csv)

    # ---------------------------------------------------------
    # store results
    # ---------------------------------------------------------
    print("Store results...")
    store_results(csv_text, out_dir)

    # reference benchmarks
    if sys.platform.startswith("linux"):
        cmd = [
            "taskset", "-c", "0",
            sys.executable,
            "scripts/run_reference_benchmarks.py",
        ]
    else:
        cmd = [
            sys.executable,
            "scripts/run_reference_benchmarks.py",
        ]

    subprocess.run(cmd, check=True)

    print("Done.")


if __name__ == "__main__":
    raise SystemExit(main())
