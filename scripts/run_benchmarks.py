#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import sys
import os
import pandas as pd
from io import StringIO


BENCHMARK_GROUPS = [
    "t1_c1",
    "t2_c1",
    "t2_c2",
    "t3_c1",
    "t3_c2",
    "t3_c3",
]

def run_all_benchmarks(executable: Path):
    cmd = [
        str(executable),
        "--benchmark_repetitions=5",
        "--benchmark_format=csv",
        "--benchmark_report_aggregates_only=false",
    ]
    print(cmd)

    if sys.platform.startswith("linux"):
        cmd = ["taskset", "-c", "0"] + cmd

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )

    return result.stdout


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
    all_csv = []

    for exe in executables:
        print(f"Running {exe.name}...")

        cmd = [
            str(exe),
            "--benchmark_repetitions=1",
            "--benchmark_format=csv",
            "--benchmark_report_aggregates_only=false",
        ]

        if sys.platform.startswith("linux"):
            cmd = ["taskset", "-c", "0"] + cmd

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )

        all_csv.append(result.stdout)

    csv_text = "\n".join(all_csv)

    # ---------------------------------------------------------
    # store results
    # ---------------------------------------------------------
    print("Store results...")
    store_results(csv_text, out_dir)

    # reference benchmarks
    subprocess.run(
        [
            sys.executable,
            "scripts/run_reference_benchmarks.py"
        ],
        check=True
    )

    print("Done.")


if __name__ == "__main__":
    raise SystemExit(main())
