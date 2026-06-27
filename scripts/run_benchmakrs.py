#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import sys
import os


BENCHMARKS = [
    ("t1_c1_bench", "t1_c1"),

    ("t2_c1_bench", "t2_c1"),
    ("t2_c2_bench", "t2_c2"),

    ("t3_c1_bench", "t3_c1"),
    ("t3_c2_bench", "t3_c2"),
    ("t3_c3_bench", "t3_c3"),
]


def run_benchmark(executable: Path, name: str, output: Path):
    print(f"Running {name}...")

    cmd = [
        str(executable),
        "--benchmark_repetitions=10",
        "--benchmark_format=csv",
        "--benchmark_report_aggregates_only=false",
    ]

    # Linux CPU pinning equivalent to taskset
    if sys.platform.startswith("linux"):
        cmd = ["taskset", "-c", "0"] + cmd

    with output.open("w") as f:
        subprocess.run(
            cmd,
            stdout=f,
            check=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preset")
    args = parser.parse_args()

    bin_dir = Path("build") / args.preset / "bin" / "benchmarks"
    out_dir = Path("results") / "benchmarking"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not bin_dir.exists():
        print(f"Benchmark directory not found: {bin_dir}")
        return 1

    for exe, name in BENCHMARKS:
        executable = bin_dir / exe

        if os.name == "nt":
            executable = executable.with_suffix(".exe")

        if not executable.exists():
            print(f"Missing benchmark: {executable}")
            continue

        run_benchmark(
            executable,
            name,
            out_dir / f"{name}_results.csv"
        )

    # numpy benchmark
    subprocess.run(
        [
            sys.executable,
            "python/run_reference_benchmarks.py"
        ],
        check=True
    )

    print("All benchmarks completed.")


if __name__ == "__main__":
    raise SystemExit(main())
