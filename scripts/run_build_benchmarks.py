#!/usr/bin/env python3

import subprocess
import time
from pathlib import Path
import sys

ITERATIONS = 1

GENERATED = Path("examples/generated")
BUILD_DIR = Path("build/tmp")
RESULTS = Path("results/compile_report.csv")


def compile_file(source: Path, output: Path):

    cmd = [
        "clang++",
        "-std=c++23",
        "-O3",
        "-mavx",
        "-march=native",
        "-I./include",
        "-c",
        str(source),
        "-o",
        str(output),
    ]

    start = time.perf_counter()

    subprocess.run(
        cmd,
        check=True
    )

    end = time.perf_counter()

    return end - start


def main():
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_examples.py"
        ],
        check=True
    )

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS.open("w") as f:
        f.write("name,build_time\n")

    for source in sorted(GENERATED.glob("*.e.cpp")):

        name = source.stem.replace(".e", "")

        print(f"Compiling {name}")

        output = BUILD_DIR / f"{name}.o"

        for _ in range(ITERATIONS):
            elapsed = compile_file(
                source,
                output
            )

            with RESULTS.open("a") as f:
                f.write(
                    f"{name},{elapsed:.6f}\n"
                )


if __name__ == "__main__":
    main()
