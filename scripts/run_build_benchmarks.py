#!/usr/bin/env python3

import subprocess
import argparse
import time
from pathlib import Path
import sys
from config import COMPILE_ITERATIONS

COMPILE_ITERATIONS = 3

GENERATED = Path("examples/generated")

def compile_file(preset, source: Path, output: Path):

    if 'clang' in preset:
        compiler = 'clang++'
    elif 'gcc' in preset:
        compiler = 'g++'
    else:
        raise ValueError(f"Preset not implemented: {preset}")

    if 'release' in preset:
        opt = '-O3'
    elif 'relwdebinfo' in preset:
        opt = '-O2'
    elif 'debug' in preset:
        opt = '-O0'
    else:
        raise ValueError(f"Preset not implemented: {preset}")

    cmd = [
        compiler,
        "-std=c++23",
        opt,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("preset")
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_examples.py"
        ],
        check=True
    )

    BUILD_DIR = Path(f"build/{args.preset}/tmp")
    RESULTS = Path(f"results/{args.preset}/compile_report.csv")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS.open("w") as f:
        f.write("name,build_time\n")

    for source in sorted(GENERATED.glob("*.e.cpp")):

        name = source.stem.replace(".e", "")

        print(f"Compiling {name}")

        output = BUILD_DIR / f"{name}.o"

        for _ in range(COMPILE_ITERATIONS):
            elapsed = compile_file(
                args.preset,
                source,
                output
            )

            with RESULTS.open("a") as f:
                f.write(
                    f"{name},{elapsed:.6f}\n"
                )


if __name__ == "__main__":
    main()
