#!/usr/bin/env bash

set -euo pipefail

BIN="./build/gcc-release/bin/benchmarks"
OUT_DIR="results/benchmarking"

mkdir -p "$OUT_DIR"

run_bench () {
    local exe=$1
    local name=$2

    echo "Running $name..."

    taskset -c 0 "$exe" \
        --benchmark_repetitions=10 \
        --benchmark_format=csv \
        --benchmark_report_aggregates_only=false \
        > "${OUT_DIR}/${name}_results.csv"
}

# T1
run_bench "${BIN}/t1_c1_bench" "t1_c1"

# T2
run_bench "${BIN}/t2_c1_bench" "t2_c1"
run_bench "${BIN}/t2_c2_bench" "t2_c2"

# T3
run_bench "${BIN}/t3_c1_bench" "t3_c1"
run_bench "${BIN}/t3_c2_bench" "t3_c2"
run_bench "${BIN}/t3_c3_bench" "t3_c3"

# numpy
venv/bin/python python/run_reference_benchmarks.py

echo "All benchmarks completed."
