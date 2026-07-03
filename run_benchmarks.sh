#!/usr/bin/env bash

set -euo pipefail

preset="$1"

if [ -z "$preset" ]; then
    echo "Usage: $0 <preset>"
    exit 1
fi

venv/bin/python scripts/run_benchmarks.py $preset

# BIN="./build/$preset/bin/benchmarks"
# OUT_DIR="results/benchmarking"
#
# mkdir -p "$OUT_DIR"
#
# run_bench () {
#     local exe=$1
#     local name=$2
#
#     echo "Running $name..."
#
#     taskset -c 0 "$exe" \
#         --benchmark_repetitions=5 \
#         --benchmark_format=csv \
#         --benchmark_report_aggregates_only=false \
#         --benchmark_min_time=1x \
#         > "${OUT_DIR}/${name}_results.csv"
# }
#
# # T1
# run_bench "${BIN}/t1_c1_bench" "t1_c1"
#
# # T2
# run_bench "${BIN}/t2_c1_bench" "t2_c1"
# run_bench "${BIN}/t2_c2_bench" "t2_c2"
#
# # T3
# run_bench "${BIN}/t3_c1_bench" "t3_c1"
# run_bench "${BIN}/t3_c2_bench" "t3_c2"
# run_bench "${BIN}/t3_c3_bench" "t3_c3"
#
# # numpy
# venv/bin/python scripts/run_reference_benchmarks.py
#
# echo "All benchmarks completed."
