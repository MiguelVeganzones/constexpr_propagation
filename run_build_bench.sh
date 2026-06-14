#!/usr/bin/env bash

set -euo pipefail

# regenerate examples
venv/bin/python python/generate_examples.py

# prepare output dirs
mkdir -p tmp_build

# clean results file
results_file="results/compile_report.csv"
mkdir -p "$(dirname "$results_file")"
echo "name,build_time,max_rss_kb" > "$results_file"

# Ctrl+C handler
cleanup() {
    echo ""
    echo "Interrupted. Exiting cleanly."
    exit 1
}
trap cleanup INT

for file in examples/generated/*.e.cpp; do
    name=$(basename "$file" .e.cpp)

    echo "Compiling $name"

    for i in {1..10}; do
        (
            /usr/bin/time \
                -f "$name,%e,%M" \
                g++ \
                    -std=c++23 \
                    -O3 \
                    -I./include \
                    -c "$file" \
                    -o "tmp_build/$name.o"
        ) 2>> "$results_file"
    done
done
