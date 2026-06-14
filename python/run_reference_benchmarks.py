import numpy as np
import time
import csv
import config


OUTPUT_FILE = "results/benchmarking/np_np_results.csv"
REPEATS = 10
WARMUP = 3


def run_case(name, a_shape, b_shape, cis):
    # deterministic data (same as your C++ pattern style)
    a = np.arange(np.prod(a_shape), dtype=np.float32).reshape(a_shape)
    b = np.arange(np.prod(b_shape), dtype=np.float32).reshape(b_shape)

    axes_a = [i for i, _ in cis]
    axes_b = [j for _, j in cis]

    # warmup (important to avoid first-call bias)
    for _ in range(WARMUP):
        np.tensordot(a, b, axes=(axes_a, axes_b))

    rows = []

    for _ in range(REPEATS):
        t0 = time.perf_counter()
        np.tensordot(a, b, axes=(axes_a, axes_b))
        t1 = time.perf_counter()

        real_time_ns = (t1 - t0) * 1e9

        # mimic google benchmark CSV schema
        rows.append([
            f"BM_tc_{name}_np_np",
            1,                  # iterations (logical single op)
            real_time_ns,
            real_time_ns,      # cpu_time ~ real_time for NumPy
            "ns",
            "",
            "",
            "",
            "",
            ""
        ])

    return rows


def main():
    all_rows = []

    for name, a_shape, b_shape, cis in config.samples:
        rows = run_case(name, a_shape, b_shape, cis)
        all_rows.extend(rows)
        print(f"done {name}")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        # exact match to google benchmark CSV header
        writer.writerow([
            "name",
            "iterations",
            "real_time",
            "cpu_time",
            "time_unit",
            "bytes_per_second",
            "items_per_second",
            "label",
            "error_occurred",
            "error_message"
        ])

        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
