import numpy as np
import time
import csv
import config
import os


OUTPUT_FILE = "results/benchmarking/np_np_results.csv"


def run_case(name, a_shape, b_shape, cis):
    a = np.ones(np.prod(a_shape), dtype=np.float32).reshape(a_shape)
    b = np.ones(np.prod(b_shape), dtype=np.float32).reshape(b_shape)

    axes_a = [i for i, _ in cis]
    axes_b = [j for _, j in cis]

    for _ in range(config.BENCHMARK_WARMUPS):
        np.tensordot(a, b, axes=(axes_a, axes_b))

    rows = []

    for _ in range(config.BENCHMARK_ITERATIONS):
        t0 = time.perf_counter()
        np.tensordot(a, b, axes=(axes_a, axes_b))
        t1 = time.perf_counter()

        real_time_ns = (t1 - t0) * 1e9

        # mimic google benchmark CSV schema
        rows.append([
            f"BM_tc_{name}_np_np",
            1,
            real_time_ns,
            "ns",
        ])

    return rows


def main():
    all_rows = []

    for s in config.samples:
        print(f"running {s.name}: {s.memory_mb:.2f} MB, {s.flops} ops")
        rows = run_case(s.name, s.a_shape, s.b_shape, s.cis)
        all_rows.extend(rows)


    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
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
