import numpy as np
import pathlib
from dataclasses import dataclass
from typing import Tuple, List
import config
from math import prod
import csv


# =========================================================
# BACKENDS
# =========================================================

TENSOR_IMPLS = {
    "t1": {
        "includes": """
#include "tensor/v1/tensor.hpp"
#include "tensor/v1/utils.hpp"
#include <iostream>
""",
        "type_defs": """
using a_t = v1::tensor<F>;
using b_t = v1::tensor<F>;
""",
        "construct": """
a_t a(a_shape);
b_t b(b_shape);
const auto cis = v1::utils::types::contraction_index_set<std::size_t>(cis_data);
auto c = v1::utils::types::allocate_output_uninitialized(a, b, cis);
"""
    },

    "t2": {
        "includes": """
#include "tensor/v2/tensor.hpp"
#include "tensor/v2/utils.hpp"
#include <iostream>
""",
        "type_defs": """
using a_t = v2::tensor<F, a_rank>;
using b_t = v2::tensor<F, b_rank>;
""",
        "construct": """
a_t a(a_shape);
b_t b(b_shape);
constexpr auto cis = v2::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v2::utils::types::allocate_output_uninitialized(a, b, cis);
"""
    },

    "t3": {
        "includes": """
#include "tensor/v3/static_layout.hpp"
#include "tensor/v3/static_shape.hpp"
#include "tensor/v3/tensor.hpp"
#include "tensor/v3/utils.hpp"
#include <iostream>
""",
        "type_defs": """
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;
""",
        "construct": """
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);
"""
    }
}

CONTRACTION_IMPLS = {
    "c1": {
        "include": '#include "tensor/v1/tensor_operations.hpp"',
        "call": "v1::tensor_contraction(a, b, c, cis)"
    },

    "c2": {
        "include": '#include "tensor/v2/tensor_operations.hpp"',
        "call": "v2::tensor_contraction(a, b, c, cis)"
    },

    "c3": {
        "include": '#include "tensor/v3/tensor_operations.hpp"',
        "call": "v3::tensor_contraction<cis>(a, b, c)"
    }
}

BENCHMARK_TEMPLATE = """
static void BM_tc_{name}_{backend}(benchmark::State& state)
{{
    using F = float;

    constexpr std::array a_shape{{{a_shape}}};
    constexpr std::array b_shape{{{b_shape}}};
    constexpr std::array cis_data{{{cis}}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    {type_defs}

    {construct}

    std::ranges::fill(a.buffer(), F{{1}});
    std::ranges::fill(b.buffer(), F{{1}});

    std::cerr << "Running {name}_{backend} [{memory:.2e} MB] [{flops:.2e} flops]\\n";

    for (auto _ : state)
    {{
        {contraction};
        benchmark::DoNotOptimize(c);
    }}
}}
BENCHMARK(BM_tc_{name}_{backend});
"""

COMBINATIONS = [
    ("t1", "c1"),

    ("t2", "c1"),
    ("t2", "c2"),

    ("t3", "c1"),
    ("t3", "c2"),
    ("t3", "c3"),
]

# =========================================================
# CASE GENERATION
# =========================================================


def build_include_block(tensor_impl, contraction_impl):
    return f"""
{CONTRACTION_IMPLS[contraction_impl]["include"]}
{TENSOR_IMPLS[tensor_impl]["includes"]}
#include <benchmark/benchmark.h>
#include <numeric>
"""

def get_snippet(tensor_impl, contraction_impl):
    tensor = TENSOR_IMPLS[tensor_impl]
    contraction = CONTRACTION_IMPLS[contraction_impl]

    return {
        "type_defs": tensor["type_defs"],
        "construct": tensor["construct"],
        "contraction": contraction["call"]
    }


def compute_flops(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "name",
            "output_size",
            "reduction_size",
            "flops",
            "memory",
        ])

        for s in config.samples:
            writer.writerow([
                s.name,
                s.output_size,
                s.reduction_size,
                s.flops,
                s.memory_bytes,
            ])


# =========================================================
# RENDERER
# =========================================================

BENCHMARK_MAIN_FILE = """
#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    benchmark::ConsoleReporter console;
    benchmark::CSVReporter csv;

    std::ofstream out("results.csv");
    csv.SetOutputStream(&out);

    benchmark::RunSpecifiedBenchmarks(&console, &csv);
}
"""

def render(case, backend_name, snippet):
    a_shape = ", ".join(
        f"{x}uz"
        for x in case.a_shape
    )
    b_shape = ", ".join(
        f"{x}uz"
        for x in case.b_shape
    )
    cis = ", ".join(
        f"std::pair{{{i}uz,{j}uz}}"
        for i, j in case.cis
    )

    return BENCHMARK_TEMPLATE.format(
        backend=backend_name.upper(),
        name=case.name,
        a_shape=a_shape,
        b_shape=b_shape,
        cis=cis,
        memory=case.memory_mb,
        flops=case.flops,
        type_defs=snippet["type_defs"],
        construct=snippet["construct"],
        contraction=snippet["contraction"],
    )


# =========================================================
# EMITTER
# =========================================================

def emit_case_file(
    backend_name,
    include_block,
    snippet,
    case,
):
    text = []
    text.append(
        include_block.strip()
    )
    text.append(
        render(
            case,
            backend_name,
            snippet,
        )
    )

    filename = (
        f"benchmarks/generated/"
        f"{case.name}_{backend_name}.b.cpp"
    )

    pathlib.Path(filename).write_text(
        "\n".join(text)
    )


def emit_main(out_dir="benchmarks/generated"):
    path = pathlib.Path(out_dir) / "main.cpp"

    path.write_text(
        "#include <benchmark/benchmark.h>\n\n"
        "BENCHMARK_MAIN();\n"
    )


# =========================================================
# MAIN
# =========================================================

def main():
    cases = config.samples

    out_dir = pathlib.Path("benchmarks/generated")
    out_dir.mkdir(parents=True, exist_ok=True)

    emit_main(out_dir)

    for tensor_impl, contraction_impl in COMBINATIONS:
        backend_name = f"{tensor_impl}_{contraction_impl}"

        include_block = build_include_block(tensor_impl, contraction_impl)
        snippet = get_snippet(tensor_impl, contraction_impl)

        for case in cases:
            emit_case_file(
                backend_name,
                include_block,
                snippet,
                case,
            )

    compute_flops("benchmarks/generated/benchmark_info.csv")

if __name__ == "__main__":
    main()
