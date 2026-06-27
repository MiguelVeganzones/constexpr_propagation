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

    const std::array<F, {a_size}> a_data{{{a_data}}};
    const std::array<F, {b_size}> b_data{{{b_data}}};

    constexpr std::array a_shape{{{a_shape}}};
    constexpr std::array b_shape{{{b_shape}}};
    constexpr std::array cis_data{{{cis}}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    {type_defs}

    {construct}

    std::ranges::copy(a_data, a.buffer().begin());
    std::ranges::copy(b_data, b.buffer().begin());

    for (auto _ : state)
    {{
        {contraction};
        benchmark::ClobberMemory();
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
# CASE MODEL
# =========================================================

@dataclass
class Case:
    name: str
    a_shape: Tuple[int, ...]
    b_shape: Tuple[int, ...]
    cis: List[Tuple[int, int]]
    a: np.ndarray
    b: np.ndarray


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

def generate_cases():
    cases = []

    def make_tensor(shape):
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    def add(name, a_shape, b_shape, cis):
        a = make_tensor(a_shape)
        b = make_tensor(b_shape)

        cases.append(Case(
            name=name,
            a_shape=a_shape,
            b_shape=b_shape,
            cis=cis,
            a=a,
            b=b
        ))

    for name, size_a, size_b, cis in config.samples:
        add(name, size_a, size_b, cis)

    return cases

def compute_flops(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "name",
            "output_size",
            "reduction_size",
            "multiplies",
            "adds",
            "flops",
            "fma_flops",
        ])

        for name, shape_a, shape_b, pairs in config.samples:
            info = contraction_flops(shape_a, shape_b, pairs)
            writer.writerow([
                name,
                info["output_size"],
                info["reduction_size"],
                info["multiplies"],
                info["adds"],
                info["flops"],
                info["fma_flops"],
            ])

def contraction_flops(shapeA, shapeB, pairs):
    contractedA = {i for i, _ in pairs}
    contractedB = {j for _, j in pairs}

    # sanity check
    for ia, ib in pairs:
        assert shapeA[ia] == shapeB[ib]

    output_size = (
        prod(shapeA[i] for i in range(len(shapeA)) if i not in contractedA)
        * prod(shapeB[j] for j in range(len(shapeB)) if j not in contractedB)
    )

    reduction_size = prod(shapeA[i] for i, _ in pairs)

    multiplications = output_size * reduction_size
    additions = output_size * (reduction_size - 1)

    return {
        "output_size": output_size,
        "reduction_size": reduction_size,
        "multiplies": multiplications,
        "adds": additions,
        "flops": multiplications + additions,
        "fma_flops": 2 * multiplications,
    }

    print(
        f"Generated {len(cases)} cases for "
        f"{len(COMBINATIONS)} backend combinations"
    )


# =========================================================
# RENDERER
# =========================================================

def render(case, backend_name, snippet):
    a = ", ".join(f"{x}f" for x in case.a.flatten())
    b = ", ".join(f"{x}f" for x in case.b.flatten())

    a_shape = ", ".join(f"{x}uz" for x in case.a_shape)
    b_shape = ", ".join(f"{x}uz" for x in case.b_shape)

    cis = ", ".join(
        f"std::pair{{{i}uz,{j}uz}}"
        for i, j in case.cis
    )

    return BENCHMARK_TEMPLATE.format(
        backend=backend_name.upper(),
        name=case.name,
        a_data=a,
        b_data=b,
        a_size=len(case.a.flatten()),
        b_size=len(case.b.flatten()),
        a_shape=a_shape,
        b_shape=b_shape,
        cis=cis,
        type_defs=snippet["type_defs"],
        construct=snippet["construct"],
        contraction=snippet["contraction"],
    )


# =========================================================
# EMITTER
# =========================================================

def emit_file(
    backend_name,
    include_block,
    snippet,
    cases,
):
    out = [include_block.strip()]
    for case in cases:
        out.append(
            render(
                case,
                backend_name,
                snippet,
            )
        )
    out.append("BENCHMARK_MAIN();")
    pathlib.Path(
        f"benchmarks/generated/{backend_name}_bench.b.cpp"
    ).write_text("\n".join(out))


# =========================================================
# MAIN
# =========================================================

def main():
    cases = generate_cases()

    pathlib.Path(
        "benchmarks/generated"
    ).mkdir(parents=True, exist_ok=True)

    for tensor_impl, contraction_impl in COMBINATIONS:

        backend_name = f"{tensor_impl}_{contraction_impl}"

        include_block = build_include_block(
            tensor_impl,
            contraction_impl,
        )

        snippet = get_snippet(
            tensor_impl,
            contraction_impl,
        )

        emit_file(
            backend_name=backend_name,
            include_block=include_block,
            snippet=snippet,
            cases=cases,
        )

    compute_flops("benchmarks/generated/benchmark_info.csv")

if __name__ == "__main__":
    main()
