import numpy as np
import pathlib
from dataclasses import dataclass
from typing import Tuple, List


# =========================================================
# BACKENDS
# =========================================================

TENSOR_IMPLS = {
    "v1": {
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
const auto cis =
    v1::utils::types::contraction_index_set<std::size_t>(cis_data);
"""
    },

    "v2": {
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
constexpr auto cis =
    v2::utils::types::contraction_index_set<std::size_t, order>(cis_data);
"""
    },

    "v3": {
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
constexpr auto cis =
    v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
"""
    }
}

CONTRACTION_IMPLS = {
    "v1": {
        "include": '#include "tensor/v1/tensor_operations.hpp"',
        "call": "v1::tensor_contraction(a,b,cis)"
    },

    "v2": {
        "include": '#include "tensor/v2/tensor_operations.hpp"',
        "call": "v2::tensor_contraction(a,b,cis)"
    },

    "v3": {
        "include": '#include "tensor/v3/tensor_operations.hpp"',
        "call": "v3::tensor_contraction<cis>(a,b)"
    }
}

BENCHMARK_TEMPLATE = """
static void BM_tc_{name}_{container}_{backend}(benchmark::State& state)
{{
    using F = float;

    constexpr std::array a_data{{{a_data}}};
    constexpr std::array b_data{{{b_data}}};

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
        benchmark::DoNotOptimize({contraction});
    }}
}}
BENCHMARK(BM_tc_{name}_{container}_{backend});
"""

COMBINATIONS = [
    ("v1", "v1"),

    ("v2", "v1"),
    ("v2", "v2"),

    ("v3", "v1"),
    ("v3", "v2"),
    ("v3", "v3"),
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

    add("2_2_1", (2, 2), (2, 2), [(1, 0)])
    add("3_3_2", (3, 3, 3), (3, 3, 3), [(2, 0), (1, 1)])
    add("4_4_3", (4, 4, 4, 4), (4, 4, 4, 4), [(3, 0), (1, 2), (2, 1)])
    add("4_4_2", (4, 4, 4, 4), (4, 4, 4, 4), [(3, 0), (1, 3)])
    add("5_5_4", (5, 5, 5, 5, 5), (5, 5, 5, 5, 5), [(3, 0), (1, 2), (2, 1), (4, 4)])

    return cases


# =========================================================
# RENDERER
# =========================================================

def render(case, container, backend_name, snippet):
    a = ", ".join(f"{x}f" for x in case.a.flatten())
    b = ", ".join(f"{x}f" for x in case.b.flatten())

    a_shape = ", ".join(f"{x}uz" for x in case.a_shape)
    b_shape = ", ".join(f"{x}uz" for x in case.b_shape)

    cis = ", ".join(
        f"std::pair{{{i}uz,{j}uz}}"
        for i, j in case.cis
    )

    return BENCHMARK_TEMPLATE.format(
        container=container.upper(),
        backend=backend_name.upper(),
        name=case.name,
        a_data=a,
        b_data=b,
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
    container,
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
                container,
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
            container="t",
            backend_name=backend_name,
            include_block=include_block,
            snippet=snippet,
            cases=cases,
        )

    print(
        f"Generated {len(cases)} cases for "
        f"{len(COMBINATIONS)} backend combinations"
    )


if __name__ == "__main__":
    main()
