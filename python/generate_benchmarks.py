import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List


# =========================================================
# BACKENDS
# =========================================================

BACKEND_INCLUDES = {
    "v1": """
#include "tensor/v1/tensor.hpp"
#include "tensor/v1/tensor_operations.hpp"
#include "tensor/v1/utils.hpp"
#include <benchmark/benchmark.h>
#include <numeric>
""",

    "v2": """
#include "tensor/v2/tensor.hpp"
#include "tensor/v2/tensor_operations.hpp"
#include "tensor/v2/utils.hpp"
#include <benchmark/benchmark.h>
#include <numeric>
""",

    "v3": """
#include "tensor/v3/static_layout.hpp"
#include "tensor/v3/static_shape.hpp"
#include "tensor/v3/tensor.hpp"
#include "tensor/v3/tensor_operations.hpp"
#include "tensor/v3/utils.hpp"
#include <benchmark/benchmark.h>
#include <numeric>
"""
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
"""


BACKEND_SNIPPETS = {
    "v1": {
        "type_defs": """
    using a_t = v1::tensor<F>;
    using b_t = v1::tensor<F>;
""",
        "construct": """
    a_t a(a_shape);
    b_t b(b_shape);
    const auto cis = v1::utils::types::contraction_index_set<std::size_t>(cis_data);
""",
        "contraction": "tensor_contraction(a,b,cis)"
    },

    "v2": {
        "type_defs": """
    using a_t = v2::tensor<F, a_rank>;
    using b_t = v2::tensor<F, b_rank>;
""",
        "construct": """
    a_t a(a_shape);
    b_t b(b_shape);
    constexpr auto cis = v2::utils::types::contraction_index_set<std::size_t, order>(cis_data);
""",
        "contraction": "tensor_contraction(a,b,cis)"
    },

    "v3": {
        "type_defs": """
    using a_t = v3::tensor<F, v3::static_layout<v3::static_shape<a_shape>>>;
    using b_t = v3::tensor<F, v3::static_layout<v3::static_shape<b_shape>>>;
""",
        "construct": """
    a_t a{};
    b_t b{};
    constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
""",
        "contraction": "tensor_contraction<cis>(a,b)"
    }
}


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

    add("case_mm", (2, 3, 4), (4, 3, 2), [(2, 0), (1, 1)])
    add("case_rank3", (2, 2, 3), (3, 2, 2), [(2, 0), (1, 1)])
    add("case_rank4a", (4, 2, 3, 6), (6, 3, 2, 2), [(3, 0), (1, 2), (2, 1)])
    add("case_rank4b", (2, 4, 3, 5), (5, 3, 2, 4), [(3, 0), (1, 3)])
    add("case_rank5", (2, 2, 3, 6, 2), (6, 3, 2, 2, 2), [(3, 0), (1, 2), (2, 1), (4, 4)])
    add("case_rank6", (2, 2, 3, 6, 3, 2), (6, 3, 2, 2, 2, 3), [(3, 0), (1, 2), (2, 1), (4, 5), (5, 4)])
    add("case_identity_like", (3, 3), (3, 3), [(1, 0)])

    return cases


# =========================================================
# RENDERER
# =========================================================

def render(case, container, backend):
    a = ", ".join(f"{x}f" for x in case.a.flatten())
    b = ", ".join(f"{x}f" for x in case.b.flatten())

    a_shape = ", ".join(f"{x}uz" for x in case.a_shape)
    b_shape = ", ".join(f"{x}uz" for x in case.b_shape)

    cis = ", ".join(
        f"std::pair{{{i}uz,{j}uz}}"
        for i, j in case.cis
    )

    snip = BACKEND_SNIPPETS[backend]

    return BENCHMARK_TEMPLATE.format(
        container=container.upper(),
        backend=backend.upper(),
        name=case.name,
        a_data=a,
        b_data=b,
        a_shape=a_shape,
        b_shape=b_shape,
        cis=cis,
        type_defs=snip["type_defs"],
        construct=snip["construct"],
        contraction=snip["contraction"]
    )


# =========================================================
# EMITTER
# =========================================================

def emit_file(container, backend, cases):
    out = [BACKEND_INCLUDES[backend].strip()]

    for c in cases:
        out.append(render(c, container, backend))

    Path(f"benchmarks/generated/{backend}_bench.b.cpp").write_text("\n".join(out))


# =========================================================
# MAIN
# =========================================================

def main():
    cases = generate_cases()

    for backend in BACKEND_INCLUDES.keys():
        emit_file("t", backend, cases)

    print(f"Generated {len(cases)} cases for v1/v2/v3")


if __name__ == "__main__":
    main()
