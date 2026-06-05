#include "tensor/v3/static_layout.hpp"
#include "tensor/v3/static_shape.hpp"
#include "tensor/v3/tensor.hpp"
#include "tensor/v3/tensor_operations.hpp"
#include "tensor/v3/utils.hpp"
#include <benchmark/benchmark.h>

using F = float;

static void BM_tensor_contraction_v3(benchmark::State& state)
{
    constexpr auto a_sizes = std::array{ 2uz, 2uz, 3uz };
    constexpr auto b_sizes = std::array{ 3uz, 2uz, 2uz };

    using ShapeA = v3::static_shape<a_sizes>;
    using ShapeB = v3::static_shape<b_sizes>;

    using A = v3::tensor<F, v3::static_layout<ShapeA>>;
    using B = v3::tensor<F, v3::static_layout<ShapeB>>;

    A a{};
    B b{};

    std::iota(a.buffer().begin(), a.buffer().end(), F{});
    std::iota(b.buffer().begin(), b.buffer().end(), F{});

    constexpr auto cis_data = std::array{
        std::pair{ 2uz, 0uz },
        std::pair{ 1uz, 1uz }
    };
    constexpr auto cis =
        v3::utils::types::contraction_index_set<std::size_t, cis_data.size()>(cis_data);

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(tensor_contraction<cis>(a, b));
    }
}

BENCHMARK(BM_tensor_contraction_v3);
BENCHMARK_MAIN();
