#include "tensor/v3/tensor_operations.hpp"

#include "tensor/v3/static_layout.hpp"
#include "tensor/v3/static_shape.hpp"
#include "tensor/v3/tensor.hpp"
#include "tensor/v3/utils.hpp"

#include <benchmark/benchmark.h>
#include <numeric>

static void BM_tc_2r2s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_2r2s1o_T3_C3);


static void BM_tc_2r4s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_2r4s1o_T3_C3);


static void BM_tc_2r8s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_2r8s1o_T3_C3);


static void BM_tc_2r16s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{16uz, 16uz};
    constexpr std::array b_shape{16uz, 16uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_2r16s1o_T3_C3);


static void BM_tc_2r32s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{32uz, 32uz};
    constexpr std::array b_shape{32uz, 32uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_2r32s1o_T3_C3);


static void BM_tc_2r64s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{64uz, 64uz};
    constexpr std::array b_shape{64uz, 64uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_2r64s1o_T3_C3);


static void BM_tc_3r2s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_3r2s1o_T3_C3);


static void BM_tc_3r4s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_3r4s1o_T3_C3);


static void BM_tc_3r8s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz, 8uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_3r8s1o_T3_C3);


static void BM_tc_3r16s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{16uz, 16uz, 16uz};
    constexpr std::array b_shape{16uz, 16uz, 16uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_3r16s1o_T3_C3);


static void BM_tc_3r32s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{32uz, 32uz, 32uz};
    constexpr std::array b_shape{32uz, 32uz, 32uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_3r32s1o_T3_C3);


static void BM_tc_3r2s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_3r2s2o_T3_C3);


static void BM_tc_3r4s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_3r4s2o_T3_C3);


static void BM_tc_3r8s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz, 8uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_3r8s2o_T3_C3);


static void BM_tc_3r16s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{16uz, 16uz, 16uz};
    constexpr std::array b_shape{16uz, 16uz, 16uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_3r16s2o_T3_C3);


static void BM_tc_3r32s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{32uz, 32uz, 32uz};
    constexpr std::array b_shape{32uz, 32uz, 32uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_3r32s2o_T3_C3);


static void BM_tc_4r2s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r2s1o_T3_C3);


static void BM_tc_4r4s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r4s1o_T3_C3);


static void BM_tc_4r8s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz, 8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz, 8uz, 8uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r8s1o_T3_C3);


static void BM_tc_4r16s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{16uz, 16uz, 16uz, 16uz};
    constexpr std::array b_shape{16uz, 16uz, 16uz, 16uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r16s1o_T3_C3);


static void BM_tc_4r2s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r2s2o_T3_C3);


static void BM_tc_4r4s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r4s2o_T3_C3);


static void BM_tc_4r8s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz, 8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz, 8uz, 8uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r8s2o_T3_C3);


static void BM_tc_4r16s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{16uz, 16uz, 16uz, 16uz};
    constexpr std::array b_shape{16uz, 16uz, 16uz, 16uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r16s2o_T3_C3);


static void BM_tc_4r2s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r2s3o_T3_C3);


static void BM_tc_4r4s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r4s3o_T3_C3);


static void BM_tc_4r8s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz, 8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz, 8uz, 8uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r8s3o_T3_C3);


static void BM_tc_4r16s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{16uz, 16uz, 16uz, 16uz};
    constexpr std::array b_shape{16uz, 16uz, 16uz, 16uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r16s3o_T3_C3);


static void BM_tc_5r2s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r2s1o_T3_C3);


static void BM_tc_5r3s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array b_shape{3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r3s1o_T3_C3);


static void BM_tc_5r4s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r4s1o_T3_C3);


static void BM_tc_5r8s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz, 8uz, 8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz, 8uz, 8uz, 8uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r8s1o_T3_C3);


static void BM_tc_5r16s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{16uz, 16uz, 16uz, 16uz, 16uz};
    constexpr std::array b_shape{16uz, 16uz, 16uz, 16uz, 16uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r16s1o_T3_C3);


static void BM_tc_5r2s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r2s2o_T3_C3);


static void BM_tc_5r3s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array b_shape{3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r3s2o_T3_C3);


static void BM_tc_5r4s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r4s2o_T3_C3);


static void BM_tc_5r8s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz, 8uz, 8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz, 8uz, 8uz, 8uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r8s2o_T3_C3);


static void BM_tc_5r2s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r2s3o_T3_C3);


static void BM_tc_5r3s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array b_shape{3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r3s3o_T3_C3);


static void BM_tc_5r4s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r4s3o_T3_C3);


static void BM_tc_5r8s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz, 8uz, 8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz, 8uz, 8uz, 8uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r8s3o_T3_C3);


static void BM_tc_5r2s4o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}, std::pair{3uz,3uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r2s4o_T3_C3);


static void BM_tc_5r3s4o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array b_shape{3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}, std::pair{3uz,3uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r3s4o_T3_C3);


static void BM_tc_5r4s4o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}, std::pair{3uz,3uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r4s4o_T3_C3);


static void BM_tc_5r8s4o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz, 8uz, 8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz, 8uz, 8uz, 8uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}, std::pair{3uz,3uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r8s4o_T3_C3);


static void BM_tc_6r2s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r2s1o_T3_C3);


static void BM_tc_6r3s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{3uz, 3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array b_shape{3uz, 3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r3s1o_T3_C3);


static void BM_tc_6r4s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r4s1o_T3_C3);


static void BM_tc_6r5s1o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{5uz, 5uz, 5uz, 5uz, 5uz, 5uz};
    constexpr std::array b_shape{5uz, 5uz, 5uz, 5uz, 5uz, 5uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r5s1o_T3_C3);


static void BM_tc_6r2s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r2s2o_T3_C3);


static void BM_tc_6r3s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{3uz, 3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array b_shape{3uz, 3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r3s2o_T3_C3);


static void BM_tc_6r4s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r4s2o_T3_C3);


static void BM_tc_6r5s2o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{5uz, 5uz, 5uz, 5uz, 5uz, 5uz};
    constexpr std::array b_shape{5uz, 5uz, 5uz, 5uz, 5uz, 5uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r5s2o_T3_C3);


static void BM_tc_6r2s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array b_shape{2uz, 2uz, 2uz, 2uz, 2uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r2s3o_T3_C3);


static void BM_tc_6r3s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{3uz, 3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array b_shape{3uz, 3uz, 3uz, 3uz, 3uz, 3uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r3s3o_T3_C3);


static void BM_tc_6r4s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r4s3o_T3_C3);


static void BM_tc_6r5s3o_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{5uz, 5uz, 5uz, 5uz, 5uz, 5uz};
    constexpr std::array b_shape{5uz, 5uz, 5uz, 5uz, 5uz, 5uz};
    constexpr std::array cis_data{std::pair{0uz,0uz}, std::pair{1uz,1uz}, std::pair{2uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r5s3o_T3_C3);


static void BM_tc_3r2s1orect_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 4uz, 8uz};
    constexpr std::array b_shape{8uz, 4uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,2uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_3r2s1orect_T3_C3);


static void BM_tc_5r4s1ostride_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 8uz, 2uz, 8uz, 4uz};
    constexpr std::array b_shape{4uz, 8uz, 2uz, 8uz, 4uz};
    constexpr std::array cis_data{std::pair{0uz,4uz}, std::pair{1uz,3uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r4s1ostride_T3_C3);


static void BM_tc_6r4s2ostride_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 4uz, 8uz, 2uz, 4uz, 8uz};
    constexpr std::array b_shape{8uz, 4uz, 2uz, 8uz, 4uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,5uz}, std::pair{1uz,4uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_6r4s2ostride_T3_C3);


static void BM_tc_4r2s2omixed_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{2uz, 4uz, 2uz, 4uz};
    constexpr std::array b_shape{4uz, 2uz, 4uz, 2uz};
    constexpr std::array cis_data{std::pair{0uz,1uz}, std::pair{2uz,3uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r2s2omixed_T3_C3);


static void BM_tc_4r8s1operm_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz, 8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz, 8uz, 8uz};
    constexpr std::array cis_data{std::pair{3uz,3uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r8s1operm_T3_C3);


static void BM_tc_4r8s1operm2_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{8uz, 8uz, 8uz, 8uz};
    constexpr std::array b_shape{8uz, 8uz, 8uz, 8uz};
    constexpr std::array cis_data{std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_4r8s1operm2_T3_C3);


static void BM_tc_5r4s2operm_T3_C3(benchmark::State& state)
{
    using F = float;

    constexpr std::array a_shape{4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array b_shape{4uz, 4uz, 4uz, 4uz, 4uz};
    constexpr std::array cis_data{std::pair{3uz,3uz}, std::pair{1uz,1uz}};

    [[maybe_unused]] constexpr auto a_rank = a_shape.size();
    [[maybe_unused]] constexpr auto b_rank = b_shape.size();
    [[maybe_unused]] constexpr auto order = cis_data.size();

    
using a_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<a_shape>>
>;

using b_t = v3::tensor<
    F,
    v3::static_layout<v3::static_shape<b_shape>>
>;


    
a_t a{};
b_t b{};
constexpr auto cis = v3::utils::types::contraction_index_set<std::size_t, order>(cis_data);
auto c = v3::utils::allocate_output_uninitialized<cis>(a, b);


    std::ranges::fill(a.buffer(), F{1});
    std::ranges::fill(b.buffer(), F{1});

    for (auto _ : state)
    {
        v3::tensor_contraction<cis>(a, b, c);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_tc_5r4s2operm_T3_C3);

BENCHMARK_MAIN();