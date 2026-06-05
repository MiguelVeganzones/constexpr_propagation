#include "common/tensor_printing.hpp"
#include "tensor/v1/tensor_operations.hpp"
#include "tensor/v2/utils.hpp"
#include "tensor/v3/static_layout.hpp"
#include "tensor/v3/static_shape.hpp"
#include "tensor/v3/tensor.hpp"
#include "utility/parser.hpp"
#include <iostream>
#include <print>
#include <ranges>
#include <utility>
#include <vector>

struct cmd_opts
{
    using size_type = std::size_t;
    std::vector<std::pair<size_type, size_type>> cis;
};

auto main(int argc, char* argv[]) -> int
{
    using namespace v3;
    using namespace containers::print;
    using F                = float;
    constexpr auto Order   = 2uz;
    constexpr auto a_sizes = std::array{ 2uz, 2uz, 2uz };
    constexpr auto b_sizes = std::array{ 2uz, 2uz, 3uz };
    using Shape_A          = static_shape<a_sizes>;
    using Shape_B          = static_shape<b_sizes>;
    using Layout_A         = static_layout<Shape_A>;
    using Layout_B         = static_layout<Shape_B>;
    using tensor_a_t       = tensor<F, Layout_A>;
    using tensor_b_t       = tensor<F, Layout_B>;
    using size_type        = std::
        common_type_t<typename tensor_a_t::size_type, typename tensor_b_t::size_type>;

    auto const opts = utility::cmd::parse_options<cmd_opts>({ argv, argv + argc });
    std::println("cx.a_sizes is {}", a_sizes);
    std::println("cx.b_sizes is {}", b_sizes);

    tensor_a_t a{};
    tensor_b_t b{};
    std::ranges::iota(a.buffer(), 0);
    std::ranges::iota(b.buffer(), 0);
    std::cout << "a\n" << a << '\n';
    std::cout << "b\n" << b << '\n';
    const auto cis = v2::utils::types::contraction_index_set<size_type, Order>(opts.cis);
    auto const c   = v1::tensor_contraction(a, b, cis);
    std::cout << "c\n" << c << '\n';
}
