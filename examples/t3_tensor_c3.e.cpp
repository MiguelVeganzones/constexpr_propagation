#include "common/tensor_printing.hpp"
#include "tensor/v3/static_layout.hpp"
#include "tensor/v3/static_shape.hpp"
#include "tensor/v3/tensor.hpp"
#include "tensor/v3/tensor_operations.hpp"
#include "tensor/v3/utils.hpp"
#include <iostream>
#include <print>
#include <ranges>
#include <utility>
#include <vector>

auto main() -> int
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

    std::println("cx.sizes is {}", a_sizes);
    std::println("cx.sizes is {}", b_sizes);

    tensor_a_t a{};
    tensor_b_t b{};
    std::ranges::iota(a.buffer(), 0);
    std::ranges::iota(b.buffer(), 0);
    std::cout << "a\n" << a << '\n';
    std::cout << "b\n" << b << '\n';
    constexpr auto cis = utils::types::contraction_index_set<size_type, Order>(std::array{
        std::pair{ 0uz, 1uz },
        std::pair{ 1uz, 0uz }
    });
    auto const     c   = tensor_contraction<cis>(a, b);
    std::cout << "c\n" << c << '\n';
}
