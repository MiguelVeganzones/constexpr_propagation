#include "common/tensor_printing.hpp"
#include "tensor/v3/static_layout.hpp"
#include "tensor/v3/static_shape.hpp"
#include "tensor/v3/tensor.hpp"
#include "tensor/v3/tensor_operations.hpp"
#include "tensor/v3/utils.hpp"
#include <ranges>
#include <utility>
#include <vector>

auto main() -> int
{
    using namespace v3;
    using namespace containers::print;
    using F                = float;
    constexpr auto a_sizes = std::array{ 32uz, 32uz, 32uz };
    constexpr auto b_sizes = std::array{ 32uz, 32uz, 32uz };
    using Shape_A          = static_shape<a_sizes>;
    using Shape_B          = static_shape<b_sizes>;
    using Layout_A         = static_layout<Shape_A>;
    using Layout_B         = static_layout<Shape_B>;
    using tensor_a_t       = tensor<F, Layout_A>;
    using tensor_b_t       = tensor<F, Layout_B>;
    using size_type        = std::
        common_type_t<typename tensor_a_t::size_type, typename tensor_b_t::size_type>;

    tensor_a_t a{};
    tensor_b_t b{};
    std::ranges::iota(a.buffer(), 0);
    std::ranges::iota(b.buffer(), 0);
    constexpr auto cis_data = std::array{
        std::pair{ 0uz, 0uz },
        std::pair{ 1uz, 1uz },
    };
    constexpr auto cis =
        utils::types::contraction_index_set<size_type, cis_data.size()>(cis_data);
    auto const c = tensor_contraction<cis>(a, b);
    return (int)c.buffer()[0];
}
