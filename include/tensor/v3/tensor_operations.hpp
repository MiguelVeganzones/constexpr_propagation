#ifndef INCLUDED_TENSOR_OPERATIONS_V3
#define INCLUDED_TENSOR_OPERATIONS_V3

#include "common/container_concepts.hpp"
#include "tensor_iteration.hpp"
#include "utils.hpp"

namespace v3
{

template <containers::concepts::ContractionIndexSet auto CIS>
constexpr auto tensor_contraction(
    containers::concepts::StaticContainer auto const& a,
    containers::concepts::StaticContainer auto const& b,
    containers::concepts::StaticContainer auto&       c
) -> void
{
    using a_t        = std::remove_cvref_t<decltype(a)>;
    using b_t        = std::remove_cvref_t<decltype(b)>;
    using c_t        = std::remove_cvref_t<decltype(c)>;
    using result_t   = utils::types::tensor_contraction_result<a_t, b_t, CIS>;
    using value_type = typename c_t::value_type;
    // TODO_ Check output shape

    iteration::shaped_for<
        typename result_t::outter_loop_t,
        result_t::s_a_free_strides,
        result_t::s_b_free_strides,
        c_t::strides()>(
        [&a, &b](auto& out, auto const out_idx, auto const a_base, auto const b_base)
        {
            value_type sum{};
            iteration::shaped_for_inner<
                typename result_t::inner_loop_t,
                result_t::s_a_contract_strides,
                result_t::s_b_contract_strides>(
                a_base,
                b_base,
                [&a, &b]
                (auto& e, auto const a_offset, auto const b_offset)
                { //
                    e += a.buffer()[a_offset] * b.buffer()[b_offset];
                },
                sum
            );
            out[out_idx] = sum;
        },
        c
    );
}

} // namespace v3

#endif // INCLUDED_TENSOR_OPERATIONS_V3
