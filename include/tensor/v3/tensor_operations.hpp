#ifndef INCLUDED_TENSOR_OPERATIONS_V3
#define INCLUDED_TENSOR_OPERATIONS_V3

#include "common/container_concepts.hpp"
#include "tensor_iteration.hpp"
#include "utils.hpp"

namespace v3
{

template <
    containers::concepts::ContractionIndexSet auto CIS,
    containers::concepts::StaticContainer          A,
    containers::concepts::StaticContainer          B>
[[nodiscard]]
constexpr auto tensor_contraction(A const& a, B const& b) noexcept -> utils::types::
    tensor_contraction_result_t<std::remove_cvref_t<A>, std::remove_cvref_t<B>, CIS>
{
    using tensor_a_t = std::remove_cvref_t<A>;
    using tensor_b_t = std::remove_cvref_t<B>;
    using result_t = utils::types::tensor_contraction_result<tensor_a_t, tensor_b_t, CIS>;
    using tensor_c_t = typename result_t::type;

#ifndef NDEBUG
    for (auto const& [a_axis, b_axis] : CIS)
    {
        assert(tensor_a_t::size(a_axis) == tensor_b_t::size(b_axis));
    }
#endif

    tensor_c_t c{};

    iteration::shaped_for<
        typename result_t::outter_loop_t,
        result_t::s_a_free_strides,
        result_t::s_b_free_strides>(
        [&a, &b](auto& out, auto const& out_idxs, auto const a_base, auto const b_base)
        {
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
                out[out_idxs]
            );
        },
        c
    );
    return c;
}

} // namespace v3

#endif // INCLUDED_TENSOR_OPERATIONS_V3
