#ifndef INCLUDED_TENSOR_OPERATIONS_V2
#define INCLUDED_TENSOR_OPERATIONS_V2

#include "common/container_concepts.hpp"
#include "tensor.hpp"
#include "tensor_iteration.hpp"
#include <array>
#include <concepts>
#include <ranges>

namespace v2
{

auto tensor_contraction(
    containers::concepts::StaticRankContainer auto const& a,
    containers::concepts::StaticRankContainer auto const& b,
    containers::concepts::ContractionIndexSet auto const& cis
)
{
    using a_t = std::remove_cvref_t<decltype(a)>;
    using b_t = std::remove_cvref_t<decltype(b)>;
    using value_type =
        std::common_type_t<typename a_t::value_type, typename b_t::value_type>;
    using size_type =
        std::common_type_t<typename a_t::size_type, typename b_t::size_type>;
    constexpr auto      s_order    = std::remove_cvref_t<decltype(cis)>::order();
    constexpr size_type s_out_rank = a_t::rank() + b_t::rank() - 2 * s_order;
    using tensor_c_t               = v2::tensor<value_type, s_out_rank>;

#ifndef NDEBUG
    for (auto const& [a_axis, b_axis] : cis)
    {
        assert(a.size(a_axis) == b.size(b_axis));
    }
#endif

    std::array<bool, a_t::rank()> a_contracted{};
    std::array<bool, b_t::rank()> b_contracted{};
    for (auto const& [a_idx, b_idx] : cis)
    {
        a_contracted[a_idx] = true;
        b_contracted[b_idx] = true;
    }
    std::array<size_type, s_out_rank> out_sizes{};
    size_type                         k{};
    for (size_type i{}; i != a_t::rank(); ++i)
    {
        if (!a_contracted[i]) out_sizes[k++] = a.size(i);
    }
    for (size_type i{}; i != b_t::rank(); ++i)
    {
        if (!b_contracted[i]) out_sizes[k++] = b.size(i);
    }
    tensor_c_t c(out_sizes);

    std::array<size_type, a_t::rank() - s_order> a_free_axes{};
    std::array<size_type, b_t::rank() - s_order> b_free_axes{};
    k = 0;
    for (size_type i{}; i != a_t::rank(); ++i)
    {
        if (!a_contracted[i])
        {
            a_free_axes[k] = i;
            k++;
        }
    }
    k = 0;
    for (size_type i{}; i != b_t::rank(); ++i)
    {
        if (!b_contracted[i])
        {
            b_free_axes[k] = i;
            k++;
        }
    }

    std::array<size_type, s_order> contract_sizes;
    k = 0;
    for (auto const& [a_axis, _] : cis)
    {
        contract_sizes[k++] = a.size(a_axis);
    }

    std::array<size_type, s_order> a_contract_strides{};
    std::array<size_type, s_order> b_contract_strides{};
    for (size_type i{}; i != s_order; ++i)
    {
        auto const& [a_axis, b_axis] = cis[i];
        a_contract_strides[i]        = a.stride(a_axis);
        b_contract_strides[i]        = b.stride(b_axis);
    }

    iteration::shaped_for(
        out_sizes,
        [&a,
         &b,
         &a_free_axes,
         &b_free_axes,
         &contract_sizes,
         &a_contract_strides,
         &b_contract_strides](auto& out, auto const& out_idxs)
        {
            size_type a_base{};
            size_type b_base{};
            size_type ki = 0;
            for (auto a_axis : a_free_axes)
                a_base += out_idxs[ki++] * a.stride(a_axis);
            for (auto b_axis : b_free_axes)
                b_base += out_idxs[ki++] * b.stride(b_axis);
            iteration::shaped_for_inner(
                contract_sizes,
                a_contract_strides,
                b_contract_strides,
                [&a, &b, &a_base, &b_base] //
                (auto& e, auto const a_offset, auto const b_offset)
                {
                    //
                    e += a.buffer()[a_base + a_offset] * b.buffer()[b_base + b_offset];
                },
                out[out_idxs]
            );
        },
        c
    );
    return c;
}

} // namespace v2

#endif // INCLUDED_TENSOR_OPERATIONS_V2
