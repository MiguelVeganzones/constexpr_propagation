#ifndef INCLUDED_TENSOR_OPERATIONS_V1
#define INCLUDED_TENSOR_OPERATIONS_V1

#include "tensor.hpp"
#include <algorithm>
#include <concepts>
#include <ranges>
#include <utility>
#include <vector>

namespace v1
{

template <std::integral I>
auto increment_index(
    std::ranges::range auto&       idx,
    std::ranges::range auto const& sizes
) noexcept -> bool
{
    for (I d = static_cast<I>(idx.size()); d-- > 0;)
    {
        if (idx[d] + 1 < sizes[d])
        {
            ++idx[d];
            return true;
        }
        idx[d] = I{};
    }
    return false;
}

auto tensor_contraction(
    containers::concepts::Container auto const&           a,
    containers::concepts::Container auto const&           b,
    containers::concepts::ContractionIndexSet auto const& cis
)
{
    using a_t = std::remove_cvref_t<decltype(a)>;
    using b_t = std::remove_cvref_t<decltype(b)>;
    using value_type =
        std::common_type_t<typename a_t::value_type, typename b_t::value_type>;
    using size_type =
        std::common_type_t<typename a_t::size_type, typename b_t::size_type>;
    using tensor_c_t = v1::tensor<value_type>;

#ifndef NDEBUG
    for (auto const& [a_axis, b_axis] : cis)
    {
        assert(a.size(a_axis) == b.size(b_axis));
    }
#endif

    const auto      order    = cis.order();
    const size_type out_rank = a.rank() + b.rank() - 2 * order;
    assert(out_rank > 0);
    std::vector<size_type> out_sizes;
    out_sizes.reserve(out_rank);

    std::vector<bool> a_contracted(a.rank(), false);
    std::vector<bool> b_contracted(b.rank(), false);
    for (auto const& [a_idx, b_idx] : cis)
    {
        a_contracted[a_idx] = true;
        b_contracted[b_idx] = true;
    }
    for (size_type i{}; i != a.rank(); ++i)
    {
        if (!a_contracted[i]) out_sizes.push_back(a.size(i));
    }
    for (size_type i{}; i != b.rank(); ++i)
    {
        if (!b_contracted[i]) out_sizes.push_back(b.size(i));
    }
    tensor_c_t c(out_sizes);

    std::vector<size_type> a_free_axes;
    std::vector<size_type> b_free_axes;
    a_free_axes.reserve(a.rank() - order);
    b_free_axes.reserve(b.rank() - order);
    for (size_type i{}; i != a.rank(); ++i)
    {
        if (!a_contracted[i]) a_free_axes.push_back(i);
    }
    for (size_type i{}; i != b.rank(); ++i)
    {
        if (!b_contracted[i]) b_free_axes.push_back(i);
    }

    std::vector<size_type> contract_sizes;
    contract_sizes.reserve(order);
    for (auto const& [a_axis, _] : cis)
        contract_sizes.push_back(a.size(a_axis));

    std::vector<size_type> out_idx(c.rank(), 0);
    std::vector<size_type> a_idx(a.rank(), 0);
    std::vector<size_type> b_idx(b.rank(), 0);
    std::vector<size_type> contract_idx(order, 0);

    do
    {
        size_type k = 0;
        for (auto a_axis : a_free_axes)
            a_idx[a_axis] = out_idx[k++];
        for (auto b_axis : b_free_axes)
            b_idx[b_axis] = out_idx[k++];

        value_type sum{};
        std::ranges::fill(contract_idx, size_type{});
        do
        {
            for (size_type i = 0; i != order; ++i)
            {
                auto const& [a_axis, b_axis] = cis[i];
                a_idx[a_axis]                = contract_idx[i];
                b_idx[b_axis]                = contract_idx[i];
            }
            sum += a[a_idx] * b[b_idx];
        } while (increment_index<size_type>(contract_idx, contract_sizes));
        c[out_idx] = sum;
    } while (increment_index<size_type>(out_idx, out_sizes));
    return c;
}

} // namespace v1

#endif // INCLUDED_TENSOR_OPERATIONS_V1
