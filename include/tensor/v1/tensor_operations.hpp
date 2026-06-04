#pragma once
#include "tensor.hpp"
#include <concepts>
#include <utility>

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

template <typename T>
[[gnu::noinline]]
auto tensor_contraction(
    tensor<T> const& A,
    tensor<T> const& B,
    std::vector<
        std::pair<typename tensor<T>::index_t, typename tensor<T>::index_t>> const& cis
)
{
    using tensor_t  = tensor<T>;
    using size_type = typename tensor_t::size_type;
    using index_t   = typename tensor_t::index_t;

#ifndef NDEBUG
    for (auto const& [a_axis, b_axis] : cis)
    {
        assert(A.size(a_axis) == B.size(b_axis));
    }
#endif

    const size_type out_rank = A.rank() + B.rank() - 2 * cis.size();
    assert(out_rank > 0);
    std::vector<size_type> out_sizes;
    out_sizes.reserve(out_rank);

    std::vector<bool> a_contracted(A.rank(), false);
    std::vector<bool> b_contracted(B.rank(), false);
    for (auto const& [a, b] : cis)
    {
        a_contracted[a] = true;
        b_contracted[b] = true;
    }
    for (size_type i{}; i != A.rank(); ++i)
    {
        if (!a_contracted[i]) out_sizes.push_back(A.size(i));
    }
    for (size_type i{}; i != B.rank(); ++i)
    {
        if (!b_contracted[i]) out_sizes.push_back(B.size(i));
    }
    tensor_t C(out_sizes);

    std::vector<size_type> a_free_axes;
    std::vector<size_type> b_free_axes;
    a_free_axes.reserve(A.rank() - cis.size());
    b_free_axes.reserve(B.rank() - cis.size());
    for (size_type i{}; i != A.rank(); ++i)
    {
        if (!a_contracted[i]) a_free_axes.push_back(i);
    }
    for (size_type i{}; i != B.rank(); ++i)
    {
        if (!b_contracted[i]) b_free_axes.push_back(i);
    }

    std::vector<size_type> contract_sizes;
    contract_sizes.reserve(cis.size());
    for (auto const& [a_axis, _] : cis)
        contract_sizes.push_back(A.size(a_axis));

    std::vector<size_type> out_idx(C.rank(), 0);
    std::vector<size_type> a_idx(A.rank(), 0);
    std::vector<size_type> b_idx(B.rank(), 0);
    std::vector<size_type> contract_idx(cis.size(), 0);

    do
    {
        size_type k = 0;
        for (auto a_axis : a_free_axes)
            a_idx[a_axis] = out_idx[k++];
        for (auto b_axis : b_free_axes)
            b_idx[b_axis] = out_idx[k++];

        T sum{};
        std::ranges::fill(contract_idx, size_type{});
        do
        {
            for (size_type i = 0; i != cis.size(); ++i)
            {
                auto const& [a_axis, b_axis] = cis[i];
                a_idx[a_axis]                = contract_idx[i];
                b_idx[b_axis]                = contract_idx[i];
            }
            sum += A[a_idx] * B[b_idx];
        } while (increment_index<index_t>(contract_idx, contract_sizes));
        C[out_idx] = sum;
    } while (increment_index<index_t>(out_idx, out_sizes));
    return C;
}

} // namespace v1::tensor
