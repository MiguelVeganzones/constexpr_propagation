#pragma once
#include "tensor.hpp"
#include "tensor_iteration.hpp"
#include <array>
#include <concepts>
#include <ranges>
#include <utility>

namespace v2
{

template <std::integral Index_Type, std::integral auto Order>
struct contraction_index_set
{
public:
    using index_t                 = Index_Type;
    using size_type               = index_t;
    static constexpr auto s_order = Order;
    using index_pair_t            = std::pair<index_t, index_t>;
    using container_t             = std::array<index_pair_t, s_order>;
    using const_iterator          = typename container_t::const_iterator;
    using iterator                = typename container_t::iterator;
    using value_type              = typename container_t::value_type;

    static_assert(s_order >= 0);

public:
    explicit constexpr contraction_index_set(
        std::ranges::sized_range auto const& index_pairs
    )
        requires std::same_as<
            index_pair_t,
            std::ranges::range_value_t<std::remove_cvref_t<decltype(index_pairs)>>>
        : indices_{}
    {
        std::ranges::copy(index_pairs, std::begin(indices_));
    }

    explicit constexpr contraction_index_set(
        std::same_as<index_pair_t> auto const&... index_pairs
    )
        requires(sizeof...(index_pairs) == s_order)
        : indices_{ index_pairs... }
    {
    }

    [[nodiscard]]
    static constexpr auto order() noexcept -> auto
    {
        return s_order;
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const i) const noexcept -> index_pair_t const&

    {
        assert(i < s_order);
        if constexpr (std::is_signed_v<index_t>)
        {
            assert(i >= index_t{});
        }
        return indices_[i];
    }

    [[nodiscard]]
    constexpr auto cbegin() const noexcept -> const_iterator
    {
        return std::cbegin(indices_);
    }

    [[nodiscard]]
    constexpr auto cend() const noexcept -> const_iterator
    {
        return std::cend(indices_);
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
        return std::begin(indices_);
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
        return std::end(indices_);
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
        return std::begin(indices_);
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
        return std::end(indices_);
    }

public:
    container_t indices_;
};

template <typename T, std::size_t Rank_A, std::size_t Rank_B, std::size_t Order>
[[gnu::noinline]]
auto tensor_contraction(
    tensor<T, Rank_A> const&                                                 A,
    tensor<T, Rank_B> const&                                                 B,
    contraction_index_set<typename tensor<T, Rank_A>::index_t, Order> const& cis
)
{
    using tensor_a_t               = tensor<T, Rank_A>;
    using tensor_b_t               = tensor<T, Rank_B>;
    using size_type                = typename tensor_a_t::size_type;
    constexpr size_type s_out_rank = tensor_a_t::rank() + tensor_b_t::rank() - 2 * Order;
    using tensor_c_t               = tensor<T, s_out_rank>;

#ifndef NDEBUG
    for (auto const& [a_axis, b_axis] : cis)
    {
        assert(A.size(a_axis) == B.size(b_axis));
    }
#endif

    std::array<bool, tensor_a_t::rank()> a_contracted{};
    std::array<bool, tensor_b_t::rank()> b_contracted{};
    for (auto const& [a, b] : cis)
    {
        a_contracted[a] = true;
        b_contracted[b] = true;
    }
    std::array<size_type, s_out_rank> out_sizes{};
    size_type                         k{};
    for (size_type i{}; i != tensor_a_t::rank(); ++i)
    {
        if (!a_contracted[i]) out_sizes[k++] = A.size(i);
    }
    for (size_type i{}; i != tensor_b_t::rank(); ++i)
    {
        if (!b_contracted[i]) out_sizes[k++] = B.size(i);
    }
    tensor_c_t C(out_sizes);

    std::array<size_type, tensor_a_t::rank() - Order> a_free_axes{};
    std::array<size_type, tensor_b_t::rank() - Order> b_free_axes{};
    k = 0;
    for (size_type i{}; i != tensor_a_t::rank(); ++i)
    {
        if (!a_contracted[i])
        {
            a_free_axes[k] = i;
            k++;
        }
    }
    k = 0;
    for (size_type i{}; i != tensor_b_t::rank(); ++i)
    {
        if (!b_contracted[i])
        {
            b_free_axes[k] = i;
            k++;
        }
    }

    std::array<size_type, Order> contract_sizes;
    k = 0;
    for (auto const& [a_axis, _] : cis)
    {
        contract_sizes[k++] = A.size(a_axis);
    }

    std::array<size_type, Order> a_contract_strides{};
    std::array<size_type, Order> b_contract_strides{};
    for (size_type i{}; i != Order; ++i)
    {
        auto const& [a_axis, b_axis] = cis[i];
        a_contract_strides[i]        = A.stride(a_axis);
        b_contract_strides[i]        = B.stride(b_axis);
    }

    iteration::shaped_for(
        out_sizes,
        [&A,
         &B,
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
                a_base += out_idxs[ki++] * A.stride(a_axis);
            for (auto b_axis : b_free_axes)
                b_base += out_idxs[ki++] * B.stride(b_axis);
            iteration::shaped_for_inner(
                contract_sizes,
                a_contract_strides,
                b_contract_strides,
                [&A, &B, &a_base, &b_base] //
                (auto& e, auto const a_offset, auto const b_offset)
                {
                    //
                    e += A.buffer()[a_base + a_offset] * B.buffer()[b_base + b_offset];
                },
                out[out_idxs]
            );
        },
        C
    );
    return C;
}

} // namespace v2
