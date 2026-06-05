#ifndef INCLUDED_CONTAINER_UTILS_V3
#define INCLUDED_CONTAINER_UTILS_V3

#include "common/container_concepts.hpp"
#include "loop_control.hpp"
#include "static_layout.hpp"
#include "static_shape.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

namespace v3::utils
{

namespace types
{

namespace sequences
{

namespace detail
{

template <typename, typename>
struct concatenate_t_impl;

template <std::integral I, I... As, I... Bs>
struct concatenate_t_impl<
    std::integer_sequence<I, As...>,
    std::integer_sequence<I, Bs...>>
{
    using type = std::integer_sequence<I, As..., Bs...>;
};

template <
    containers::concepts::Container auto S1,
    containers::concepts::Container auto S2>
struct concatenate_v_impl
{
    static constexpr auto value = []
    {
        using size_type = std::common_type_t<
            typename decltype(S1)::value_type,
            typename decltype(S2)::value_type>;
        constexpr auto           N = std::ranges::size(S1) + std::ranges::size(S2);
        std::array<size_type, N> ret{};
        auto next = std::copy(std::cbegin(S1), std::cend(S1), std::begin(ret));
        std::copy(std::cbegin(S2), std::cend(S2), next);
        std::ranges::copy(S2, next);
        return ret;
    }();
};

} // namespace detail

template <typename A, typename B>
using concatenate_t = typename detail::concatenate_t_impl<A, B>::type;

template <auto A, auto B>
static constexpr auto concatenate_v = detail::concatenate_v_impl<A, B>::value;

} // namespace sequences

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
                     std::ranges::range_value_t<
                         std::remove_cvref_t<decltype(index_pairs)>>> &&
                 (std::ranges::size(index_pairs) == s_order)
        : indices_{}
    {
        std::ranges::copy(index_pairs, std::begin(indices_));
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

template <
    containers::concepts::StaticContainer          A,
    containers::concepts::StaticContainer          B,
    containers::concepts::ContractionIndexSet auto CIS>
struct tensor_contraction_result
{
    using a_t        = A;
    using b_t        = B;
    using size_type  = std::common_type_t<typename A::size_type, typename B::size_type>;
    using value_type = std::common_type_t<typename A::value_type, typename B::value_type>;
    static constexpr auto s_order = CIS.order();

    static_assert(std::ranges::all_of(
        CIS,
        [](auto const& p) { return A::size(p.first) == B::size(p.second); }
    ));

    // Contracted bool flags
    static constexpr auto s_a_contracted = []
    {
        using ret_t = std::array<bool, a_t::rank()>;
        ret_t ret{};
        for (auto const& [a, _] : CIS)
        {
            ret[a] = true;
        }
        return ret;
    }();
    static constexpr auto s_b_contracted = []
    {
        using ret_t = std::array<bool, b_t::rank()>;
        ret_t ret{};
        for (auto const& [_, b] : CIS)
        {
            ret[b] = true;
        }
        return ret;
    }();

    // Sizes
    static constexpr auto s_out_sizes = []
    {
        constexpr size_type s_in_rank  = A::rank() + B::rank();
        constexpr size_type s_out_rank = s_in_rank - 2 * s_order;
        using ret_t                    = std::array<size_type, s_out_rank>;
        ret_t ret{};

        size_type k{};
        for (size_type i{}; i != a_t::rank(); ++i)
        {
            if (!s_a_contracted[i]) ret[k++] = a_t::size(i);
        }
        for (size_type i{}; i != b_t::rank(); ++i)
        {
            if (!s_b_contracted[i]) ret[k++] = b_t::size(i);
        }
        assert(k == s_out_rank);

        return ret;
    }();

    static constexpr auto s_contract_sizes = []
    {
        using ret_t = std::array<size_type, CIS.order()>;
        ret_t     ret{};
        size_type k{};
        for (auto const& [a_axis, b_axis] : CIS)
        {
            // // TODO: c++26 Temaplte for
            // static_assert(a_t::size(a_axis) == b_t::size(b_axis));
            ret[k++] = a_t::size(a_axis);
        }
        return ret;
    }();

    // Idk
    static constexpr auto s_a_free_axes = []
    {
        using ret_t = std::array<size_type, a_t::rank() - s_order>;
        ret_t ret{};
        for (size_type i{}, k{}; i != a_t::rank(); ++i)
        {
            if (!s_a_contracted[i])
            {
                ret[k] = i;
                k++;
            }
        }
        return ret;
    }();
    static constexpr auto s_b_free_axes = []
    {
        using ret_t = std::array<size_type, b_t::rank() - s_order>;
        ret_t ret{};
        for (size_type i{}, k{}; i != b_t::rank(); ++i)
        {
            if (!s_b_contracted[i])
            {
                ret[k] = i;
                k++;
            }
        }
        return ret;
    }();

    static constexpr auto s_a_contract_strides = []
    {
        using ret_t = std::array<size_type, s_order>;
        ret_t ret{};
        for (size_type i{}; i != s_order; ++i)
        {
            auto const& [a_axis, _] = CIS[i];
            ret[i]                  = a_t::stride(a_axis);
        }
        return ret;
    }();

    static constexpr auto s_b_contract_strides = []
    {
        using ret_t = std::array<size_type, s_order>;
        ret_t ret{};
        for (size_type i{}; i != s_order; ++i)
        {
            auto const& [_, b_axis] = CIS[i];
            ret[i]                  = b_t::stride(b_axis);
        }
        return ret;
    }();

    using outter_loop_t = loop_control<static_shape<s_out_sizes>, 0, -1, 1>;
    using inner_loop_t  = loop_control<static_shape<s_contract_sizes>, 0, -1, 1>;

    using type = tensor<value_type, static_layout<static_shape<s_out_sizes>>>;
};

template <
    containers::concepts::StaticContainer          A,
    containers::concepts::StaticContainer          B,
    containers::concepts::ContractionIndexSet auto CIS>
using tensor_contraction_result_t = typename tensor_contraction_result<A, B, CIS>::type;

} // namespace types

} // namespace v3::utils

#endif // INCLUDED_CONTAINER_UTILS
