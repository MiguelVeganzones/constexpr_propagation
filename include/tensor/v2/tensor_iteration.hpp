#ifndef INCLUDED_CONTAINER_MANIPULATIONS
#define INCLUDED_CONTAINER_MANIPULATIONS

#include "tensor.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <numeric>
#include <ranges>
#include <type_traits>

#define INLINE_ITERATION [[gnu::always_inline, gnu::flatten]]

// #define INLINE_ITERATION [[gnu::noinline]]

namespace v2::iteration
{

namespace detail
{

template <std::integral Index_Type, Index_Type I, std::integral auto Order>
INLINE_ITERATION constexpr auto shaped_for_impl(
    std::ranges::sized_range auto const& limits,
    std::ranges::sized_range auto const& a_strides,
    std::ranges::sized_range auto const& b_strides,
    std::ranges::sized_range auto const& c_strides,
    std::ranges::sized_range auto&       idxs,
    Index_Type                           out_idx,
    Index_Type                           a_base,
    Index_Type                           b_base,
    auto&&                               fn,
    auto&&... args
) noexcept -> void
{
    using index_t = Index_Type;
    if constexpr (I == index_t{ Order })
    {
        static_assert(
            std::
                is_invocable_v<decltype(fn), decltype(args)..., index_t, index_t, index_t>
        );
        std::invoke(
            std::forward<decltype(fn)>(fn),
            std::forward<decltype(args)>(args)...,
            out_idx,
            a_base,
            b_base
        );
    }
    else
    {
        for (idxs[I] = 0; idxs[I] != limits[I]; ++idxs[I])
        {
            shaped_for_impl<index_t, I + index_t{ 1 }, Order>(
                limits,
                a_strides,
                b_strides,
                c_strides,
                idxs,
                out_idx,
                a_base,
                b_base,
                std::forward<decltype(fn)>(fn),
                std::forward<decltype(args)>(args)...
            );
            out_idx += c_strides[I];
            if constexpr (I < a_strides.size())
            {
                a_base += a_strides[I];
            }
            else
            {
                b_base += b_strides[I - a_strides.size()];
            }
        }
    }
}

template <std::integral Index_Type, Index_Type I, std::integral auto Rank>
INLINE_ITERATION constexpr auto shaped_for_inner_impl(
    std::ranges::sized_range auto const& limits,
    std::ranges::sized_range auto const& a_strides,
    std::ranges::sized_range auto const& b_strides,
    Index_Type                           a_index,
    Index_Type                           b_index,
    auto&&                               fn,
    auto&&... args
) noexcept -> void
{
    using index_t = Index_Type;
    if constexpr (I == index_t{ Rank })
    {
        static_assert(std::invocable<decltype(fn), decltype(args)..., index_t, index_t>);
        std::invoke(
            std::forward<decltype(fn)>(fn),
            std::forward<decltype(args)>(args)...,
            a_index,
            b_index
        );
    }
    else
    {
        for (index_t i{}; i != limits[I]; ++i)
        {
            shaped_for_inner_impl<index_t, I + index_t{ 1 }, Rank>(
                limits,
                a_strides,
                b_strides,
                a_index,
                b_index,
                std::forward<decltype(fn)>(fn),
                std::forward<decltype(args)>(args)...
            );
            a_index += a_strides[I];
            b_index += b_strides[I];
        }
    }
}

} // namespace detail

template <std::integral auto Order>
INLINE_ITERATION constexpr auto shaped_for(
    std::ranges::sized_range auto const& limits,
    std::ranges::sized_range auto const& a_strides,
    std::ranges::sized_range auto const& b_strides,
    std::ranges::sized_range auto const& c_strides,
    auto&&                               fn,
    auto&&... args
) noexcept -> void
{
    using index_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(limits)>>;
    std::array<index_t, Order> idxs{};
    detail::shaped_for_impl<index_t, index_t{}, Order>(
        limits,
        a_strides,
        b_strides,
        c_strides,
        idxs,
        index_t{},
        index_t{},
        index_t{},
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(args)>(args)...
    );
}

template <std::integral auto Rank>
INLINE_ITERATION constexpr auto shaped_for_inner(
    std::ranges::sized_range auto const& limits,
    std::ranges::sized_range auto const& a_strides,
    std::ranges::sized_range auto const& b_strides,
    auto&&                               fn,
    auto&&... args
) noexcept -> void
{
    using index_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(limits)>>;
    detail::shaped_for_inner_impl<index_t, index_t{}, Rank>(
        limits,
        a_strides,
        b_strides,
        index_t{},
        index_t{},
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(args)>(args)...
    );
}

} // namespace v2::iteration

#endif // INCLUDED_CONTAINER_MANIPULATIONS
