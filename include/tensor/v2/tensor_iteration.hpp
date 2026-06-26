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

namespace v2::iteration
{

namespace detail
{

template <std::integral Size_Type, Size_Type I, Size_Type Order>
constexpr auto shaped_for_impl(
    std::ranges::sized_range auto const& limits,
    std::ranges::sized_range auto&       idxs,
    auto&&                               fn,
    auto&&... args
) noexcept -> void
{
    using size_type = Size_Type;
    if constexpr (I == Order)
    {
        static_assert(std::invocable<decltype(fn), decltype(args)..., decltype(idxs)>);
        std::invoke(
            std::forward<decltype(fn)>(fn), std::forward<decltype(args)>(args)..., idxs
        );
    }
    else
    {
        for (idxs[I] = 0; idxs[I] != limits[I]; ++idxs[I])
        {
            shaped_for_impl<size_type, I + size_type{ 1 }, Order>(
                limits,
                idxs,
                std::forward<decltype(fn)>(fn),
                std::forward<decltype(args)>(args)...
            );
        }
    }
}

template <std::integral Size_Type, Size_Type I, std::integral auto Rank>
constexpr auto shaped_for_inner_impl(
    std::ranges::sized_range auto const& limits,
    std::ranges::sized_range auto const& a_strides,
    std::ranges::sized_range auto const& b_strides,
    Size_Type                            a_index,
    Size_Type                            b_index,
    auto&&                               fn,
    auto&&... args
) noexcept -> void
{
    using size_type = Size_Type;
    if constexpr (I == size_type{ Rank })
    {
        static_assert(
            std::invocable<decltype(fn), decltype(args)..., size_type, size_type>
        );
        std::invoke(
            std::forward<decltype(fn)>(fn),
            std::forward<decltype(args)>(args)...,
            a_index,
            b_index
        );
    }
    else
    {
        for (size_type i{}; i != limits[I]; ++i)
        {
            shaped_for_inner_impl<size_type, I + size_type{ 1 }, Rank>(
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
constexpr auto shaped_for(
    std::ranges::sized_range auto const& limits,
    auto&&                               fn,
    auto&&... args
) noexcept -> void
{
    using size_type = std::ranges::range_value_t<std::remove_cvref_t<decltype(limits)>>;
    std::array<size_type, Order> idxs{};
    detail::shaped_for_impl<size_type, size_type{}, size_type{ Order }>(
        limits,
        idxs,
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(args)>(args)...
    );
}

template <std::integral auto Rank>
constexpr auto shaped_for_inner(
    std::ranges::sized_range auto const& limits,
    std::ranges::sized_range auto const& a_strides,
    std::ranges::sized_range auto const& b_strides,
    auto&&                               fn,
    auto&&... args
) noexcept -> void
{
    using size_type = std::ranges::range_value_t<std::remove_cvref_t<decltype(limits)>>;
    detail::shaped_for_inner_impl<size_type, size_type{}, Rank>(
        limits,
        a_strides,
        b_strides,
        size_type{},
        size_type{},
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(args)>(args)...
    );
}

} // namespace v2::iteration

#endif // INCLUDED_CONTAINER_MANIPULATIONS
