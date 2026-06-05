#ifndef INCLUDED_CONTAINER_ITERATION_V3
#define INCLUDED_CONTAINER_ITERATION_V3

#include "common/container_concepts.hpp"
#include "utils.hpp"
#include "loop_control.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <numeric>
#include <type_traits>

namespace v3::iteration
{

namespace detail
{

template <
    containers::concepts::LoopControl Loop_Control,
    std::integral auto                I,
    std::integral                     Index_Type>
[[gnu::always_inline, gnu::flatten]]
constexpr auto shaped_for_impl(
    std::array<Index_Type, Loop_Control::rank()>& idxs,
    auto&&                                        fn,
    auto&&... args
) noexcept -> void
{
    using loop_t = Loop_Control;
    using rank_t = typename loop_t::rank_t;

    if constexpr (rank_t{ I } == loop_t::rank())
    {
        static_assert(std::invocable<decltype(fn), decltype(args)..., decltype(idxs)>);
        std::invoke(
            std::forward<decltype(fn)>(fn), std::forward<decltype(args)>(args)..., idxs
        );
    }
    else
    {
        for (idxs[I] = loop_t::start(I); idxs[I] != loop_t::end(I);
             idxs[I] += loop_t::stride(I))
        {
            shaped_for_impl<loop_t, I + rank_t{ 1 }, Index_Type>(
                idxs,
                std::forward<decltype(fn)>(fn),
                std::forward<decltype(args)>(args)...
            );
        }
    }
}

template <
    containers::concepts::LoopControl Loop_Control,
    std::ranges::sized_range auto     A_Strides,
    std::ranges::sized_range auto     B_Strides,
    std::integral auto                I,
    std::integral                     Index_Type>
    requires(std::ranges::size(A_Strides) == Loop_Control::rank()) &&
            (std::ranges::size(B_Strides) == Loop_Control::rank())
constexpr auto shaped_for_inner_impl(
    Index_Type a_index,
    Index_Type b_index,
    auto&&     fn,
    auto&&... args
) noexcept -> void
{
    using loop_t  = Loop_Control;
    using rank_t  = typename loop_t::rank_t;
    using index_t = Index_Type;

    if constexpr (I == loop_t::rank())
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
        for (auto i = loop_t::start(I); i != loop_t::end(I); i += loop_t::stride(I))
        {
            shaped_for_inner_impl<Loop_Control, A_Strides, B_Strides, I + rank_t{ 1 }>(
                a_index,
                b_index,
                std::forward<decltype(fn)>(fn),
                std::forward<decltype(args)>(args)...
            );
            a_index += A_Strides[I];
            b_index += B_Strides[I];
        }
    }
}

} // namespace detail

template <containers::concepts::LoopControl Loop_Control>
[[gnu::always_inline, gnu::flatten]]
constexpr auto shaped_for(auto&& fn, auto&&... args) noexcept -> void
{
    using loop_t = Loop_Control;
    using rank_t = typename loop_t::rank_t;
    std::array<typename loop_t::index_t, loop_t::rank()> idxs{};
    detail::shaped_for_impl<Loop_Control, rank_t{}>(
        idxs, std::forward<decltype(fn)>(fn), std::forward<decltype(args)>(args)...
    );
}

template <
    containers::concepts::LoopControl Loop_Control,
    std::ranges::sized_range auto     A_Strides,
    std::ranges::sized_range auto     B_Strides>
    requires(std::ranges::size(A_Strides) == Loop_Control::rank()) &&
            (std::ranges::size(B_Strides) == Loop_Control::rank())
constexpr auto shaped_for_inner(auto&& fn, auto&&... args) noexcept -> void
{
    using loop_t  = Loop_Control;
    using rank_t  = typename loop_t::rank_t;
    using index_t = typename loop_t::index_t;
    detail::shaped_for_inner_impl<loop_t, A_Strides, B_Strides, rank_t{}>(
        index_t{},
        index_t{},
        std::forward<decltype(fn)>(fn),
        std::forward<decltype(args)>(args)...
    );
}

} // namespace v3::iteration

#endif // INCLUDED_CONTAINER_ITERATION_V3
