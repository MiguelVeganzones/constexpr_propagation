#pragma once
#include <concepts>
#include <utility>

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

template <concepts::Container auto S1, concepts::Container auto S2>
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
