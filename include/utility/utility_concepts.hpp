#pragma once
#include <concepts>
#include <type_traits>
#include <ranges>

namespace utility::concepts
{

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <typename C>
concept Container = requires {
    typename C::size_type;
    typename C::value_type;
} && std::ranges::sized_range<C>;

template <typename S>
concept StaticShape = requires {
    typename S::size_type;
    typename S::index_t;
    typename S::rank_t;
    { S::rank() } -> std::same_as<typename S::rank_t>;
    { S::elements() } -> std::same_as<typename S::size_type>;
    { S::sizes() };
};

template <typename A>
concept StaticContainer = Container<A> && StaticShape<A>;

} // namespace utility::concepts
