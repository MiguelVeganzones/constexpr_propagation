#ifndef INCLUDED_CONTAINER_CONCEPTS
#define INCLUDED_CONTAINER_CONCEPTS

#include <concepts>
#include <ranges>
#include <type_traits>

namespace containers::concepts
{

template <typename C>
concept Container = requires {
    typename C::size_type;
    typename C::value_type;
} && std::ranges::sized_range<C>;

template <typename I>
concept MultiIndex = requires() {
    { I::elements() } -> std::same_as<typename I::size_type>;
    { I::rank() } -> std::same_as<typename I::rank_t>;
} && std::ranges::range<I>;

template <typename L>
concept StaticLayout = requires {
    typename L::size_type;
    typename L::index_t;
    typename L::rank_t;
    { L::rank() } -> std::same_as<typename L::rank_t>;
    { L::elements() } -> std::same_as<typename L::size_type>;
    { L::flat_size() } -> std::same_as<typename L::size_type>;
    { L::sizes() };
    {
        L::size(std::declval<typename L::index_t>())
    } -> std::same_as<typename L::size_type>;
    { L::strides() };
    {
        L::stride(std::declval<typename L::index_t>())
    } -> std::same_as<typename L::size_type>;
    // {
    //     L::linear_index(std::declval<typename L::index_t[L::rank()]>())
    // } -> std::same_as<typename L::index_t>;
};

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

template <typename C>
concept LoopControl = requires {
    typename C::index_t;
    {
        C::start(std::declval<typename C::index_t>())
    } -> std::same_as<typename C::index_t>;
    { C::end(std::declval<typename C::index_t>()) } -> std::same_as<typename C::index_t>;
    {
        C::stride(std::declval<typename C::index_t>())
    } -> std::same_as<typename C::index_t>;
};

template <typename CIS>
concept ContractionIndexSet = requires(CIS const& cis) {
    typename CIS::index_t;
    {
        cis[std::declval<typename CIS::index_t>()]
    } -> std::convertible_to<std::pair<typename CIS::index_t, typename CIS::index_t>>;
};

} // namespace containers::concepts

#endif // INCLUDED_CONTAINER_CONCEPTS
