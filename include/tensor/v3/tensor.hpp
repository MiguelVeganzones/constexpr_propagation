#ifndef INCLUDED_STATIC_TENSOR_V3
#define INCLUDED_STATIC_TENSOR_V3

#include "common/container_concepts.hpp"
#include "utility/utility_concepts.hpp"
#include <cassert>
#include <ranges>
#include <type_traits>

namespace v3
{

template <utility::concepts::Arithmetic T, containers::concepts::StaticLayout Layout>
struct tensor
{
    using value_type      = std::remove_cv_t<T>;
    using layout_t        = Layout;
    using shape_t         = typename layout_t::shape_t;
    using size_type       = typename layout_t::size_type;
    using index_t         = typename layout_t::index_t;
    using rank_t          = typename layout_t::rank_t;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;

    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::is_standard_layout_v<T>);

    // template <typename U>
    // using rebind_t = tensor<U, Layout>;

    [[nodiscard]]
    static constexpr auto flat_size() noexcept -> size_type
    {
        return layout_t::flat_size();
    }

    [[nodiscard]]
    static constexpr auto elements() noexcept -> size_type
    {
        return layout_t::elements();
    }

    [[nodiscard]]
    static constexpr auto rank() noexcept -> rank_t
    {
        return layout_t::rank();
    }

    [[nodiscard]]
    static constexpr auto sizes() noexcept -> auto const&
    {
        return layout_t::sizes();
    }

    [[nodiscard]]
    static constexpr auto size(index_t const i) noexcept -> size_type
    {
        if (!std::is_constant_evaluated())
        {
            assert(i < rank());
        }
        return layout_t::size(i);
    }

    [[nodiscard]]
    static constexpr auto strides() noexcept -> auto const&
    {
        return layout_t::strides();
    }

    [[nodiscard]]
    static constexpr auto stride(index_t const i) noexcept -> size_type
    {
        assert(i < rank());
        return layout_t::stride(i);
    }

    template <std::integral... I>
        requires(sizeof...(I) == rank())
    [[nodiscard]]
    static constexpr auto linear_index(I&&... idxs) noexcept -> index_t
    {
        return layout_t::linear_index(std::forward<decltype(idxs)>(idxs)...);
    }

    [[nodiscard]]
    static constexpr auto
        linear_index(std::ranges::contiguous_range auto const& idxs) noexcept -> index_t

    {
        // TODO
        // assert(std::ranges::size(idxs) == rank());
        return layout_t::linear_index(idxs);
    }

    [[nodiscard]]
    constexpr auto
        operator[](std::ranges::contiguous_range auto const& idxs) const noexcept
        -> const_reference
    // TODO: @Miguel
    // requires(std::ranges::size(idxs) == rank() &&
    // std::is_same_v<std::ranges::range_value_t<decltype(idxs)>, index_t>)
    {
        return buffer_[linear_index(idxs)];
    }

    [[nodiscard]]
    constexpr auto operator[](std::ranges::contiguous_range auto const& idxs) noexcept
        -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](idxs));
    }

    [[nodiscard]]
    constexpr auto underlying_at(const index_t i) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).underlying_at(i));
    }

    [[nodiscard]]
    constexpr auto underlying_at(const index_t i) const noexcept -> const_reference
    {
        static_assert(
            sizeof(tensor) == sizeof(value_type) * elements(),
            "Container must be compact to bypass the subscript operator!"
        );
        assert(i < elements());
        if constexpr (std::is_signed_v<index_t>)
        {
            assert(i >= index_t{});
        }
        return buffer_[i];
    }

    template <typename... I>
        requires(sizeof...(I) == rank()) && (std::integral<std::remove_cvref_t<I>> && ...)
    [[nodiscard]]
    constexpr auto operator[](I const&... idxs) const noexcept -> const_reference
    {
        return buffer_[linear_index(static_cast<index_t>(idxs)...)];
    }

    template <typename... I>
        requires(sizeof...(I) == rank()) && (std::integral<std::remove_cvref_t<I>> && ...)
    [[nodiscard]]
    constexpr auto operator[](I const&... idxs) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](idxs...));
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const linear_idx) const noexcept -> const_reference
    {
        return buffer_[linear_idx];
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const linear_idx) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](linear_idx));
    }

    [[nodiscard]]
    constexpr auto cbegin() const noexcept -> const_iterator
    {
        return std::cbegin(buffer_);
    }

    [[nodiscard]]
    constexpr auto cend() const noexcept -> const_iterator
    {
        return std::cend(buffer_);
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
        return std::begin(buffer_);
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
        return std::end(buffer_);
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
        return std::begin(buffer_);
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
        return std::end(buffer_);
    }

    [[nodiscard]]
    constexpr auto buffer() const noexcept -> std::span<value_type const, flat_size()>
    {
        return buffer_;
    }

    [[nodiscard]]
    constexpr auto buffer() noexcept -> std::span<value_type, flat_size()>
    {
        return buffer_;
    }

    // TODO: Alignment?
    value_type buffer_[flat_size()];
};

} // namespace v3

#endif // INCLUDED_STATIC_TENSOR_V3
