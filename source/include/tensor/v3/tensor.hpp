#ifndef INCLUDED_STATIC_TENSOR_V3
#define INCLUDED_STATIC_TENSOR_V3

#include "common/container_concepts.hpp"
#include "utility/utility_concepts.hpp"
#include <cassert>
#include <memory>
#include <ranges>
#include <span>
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

    static constexpr auto s_stack_threshold = 1uz << 17uz;

    static_assert(std::is_trivially_copyable_v<value_type>);
    static_assert(std::is_standard_layout_v<value_type>);

    // template <typename U>
    // using rebind_t = tensor<U, Layout>;

    template <size_type Flat_Size>
    struct stack_storage
    {
        value_type data_[Flat_Size];
    };

    template <size_type Flat_Size>
    struct heap_storage
    {
        heap_storage()
            : data_(std::make_unique<value_type[]>(Flat_Size))
        {
        }

        std::unique_ptr<value_type[]> data_;
    };

    template <size_type Flat_Size>
    inline static constexpr bool use_stack_v =
        Flat_Size * sizeof(value_type) < (s_stack_threshold);

    template <size_type Flat_Size>
    using storage_t = std::conditional_t<
        use_stack_v<Flat_Size>,
        stack_storage<Flat_Size>,
        heap_storage<Flat_Size>>;

    template <size_type Flat_Size>
    struct buffer_t : storage_t<Flat_Size>
    {
        using base_t = storage_t<Flat_Size>;

        auto get() noexcept -> iterator
        {
            if constexpr (use_stack_v<Flat_Size>)
                return this->data_;
            else
                return this->data_.get();
        }

        auto get() const noexcept -> const_iterator
        {
            if constexpr (use_stack_v<Flat_Size>)
                return this->data_;
            else
                return this->data_.get();
        }

        [[nodiscard]]
        auto span() noexcept -> std::span<value_type, Flat_Size>
        {
            return std::span<value_type, Flat_Size>(std::span<value_type>{ get(),
                                                                           Flat_Size });
        }

        [[nodiscard]]
        auto span() const noexcept -> std::span<value_type const, Flat_Size>
        {
            return std::span<value_type const, Flat_Size>(std::span<value_type const>{
                get(), Flat_Size });
        }
    };

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
        return buffer_.get()[linear_index(idxs)];
    }

    [[nodiscard]]
    constexpr auto operator[](std::ranges::contiguous_range auto const& idxs) noexcept
        -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](idxs));
    }

    template <typename... I>
        requires(sizeof...(I) == rank()) && (std::integral<std::remove_cvref_t<I>> && ...)
    [[nodiscard]]
    constexpr auto operator[](I const&... idxs) const noexcept -> const_reference
    {
        return buffer_.get()[linear_index(static_cast<index_t>(idxs)...)];
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
        return buffer_.get()[linear_idx];
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const linear_idx) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](linear_idx));
    }

    [[nodiscard]]
    constexpr auto cbegin() const noexcept -> const_iterator
    {
        return buffer_.get();
    }

    [[nodiscard]]
    constexpr auto cend() const noexcept -> const_iterator
    {
        return buffer_.get() + flat_size();
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
        return buffer_.get();
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
        return buffer_.get() + flat_size();
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
        return buffer_.get();
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
        return buffer_.get() + flat_size();
    }

    [[nodiscard]]
    constexpr auto buffer() const noexcept -> std::span<value_type const, flat_size()>
    {
        return buffer_.span();
    }

    [[nodiscard]]
    constexpr auto buffer() noexcept -> std::span<value_type, flat_size()>
    {
        return buffer_.span();
    }

    buffer_t<flat_size()> buffer_{};
};

} // namespace v3

#endif // INCLUDED_STATIC_TENSOR_V3
