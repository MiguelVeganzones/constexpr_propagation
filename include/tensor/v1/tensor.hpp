#ifndef INCLUDED_STATIC_TENSOR_V1
#define INCLUDED_STATIC_TENSOR_V1

#include "common/container_concepts.hpp"
#include "utility/utility_concepts.hpp"
#include <cassert>
#include <concepts>
#include <memory>
#include <numeric>
#include <utility>

namespace v1
{

template <utility::concepts::Arithmetic T>
struct tensor
{
    using value_type      = T;
    using size_type       = std::size_t;
    using index_t         = size_type;
    using rank_t          = size_type;
    using const_iterator  = value_type const*;
    using iterator        = value_type*;
    using const_reference = value_type const&;
    using reference       = value_type&;

    static_assert(std::is_trivially_copyable_v<value_type>);
    static_assert(std::is_standard_layout_v<value_type>);

    explicit tensor(std::ranges::sized_range auto const& dims) noexcept
        requires std::is_same_v<
                     std::ranges::range_value_t<std::remove_cvref_t<decltype(dims)>>,
                     size_type>
        : rank_{ std::ranges::size(dims) }
        , flat_size_{}
        , metadata_{ std::make_unique<index_t[]>(2 * rank_) }
    {
        assert(rank_ > 0);
        auto sizes   = metadata_.get();
        auto strides = sizes + rank_;
        std::ranges::copy(dims, sizes);
        flat_size_         = sizes[rank_ - 1];
        strides[rank_ - 1] = size_type{ 1 };
        for (size_type i = rank_ - 1; i-- > size_type{};)
        {
            strides[i] = strides[i + 1] * sizes[i + 1];
            flat_size_ *= sizes[i];
        }
        buffer_ = std::make_unique<value_type[]>(flat_size_);
    }

    auto operator[](this auto&& self, std::ranges::range auto const& idxs) noexcept
        -> decltype(auto)
    {
        assert(std::ranges::size(idxs) == self.rank_);
        auto const i = std::inner_product(
            std::cbegin(idxs), std::cend(idxs), self.strides_data(), index_t{}
        );
        assert(i < self.flat_size_);
        if constexpr (std::is_const_v<std::remove_reference_t<decltype(self)>>)
            return static_cast<value_type const&>(self.buffer_[i]);
        else
            return self.buffer_[i];
    }

    [[nodiscard]]
    auto sizes() const noexcept -> std::span<size_type const>
    {
        return std::span{ metadata_.get(), metadata_.get() + rank_ };
    }

    [[nodiscard]]
    auto strides() const noexcept -> std::span<size_type const>
    {
        return std::span{ metadata_.get() + rank_, metadata_.get() + 2 * rank_ };
    }

    [[nodiscard]]
    auto buffer() const noexcept -> std::span<value_type const>
    {
        return std::span(buffer_.get(), flat_size_);
    }

    [[nodiscard]]
    auto buffer() noexcept -> std::span<value_type>
    {
        return std::span(buffer_.get(), flat_size_);
    }

    [[nodiscard]]
    auto rank() const noexcept -> size_type
    {
        return rank_;
    }

    [[nodiscard]]
    auto sizes_data() const noexcept -> size_type const*
    {
        return metadata_.get();
    }

    [[nodiscard]]
    auto strides_data() const noexcept -> size_type const*
    {
        return metadata_.get() + rank_;
    }

    [[nodiscard]]
    auto size(index_t rank) const noexcept -> size_type

    {
        assert(rank < rank_);
        return metadata_[rank];
    }

    [[nodiscard]]
    auto stride(index_t rank) const noexcept -> size_type
    {
        assert(rank < rank_);
        return metadata_[rank_ + rank];
    }

    [[nodiscard]]
    constexpr auto cbegin() const noexcept -> const_iterator
    {
        return buffer_.get();
    }

    [[nodiscard]]
    constexpr auto cend() const noexcept -> const_iterator
    {
        return buffer_.get() + flat_size_;
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
        return buffer_.get();
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
        return buffer_.get() + flat_size_;
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
        return buffer_.get();
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
        return buffer_.get() + flat_size_;
    }

    size_type                     rank_;
    size_type                     flat_size_;
    std::unique_ptr<size_type[]>  metadata_;
    std::unique_ptr<value_type[]> buffer_;
};

} // namespace v1

#endif // INCLUDED_STATIC_TENSOR_V1
