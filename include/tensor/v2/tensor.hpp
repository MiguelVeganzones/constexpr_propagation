#ifndef INCLUDED_STATIC_RANK_TENSOR_V2
#define INCLUDED_STATIC_RANK_TENSOR_V2

#include "common/container_concepts.hpp"
#include "utility/utility_concepts.hpp"
#include <cassert>
#include <memory>
#include <numeric>
#include <span>
#include <type_traits>

namespace v2
{

template <utility::concepts::Arithmetic T, std::unsigned_integral auto Rank>
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

    inline static constexpr size_type s_rank = Rank;

    static_assert(Rank > 0);
    static_assert(std::is_trivially_copyable_v<value_type>);
    static_assert(std::is_standard_layout_v<value_type>);

    explicit tensor(std::ranges::sized_range auto const& dims) noexcept
        requires std::is_same_v<
                     std::ranges::range_value_t<std::remove_cvref_t<decltype(dims)>>,
                     size_type> &&
                 std::ranges::sized_range<decltype(dims)>
        : flat_size_{}
    {
        assert(std::ranges::size(dims) == Rank);
        std::ranges::copy(dims, sizes_);
        flat_size_           = sizes_[s_rank - 1];
        strides_[s_rank - 1] = size_type{ 1 };
        for (size_type i = s_rank - 1; i-- > size_type{};)
        {
            strides_[i] = strides_[i + 1] * sizes_[i + 1];
            flat_size_ *= sizes_[i];
        }
        buffer_ = std::make_unique<value_type[]>(flat_size_);
    }

    auto operator[](this auto&& self, std::ranges::range auto const& idxs) noexcept
        -> decltype(auto)
    {
        assert(std::ranges::size(idxs) == s_rank);
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
    static constexpr auto rank() noexcept -> size_type
    {
        return s_rank;
    }

    [[nodiscard]]
    auto sizes() const noexcept -> std::span<size_type const>
    {
        return std::span{ sizes_ };
    }

    [[nodiscard]]
    auto strides() const noexcept -> std::span<size_type const>
    {
        return std::span{ strides_ };
    }

    [[nodiscard]]
    auto sizes_data() const noexcept -> size_type const*
    {
        return sizes_;
    }

    [[nodiscard]]
    auto strides_data() const noexcept -> size_type const*
    {
        return strides_;
    }

    [[nodiscard]]
    auto size(index_t rank) const noexcept -> size_type

    {
        assert(rank < s_rank);
        return sizes_[rank];
    }

    [[nodiscard]]
    auto stride(index_t rank) const noexcept -> size_type
    {
        assert(rank < s_rank);
        return strides_[rank];
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

    size_type                     flat_size_;
    size_type                     sizes_[s_rank];
    size_type                     strides_[s_rank];
    std::unique_ptr<value_type[]> buffer_;
};

} // namespace v2

#endif // INCLUDED_STATIC_RANK_TENSOR_V2
