#ifndef INCLUDED_MULTI_INDEX
#define INCLUDED_MULTI_INDEX

#include <algorithm>
#include <concepts>
#include <iterator>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace containers::midx
{

template <std::integral Index_Type>
struct multi_index
{
    using size_type       = Index_Type;
    using index_t         = size_type;
    using rank_t          = size_type;
    using const_iterator  = std::vector<size_type>::const_iterator;
    using iterator        = std::vector<size_type>::iterator;
    using const_reference = std::vector<size_type>::const_reference;
    using reference       = std::vector<size_type>::reference;

    template <std::input_iterator It1, std::input_iterator It2>
    explicit multi_index(It1 sizes, It2 strides, size_type rank) noexcept
        : sizes_(sizes, rank)
        , strides_(strides, rank)
        , idxs_(rank, size_type{})
        , rank_{ rank }
    {
    }

    std::span<size_type const> sizes_;
    std::span<size_type const> strides_;
    std::vector<size_type>     idxs_;
    size_type const            rank_;

    [[nodiscard]]
    auto rank() const noexcept -> rank_t
    {
        return rank_;
    }

    [[nodiscard]]
    auto sizes() const noexcept -> auto const&
    {
        return sizes_;
    }

    [[nodiscard]]
    auto size(index_t const i) const noexcept -> size_type
    {
        return sizes_[i];
    }

    auto reset() noexcept -> void
    {
        std::ranges::fill(idxs_, index_t{});
    }

    struct increment_result_t
    {
        constexpr auto incremented_idx() const noexcept -> rank_t
        {
            return incremented_idx_;
        }

        constexpr auto reverse_incremented_idx() const noexcept -> rank_t
        {
            return incremented_idx_ == rank_ ? rank_
                                             : rank_t{ rank_ - incremented_idx_ } - 1;
        }

        constexpr operator bool() const noexcept
        {
            return incremented_idx_ != rank_;
        }

        constexpr auto is_fastest() const noexcept -> bool
        {
            return incremented_idx_ == rank_t{ rank_ - 1 };
        }

        rank_t incremented_idx_;
        rank_t rank_;
    };

    [[nodiscard]]
    constexpr auto increment() noexcept -> increment_result_t
    {
        for (rank_t d = rank_; d-- > 0;)
        {
            if (idxs_[d] != sizes_[d] - 1)
            {
                ++idxs_[d];
                return { d, rank_ };
            }
            else
            {
                idxs_[d] = index_t{};
            }
        }
        return { rank_, rank_ };
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const i) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](i));
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const i) const noexcept -> const_reference
    {
        return idxs_[i];
    }

    [[nodiscard]]
    constexpr auto cbegin() const noexcept -> const_iterator
    {
        return std::cbegin(idxs_);
    }

    [[nodiscard]]
    constexpr auto cend() const noexcept -> const_iterator
    {
        return std::cend(idxs_);
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
        return std::begin(idxs_);
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
        return std::end(idxs_);
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
        return std::begin(idxs_);
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
        return std::end(idxs_);
    }

    [[nodiscard]]
    constexpr auto data() const noexcept -> auto const&
    {
        return idxs_;
    }

    [[nodiscard]]
    constexpr auto operator<=>(multi_index const&) const noexcept = default;
};

} // namespace containers::midx

#endif // INCLUDED_MULTI_INDEX_V1
