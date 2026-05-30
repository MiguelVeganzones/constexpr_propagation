#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <iostream>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace v1::tensor
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

    explicit multi_index(
        size_type const* const sizes,
        size_type const* const strides,
        size_type              rank
    ) noexcept
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

    [[nodiscard]]
    auto increment() noexcept -> bool
    {
        for (rank_t d = rank_; d-- > 0;)
        {
            if (idxs_[d] != sizes_[d] - 1)
            {
                ++idxs_[d];
                return true;
            }
            else
            {
                idxs_[d] = index_t{};
            }
        }
        return false;
    }

    auto offset() const -> index_t
    {
        size_type off = 0;
        for (rank_t i = 0; i < rank_; ++i)
            off += idxs_[i] * strides_[i];
        return off;
    }

    [[nodiscard]]
    auto operator[](index_t const i) noexcept -> reference
    {
        return const_cast<reference>(std::as_const(*this).operator[](i));
    }

    [[nodiscard]]
    auto operator[](index_t const i) const noexcept -> const_reference
    {
        return idxs_[i];
    }

    [[nodiscard]]
    auto cbegin() const noexcept -> const_iterator
    {
        return std::cbegin(idxs_);
    }

    [[nodiscard]]
    auto cend() const noexcept -> const_iterator
    {
        return std::cend(idxs_);
    }

    [[nodiscard]]
    auto begin() const noexcept -> const_iterator
    {
        return std::begin(idxs_);
    }

    [[nodiscard]]
    auto end() const noexcept -> const_iterator
    {
        return std::end(idxs_);
    }

    [[nodiscard]]
    auto begin() noexcept -> iterator
    {
        return std::begin(idxs_);
    }

    [[nodiscard]]
    auto end() noexcept -> iterator
    {
        return std::end(idxs_);
    }

    [[nodiscard]]
    auto data() const noexcept -> auto const&
    {
        return idxs_;
    }

    [[nodiscard]]
    auto operator<=>(multi_index const&) const noexcept = default;
};

} // namespace v1::tensor
