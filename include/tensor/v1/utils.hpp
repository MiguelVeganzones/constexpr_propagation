#ifndef INCLUDED_CONTAINER_UTILS_V1
#define INCLUDED_CONTAINER_UTILS_V1

#include <ranges>
#include <utility>
#include <vector>

namespace v1::utils::types
{

template <std::integral Index_Type>
struct contraction_index_set
{
public:
    using index_t        = Index_Type;
    using size_type      = index_t;
    using index_pair_t   = std::pair<index_t, index_t>;
    using container_t    = std::vector<index_pair_t>;
    using const_iterator = typename container_t::const_iterator;
    using iterator       = typename container_t::iterator;
    using value_type     = typename container_t::value_type;

public:
    explicit constexpr contraction_index_set(
        std::ranges::sized_range auto const& index_pairs
    )
        requires std::same_as<
            index_pair_t,
            std::ranges::range_value_t<std::remove_cvref_t<decltype(index_pairs)>>>
        : indices_(std::from_range, index_pairs)
    {
        std::ranges::copy(index_pairs, std::begin(indices_));
    }

    [[nodiscard]]
    constexpr auto order() const noexcept -> auto
    {
        return indices_.size();
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const i) const noexcept -> index_pair_t const&

    {
        assert(i < order());
        if constexpr (std::is_signed_v<index_t>)
        {
            assert(i >= index_t{});
        }
        return indices_[i];
    }

    [[nodiscard]]
    constexpr auto cbegin() const noexcept -> const_iterator
    {
        return std::cbegin(indices_);
    }

    [[nodiscard]]
    constexpr auto cend() const noexcept -> const_iterator
    {
        return std::cend(indices_);
    }

    [[nodiscard]]
    constexpr auto begin() const noexcept -> const_iterator
    {
        return std::begin(indices_);
    }

    [[nodiscard]]
    constexpr auto end() const noexcept -> const_iterator
    {
        return std::end(indices_);
    }

    [[nodiscard]]
    constexpr auto begin() noexcept -> iterator
    {
        return std::begin(indices_);
    }

    [[nodiscard]]
    constexpr auto end() noexcept -> iterator
    {
        return std::end(indices_);
    }

public:
    container_t indices_;
};

} // namespace v2::utils::types

#endif // INCLUDED_CONTAINER_UTILS_V1
