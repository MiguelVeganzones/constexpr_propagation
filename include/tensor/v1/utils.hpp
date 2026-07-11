#ifndef INCLUDED_CONTAINER_UTILS_V1
#define INCLUDED_CONTAINER_UTILS_V1

#include "common/container_concepts.hpp"
#include "tensor/v1/tensor.hpp"
#include <vector>

namespace v1::utils::types
{

auto allocate_output_uninitialized(
    containers::concepts::Container auto const&           a,
    containers::concepts::Container auto const&           b,
    containers::concepts::ContractionIndexSet auto const& cis
) -> auto
{
    using a_t = std::remove_cvref_t<decltype(a)>;
    using b_t = std::remove_cvref_t<decltype(b)>;
    using value_type =
        std::common_type_t<typename a_t::value_type, typename b_t::value_type>;
    using size_type =
        std::common_type_t<typename a_t::size_type, typename b_t::size_type>;
    using c_t = v1::tensor<value_type>;

#ifndef NDEBUG
    for (auto const& [a_axis, b_axis] : cis)
    {
        assert(a.size(a_axis) == b.size(b_axis));
    }
#endif

    const auto      order    = cis.order();
    const size_type out_rank = a.rank() + b.rank() - 2 * order;
    assert(out_rank > 0);
    std::vector<size_type> out_sizes;
    out_sizes.reserve(out_rank);

    std::vector<bool> a_contracted(a.rank(), false);
    std::vector<bool> b_contracted(b.rank(), false);
    for (auto const& [a_idx, b_idx] : cis)
    {
        a_contracted[a_idx] = true;
        b_contracted[b_idx] = true;
    }
    for (size_type i{}; i != a.rank(); ++i)
    {
        if (!a_contracted[i]) out_sizes.push_back(a.size(i));
    }
    for (size_type i{}; i != b.rank(); ++i)
    {
        if (!b_contracted[i]) out_sizes.push_back(b.size(i));
    }
    return c_t(out_sizes);
}

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

} // namespace v1::utils::types

#endif // INCLUDED_CONTAINER_UTILS_V1
