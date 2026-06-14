#ifndef INCLUDED_CONTAINER_UTILS_V2
#define INCLUDED_CONTAINER_UTILS_V2

#include <array>
#include <ranges>
#include <utility>

namespace v2::utils::types
{

auto allocate_output_uninitialized(
    containers::concepts::Container auto const&           a,
    containers::concepts::Container auto const&           b,
    containers::concepts::ContractionIndexSet auto const& cis
)
{
    using a_t = std::remove_cvref_t<decltype(a)>;
    using b_t = std::remove_cvref_t<decltype(b)>;
    using value_type =
        std::common_type_t<typename a_t::value_type, typename b_t::value_type>;
    using size_type =
        std::common_type_t<typename a_t::size_type, typename b_t::size_type>;
    constexpr auto      s_order    = std::remove_cvref_t<decltype(cis)>::order();
    constexpr size_type s_out_rank = a_t::rank() + b_t::rank() - 2 * s_order;
    using tensor_c_t               = v2::tensor<value_type, s_out_rank>;

#ifndef NDEBUG
    for (auto const& [a_axis, b_axis] : cis)
    {
        assert(a.size(a_axis) == b.size(b_axis));
    }
#endif

    std::array<bool, a_t::rank()> a_contracted{};
    std::array<bool, b_t::rank()> b_contracted{};
    for (auto const& [a_idx, b_idx] : cis)
    {
        a_contracted[a_idx] = true;
        b_contracted[b_idx] = true;
    }
    std::array<size_type, s_out_rank> out_sizes{};
    size_type                         k{};
    for (size_type i{}; i != a_t::rank(); ++i)
    {
        if (!a_contracted[i]) out_sizes[k++] = a.size(i);
    }
    for (size_type i{}; i != b_t::rank(); ++i)
    {
        if (!b_contracted[i]) out_sizes[k++] = b.size(i);
    }
    return tensor_c_t(out_sizes);
}

template <std::integral Index_Type, std::integral auto Order>
struct contraction_index_set
{
public:
    using index_t                 = Index_Type;
    using size_type               = index_t;
    static constexpr auto s_order = Order;
    using index_pair_t            = std::pair<index_t, index_t>;
    using container_t             = std::array<index_pair_t, s_order>;
    using const_iterator          = typename container_t::const_iterator;
    using iterator                = typename container_t::iterator;
    using value_type              = typename container_t::value_type;

    static_assert(s_order >= 0);

public:
    explicit constexpr contraction_index_set(
        std::ranges::sized_range auto const& index_pairs
    )
        requires std::same_as<
            index_pair_t,
            std::ranges::range_value_t<std::remove_cvref_t<decltype(index_pairs)>>>
        : indices_{}
    {
        std::ranges::copy(index_pairs, std::begin(indices_));
    }

    explicit constexpr contraction_index_set(
        std::same_as<index_pair_t> auto const&... index_pairs
    )
        requires(sizeof...(index_pairs) == s_order)
        : indices_{ index_pairs... }
    {
    }

    [[nodiscard]]
    static constexpr auto order() noexcept -> auto
    {
        return s_order;
    }

    [[nodiscard]]
    constexpr auto operator[](index_t const i) const noexcept -> index_pair_t const&

    {
        assert(i < s_order);
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

#endif // INCLUDED_CONTAINER_UTILS_V2
