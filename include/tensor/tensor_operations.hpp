#pragma once
#include <concepts>
#include <utility>

namespace tensor
{

namespace detail
{

template <concepts::StaticContainer A, concepts::StaticContainer B>
struct tensor_product_result
{
    using a_t                        = typename A::value_type;
    using b_t                        = typename B::value_type;
    using value_type                  = typename std::common_type_t<a_t, b_t>;
    static constexpr auto result_size = sequences::concatenate_v<A::sizes(), B::sizes()>;
    using type = static_tensor<value_type, static_layout<static_shape<result_size>>>;
};

template <concepts::StaticContainer A, concepts::StaticContainer B>
using tensor_product_result_t = typename tensor_product_result<A, B>::type;

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

template <
    concepts::StaticContainer          A,
    concepts::StaticContainer          B,
    concepts::ContractionIndexSet auto CIS>
struct tensor_contraction_result
{
    using size_type  = std::common_type_t<typename A::size_type, typename B::size_type>;
    using value_type = std::common_type_t<typename A::value_type, typename B::value_type>;
    static_assert(std::ranges::all_of(
        CIS,
        [](auto const& p) { return A::size(p.first) == B::size(p.second); }
    ));

    static constexpr auto s_out_sizes = []
    {
        constexpr size_type s_in_rank  = A::rank() + B::rank();
        constexpr size_type s_out_rank = s_in_rank - 2 * CIS.order();
        using ret_t                    = std::array<size_type, s_out_rank>;
        // constexpr auto sizes           = sequences::concatenate_v<A::sizes(),
        // B::sizes()>;
        ret_t ret{};

        size_type k = 0;
        for (auto i = size_type{}; i != A::rank(); ++i)
        {
            if (std::ranges::find_if(CIS, [&i](auto const& p) { return p.first == i; }) !=
                std::cend(CIS))
            {
                continue;
            }
            ret[k++] = A::size(i);
        }
        for (auto j = size_type{}; j != B::rank(); ++j)
        {
            if (std::ranges::find_if(
                    CIS, [&j](auto const& p) { return p.second == j; }
                ) != std::cend(CIS))
            {
                continue;
            }
            ret[k++] = B::size(j);
        }
        assert(k == s_out_rank);

        return ret;
    }();

    using type = static_tensor<value_type, static_layout<static_shape<s_out_sizes>>>;
};

template <
    concepts::StaticContainer          A,
    concepts::StaticContainer          B,
    concepts::ContractionIndexSet auto CIS>
using tensor_contraction_result_t = typename tensor_contraction_result<A, B, CIS>::type;


template <
    concepts::ContractionIndexSet auto CIS,
    concepts::StaticContainer          A,
    concepts::StaticContainer          B>
[[nodiscard, deprecated("Not implemented")]]
constexpr auto
    contraction([[maybe_unused]] A const& a, [[maybe_unused]] B const& b) noexcept
    -> utils::types::tensor::
        tensor_contraction_result_t<std::remove_cvref_t<A>, std::remove_cvref_t<B>, CIS>
{
    using a_t                                   = std::remove_cvref_t<A>;
    using b_t                                   = std::remove_cvref_t<B>;
    static constexpr auto contraction_index_set = CIS;
    using ret_t                                 = utils::types::tensor::
        tensor_contraction_result_t<a_t, b_t, contraction_index_set>;
    // using reduction_loop_shape_t = containers::static_shape<>;

    ret_t ret{};
    containers::manipulators::apply(
        ret,
        [](auto& out, auto const& idxs)
        {
            out[idxs] = 1;
            utility::error_handling::assert_unreachable();
        }
    );
    return ret;
}

} // namespace tensor
