#ifndef INCLUDED_TENSOR_PRINTING
#define INCLUDED_TENSOR_PRINTING

#include "container_concepts.hpp"
#include "multi_index.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <optional>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace containers::print
{

struct abs
{
    constexpr auto operator()(auto const& x) const -> auto
    {
        return std::abs(x);
    }
};

auto operator<<(std::ostream& os, concepts::PrintableTensor auto const& t) noexcept
    -> std::ostream&
{
    using tensor_t             = std::remove_cvref_t<decltype(t)>;
    using value_type           = typename tensor_t::value_type;
    const auto        rank     = t.rank();
    static const auto newlines = [rank]
    {
        static const auto pool = [rank] constexpr -> auto
        {
            std::vector<char> arr(rank);
            for (auto& e : arr)
            {
                e = '\n';
            }
            return arr;
        }();
        std::vector<std::string_view> arr(rank);
        for (std::size_t d = 0; d != rank; ++d)
        {
            arr[d] = std::string_view(pool.data(), d + 1);
        }
        return arr;
    }();
    static const auto prefixes = [rank]
    {
        static const auto pool = [rank] constexpr -> auto
        {
            std::vector<char> arr(rank * 2);
            for (std::size_t d = 0; d != rank; ++d)
            {
                arr[d]        = ' ';
                arr[d + rank] = '[';
            }
            return arr;
        }();
        std::vector<std::string_view> arr(rank);
        for (std::size_t d = 0; d != rank; ++d)
        {
            arr[d] = std::string_view(pool.data() + d + 1, rank);
        }
        return arr;
    }();
    static const auto postfix = [rank]
    {
        static const auto pool = [rank] constexpr -> auto
        {
            std::vector<char> arr(rank);
            for (auto& e : arr)
            {
                e = ']';
            }
            return arr;
        }();
        std::vector<std::string_view> arr(rank);
        for (std::size_t d = 0; d != rank; ++d)
        {
            arr[d] = std::string_view(pool.data(), d + 1);
        }
        return arr;
    }();

    auto multi_idx =
        midx::multi_index(std::begin(t.sizes()), std::begin(t.strides()), t.rank());

    std::optional<int> width = std::nullopt;
    if constexpr (std::is_arithmetic_v<value_type>)
    {
        width = (int)(std::ceil(
                          std::log10(
                              std::ranges::max(t.buffer() | std::views::transform(abs{}))
                          )
                      ) +
                      1);
    }

    os << prefixes[rank - 1];
    while (true)
    {
        if constexpr (std::is_arithmetic_v<value_type>)
        {
            os << std::setw(width.value()) << std::setfill(' ');
        }
        os << t[multi_idx];
        auto res = multi_idx.increment();
        if (!res)
        {
            break;
        }
        if (res.is_fastest())
        {
            os << ", ";
        }
        else
        {
            const auto i = res.reverse_incremented_idx() - 1;
            os << postfix[i];
            os << newlines[i];
            os << prefixes[i];
        }
    }
    os << postfix[rank - 1];
    return os;
}

} // namespace containers::print

#endif // INCLUDED_TENSOR_PRINTING
