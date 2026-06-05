#pragma once
#include <gtest/gtest.h>
#include <ranges>
#include <span>

inline auto compare(
    std::ranges::sized_range auto const& a,
    std::ranges::sized_range auto const& b
) -> void
{
    ASSERT_EQ(a.size(), b.size());

    for (std::size_t i = 0uz; i != a.size(); ++i)
    {
        EXPECT_NEAR(a[i], b[i], 1e-5);
    }
}
