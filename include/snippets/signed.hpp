#include <concepts>

template <std::integral I>
constexpr auto signed(I a) noexcept -> I
{
    return std::is_signed_v<I> ? (a > { 0 } - a < { 0 }) : a > 0;
}
