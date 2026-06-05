#ifndef INCLUDED_TIMER
#define INCLUDED_TIMER

#include <chrono>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace timer
{

using time_int_t = std::uint_fast64_t;
using clock_t    = std::chrono::steady_clock;
using duration_t = std::chrono::duration<time_int_t, std::micro>;

[[nodiscard, gnu::always_inline]]
inline auto time(auto&& fn, auto&&... args) noexcept(noexcept(fn)) -> duration_t
    requires std::is_invocable_r_v<void, decltype(fn), decltype(args)...>
{
    const auto start = clock_t::now();
    std::invoke(std::forward<decltype(fn)>(fn), std::forward<decltype(args)>(args)...);
    return std::chrono::duration_cast<duration_t>(clock_t::now() - start);
}

} // namespace timer

#endif // INCLUDED_TIMER
