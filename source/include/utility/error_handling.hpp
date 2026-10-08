#ifndef INCLUDED_ERROR_HANDLING_UTILITY
#define INCLUDED_ERROR_HANDLING_UTILITY

#ifdef NDEBUG
#    include <utility>
#endif
#include <cassert>

namespace utility::error_handling
{

[[noreturn]]
inline auto assert_unreachable() -> void
{
#ifdef NDEBUG
    std::unreachable();
#endif
    assert(false);
}

} // namespace utility::error_handling

#endif // INCLUDED_ERROR_HANDLING_UTILITY
