#ifndef INCLUDED_UTILITY_CONCEPTS
#define INCLUDED_UTILITY_CONCEPTS

#include <type_traits>

namespace utility::concepts
{

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

} // namespace utility::concepts

#endif // INCLUDED_UTILITY_CONCEPTS
