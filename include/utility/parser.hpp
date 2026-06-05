#ifndef INCLUDED_CMD_PARSER_UTILITY
#define INCLUDED_CMD_PARSER_UTILITY

#include <algorithm>
#include <concepts>
#include <iostream>
#include <istream>
#include <meta>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace utility::cmd
{

template <typename T, typename U>
    requires(std::is_integral_v<T> && std::is_integral_v<U>)
std::istream& operator>>(std::istream& is, std::pair<T, U>& p)
{
    char c{};
    T    a{};
    U    b{};
    if (!(is >> c) || c != '(') return (is.setstate(std::ios::failbit), is);
    if (!(is >> a)) return (is.setstate(std::ios::failbit), is);
    if (!(is >> c) || c != '.') return (is.setstate(std::ios::failbit), is);
    if (!(is >> b)) return (is.setstate(std::ios::failbit), is);
    if (!(is >> c) || c != ')') return (is.setstate(std::ios::failbit), is);
    p = { a, b };
    return is;
}

template <typename T>
std::istream& operator>>(std::istream& is, std::vector<T>& vec)
{
    vec.clear();
    std::string input;
    is >> input;
    std::stringstream ss(input);
    std::string       token;
    while (std::getline(ss, token, ','))
    {
        std::stringstream item(token);
        T                 value{};
        if (!(item >> value))
        {
            is.setstate(std::ios::failbit);
            return is;
        }
        vec.push_back(std::move(value));
    }
    return is;
}

template <typename Opts>
auto parse_options(std::vector<std::string_view> const& args) -> Opts
{
    Opts opts;

    constexpr auto ctx = std::meta::access_context::current();
    template for (constexpr auto dm :
                  define_static_array(nonstatic_data_members_of(^^Opts, ctx)))
    {
        auto it = std::ranges::find_if(
            args,
            [](std::string_view arg)
            { return arg.starts_with("--") && arg.substr(2) == identifier_of(dm); }
        );

        if (it != args.end() && it + 1 != args.end() && !(it + 1)->starts_with("--"))
        {
            auto iss = std::stringstream(it[1]);
            if (iss >> opts.[:dm:]; !iss)
            {
                std::println(
                    "Error: Value supplied for {} is not a valid {}",
                    *it,
                    display_string_of(type_of(dm))
                );
                std::exit(EXIT_FAILURE);
            }
        }
    }
    return opts;
}

} // namespace cmd

#endif // INCLUDED_CMD_PARSER_UTILITY
