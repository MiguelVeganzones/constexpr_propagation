#include "tensor/v1/tensor.hpp"
#include "tensor/v1/tensor_operations.hpp"
#include "utility/parser.hpp"
#include <iostream>
#include <print>
#include <ranges>
#include <utility>
#include <vector>

struct cmd_opts
{
    using size_type = std::size_t;
    std::vector<size_type>                       a_sizes;
    std::vector<size_type>                       b_sizes;
    std::vector<std::pair<size_type, size_type>> cis;
};

auto main(int argc, char* argv[]) -> int
{
    using namespace v1;
    using F        = float;
    using tensor_t = tensor<F>;

    auto const opts = cmd::parse_options<cmd_opts>({ argv, argv + argc });
    std::println("opts.sizes is {}", opts.a_sizes);
    std::println("opts.sizes is {}", opts.b_sizes);

    tensor_t a(opts.a_sizes);
    tensor_t b(opts.b_sizes);
    std::ranges::iota(a.buffer(), 0);
    std::ranges::iota(b.buffer(), 0);
    std::cout << a << '\n';
    std::cout << b << '\n';
    auto const c  =  tensor_contraction(a, b, opts.cis);
    std::cout << c << '\n';
}
