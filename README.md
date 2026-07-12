# constexpr_propagation

Experimental study of how compile-time information affects code
generation and performance in C++ numerical kernels.

This project explores the trade-offs between runtime flexibility and
compile-time specialization using template metaprogramming and `constexpr`
evaluation.

The main case study is tensor contraction, where tensor properties such as
rank, dimensions, and contraction indices are progressively moved from runtime
to compile time.

## Warning

This project makes extensive use of the `constexpr` capabilities of the compiler.
Only modern compilers are supported.
`gcc-16.1.1` and `clang-22.1.6` have been explicitly tested.

## Motivation

Modern high-performance applications rely on efficient machine code generation
to afford high-level abstractions.

This project investigates how providing additional information to the compiler
affects:

- runtime performance
- generated assembly
- code size
- compilation cost
- compiler optimizations such as constant propagation

The goal is to quantify the cost-benefit trade-off of compile-time specialization.

## Experimental Variants

The implementation contains three levels of specialization.

### v1 — Runtime Rank

Tensor rank and sizes are runtime values.

### v2 — Compile-Time Rank

Tensor and contraction ranks are known at compile time.

### v3 — Compile-Time Sizes

Tensor sizes and contraction structure are known at compile time.

## Repository Structure

```text
.
.
├── benchmarks/
├── build/
├── cmake/
├── dumps/
├── examples/
├── include/
│   ├── common/
│   │   ├── container_concepts.hpp
│   │   ├── multi_index.hpp
│   │   ├── tensor_printing.hpp
│   │   └── timer.hpp
│   ├── tensor/
│   │   ├── v1/
│   │   │   ├── tensor.hpp
│   │   │   ├── tensor_operations.hpp
│   │   │   └── utils.hpp
│   │   ├── v2/
│   │   │   ├── tensor.hpp
│   │   │   ├── tensor_iteration.hpp
│   │   │   ├── tensor_operations.hpp
│   │   │   └── utils.hpp
│   │   └── v3/
│   │       ├── loop_control.hpp
│   │       ├── static_layout.hpp
│   │       ├── static_shape.hpp
│   │       ├── tensor.hpp
│   │       ├── tensor_iteration.hpp
│   │       ├── tensor_operations.hpp
│   │       └── utils.hpp
│   └── utility/
│       ├── error_handling.hpp
│       ├── parser.hpp
│       └── utility_concepts.hpp
├── processing/
├── report/
│   └── report.pdf
├── scripts/
│   ├── config.py
│   ├── generate_benchmarks.py
│   ├── generate_examples.py
│   ├── generate_tests.py
│   └── run_reference_benchmarks.py
├── results
├── tests
├── venv
├── Makefile
├── CMakeLists.txt
├── CMakePresets.json
├── compiler_flags.txt
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

### Software

- C++ compiler with C++23 support
- CMake
- Python 3
- Google Benchmark
- Google Test

## Config

The maximum size and FLOP count of the examples and benchamrks can be configured
in `./scripts/config.py` using the variables `MAX_FLOPS` and `MAX_MEMORY`.

## Building

### Setup python environment

```bash
python -m venv ./venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### Available presets:
```text
	gcc-debug
	clang-debug
	gcc-relwdebinfo
	clang-relwdebinfo
	gcc-release
	clang-release
```

### CMake presets

Cross platform option

```bash
cmake --preset <preset>
cmake --build --preset <preset>
```

### Makefile driver

Linux convenience driver

Generate benchmark files
```bash
make generate
```

Configure preset
```bash
make configure-<preset>
```

Build preset, depends on `generate` and `configure-<preset>`
```bash
make build-<preset>
```

Run tests, depends on `build-<preset>`
```bash
make test-<preset>
```

Run benchmarks, depends on `test-<preset>`
```bash
make run-<preset>
```

## Running Tests

Run tests:

```bash
make test-<preset>
```

or:

```bash
ctest --test-dir build/<preset>/tests
```

## Running Benchmarks

Generate and run benchmarks:

### Cross platform driver

```bash
python scripts/run_benchmakrs.py <preset>
python scripts/run_build_benchmarks.py
python scripts/run_reference_benchmarks.py
```

### Linux convenience driver

```bash
make run-<preset>
```

Or just run benchmarks:

```bash
venv/bin/python ./scripts/run_benchmarks.py <preset>
```

### Cleaning generating files

Generated files and build outputs can be cleared with
```bash
make clean
```

## Benchmark Kernel

The reference kernel is tensor contraction.

The operation is:

$$ C_{\mathbf{i},\mathbf{j}} = \sum_{\mathbf{k}} A_{\mathbf{i},\mathbf{k}} B_{\mathbf{k},\mathbf{j}} $$

The implementation evaluates how different amounts of compile-time information
affect generated code and execution efficiency.

The benchmark suite varies:

- tensor rank
- tensor dimensions
- number of contraction indices
- contraction patterns

## Results

Results include:

- execution time
- scaling behavior
- compilation time
- binary size
- generated assembly inspection

Results and plots:

```text
python processing/process.py <preset>
```

## Related Work

This project was developed as part of an HPC seminar.

Topic:

> Cost-benefit analysis of template metaprogramming and compile-time
> specialization in numerical kernels.

## License

This project is licensed under the [GNU General Public License v3](./LICENSE).

For more information, please see the [LICENSE](./LICENSE) file.
