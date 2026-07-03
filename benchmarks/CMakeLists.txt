include(FetchContent)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.zip
)

FetchContent_MakeAvailable(benchmark)

file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
    ./**/*.b.cpp
)

add_executable(benchmarks
    ${EXAMPLE_SOURCES}
)

target_link_libraries(benchmarks PRIVATE
    hpc_seminar
    hpc_seminar_benchmarks
    benchmark::benchmark
    benchmark::benchmark_main
)

set_target_properties(benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR_BENCHMARKS}
)

# foreach(src ${EXAMPLE_SOURCES})
#     get_filename_component(target ${src} NAME_WE)
#     add_executable(${target} ${src})
#     target_link_libraries(${target} PRIVATE
#         hpc_seminar
#         hpc_seminar_benchmarks
#         benchmark::benchmark
#         benchmark::benchmark_main
#     )
#     set_target_properties(${target} PROPERTIES
#         RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR_BENCHMARKS}
#     )
# endforeach()

