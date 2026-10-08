include(FetchContent)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.zip
)

FetchContent_MakeAvailable(benchmark)

# Prevent -Wpadded -Wswitch-default to propagate to benchmark files
if(TARGET benchmark)
    set_target_properties(benchmark PROPERTIES
        SYSTEM TRUE
    )
endif()

set(GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

set(BENCHMARK_MANIFEST
    "${GENERATED_DIR}/benchmarks.cmake"
)

set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/scripts/generate_benchmarks.py"
        "${CMAKE_SOURCE_DIR}/scripts/config.py"
)

message(STATUS "CONFIGURING BENCHMARKS")
execute_process(
    COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/scripts/generate_benchmarks.py"
        "--out_dir=${GENERATED_DIR}"
        "--manifest_path=${BENCHMARK_MANIFEST}"
    COMMAND_ERROR_IS_FATAL ANY
)

include("${BENCHMARK_MANIFEST}")

add_executable(benchmarks
    ${GENERATED_BENCHMARK_SOURCES}
    "${GENERATED_DIR}/init.cpp"
)

target_link_libraries(benchmarks PRIVATE
    hpc_seminar
    compiler_options
    compiler_warnings
    benchmark::benchmark
    benchmark::benchmark_main
)
