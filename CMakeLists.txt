cmake_minimum_required(VERSION 3.25)
project(hpc_seminar VERSION 0.1 LANGUAGES CXX)

include(CTest)

# ----------------------------
# Global settings
# ----------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    endif()
endif()

# ----------------------------
# Output dir
# ----------------------------
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/examples)
set(OUTPUT_DIR_BENCHMARKS ${CMAKE_BINARY_DIR}/bin/benchmarks)

# ----------------------------
# Options
# ----------------------------
include(cmake/compiler_options.cmake)
include(cmake/compiler_warnings.cmake)

# ----------------------------
# Print Status
# ----------------------------
message(STATUS "----------------------------")
message(STATUS "hpc_seminar configuration summary:")
message(STATUS "  Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Output dir       : ${OUTPUT_DIR}")
message(STATUS "----------------------------")

# ----------------------------
# Core taget
# ----------------------------
add_library(hpc_seminar INTERFACE)

target_compile_features(hpc_seminar INTERFACE cxx_std_26)

target_include_directories(hpc_seminar INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(hpc_seminar
    INTERFACE compiler_options
)

# ----------------------------
# Policy targets
# ----------------------------
add_library(hpc_seminar_examples INTERFACE)
add_library(hpc_seminar_tests INTERFACE)
add_library(hpc_seminar_benchmarks INTERFACE)

# ----------------------------
# Compiler flags
# ----------------------------
if(NOT DEFINED WARNINGS)
    message(FATAL_ERROR "Compiler warning flags not provided. Use a toolchain file.")
endif()
if(NOT DEFINED DIAGNOSTICS)
    message(FATAL_ERROR "Compiler diagnostic flags not provided. Use a toolchain file.")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------
# CMake Subdirectories
# ----------------------------

add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(benchmarks)
