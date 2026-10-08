cmake_minimum_required(VERSION 4.4)
project(hpc_seminar
    VERSION 1.0
    LANGUAGES CXX
)

get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT isMultiConfig)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Python COMPONENTS Interpreter)

include(CTest)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/bin"
)

include(cmake/compiler_options.cmake)
include(cmake/compiler_warnings.cmake)
include(cmake/compiler_sanitizers.cmake)

add_library(hpc_seminar INTERFACE)

target_compile_features(hpc_seminar
    INTERFACE
        cxx_std_26
)

target_include_directories(hpc_seminar
    INTERFACE
        "${PROJECT_SOURCE_DIR}/include"
)

add_subdirectory(examples)
add_subdirectory(benchmarks)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
