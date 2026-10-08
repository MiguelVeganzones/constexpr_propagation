include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)

FetchContent_MakeAvailable(googletest)
include(GoogleTest)

set(GENERATED_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/scripts/generate_tests.py
        ${CMAKE_SOURCE_DIR}/scripts/config.py
)

message(STATUS "CONFIGURING TESTS")
execute_process(
    COMMAND
        ${Python_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/generate_tests.py
            --out_dir=${GENERATED_DIR}
    COMMAND_ERROR_IS_FATAL ANY
)

file(GLOB_RECURSE TEST_SOURCES
    CONFIGURE_DEPENDS
    "${GENERATED_DIR}/*.t.cpp"
)

foreach(src IN LISTS TEST_SOURCES)

    cmake_path(GET src STEM  target)

    add_executable(${target}
        "${src}"
    )

    target_include_directories(${target} PRIVATE "${CMAKE_SOURCE_DIR}/tests")

    target_link_libraries(${target}
        PRIVATE
            hpc_seminar
            compiler_options
            compiler_sanitizers
            GTest::gtest_main
    )

    gtest_discover_tests(${target})
endforeach()
