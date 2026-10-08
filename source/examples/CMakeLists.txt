set(GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/scripts/generate_examples.py"
        "${CMAKE_SOURCE_DIR}/scripts/config.py"
)

message(STATUS "CONFIGURING EXAMPLES")
execute_process(
    COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/scripts/generate_examples.py"
        "--out_dir=${GENERATED_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
)

file(GLOB_RECURSE EXAMPLE_SOURCES
    CONFIGURE_DEPENDS
    "${GENERATED_DIR}/*.e.cpp"
)

foreach(src IN LISTS EXAMPLE_SOURCES)

    cmake_path(GET src STEM  target)

    add_library(${target} OBJECT "${src}")

    target_link_libraries(${target}
        PRIVATE
            hpc_seminar
            compiler_options
            compiler_warnings
            assembly_options
    )

endforeach()
