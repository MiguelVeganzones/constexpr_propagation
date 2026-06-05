file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
    *.e.cpp
)

foreach(src ${EXAMPLE_SOURCES})
    get_filename_component(target ${src} NAME_WE)
    add_executable(${target} ${src})
    target_link_libraries(${target} PRIVATE
        hpc_seminar
        hpc_seminar_examples
        compiler_warnings
    )
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR}
    )
endforeach()
