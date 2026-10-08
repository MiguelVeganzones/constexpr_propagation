# cmake/compiler_sanitizers.cmake

add_library(compiler_sanitizers INTERFACE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    set(HPC_SANITIZER_FLAGS
        -fsanitize=address
        -fsanitize=bounds
        -fsanitize=float-cast-overflow
        -fsanitize=float-divide-by-zero
        -fsanitize=integer-divide-by-zero
        -fsanitize=null
        -fsanitize=signed-integer-overflow
        -fsanitize=undefined
    )

    # LeakSanitizer is integrated with AddressSanitizer on supported
    # platforms. Keep the explicit standalone flag only where the
    # original project policy requested it.
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
        list(APPEND HPC_SANITIZER_FLAGS
            -fsanitize=leak
        )
    endif()

    # Compile-time instrumentation.
    target_compile_options(compiler_sanitizers
        INTERFACE
            ${HPC_SANITIZER_FLAGS}
    )

    # Sanitizer runtimes also need to participate in linking.
    target_link_options(compiler_sanitizers
        INTERFACE
            ${HPC_SANITIZER_FLAGS}
    )

else()

    message(FATAL_ERROR
        "Unsupported C++ compiler for sanitizers: "
        "${CMAKE_CXX_COMPILER_ID}"
    )

endif()
