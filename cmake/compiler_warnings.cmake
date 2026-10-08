# cmake/compiler_warnings.cmake

add_library(compiler_warnings INTERFACE)

# ----------------------------
# Common warnings
# ----------------------------

target_compile_options(compiler_warnings
    INTERFACE
        -Werror
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wdangling-else
        -Wdouble-promotion
        -Wfloat-equal
        -Wformat
        -Winvalid-pch
        -Wmisleading-indentation
        -Wnull-dereference
        -Wodr
        -Wpadded
        -Wpointer-arith
        -Wredundant-decls
        -Wshadow
        -Wswitch-default
        -Wswitch-enum
        -Wuninitialized
        -Wvla
)

# ----------------------------
# Compiler-specific warnings
# ----------------------------

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

    target_compile_options(compiler_warnings
        INTERFACE
            -Wrestrict
            -Wreturn-local-addr
    )

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    target_compile_options(compiler_warnings
        INTERFACE
            -Wreturn-stack-address
    )

else()

    message(FATAL_ERROR
        "Unsupported C++ compiler: ${CMAKE_CXX_COMPILER_ID}"
    )

endif()
