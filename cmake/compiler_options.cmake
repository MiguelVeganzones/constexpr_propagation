# cmake/compiler_options.cmake

add_library(compiler_options INTERFACE)

add_library(assembly_options INTERFACE)

# ----------------------------
# Common options
# ----------------------------

target_compile_options(compiler_options
    INTERFACE
        -fno-exceptions
        -march=native

        -fvisibility=hidden
)

# ----------------------------
# Assembly options (object file builds)
# ----------------------------

target_compile_options(assembly_options
    INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
            -fverbose-asm
        >
)

# ----------------------------
# Compiler-specific options
# ----------------------------

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

    target_compile_options(compiler_options
        INTERFACE

            # ----------------------------
            # Diagnostics
            # ----------------------------

            $<$<CONFIG:Debug>:
                -fconcepts-diagnostics-depth=3
                -fdiagnostics-color=auto
                -fdiagnostics-path-format=inline-events
                -fdiagnostics-show-caret
                -fdiagnostics-show-template-tree
            >

            # ----------------------------
            # Debug
            # ----------------------------

            $<$<CONFIG:Debug>:
                -fno-omit-frame-pointer
                -fvar-tracking
                -fvar-tracking-assignments
                -ggdb3
                -gvariable-location-views
                -ginline-points
                -gstatement-frontiers

                -ftime-report

                -fmax-errors=15
                -fno-eliminate-unused-debug-symbols
                -fno-inline
                -fno-default-inline
            >

            # ----------------------------
            # Release
            # ----------------------------

            $<$<CONFIG:Release>:
                -ffast-math
                -fassociative-math
                -fomit-frame-pointer
            >

            # ----------------------------
            # RelWithDebInfo
            # ----------------------------

            $<$<CONFIG:RelWithDebInfo>:
                -ffast-math
                -fstrength-reduce
                -fassociative-math
            >
    )

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    target_compile_options(compiler_options
        INTERFACE

            # ----------------------------
            # Diagnostics
            # ----------------------------

            $<$<CONFIG:Debug>:
                -fdiagnostics-color=auto
                -fdiagnostics-show-caret
                -fdiagnostics-show-template-tree
            >

            # ----------------------------
            # Debug
            # ----------------------------

            $<$<CONFIG:Debug>:
                -fno-omit-frame-pointer
                -g
                -fno-inline

                -fmax-errors=15
            >

            # ----------------------------
            # Release
            # ----------------------------

            $<$<CONFIG:Release>:
                -ffast-math
                -fassociative-math
                -fomit-frame-pointer
            >

            # ----------------------------
            # RelWithDebInfo
            # ----------------------------

            $<$<CONFIG:RelWithDebInfo>:
                -ffast-math
                -fassociative-math
            >
    )

else()

    message(FATAL_ERROR
        "Unsupported C++ compiler: ${CMAKE_CXX_COMPILER_ID}"
    )

endif()
