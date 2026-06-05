# ----------------------------
# Toolchain selection
# ----------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    include(${CMAKE_CURRENT_LIST_DIR}/compiler/gcc-flags.cmake)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    include(${CMAKE_CURRENT_LIST_DIR}/compiler/clang-flags.cmake)
else()
    message(FATAL_ERROR "Unsupported compiler")
endif()

# ----------------------------
# Global compiler options target
# ----------------------------
add_library(compiler_warnings INTERFACE)

target_compile_options(compiler_warnings INTERFACE
    ${WARNINGS}
)
