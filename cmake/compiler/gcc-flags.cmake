# ----------------------------
# Compiler flags
# ----------------------------

set(COMMON_FLAGS
    -fverbose-asm
    -freflection
)

set(WARNINGS
    -fbounds-check
    -Wall
    -Wconversion
    -Werror
    -Wextra
    -Wfloat-equal
    -Wnull-dereference
    -Wodr
    -Wpointer-arith
    -Wrestrict
    -Wuninitialized
)

set(DIAGNOSTICS
	-fconcepts-diagnostics-depth=3
	-fdiagnostics-color=auto
	-fdiagnostics-path-format=inline-events
	-fdiagnostics-show-caret
	-fdiagnostics-show-template-tree
    -ftime-report
)

set(DEBUG_INFO
	-fno-omit-frame-pointer
	-fvar-tracking
	-fvar-tracking-assignments
	-ggdb3
	-gvariable-location-views
	-ginline-points
	-gstatement-frontiers
)

set(DEBUG_FLAGS
    -freflection
	-ffinite-math-only
	-fmax-errors=15
	-fno-eliminate-unused-debug-symbols
	-fno-inline
	-fno-default-inline
	-march=native
	-O0
)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    list(APPEND DEBUG_FLAGS -mavx)
endif()

set(RELEASE_FLAGS
	-fno-math-errno
	-ffast-math
	-fno-trapping-math
	-fstrength-reduce
	-march=native
	-O3
)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    list(APPEND RELEASE_FLAGS -mavx)
endif()

set(RELWITHDEBINFO_FLAGS
	-fno-math-errno
	-fno-trapping-math
	-fstrength-reduce
	-march=native
	-O2
)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    list(APPEND RELWITHDEBINFO_FLAGS -mavx)
endif()
