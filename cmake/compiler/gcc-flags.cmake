# ----------------------------
# Compiler flags
# ----------------------------

set(COMMON_FLAGS
    -fverbose-asm
    -freflection
    -fno-exceptions
)

set(WARNINGS
    -fbounds-check
    -fvisibility=hidden
    -pedantic
    -Wall
    -Wconversion
    -Wdangling-else
    -Wdouble-promotion
    -Werror
    -Wextra
    -Wfloat-equal
    -Wformat
    -Winvalid-pch
    -Wmisleading-indentation
    -Wnull-dereference
    -Wodr
    -Wpadded
    -Wpointer-arith
    -Wredundant-decls
    -Wrestrict
    -Wreturn-local-addr
    -Wshadow
    -Wswitch-default
    -Wswitch-enum
    -Wuninitialized
    -Wvla
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

set(SANITIZERS
	-fsanitize=address
	-fsanitize=bounds
	-fsanitize=float-cast-overflow
	-fsanitize=float-divide-by-zero
	-fsanitize=integer-divide-by-zero
	-fsanitize=null
	-fsanitize=signed-integer-overflow
	-fsanitize=undefined
)
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    list(APPEND SANITIZERS -fsanitize=leak)
endif()



