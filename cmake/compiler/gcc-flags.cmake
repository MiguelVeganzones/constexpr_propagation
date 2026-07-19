# ----------------------------
# Compiler flags
# ----------------------------

set(COMMON_FLAGS
    -fverbose-asm
    # -freflection
    -fno-exceptions
	-march=native
    -mavx
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
)

set(DEBUG_INFO
	-fno-omit-frame-pointer
	-fvar-tracking
	-fvar-tracking-assignments
	-ggdb3
	-gvariable-location-views
	-ginline-points
	-gstatement-frontiers
    -ftime-report
)

set(DEBUG_FLAGS
	-ffinite-math-only
	-fmax-errors=15
	-fno-eliminate-unused-debug-symbols
	-fno-inline
	-fno-default-inline
	-O0
)

set(RELEASE_FLAGS
	-fno-math-errno
	-ffast-math
    -fassociative-math
	-fno-trapping-math
	-fstrength-reduce
	-O3
	-fomit-frame-pointer
)

set(RELWITHDEBINFO_FLAGS
	-fno-math-errno
	-fno-trapping-math
	-fstrength-reduce
	-O2
)

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



