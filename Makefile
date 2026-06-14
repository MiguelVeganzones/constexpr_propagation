fresh: clean generate

.PHONY: generate clean all-builds all-tests

PYTHON := venv/bin/python

PRESETS := \
	gcc-debug \
	clang-debug \
	gcc-relwdebinfo \
	clang-relwdebinfo \
	gcc-release \
	clang-release \

generate:
	$(PYTHON) python/generate_tests.py
	$(PYTHON) python/generate_benchmarks.py

configure-%:
	cmake --preset $*

build-%: generate
	cmake --build --preset $*

test-%:
	ctest --test-dir build/$*/tests

run-%: configure-% build-% test-%

ALL_CONFIGS := $(addprefix configure-,$(PRESETS))
ALL_BUILDS := $(addprefix build-,$(PRESETS))
ALL_TESTS  := $(addprefix test-,$(PRESETS))

all-configs: $(ALL_CONFIGS)
all-builds: $(ALL_BUILDS)
all-tests: $(ALL_TESTS)

clean:
	rm -rf build tests/generated benchmarks/generated
