fresh: clean all-tests

.PHONY: fresh clean all-configs all-builds all-tests

PYTHON := venv/bin/python

PRESETS := \
	gcc-debug \
	clang-debug \
	gcc-relwdebinfo \
	clang-relwdebinfo \
	gcc-release \
	clang-release \

GENERATE_DEPS := \
	scripts/config.py \
	scripts/generate_tests.py \
	scripts/generate_benchmarks.py \
	$(wildcard scripts/*.py)

GENERATE_STAMP := tests/generated/.stamp

$(GENERATE_STAMP): $(GENERATE_DEPS)
	$(PYTHON) scripts/generate_tests.py
	$(PYTHON) scripts/generate_benchmarks.py
	touch $@

generate: $(GENERATE_STAMP)

configure-%:
	cmake --preset $*

build-%: generate configure-%
	cmake --build --preset $*

test-%: build-%
	ctest --test-dir build/$*/tests

run-%: test-%
	python scripts/run_reference_benchmarks.py
	sh run_benchmarks.sh $*

ALL_CONFIGS := $(addprefix configure-,$(PRESETS))
ALL_BUILDS := $(addprefix build-,$(PRESETS))
ALL_TESTS  := $(addprefix test-,$(PRESETS))

all-configs: $(ALL_CONFIGS)
all-builds: $(ALL_BUILDS)
all-tests: $(ALL_TESTS)

clean:
	rm -rf build tests/generated benchmarks/generated
