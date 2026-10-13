CXX      ?= g++
MARCH    ?= native
EXTRA    ?=
# clang with -march=native on CPUs that report AVX10 warns that avx10.1-256
# "will be promoted" to 512, which -Werror turns into a failed build.
ifneq ($(findstring clang,$(shell $(CXX) --version 2>/dev/null)),)
CLANG_WARN := -Wno-unknown-warning-option -Wno-invalid-feature-combination
endif
CXXFLAGS ?= -O3 -march=$(MARCH) -std=c++20 -pthread -Wall -Wextra $(CLANG_WARN) $(EXTRA)
SANFLAGS  = -O1 -g -march=$(MARCH) -std=c++20 -pthread $(CLANG_WARN) $(EXTRA)

HEADERS := $(wildcard src/*.h)
TEST_HEADERS := $(wildcard tests/*.h)

all: ternary_moe test_engine cxxgpt

research_experiment: experiments/toy_world.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

ternary_moe: ternary_moe.cpp src/ternary.h src/thread_pool.h
	$(CXX) $(CXXFLAGS) $< -o $@

test_engine: tests/test_engine.cpp $(HEADERS) $(TEST_HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

cxxgpt: src/cxxgpt.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: all test bench asan tsan clean

test: ternary_moe test_engine
	./ternary_moe --tests
	./test_engine

bench: ternary_moe
	./ternary_moe

# Sanitizer builds run the tests only.
asan: ternary_moe.cpp tests/test_engine.cpp experiments/toy_world.cpp $(HEADERS) $(TEST_HEADERS)
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    ternary_moe.cpp -o ternary_moe_asan
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    tests/test_engine.cpp -o test_engine_asan
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    experiments/toy_world.cpp -o research_experiment_asan
	./ternary_moe_asan --tests
	./test_engine_asan
	./research_experiment_asan --steps 5 --seeds 1 --threads 2 --continual --rl-steps 5 --out runs/asan-smoke-$$$$.csv

tsan: ternary_moe.cpp tests/test_engine.cpp experiments/toy_world.cpp $(HEADERS) $(TEST_HEADERS)
	$(CXX) $(SANFLAGS) -fsanitize=thread ternary_moe.cpp -o ternary_moe_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread tests/test_engine.cpp -o test_engine_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread experiments/toy_world.cpp -o research_experiment_tsan
	./ternary_moe_tsan --tests
	./test_engine_tsan
	./research_experiment_tsan --steps 5 --seeds 1 --threads 2 --continual --rl-steps 5 --out runs/tsan-smoke-$$$$.csv

clean:
	rm -f ternary_moe test_engine cxxgpt research_experiment *_asan *_tsan
