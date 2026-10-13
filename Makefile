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

all: ternary_moe test_engine cxxgpt test_training_cli reasoning_audit reasoning_targets

reasoning_targets: experiments/reasoning_targets.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

reasoning_audit: experiments/reasoning_audit.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

research_experiment: experiments/toy_world.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

ternary_moe: ternary_moe.cpp src/ternary.h src/thread_pool.h
	$(CXX) $(CXXFLAGS) $< -o $@

test_engine: tests/test_engine.cpp $(HEADERS) $(TEST_HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

cxxgpt: src/cxxgpt.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

# The CUDA training backend (train --gpu). Needs the CUDA toolkit (nvcc,
# cuBLAS). CUDA_ARCH=native targets the GPU in this machine; name one (e.g.
# sm_89 for an RTX 40xx) to build where there is none.
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++20 -arch=$(CUDA_ARCH) -Xcompiler -Wall,-Wextra $(NVCC_EXTRA)
CUDA_LIBS ?= -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcublas -lcudart

gpu.o: src/gpu.cu src/gpu.h src/config.h src/common.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cxxgpt-gpu: src/cxxgpt.cpp gpu.o $(HEADERS)
	$(CXX) $(CXXFLAGS) -DCXXGPT_CUDA $< gpu.o -o $@ $(CUDA_LIBS)

test_gpu: tests/test_gpu.cpp gpu.o $(HEADERS) $(TEST_HEADERS)
	$(CXX) $(CXXFLAGS) $< gpu.o -o $@ $(CUDA_LIBS)

gpu: cxxgpt-gpu test_gpu

# GPU against CPU, op for op; needs an NVIDIA GPU.
gpu-test: test_gpu
	./test_gpu

.PHONY: all test bench asan tsan clean gpu gpu-test

test_training_cli: tests/test_training_cli.cpp $(HEADERS) $(TEST_HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

test: ternary_moe test_engine cxxgpt test_training_cli reasoning_audit reasoning_targets
	./ternary_moe --tests
	./test_engine
	bash tests/test_cli.sh

bench: ternary_moe
	./ternary_moe

# Sanitizer builds run the tests only.
asan: ternary_moe.cpp tests/test_engine.cpp experiments/toy_world.cpp src/cxxgpt.cpp $(HEADERS) $(TEST_HEADERS) test_training_cli
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    ternary_moe.cpp -o ternary_moe_asan
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    tests/test_engine.cpp -o test_engine_asan
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    experiments/toy_world.cpp -o research_experiment_asan
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    src/cxxgpt.cpp -o cxxgpt_asan
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    experiments/reasoning_audit.cpp -o reasoning_audit_asan
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    experiments/reasoning_targets.cpp -o reasoning_targets_asan
	./ternary_moe_asan --tests
	./test_engine_asan
	CXXGPT=./cxxgpt_asan CXXGPT_AUDIT=./reasoning_audit_asan CXXGPT_TARGETS=./reasoning_targets_asan bash tests/test_cli.sh
	./research_experiment_asan --steps 5 --seeds 1 --threads 2 --continual --rl-steps 5 --out runs/asan-smoke-$$$$.csv

tsan: ternary_moe.cpp tests/test_engine.cpp experiments/toy_world.cpp src/cxxgpt.cpp $(HEADERS) $(TEST_HEADERS) test_training_cli
	$(CXX) $(SANFLAGS) -fsanitize=thread ternary_moe.cpp -o ternary_moe_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread tests/test_engine.cpp -o test_engine_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread experiments/toy_world.cpp -o research_experiment_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread src/cxxgpt.cpp -o cxxgpt_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread experiments/reasoning_audit.cpp -o reasoning_audit_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread experiments/reasoning_targets.cpp -o reasoning_targets_tsan
	./ternary_moe_tsan --tests
	./test_engine_tsan
	CXXGPT=./cxxgpt_tsan CXXGPT_AUDIT=./reasoning_audit_tsan CXXGPT_TARGETS=./reasoning_targets_tsan bash tests/test_cli.sh
	./research_experiment_tsan --steps 5 --seeds 1 --threads 2 --continual --rl-steps 5 --out runs/tsan-smoke-$$$$.csv

clean:
	rm -f ternary_moe test_engine cxxgpt research_experiment reasoning_audit reasoning_targets test_training_cli *_asan *_tsan gpu.o cxxgpt-gpu test_gpu
