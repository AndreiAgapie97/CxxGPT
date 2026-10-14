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

all: ternary_moe test_engine cxxgpt test_training_cli reasoning_audit reasoning_targets reasoning_traces reasoning_curriculum

reasoning_traces: experiments/reasoning_traces.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

reasoning_curriculum: experiments/reasoning_curriculum.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

memory_probe: experiments/memory_probe.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

reasoning_targets: experiments/reasoning_targets.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

reasoning_audit: experiments/reasoning_audit.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

research_experiment: experiments/toy_world.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

ternary_moe: ternary_moe.cpp src/ternary.h src/thread_pool.h
	$(CXX) $(CXXFLAGS) $< -o $@

test_engine: tests/test_engine.cpp $(HEADERS) $(TEST_HEADERS) src/attention.cuh
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

gpu.o: src/gpu.cu src/gpu.h src/attention.cuh src/config.h src/common.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cxxgpt-gpu: src/cxxgpt.cpp gpu.o $(HEADERS)
	$(CXX) $(CXXFLAGS) -DCXXGPT_CUDA $< gpu.o -o $@ $(CUDA_LIBS)

test_gpu: tests/test_gpu.cpp gpu.o $(HEADERS) $(TEST_HEADERS)
	$(CXX) $(CXXFLAGS) $< gpu.o -o $@ $(CUDA_LIBS)

gpu: cxxgpt-gpu test_gpu

# GPU against CPU, op for op; needs an NVIDIA GPU.
gpu-test: test_gpu
	./test_gpu

.PHONY: all test bench asan tsan clean gpu gpu-test optimizer-protocol-test

# Optional research orchestration; training/evaluation themselves are native.
optimizer-protocol-test:
	python3 tests/test_optimizer_selection.py

test_training_cli: tests/test_training_cli.cpp $(HEADERS) $(TEST_HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

test: ternary_moe test_engine cxxgpt test_training_cli reasoning_audit reasoning_targets reasoning_traces reasoning_curriculum
	./ternary_moe --tests
	./test_engine
	bash tests/test_cli.sh

bench: ternary_moe
	./ternary_moe

# Sanitizer builds run the tests only.
asan: ternary_moe.cpp tests/test_engine.cpp experiments/toy_world.cpp src/cxxgpt.cpp $(HEADERS) $(TEST_HEADERS) test_training_cli src/attention.cuh
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
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    experiments/memory_probe.cpp -o memory_probe_asan
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    experiments/reasoning_traces.cpp -o reasoning_traces_asan
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    experiments/reasoning_curriculum.cpp -o reasoning_curriculum_asan
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    tests/test_training_cli.cpp -o test_training_cli_asan
	./ternary_moe_asan --tests
	./test_engine_asan
	CXXGPT_TRAINING_CLI=./test_training_cli_asan CXXGPT=./cxxgpt_asan CXXGPT_AUDIT=./reasoning_audit_asan CXXGPT_TARGETS=./reasoning_targets_asan CXXGPT_TRACES=./reasoning_traces_asan CXXGPT_CURRICULUM=./reasoning_curriculum_asan bash tests/test_cli.sh
	./research_experiment_asan --steps 5 --seeds 1 --threads 2 --continual --rl-steps 5 --out runs/asan-smoke-$$$$.csv
	./memory_probe_asan --mode train --ternary --experts 4 --muon --batch 2 --seq 8 --steps 1 > /dev/null
	./memory_probe_asan --mode packed --ternary --experts 4 --batch 1 > /dev/null

tsan: ternary_moe.cpp tests/test_engine.cpp experiments/toy_world.cpp src/cxxgpt.cpp $(HEADERS) $(TEST_HEADERS) test_training_cli src/attention.cuh
	$(CXX) $(SANFLAGS) -fsanitize=thread ternary_moe.cpp -o ternary_moe_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread tests/test_engine.cpp -o test_engine_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread experiments/toy_world.cpp -o research_experiment_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread src/cxxgpt.cpp -o cxxgpt_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread experiments/reasoning_audit.cpp -o reasoning_audit_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread experiments/reasoning_targets.cpp -o reasoning_targets_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread experiments/memory_probe.cpp -o memory_probe_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread experiments/reasoning_traces.cpp -o reasoning_traces_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread experiments/reasoning_curriculum.cpp -o reasoning_curriculum_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread tests/test_training_cli.cpp -o test_training_cli_tsan
	./ternary_moe_tsan --tests
	./test_engine_tsan
	CXXGPT_TRAINING_CLI=./test_training_cli_tsan CXXGPT=./cxxgpt_tsan CXXGPT_AUDIT=./reasoning_audit_tsan CXXGPT_TARGETS=./reasoning_targets_tsan CXXGPT_TRACES=./reasoning_traces_tsan CXXGPT_CURRICULUM=./reasoning_curriculum_tsan bash tests/test_cli.sh
	./research_experiment_tsan --steps 5 --seeds 1 --threads 2 --continual --rl-steps 5 --out runs/tsan-smoke-$$$$.csv
	./memory_probe_tsan --mode train --ternary --experts 4 --muon --batch 2 --seq 8 --steps 1 > /dev/null
	./memory_probe_tsan --mode packed --ternary --experts 4 --batch 1 > /dev/null

clean:
	rm -f ternary_moe test_engine cxxgpt research_experiment reasoning_audit reasoning_targets reasoning_traces reasoning_curriculum memory_probe test_training_cli *_asan *_tsan gpu.o cxxgpt-gpu test_gpu
