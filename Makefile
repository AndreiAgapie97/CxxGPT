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

ternary_moe: ternary_moe.cpp src/ternary.h src/thread_pool.h
	$(CXX) $(CXXFLAGS) $< -o $@

test_engine: tests/test_engine.cpp $(HEADERS) $(TEST_HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

cxxgpt: src/cxxgpt.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

# The CUDA training backend (train --gpu). Needs the CUDA toolkit (nvcc,
# cuBLAS). CUDA_ARCH=native targets the GPU in this machine; name one (e.g.
# sm_89 for an RTX 40xx) to build where there is none.
NVCC      ?= nvcc
CUDA_ARCH ?= native
CUDA_HOME ?= /usr/local/cuda
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

test: ternary_moe test_engine
	./ternary_moe --tests
	./test_engine

bench: ternary_moe
	./ternary_moe

# Sanitizer builds run the tests only.
asan: ternary_moe.cpp tests/test_engine.cpp $(HEADERS) $(TEST_HEADERS)
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    ternary_moe.cpp -o ternary_moe_asan
	$(CXX) $(SANFLAGS) -fsanitize=address,undefined -fno-sanitize-recover=undefined \
	    tests/test_engine.cpp -o test_engine_asan
	./ternary_moe_asan --tests
	./test_engine_asan

tsan: ternary_moe.cpp tests/test_engine.cpp $(HEADERS) $(TEST_HEADERS)
	$(CXX) $(SANFLAGS) -fsanitize=thread ternary_moe.cpp -o ternary_moe_tsan
	$(CXX) $(SANFLAGS) -fsanitize=thread tests/test_engine.cpp -o test_engine_tsan
	./ternary_moe_tsan --tests
	./test_engine_tsan

clean:
	rm -f ternary_moe test_engine cxxgpt *_asan *_tsan gpu.o cxxgpt-gpu test_gpu
