CXX      ?= g++
MARCH    ?= native
CXXFLAGS ?= -O3 -march=$(MARCH) -std=c++17 -pthread -Wall -Wextra

ternary_moe: ternary_moe.cpp src/thread_pool.h
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: test bench asan tsan clean

test: ternary_moe
	./ternary_moe --tests

bench: ternary_moe
	./ternary_moe

# Sanitizer builds run the tests only.
asan: ternary_moe.cpp src/thread_pool.h
	$(CXX) -O1 -g -march=$(MARCH) -std=c++17 -pthread -fsanitize=address,undefined \
	    -fno-sanitize-recover=undefined $< -o ternary_moe_asan
	./ternary_moe_asan --tests

tsan: ternary_moe.cpp src/thread_pool.h
	$(CXX) -O1 -g -march=$(MARCH) -std=c++17 -pthread -fsanitize=thread $< -o ternary_moe_tsan
	./ternary_moe_tsan --tests

clean:
	rm -f ternary_moe ternary_moe_asan ternary_moe_tsan
