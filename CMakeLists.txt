cmake_minimum_required(VERSION 3.20)
project(CxxGPT LANGUAGES CXX)

# CMake is only for the native Windows build. On Linux and WSL the Makefile
# is the build: it compiles with -O3 -march=native and runs every test,
# while this file would build a slower binary and only two of the tests.
if(NOT WIN32)
    message(FATAL_ERROR "On Linux and WSL, build with make (make all, make test); "
                        "CMakeLists.txt is only for the native Windows build (README).")
endif()

# clang-cl uses the Windows SDK and Microsoft C++ library, while supporting
# the existing compiler-vector fallback and function-local AVX/VNNI targets
# without enabling those instructions globally.
if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND MSVC))
    message(FATAL_ERROR "Use clang-cl: cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang-cl")
endif()

# Which CPU to compile for.
# - Empty (the default, and what CI builds): baseline x86-64, which runs on
#   any 64-bit PC. Only the ternary row kernels use AVX2/AVX-512/VNNI, picked
#   at runtime for the CPU; the fp32 GEMM and the int8/fp32 helpers stay at
#   SSE or scalar speed.
# - native (-DCXXGPT_MARCH=native, for the laptop): everything is compiled
#   for this machine's CPU, as the Linux Makefile does, so the whole engine
#   uses AVX-512 and the kernels are chosen at compile time. Only this build
#   is comparable with WSL timings. The binary may not run on another CPU.
set(CXXGPT_MARCH "" CACHE STRING "clang -march value for every file, e.g. native; empty for baseline x86-64 with runtime kernel choice")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

add_executable(cxxgpt src/cxxgpt.cpp)
add_executable(ternary_moe ternary_moe.cpp)
add_executable(test_engine tests/test_engine.cpp)
foreach(target cxxgpt ternary_moe test_engine)
    target_link_libraries(${target} PRIVATE Threads::Threads)
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
    target_compile_options(${target} PRIVATE /clang:-Wall /clang:-Wextra /clang:-Werror /clang:-Wno-unused-command-line-argument)
    if(CXXGPT_MARCH)
        # The two warning flags are the Makefile's: on CPUs that report
        # AVX10, clang's -march=native warns, and -Werror would stop.
        target_compile_options(${target} PRIVATE /clang:-march=${CXXGPT_MARCH}
            /clang:-Wno-unknown-warning-option /clang:-Wno-invalid-feature-combination)
    else()
        target_compile_definitions(${target} PRIVATE CXXGPT_RUNTIME_SIMD)
    endif()
endforeach()

enable_testing()
add_test(NAME kernels COMMAND ternary_moe --tests)
add_test(NAME engine COMMAND test_engine)
