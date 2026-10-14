cmake_minimum_required(VERSION 3.20)
project(CxxGPT LANGUAGES CXX)

# Linux keeps its Makefile. clang-cl uses the Windows SDK and Microsoft C++
# library, while supporting the existing compiler-vector fallback and
# function-local AVX/VNNI targets without enabling those instructions globally.
if(WIN32 AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND MSVC))
    message(FATAL_ERROR "Use clang-cl: cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang-cl")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

add_executable(cxxgpt src/cxxgpt.cpp)
add_executable(ternary_moe ternary_moe.cpp)
add_executable(test_engine tests/test_engine.cpp)
foreach(target cxxgpt ternary_moe test_engine)
    target_link_libraries(${target} PRIVATE Threads::Threads)
    if(WIN32)
        target_compile_definitions(${target} PRIVATE CXXGPT_RUNTIME_SIMD _CRT_SECURE_NO_WARNINGS NOMINMAX)
        target_compile_options(${target} PRIVATE /clang:-Wall /clang:-Wextra /clang:-Werror /clang:-Wno-unused-command-line-argument)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
    endif()
endforeach()

enable_testing()
add_test(NAME kernels COMMAND ternary_moe --tests)
add_test(NAME engine COMMAND test_engine)
