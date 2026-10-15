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
set(cpu_targets cxxgpt ternary_moe test_engine)

# The CUDA training backend, as the Makefile's `make gpu` builds it:
# cxxgpt-gpu (train --gpu) and test_gpu. Off by default, since CI's Windows
# runner has no CUDA; -DCXXGPT_CUDA=ON on the laptop. nvcc compiles gpu.cu
# with Microsoft's cl.exe, the only host compiler it takes on Windows; the
# rest stays clang-cl. Both halves must use the same C++ runtime (/MD), since
# std::string and std::vector cross gpu.h.
option(CXXGPT_CUDA "Build cxxgpt-gpu and test_gpu (needs the CUDA toolkit and an NVIDIA GPU to run)" OFF)
if(CXXGPT_CUDA)
    set(CXXGPT_CUDA_ARCH "native" CACHE STRING "nvcc -arch value, e.g. sm_89; native targets this machine's GPU")
    find_program(CXXGPT_NVCC nvcc HINTS "$ENV{CUDA_PATH}/bin" REQUIRED)
    find_program(CXXGPT_HOST_CL cl REQUIRED)
    cmake_path(GET CXXGPT_NVCC PARENT_PATH cuda_bin)
    cmake_path(GET cuda_bin PARENT_PATH cuda_root)
    cmake_path(GET CXXGPT_HOST_CL PARENT_PATH host_dir)
    set(gpu_obj "${CMAKE_CURRENT_BINARY_DIR}/gpu.obj")
    add_custom_command(OUTPUT "${gpu_obj}"
        COMMAND "${CXXGPT_NVCC}" -O3 -std=c++20 -arch=${CXXGPT_CUDA_ARCH} -ccbin "${host_dir}"
                -Werror all-warnings -Xcompiler "$<IF:$<CONFIG:Debug>,/MDd,/MD>,/W3,/WX"
                -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu.cu" -o "${gpu_obj}"
        DEPENDS src/gpu.cu src/gpu.h src/attention.cuh src/attention_tc.cuh src/config.h src/common.h
        COMMENT "nvcc src/gpu.cu"
        VERBATIM)
    add_executable(cxxgpt-gpu src/cxxgpt.cpp "${gpu_obj}")
    add_executable(test_gpu tests/test_gpu.cpp "${gpu_obj}")
    target_compile_definitions(cxxgpt-gpu PRIVATE CXXGPT_CUDA)
    foreach(target cxxgpt-gpu test_gpu)
        target_link_libraries(${target} PRIVATE "${cuda_root}/lib/x64/cublas.lib" "${cuda_root}/lib/x64/cudart.lib")
    endforeach()
    list(APPEND cpu_targets cxxgpt-gpu test_gpu)
endif()

foreach(target ${cpu_targets})
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
