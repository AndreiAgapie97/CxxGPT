// cuda_emu.h — runs CUDA kernels on the CPU, so their tests run on machines
// without a GPU (src/attention.cuh compiles against it).
//
// Each GPU thread of a block is an OS thread, and all of a block's threads
// run at once: __syncthreads is a barrier over the block, and a warp
// shuffle writes each lane's value to a slot, waits for the whole warp,
// reads its partner's slot and waits again. Blocks run one after another.
// It is slow, but it runs the kernel code as written, with real
// concurrency: a missing barrier shows up as a wrong result, or as a race
// under ThreadSanitizer (make tsan).
#pragma once

#include <algorithm>
#include <barrier>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace emu {

struct Dim3 {
    unsigned x = 1, y = 1, z = 1;
};

constexpr int kWarp = 32;

struct Block {
    Block(int threads, size_t smem_floats)
        : sync(threads), smem(smem_floats, 0.0f), slots(size_t(threads), 0.0f) {
        for (int w = 0; w < (threads + kWarp - 1) / kWarp; ++w)
            warps.push_back(std::make_unique<std::barrier<>>(std::min(kWarp, threads - w * kWarp)));
    }
    std::barrier<> sync;
    std::vector<float> smem;   // dynamic shared memory
    std::vector<float> slots;  // one per thread, for shuffles
    std::vector<std::unique_ptr<std::barrier<>>> warps;
};

inline thread_local Dim3 thread_idx, block_idx;
inline thread_local Block* block = nullptr;

inline float* shared_memory() { return block->smem.data(); }
inline void sync_block() { block->sync.arrive_and_wait(); }

// The value `lane` of this thread's warp passed in.
inline float shfl(float v, int lane) {
    const int i = int(thread_idx.x), w = i / kWarp;
    block->slots[size_t(i)] = v;
    block->warps[size_t(w)]->arrive_and_wait();
    const float r = block->slots[size_t(w * kWarp + lane)];
    block->warps[size_t(w)]->arrive_and_wait();
    return r;
}

// kernel<<<grid, threads, smem_bytes>>>: runs `kernel` (which calls the
// kernel function with its arguments) once per thread of every block.
inline void launch(Dim3 grid, int threads, size_t smem_bytes, const std::function<void()>& kernel) {
    for (unsigned z = 0; z < grid.z; ++z)
        for (unsigned y = 0; y < grid.y; ++y)
            for (unsigned x = 0; x < grid.x; ++x) {
                Block b(threads, smem_bytes / sizeof(float));
                std::vector<std::thread> pool;
                pool.reserve(size_t(threads));
                for (int t = 0; t < threads; ++t)
                    pool.emplace_back([&, t] {
                        thread_idx = {unsigned(t), 0, 0};
                        block_idx = {x, y, z};
                        block = &b;
                        kernel();
                    });
                for (auto& t : pool) t.join();
            }
}

}  // namespace emu
