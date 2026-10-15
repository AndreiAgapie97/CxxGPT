// CPU/OS feature checks for the Windows executable's function-local kernels.
#pragma once
#include <cstdint>
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif

struct CpuFeatures {
    bool avx2 = false, avx512f = false, avx512bw = false;
    bool avxvnni = false, avx512vnni = false;
};

// Kept separate from querying the machine so tests can cover missing CPU
// features and missing OS register support even on an AVX-512 machine.
inline CpuFeatures decode_cpu_features(uint32_t leaf1_ecx, uint32_t leaf7_ebx,
                                       uint32_t leaf7_ecx, uint32_t leaf71_eax,
                                       uint64_t xcr0) {
    const bool avx_state = (leaf1_ecx & (1u << 27)) && (leaf1_ecx & (1u << 28)) &&
                           (xcr0 & 0x6) == 0x6;
    const bool wide_state = avx_state && (xcr0 & 0xe0) == 0xe0;
    CpuFeatures out;
    out.avx2 = avx_state && (leaf7_ebx & (1u << 5));
    out.avx512f = wide_state && (leaf7_ebx & (1u << 16));
    out.avx512bw = out.avx512f && (leaf7_ebx & (1u << 30));
    out.avxvnni = out.avx2 && (leaf71_eax & (1u << 4));
    out.avx512vnni = out.avx512bw && (leaf7_ecx & (1u << 11));
    return out;
}

inline CpuFeatures cpu_features() {
    static const CpuFeatures features = [] {
        int regs[4] = {};
        auto query = [&](int leaf, int subleaf) {
#ifdef _WIN32
            __cpuidex(regs, leaf, subleaf);
#else
            unsigned a, b, c, d;
            __cpuid_count(unsigned(leaf), unsigned(subleaf), a, b, c, d);
            regs[0] = int(a); regs[1] = int(b); regs[2] = int(c); regs[3] = int(d);
#endif
        };
        query(0, 0);
        const int max_leaf = regs[0];
        if (max_leaf < 1) return CpuFeatures{};
        query(1, 0);
        const uint32_t leaf1 = uint32_t(regs[2]);
        uint64_t xcr0 = 0;
        if (leaf1 & (1u << 27)) {
#ifdef _WIN32
            // The intrinsic itself needs xsave enabled, only for this function.
            xcr0 = []() __attribute__((target("xsave"))) { return _xgetbv(0); }();
#else
            // xgetbv is only legal after CPUID reports OSXSAVE.
            unsigned lo, hi;
            asm volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
            xcr0 = uint64_t(lo) | (uint64_t(hi) << 32);
#endif
        }
        if (max_leaf < 7) return CpuFeatures{};
        query(7, 0);
        const uint32_t subleaves = uint32_t(regs[0]), ebx = uint32_t(regs[1]), ecx = uint32_t(regs[2]);
        uint32_t leaf71 = 0;
        if (subleaves >= 1) { query(7, 1); leaf71 = uint32_t(regs[0]); }
        return decode_cpu_features(leaf1, ebx, ecx, leaf71, xcr0);
    }();
    return features;
}
