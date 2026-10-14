// Portable utility checks, plus simulated CPU/OS capabilities for dispatch.
#pragma once
#include "../src/cpu_features.h"
#include "test_util.h"
#include "test_reasoning.h"
#include "test_train.h"

inline void test_portability() {
    // Independent expected truth table: each row describes a CPU/OS setup,
    // rather than reimplementing the feature decoder's bit operations.
    struct Case {
        uint32_t ecx1, ebx7, ecx7, eax71;
        uint64_t xcr0;
        bool avx2, avx512f, avx512bw, avxvnni, avx512vnni;
    };
    const uint32_t os_avx = 0x18000000, all_ebx = 0x40010020;
    const Case cases[] = {
        {0, all_ebx, 0x800, 0x10, 0xe6, false, false, false, false, false},
        {os_avx, all_ebx, 0x800, 0x10, 0x2, false, false, false, false, false},
        {os_avx, all_ebx, 0x800, 0x10, 0x6, true, false, false, true, false},
        {os_avx, 0x20, 0, 0, 0xe6, true, false, false, false, false},
        {os_avx, 0x10020, 0x800, 0, 0xe6, true, true, false, false, false},
        {os_avx, all_ebx, 0, 0, 0xe6, true, true, true, false, false},
        {os_avx, all_ebx, 0x800, 0x10, 0xe6, true, true, true, true, true},
    };
    for (const auto& c : cases) {
        const auto f = decode_cpu_features(c.ecx1, c.ebx7, c.ecx7, c.eax71, c.xcr0);
        EXPECT(f.avx2 == c.avx2 && f.avx512f == c.avx512f && f.avx512bw == c.avx512bw &&
               f.avxvnni == c.avxvnni && f.avx512vnni == c.avx512vnni,
               "CPU dispatch ignored a CPU or OS prerequisite");
    }
#ifdef CXXGPT_RUNTIME_SIMD
    struct PreferenceCase { CpuFeatures features; RowsQFn expected; };
    // Synthetic feature combinations exercise priority even when this
    // machine supports 512-bit VNNI (which otherwise hides the third row).
    const PreferenceCase preferences[] = {
        {{false, false, false, false, false}, gemv_bitplane_int8_scalar},
        {{true, false, false, false, false}, gemv_bitplane_int8_avx2},
        {{true, false, false, true, false}, gemv_bitplane_int8_vnni256},
        {{true, true, true, false, false}, gemv_bitplane_int8_avx512},
        {{true, true, true, true, false}, gemv_bitplane_int8_avx512},
        {{true, true, true, true, true}, gemv_bitplane_int8_vnni512},
    };
    for (const auto& p : preferences)
        EXPECT(preferred_int8_rows(p.features) == p.expected,
               "runtime int8 priority differs from Linux");
    const auto f = cpu_features();
    const RowsQFn expected = f.avx512vnni ? gemv_bitplane_int8_vnni512 :
                             f.avx512bw ? gemv_bitplane_int8_avx512 :
                             f.avxvnni ? gemv_bitplane_int8_vnni256 :
                             f.avx2 ? gemv_bitplane_int8_avx2 : gemv_bitplane_int8_scalar;
    EXPECT(default_int8_kernel().qrows == expected, "Windows selected the wrong int8 kernel");
    for (const auto& k : all_kernels()) {
        EXPECT(f.avx2 || (k.rows != gemv_bitplane_avx2 && k.qrows != gemv_bitplane_int8_avx2),
               "registry exposes unavailable AVX2 kernels");
        EXPECT(f.avx512f || k.rows != gemv_bitplane_avx512, "registry exposes unavailable AVX-512");
        EXPECT(f.avx512bw || k.qrows != gemv_bitplane_int8_avx512, "registry exposes unavailable AVX-512BW");
        EXPECT(f.avxvnni || k.qrows != gemv_bitplane_int8_vnni256, "registry exposes unavailable AVX-VNNI");
        EXPECT(f.avx512vnni || k.qrows != gemv_bitplane_int8_vnni512, "registry exposes unavailable AVX512-VNNI");
    }
#endif
    AlignedBuffer<float> buffer;
    float* p = buffer.get(3);
    EXPECT(uintptr_t(p) % 64 == 0, "buffer is not cache-line aligned");
    EXPECT(buffer.get(2) == p, "a smaller request unnecessarily reallocates");
    EXPECT(uintptr_t(buffer.get(100)) % 64 == 0, "grown buffer lost alignment");

    const double row[] = {2.0, 3.0};
    const MatView<double> wide_stride{row, 4294967297LL, 1};
    EXPECT(wide_stride.rs == 4294967297LL && wide_stride.at(0, 1) == 3.0,
           "matrix stride narrowed to Windows' 32-bit long");

    // Ten-digit operands exceed Windows' long. The reference reads only the
    // question and solves it; the generator uses its column/carry algorithm.
    Rng rng(717);
    const auto large = reasoning_detail::add(rng, 10);
    EXPECT(large.answer == reasoning_check::solve(large), "64-bit reasoning answer changed");
    EXPECT(std::stoll(large.answer) > 2147483647LL, "wide reasoning fixture is too small");

    const std::string path = test_path("portability-replace.ckpt");
    Config cfg = training_test_config(false);
    Model<float> model(cfg); model.init(72);
    AdamW opt; opt.m.assign(model.params.size(), 0.0f); opt.v = opt.m;
    save_checkpoint(path, model, opt, rng);
    model.params[0] = 0.25f;
    save_checkpoint(path, model, opt, rng);
    Model<float> restored(cfg);
    // Null opt exercises the 64-bit seek used to skip moment arrays.
    load_checkpoint(path, restored, nullptr, nullptr);
    EXPECT(restored.params == model.params, "checkpoint replacement kept the old file");
    FILE* file = std::fopen(path.c_str(), "rb");
    EXPECT(file != nullptr, "cannot open saved checkpoint");
    if (file) {
        EXPECT(file_seek(file, 8, SEEK_SET) == 0 && file_tell(file) == 8, "seek/tell offset changed");
        std::fclose(file);
    }
}
