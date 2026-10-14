// common.h — small utilities shared by the training engine: RNG, timer,
// aligned buffers, error checks, creating files.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <new>
#include <string>

// Fails loudly in every build type (unlike assert, which NDEBUG removes).
// For conditions that depend on input: files, configs, command lines.
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "%s:%d: check failed: %s\n  ", __FILE__,   \
                         __LINE__, #cond);                                  \
            std::fprintf(stderr, __VA_ARGS__);                              \
            std::fprintf(stderr, "\n");                                     \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// Opens a file for writing (mode "w", "wb" or "a"), first creating the
// folders on its path that are missing, so "--out data/reason" works in a
// fresh clone. Returns nullptr if the file cannot be opened.
inline FILE* create_file(const std::string& path, const char* mode) {
    const std::filesystem::path dir = std::filesystem::path(path).parent_path();
    std::error_code ec;  // on failure, fopen below fails and the caller says so
    if (!dir.empty()) std::filesystem::create_directories(dir, ec);
    return std::fopen(path.c_str(), mode);
}

// Our own RNG, so runs are reproducible across compilers and standard
// libraries (std::normal_distribution is not). xoshiro256**, seeded with
// splitmix64.
class Rng {
public:
    explicit Rng(uint64_t seed = 1) { reseed(seed); }

    void reseed(uint64_t seed) {
        for (auto& w : s_) {
            seed += 0x9E3779B97F4A7C15ull;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            w = z ^ (z >> 31);
        }
    }

    uint64_t next() {
        const uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform in [0, 1), 53 random bits.
    double uniform() { return double(next() >> 11) * 0x1.0p-53; }

    // Uniform integer in [0, n). The bias is below 2^-32 for n < 2^32.
    uint64_t below(uint64_t n) { return n ? next() % n : 0; }

    // Standard normal (Box-Muller; one of the pair is discarded for simplicity).
    double normal() {
        double u1 = uniform();
        while (u1 <= 0.0) u1 = uniform();
        const double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }

    // State, for checkpoints.
    const uint64_t* state() const { return s_; }
    void set_state(const uint64_t* s) { std::memcpy(s_, s, sizeof s_); }

private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t s_[4];
};

inline double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// A heap buffer aligned to 64 bytes (a cache line, and one AVX-512 register).
// Grows on demand and never shrinks; contents are not preserved on growth.
template <class T>
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    ~AlignedBuffer() { std::free(p_); }

    T* get(size_t n) {
        if (n > cap_) {
            std::free(p_);
            const size_t bytes = (n * sizeof(T) + 63) / 64 * 64;
            p_ = static_cast<T*>(std::aligned_alloc(64, bytes));
            if (!p_) throw std::bad_alloc();
            cap_ = n;
        }
        return p_;
    }

private:
    T* p_ = nullptr;
    size_t cap_ = 0;
};
