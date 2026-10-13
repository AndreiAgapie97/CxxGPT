// profile.h — wall-clock time per named region, off unless enabled.
//   PROFILE("attention_fwd");  // times the rest of the enclosing block
#pragma once

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "common.h"

class Profiler {
public:
    bool enabled = false;

    void add(const char* name, double seconds) {
        for (auto& e : entries_)
            if (e.name == name) {
                e.seconds += seconds;
                ++e.calls;
                return;
            }
        entries_.push_back({name, seconds, 1});
    }

    void report(int steps) const {
        double total = 0;
        for (const auto& e : entries_) total += e.seconds;
        auto sorted = entries_;
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.seconds > b.seconds; });
        std::printf("profile (ms per step, %d steps):\n", steps);
        for (const auto& e : sorted)
            std::printf("  %-22s %8.2f  %5.1f%%\n", e.name.c_str(), e.seconds * 1e3 / steps,
                        100 * e.seconds / total);
    }

    void reset() { entries_.clear(); }

private:
    struct Entry {
        std::string name;
        double seconds;
        long calls;
    };
    std::vector<Entry> entries_;
};

inline Profiler g_profiler;

class ProfileScope {
public:
    explicit ProfileScope(const char* name)
        : name_(name), t0_(g_profiler.enabled ? now_seconds() : 0.0) {}
    ~ProfileScope() {
        if (g_profiler.enabled) g_profiler.add(name_, now_seconds() - t0_);
    }

private:
    const char* name_;
    double t0_;
};

#define PROFILE_CAT2(a, b) a##b
#define PROFILE_CAT(a, b) PROFILE_CAT2(a, b)
#define PROFILE(name) ProfileScope PROFILE_CAT(profile_scope_, __LINE__)(name)
