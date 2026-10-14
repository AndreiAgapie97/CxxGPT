// thread_pool.h — persistent worker pool with a low-latency parallel_for.
#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

// A condition-variable handoff costs tens of microseconds per job (futex
// wake + scheduler), which is longer than a whole expert-sized GEMV. So the
// workers spin on an atomic between jobs and only fall back to sleeping
// after kSpinMicros without work.
static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

class ThreadPool {
public:
    explicit ThreadPool(int n) : nthreads_(std::max(1, n)) {
        for (int i = 1; i < nthreads_; ++i) workers_.emplace_back([this] { worker(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_.store(true, std::memory_order_relaxed);
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    int size() const { return nthreads_; }

    // Calls fn(row_begin, row_end) on disjoint ranges that together cover
    // [0, total), spread across the pool's threads and the caller. Ranges are
    // claimed dynamically, so a slow or preempted thread just takes fewer.
    // Safe to call from several threads (jobs run one at a time), but not
    // reentrant: fn must not call parallel_for itself. Exceptions are rethrown
    // to the caller after all claimed ranges finish; partial output is unspecified.
    // min_per_thread: below nthreads * min_per_thread items the caller runs
    // the job alone; also the smallest chunk handed out.
    void parallel_for(int total, const std::function<void(int, int)>& fn,
                      int min_per_thread = kDefaultMinPerThread) {
        if (nthreads_ == 1 || total < nthreads_ * min_per_thread) {
            fn(0, total);
            return;
        }
        std::lock_guard<std::mutex> one_job(call_m_);
        job_error_ = nullptr;
        // Close the previous job first. A worker can still hold its exhausted
        // cursor (gen, old_total); compared against the new total, that would
        // look like unclaimed rows, and its CAS would succeed until the cursor
        // changes. total_ is written last with release, so a worker that reads
        // the new total also sees the close and its claim fails.
        cursor_.store((uint64_t(gen_) << 32) | kClosed, std::memory_order_relaxed);
        job_.store(&fn, std::memory_order_relaxed);
        chunk_.store(std::max(min_per_thread, total / (nthreads_ * kChunksPerThread)),
                     std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        total_.store(total, std::memory_order_release);
        const uint32_t gen = ++gen_;
        bool wake;
        {   // under m_ so a worker about to sleep sees either this job or the notify
            std::lock_guard<std::mutex> lk(m_);
            cursor_.store(uint64_t(gen) << 32, std::memory_order_release);
            wake = sleepers_ > 0;
        }
        if (wake) cv_.notify_all();

        work(gen);
        for (unsigned spins = 0;; ++spins) {
            const int done = done_.load(std::memory_order_acquire);
            assert(done <= total && "a row was claimed twice");
            if (done == total) break;
            if (spins < 1u << 20) cpu_relax();
            else std::this_thread::yield();  // a worker got preempted mid-chunk
        }
        // Every claimed range is finished, so fn is no longer borrowed by workers.
        std::lock_guard<std::mutex> error_lock(error_m_);
        if (job_error_) std::rethrow_exception(job_error_);
    }

    static constexpr int kSpinMicros = 1000;
    static constexpr int kDefaultMinPerThread = 16;

private:
    std::mutex error_m_;
    std::exception_ptr job_error_;

    static constexpr int kChunksPerThread = 2;  // each claim costs ~0.5 us of cache traffic
    static constexpr uint32_t kClosed = 0x7FFFFFFF;  // cursor row: no job open

    // Claim and run chunks of job `gen` until none are left. The cursor packs
    // (generation << 32 | next row), so a claim made by a worker that woke up
    // late for an already-finished job fails instead of touching the next one.
    void work(uint32_t gen) {
        uint64_t c = cursor_.load(std::memory_order_acquire);
        for (;;) {
            if (uint32_t(c >> 32) != gen) return;
            const int total = total_.load(std::memory_order_acquire);
            const int r0 = int(uint32_t(c));
            if (r0 >= total) return;
            const int r1 = std::min(r0 + chunk_.load(std::memory_order_relaxed), total);
            if (!cursor_.compare_exchange_weak(c, (uint64_t(gen) << 32) | uint32_t(r1),
                                               std::memory_order_acquire))
                continue;
            // The claim succeeded, so job `gen` is still running and job_ is its.
            try {
                (*job_.load(std::memory_order_relaxed))(r0,r1);
            } catch (...) {
                std::lock_guard<std::mutex> error_lock(error_m_);
                if (!job_error_) job_error_ = std::current_exception();
            }
            done_.fetch_add(r1 - r0, std::memory_order_release);
        }
    }

    uint32_t current_gen() const {
        return uint32_t(cursor_.load(std::memory_order_acquire) >> 32);
    }

    // Spin until a job newer than `seen` is posted (or stop), then sleep.
    uint32_t wait_for_job(uint32_t seen) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::microseconds(kSpinMicros);
        for (unsigned i = 1;; ++i) {
            if (current_gen() != seen || stop_.load(std::memory_order_relaxed))
                return current_gen();
            cpu_relax();
            if (i % 256 == 0 && std::chrono::steady_clock::now() > deadline) break;
        }
        std::unique_lock<std::mutex> lk(m_);
        ++sleepers_;
        cv_.wait(lk, [&] { return current_gen() != seen || stop_.load(std::memory_order_relaxed); });
        --sleepers_;
        return current_gen();
    }

    void worker() {
        uint32_t seen = 0;
        for (;;) {
            seen = wait_for_job(seen);
            if (stop_.load(std::memory_order_relaxed)) return;
            work(seen);
        }
    }

    const int nthreads_;
    std::vector<std::thread> workers_;
    std::mutex call_m_;  // held for a whole job
    uint32_t gen_ = 0;   // guarded by call_m_

    // Job description: written before the cursor is published, read by
    // workers after they see it. Atomic only because a late worker may read
    // them while the next job is being set up (and then discard them).
    std::atomic<const std::function<void(int, int)>*> job_{nullptr};
    std::atomic<int> total_{0}, chunk_{1};

    alignas(64) std::atomic<uint64_t> cursor_{0};
    alignas(64) std::atomic<int> done_{0};  // rows finished in the current job

    alignas(64) std::mutex m_;  // sleeping only
    std::condition_variable cv_;
    int sleepers_ = 0;          // guarded by m_
    std::atomic<bool> stop_{false};
};

// The process-wide pool; main() creates it and points this at it.
inline ThreadPool* g_pool = nullptr;
inline ThreadPool& pool() {
    assert(g_pool && "create a ThreadPool and set g_pool first");
    return *g_pool;
}
