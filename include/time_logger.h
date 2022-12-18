
#pragma once

#include "spin_lock.h"
#include <mutex>
#include <vector>

inline uint64_t Now() {
#ifdef __x86_64__
  return __rdtsc();
#elif __arm__
  uint64_t val;
  asm volatile("mrs %0, cntvct_el0" : "=r"(val));
  return val;
#endif
}

struct TimeLogger {
  // saves time of first report in current epoch
  // we need epochs to reset times in all threads after each benchmark
  // just incrementing epoch
  void ReportTime(uint64_t result) {
    if (reportedEpoch < currentEpoch) {
      // lock is aquired only once per epoch
      // so it doesn't affect measurements
      std::lock_guard<SpinLock> guard(lock);
      times.push_back(result);
      reportedEpoch = currentEpoch;
    }
  }

  static std::vector<uint64_t> EndEpoch() {
    std::vector<uint64_t> res;
    std::lock_guard<SpinLock> guard(lock);
    res.swap(times);
    currentEpoch++;
    return res;
  }

private:
  size_t reportedEpoch = 0;

  // common for all threads
  static inline SpinLock lock;
  static inline std::vector<uint64_t> times; // nanoseconds
  static inline std::atomic<size_t> currentEpoch = 1;
};
