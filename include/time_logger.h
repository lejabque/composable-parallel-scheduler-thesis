
#pragma once
#include "spin_lock.h"
#include "util.h"
#include <mutex>
#include <vector>

struct TimeLogger {
  // saves time of first report in current epoch
  // we need epochs to reset times in all threads after each benchmark
  // just incrementing epoch
  void ReportTime(Timestamp result) {
    if (reportedEpoch < currentEpoch) {
      // lock is aquired only once per epoch
      // so it doesn't affect measurements
      std::lock_guard<SpinLock> guard(lock);
      times.push_back(result);
      reportedEpoch = currentEpoch;
    }
  }

  static std::vector<Timestamp> EndEpoch() {
    std::vector<Timestamp> res;
    std::lock_guard<SpinLock> guard(lock);
    res.swap(times);
    currentEpoch++;
    return res;
  }

private:
  size_t reportedEpoch = 0;

  // common for all threads
  static inline SpinLock lock;
  static inline std::vector<Timestamp> times; // nanoseconds
  static inline std::atomic<size_t> currentEpoch = 1;
};
