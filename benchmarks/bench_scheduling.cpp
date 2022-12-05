#include "parallel_for.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#define WAITING 1
#define BARRIER 2
#define RUNNING 3

namespace {

static uint64_t Now() {
#ifdef __x86_64__
  return __rdtsc();
#elif __arm__
  uint64_t val;
  asm volatile("mrs %0, cntvct_el0" : "=r"(val));
  return val;
#endif
}

struct TimeReporter {
  // saves time of first report in current epoch
  // we need epochs to reset times in all threads after each benchmark
  // just incrementing epoch
  void ReportTime(uint64_t before) {
    auto duration = Now() - before;
    if (reportedEpoch < currentEpoch) {
      // lock is aquired only once per epoch
      // so it doesn't affect measurements
      std::lock_guard<std::mutex> lock(mutex);
      times.push_back(duration);
      reportedEpoch = currentEpoch;
    }
  }

  static std::vector<uint64_t> EndEpoch() {
    std::vector<uint64_t> res;
    std::lock_guard<std::mutex> lock(mutex);
    res.swap(times);
    currentEpoch++;
    return res;
  }

private:
  size_t reportedEpoch = 0;

  // common for all threads
  static inline std::mutex mutex;
  static inline std::vector<uint64_t> times; // nanoseconds
  static inline std::atomic<size_t> currentEpoch = 1;
};
} // namespace

static thread_local TimeReporter timeReporter;

static std::vector<uint64_t> runOnce(size_t threadNum) {
  auto start = Now();
#if SCHEDULING_MEASURE_MODE == BARRIER
  std::atomic<size_t> reported = 0;
  std::vector<uint64_t> times(threadNum);
  std::mutex timesMutex;
  ParallelFor(0, threadNum, [&](size_t i) {
    auto now = Now();
    auto id = reported.fetch_add(1);
    {
      // lock is aquired after time measurement
      // so we are not afraid of performance loss
      std::lock_guard<std::mutex> lock(timesMutex);
      times[id] = now - start;
    }
    // it's ok to block here because we want
    // to measure time of all threadNum threads
    while (reported.load() != threadNum) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  std::lock_guard<std::mutex> lock(timesMutex);
  return times;
#endif
#if SCHEDULING_MEASURE_MODE == WAITING
  ParallelFor(0, threadNum, [&](size_t i) {
    timeReporter.ReportTime(start);
    // sleep for emulating work
    std::this_thread::sleep_for(std::chrono::seconds(1));
  });
  return TimeReporter::EndEpoch();
#endif
#if SCHEDULING_MEASURE_MODE == MULTITASK
  auto tasksNum = threadNum * 100;
  auto totalBenchTime = std::chrono::duration<double>(1);
  auto sleepFor = totalBenchTime * threadNum / tasksNum;
  ParallelFor(0, tasksNum, [&](size_t i) {
    timeReporter.ReportTime(start);
    // sleep for emulating work
    std::this_thread::sleep_for(sleepFor);
  });
  return TimeReporter::EndEpoch();
#endif
}

static void printTimes(size_t threadNum,
                       const std::vector<std::vector<uint64_t>> &result) {
  std::cout << "{";
  std::cout << "\"thread_num\": " << threadNum << ", ";
  std::cout << "\"results\": [";
  for (size_t i = 0; i < result.size(); ++i) {
    std::cout << "[";
    for (size_t j = 0; j < result[i].size(); ++j) {
      std::cout << result[i][j];
      if (j != result[i].size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]";
    if (i != result.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]";
  std::cout << "}" << std::endl;
}

int main(int argc, char **argv) {
  auto threadNum = GetNumThreads();
  if (argc > 1) {
    threadNum = std::stoi(argv[1]);
  }
  runOnce(threadNum); // just for warmup

  size_t repeat = 20;
  std::vector<std::vector<uint64_t>> results;
  for (size_t i = 0; i < repeat; i++) {
    auto times = runOnce(threadNum);
    std::sort(times.begin(), times.end());
    results.push_back(times);
  }
  printTimes(threadNum, results);
  return 0;
}
