#include "parallel_for.h"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

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
  // TODO: configure number of tasks to be sure that all threads are used?
  auto tasksNum = threadNum * 100;
  auto totalBenchTime = std::chrono::duration<double>(1);
  auto sleepFor = totalBenchTime * threadNum / tasksNum;
  ParallelFor(0, tasksNum, [&](size_t i) {
    timeReporter.ReportTime(start);
    // sleep for emulating work
    std::this_thread::sleep_for(sleepFor);
  });
  auto times = TimeReporter::EndEpoch();
  std::sort(times.begin(), times.end());
  return times;
}

static void printTimes(size_t threadNum, const std::vector<uint64_t> &result) {
  std::cout << "{";
  std::cout << "\"thread_num\": " << threadNum << ", ";
  std::cout << "\"used_threads\": " << result.size() << ", ";
  std::cout << "\"start_times\": [";
  for (size_t i = 0; i < result.size(); i++) {
    std::cout << result[i];
    if (i != result.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]}" << std::endl;
}

int main(int argc, char **argv) {
  auto threadNum = GetNumThreads();
  if (argc > 1) {
    threadNum = std::stoi(argv[1]);
  }
  runOnce(threadNum); // just for warmup

  size_t repeat = 20;
  std::vector<uint64_t> result(threadNum);
  for (size_t i = 0; i < repeat; i++) {
    auto times = runOnce(threadNum);
    for (size_t j = 0; j < threadNum; j++) {
      result[j] += times[j];
    }
  }
  for (size_t j = 0; j < threadNum; j++) {
    result[j] /= repeat;
  }
  printTimes(threadNum, result);
  return 0;
}
