#include "../include/parallel_for.h"
#include "../include/time_logger.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#define SLEEP 1
#define BARRIER 2
#define RUNNING 3

static thread_local TimeLogger timeLogger;

static std::vector<uint64_t> RunWithBarrier(size_t threadNum) {
  auto start = Now();
  std::atomic<size_t> reported(0);
  std::vector<uint64_t> times;
  times.reserve(threadNum);
  SpinLock timeslock;
  ParallelFor(0, threadNum, [&](size_t i) {
    auto resultTime = Now() - start;
    {
      // lock is aquired after time measurement
      // so we are not afraid of performance loss
      std::lock_guard<SpinLock> lock(timeslock);
      times.push_back(resultTime);
    }
    reported.fetch_add(1);
    // it's ok to block here because we want
    // to measure time of all threadNum threads
    while (reported.load() != threadNum) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  std::lock_guard<SpinLock> lock(timeslock);
  return times;
}

static std::vector<uint64_t> RunWithSleep(size_t threadNum) {
  auto start = Now();
  std::vector<uint64_t> times;
  times.reserve(threadNum);
  SpinLock timeslock;
  ParallelFor(0, threadNum, [&](size_t i) {
    auto resultTime = Now() - start;
    {
      // lock is aquired after time measurement
      // so we are not afraid of performance loss
      std::lock_guard<SpinLock> lock(timeslock);
      times.push_back(resultTime);
    }
    // spin 1 seconds without sleep
    auto spinStart = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - spinStart <
           std::chrono::seconds(1)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // TODO: kmp block time here? maybe spin 1 second without sleep?
  });
  std::lock_guard<SpinLock> lock(timeslock);
  return times;
}

static std::vector<uint64_t> RunMultitask(size_t threadNum) {
  auto start = Now();
  auto tasksNum = threadNum * 100;
  auto totalBenchTime = std::chrono::duration<double>(1);
  auto sleepFor = totalBenchTime * threadNum / tasksNum;
  ParallelFor(0, tasksNum, [&](size_t i) {
    timeLogger.ReportTime(Now() - start);
    // sleep for emulating work
    std::this_thread::sleep_for(sleepFor);
  });
  return TimeLogger::EndEpoch();
}

static std::vector<uint64_t> RunOnce(size_t threadNum) {
  // todo: mfence here?
  /*
  for arm:
  asm volatile(
        "DMB SY \n"  Data Memory Barrier. Full runtime operation.
        "DSB SY \n"  Data Synchronization Barrier. Full runtime operation.
        "ISB    \n"  Instruction Synchronization Barrier.
        ::: "memory"
        );
  */

#if SCHEDULING_MEASURE_MODE == BARRIER
  return RunWithBarrier(threadNum);
#endif
#if SCHEDULING_MEASURE_MODE == SLEEP
  return RunWithSleep(threadNum);
#endif
#if SCHEDULING_MEASURE_MODE == MULTITASK
  return RunMultitask(threadNum);
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
  RunOnce(threadNum); // just for warmup

  size_t repeat = 5;
  std::vector<std::vector<uint64_t>> results;
  for (size_t i = 0; i < repeat; i++) {
    auto times = RunOnce(threadNum);
    std::sort(times.begin(), times.end());
    results.push_back(times);
  }
  printTimes(threadNum, results);
  return 0;
}
