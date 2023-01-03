#include "../include/parallel_for.h"

#include <chrono>
#include <iostream>
#include <unordered_map>
#include <vector>

#define SLEEP 1
#define BARRIER 2
#define RUNNING 3

namespace {
using ThreadId = int;
struct ScheduledTask {
  ThreadId Id;
  Timestamp Time;
};
} // namespace

static std::vector<ScheduledTask> RunWithBarrier(size_t threadNum) {
  std::atomic<size_t> reported(0);
  std::vector<ScheduledTask> results(threadNum);
  auto start = Now();
  ParallelFor(0, threadNum, [&](size_t i) {
    results[i].Time = Now() - start;
    results[i].Id = GetThreadIndex();
    reported.fetch_add(1);
    // it's ok to block here because we want
    // to measure time of all threadNum threads
    while (reported.load() != threadNum) {
      // std::this_thread::sleep_for(std::chrono::milliseconds(1));
      std::this_thread::yield();
    }
  });
  return results;
}

static std::vector<ScheduledTask> RunWithSpin(size_t threadNum) {
  std::vector<ScheduledTask> results(threadNum);
  auto start = Now();
  ParallelFor(0, threadNum, [&](size_t i) {
    results[i].Time = Now() - start;
    results[i].Id = GetThreadIndex();
    // spin 1 seconds
    auto spinStart = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - spinStart <
           std::chrono::seconds(1)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      // std::this_thread::yield();
    }
    // TODO: kmp block time here? maybe spin 1 second without sleep?
  });
  return results;
}

static std::vector<ScheduledTask> RunMultitask(size_t threadNum) {
  auto tasksNum = threadNum * 100;
  auto totalBenchTime = std::chrono::duration<double>(1);
  auto sleepFor = totalBenchTime * threadNum / tasksNum;
  std::vector<ScheduledTask> results(tasksNum);
  auto start = Now();
  ParallelFor(0, tasksNum, [&](size_t i) {
    results[i].Time = Now() - start;
    results[i].Id = GetThreadIndex();
    // sleep for emulating work
    // std::this_thread::sleep_for(sleepFor);
    auto spinStart = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - spinStart < sleepFor) {
      // std::this_thread::sleep_for(std::chrono::milliseconds(1));
      std::this_thread::yield();
    }
  });
  return results;
}

static std::vector<ScheduledTask> RunOnce(size_t threadNum) {
#ifdef __x86_64__
  asm volatile("mfence" ::: "memory");
#endif
#ifdef __aarch64__
  asm volatile(
      "DMB SY \n" /* Data Memory Barrier. Full runtime operation. */
      "DSB SY \n" /* Data Synchronization Barrier. Full runtime operation. */
      "ISB    \n" /* Instruction Synchronization Barrier. */
      ::
          : "memory");
#endif

#if SCHEDULING_MEASURE_MODE == BARRIER
  return RunWithBarrier(threadNum);
#endif
#if SCHEDULING_MEASURE_MODE == SLEEP
  return RunWithSpin(threadNum);
#endif
#if SCHEDULING_MEASURE_MODE == MULTITASK
  return RunMultitask(threadNum);
#endif
}

static void
PrintResults(size_t threadNum,
             const std::vector<std::vector<ScheduledTask>> &results) {
  std::cout << "{" << std::endl;
  std::cout << "\"thread_num\": " << threadNum << "," << std::endl;
  std::cout << "\"tasks_num\": " << results[0].size() << "," << std::endl;
  std::cout << "\"results\": [" << std::endl;
  for (size_t iter = 0; iter != results.size(); ++iter) {
    auto &&res = results[iter];
    std::unordered_map<ThreadId, std::vector<std::pair<size_t, Timestamp>>>
        resultPerThread;
    for (size_t i = 0; i < res.size(); ++i) {
      auto &&task = res[i];
      resultPerThread[task.Id].emplace_back(i, task.Time);
    }
    std::cout << "  {" << std::endl;
    size_t total = 0;
    for (auto &&[id, tasks] : resultPerThread) {
      std::cout << "    \"" << id << "\": [";
      for (size_t i = 0; i != tasks.size(); ++i) {
        auto &&[index, time] = tasks[i];
        std::cout << "{\"index\": " << index << ", \"time\": " << time << "}"
                  << (i == tasks.size() - 1 ? "" : ", ");
      }
      std::cout << (++total == resultPerThread.size() ? " ]" : "],")
                << std::endl;
    }
    std::cout << (iter + 1 == results.size() ? "  }" : "  },") << std::endl;
  }
  std::cout << "]" << std::endl;
  std::cout << "}" << std::endl;
}

int main(int argc, char **argv) {
  auto threadNum = GetNumThreads();
  if (argc > 1) {
    threadNum = std::stoi(argv[1]);
  }
  RunOnce(threadNum, threadNum); // just for warmup

  std::vector<std::vector<ScheduledTask>> results;
  for (size_t i = 0; i < 10; ++i) {
    results.push_back(RunOnce(threadNum, 10 * threadNum));
  }
  PrintResults(threadNum, results);
  return 0;
}
