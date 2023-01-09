#include "../include/parallel_for.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <vector>

#define SPIN 1
#define BARRIER 2
#define RUNNING 3

namespace {
using ThreadId = int;
struct ScheduledTask {
  size_t TaskIdx;
  ThreadId Id;
  int SchedCpu;
  Timestamp Time;

  ScheduledTask() = default;

  ScheduledTask(size_t taskIdx, Timestamp start) {
    Time = Now() - start; // do it first to avoid overhead of other fields
    TaskIdx = taskIdx;
    Id = GetThreadIndex();
    SchedCpu = sched_getcpu();
  }
};
} // namespace

static std::vector<ScheduledTask> RunWithBarrier(size_t threadNum) {
  std::atomic<size_t> reported(0);
  std::vector<ScheduledTask> results(threadNum);
  auto start = Now();
  ParallelFor(0, threadNum, [&](size_t i) {
    results[i] = ScheduledTask(i, start);
    reported.fetch_add(1, std::memory_order_relaxed);
    // it's ok to block here because we want
    // to measure time of all threadNum threads
    while (reported.load(std::memory_order_relaxed) != threadNum) {
      // we don't sleep or yield here because we want to use cpu
      // like it's real workload
    }
  });
  return results;
}

static std::vector<ScheduledTask> RunWithSpin(size_t threadNum) {
  std::vector<ScheduledTask> results(threadNum);
  auto start = Now();
  ParallelFor(0, threadNum, [&](size_t i) {
    results[i] = ScheduledTask(i, start);
    // emulating work
    volatile uint64_t x = 0; // prevent compiler from optimizing out
    for (size_t i = 0; i < 100'000'000; ++i) {
      x += i;
    }
  });
  return results;
}

static std::vector<ScheduledTask> RunMultitask(size_t threadNum) {
  uint64_t totalSpin = 1'000'000'000;
  auto tasksNum = threadNum * 100;
  std::vector<ScheduledTask> results(tasksNum);
  auto start = Now();
  ParallelFor(0, tasksNum, [&](size_t i) {
    results[i] = ScheduledTask(i, start);
    // emulating work
    volatile uint64_t x = 0; // prevent compiler from optimizing out
    for (size_t i = 0; i < totalSpin / tasksNum; ++i) {
      x += i;
    }
  });
  return results;
}

static std::vector<ScheduledTask> RunOnce(size_t threadNum) {
#if defined(__x86_64__)
  asm volatile("mfence" ::: "memory");
#elif defined(__aarch64__)
  asm volatile(
      "DMB SY \n" /* Data Memory Barrier. Full runtime operation. */
      "DSB SY \n" /* Data Synchronization Barrier. Full runtime operation. */
      "ISB    \n" /* Instruction Synchronization Barrier. */
      ::
          : "memory");
#else
  static_assert(false, "Unsupported architecture");
#endif

#if SCHEDULING_MEASURE_MODE == BARRIER
  return RunWithBarrier(threadNum);
#elif SCHEDULING_MEASURE_MODE == SPIN
  return RunWithSpin(threadNum);
#elif SCHEDULING_MEASURE_MODE == MULTITASK
  return RunMultitask(threadNum);
#else
  static_assert(false, "Unsupported mode");
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
    std::unordered_map<ThreadId, std::vector<ScheduledTask>> resultPerThread;
    for (size_t i = 0; i < res.size(); ++i) {
      auto task = res[i];
      resultPerThread[task.Id].emplace_back(task);
    }
    std::cout << "  {" << std::endl;
    size_t total = 0;
    for (auto &&[id, tasks] : resultPerThread) {
      std::cout << "    \"" << id << "\": [";
      for (size_t i = 0; i != tasks.size(); ++i) {
        auto task = tasks[i];
        std::cout << "{\"index\": " << task.TaskIdx
                  << ", \"time\": " << task.Time
                  << ", \"cpu\": " << task.SchedCpu << "}"
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
  InitParallel(threadNum);
  RunOnce(threadNum); // just for warmup

  std::vector<std::vector<ScheduledTask>> results;
  for (size_t i = 0; i < 10; ++i) {
    results.push_back(RunOnce(threadNum));
  }
  PrintResults(threadNum, results);
  return 0;
}
