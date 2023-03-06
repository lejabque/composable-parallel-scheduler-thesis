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

struct Trace {
  Timestamp Start{};
  Timestamp ExecutionStart{};
  Timestamp ExecutionEnd{};
  Timestamp End{};
};

struct ScheduledTask {
  size_t TaskIdx;
  ThreadId Id;
  int SchedCpu;
  Trace Trace;

  ScheduledTask() = default;

  ScheduledTask(size_t taskIdx, Timestamp start) {
    Trace = {.Start = start, .ExecutionStart = Now()};
    TaskIdx = taskIdx;
    Id = GetThreadIndex();
    SchedCpu = sched_getcpu();
  }

  bool operator<(const ScheduledTask &other) { return TaskIdx < other.TaskIdx; }
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
      CpuRelax();
    }
    results[i].Trace.ExecutionEnd = Now();
  });
  auto end = Now();
  for (auto &&task : results) {
    task.Trace.End = end;
  }
  return results;
}

static std::vector<ScheduledTask> RunWithSpin(size_t threadNum,
                                              size_t tasksPerThread = 1) {
  uint64_t spinPerIter = 100'000'000 / tasksPerThread;
  auto tasksCount = threadNum * tasksPerThread;
  std::vector<ScheduledTask> results(tasksCount);
  auto start = Now();
  ParallelFor(0, tasksCount, [&](size_t i) {
    results[i] = ScheduledTask(i, start);
    // emulating work
    for (size_t i = 0; i < spinPerIter; ++i) {
      CpuRelax();
    }
    results[i].Trace.ExecutionEnd = Now();
  });
  auto end = Now();
  for (auto &&task : results) {
    task.Trace.End = end;
  }
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
  return RunWithSpin(threadNum, 100);
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
      std::sort(tasks.begin(), tasks.end());
      for (size_t i = 0; i != tasks.size(); ++i) {
        auto task = tasks[i];
        std::cout << "{\"index\": " << task.TaskIdx
                  << ", \"trace\": {\"start_timestamp\": " << task.Trace.Start
                  << ", \"scheduling_stage\": "
                  << task.Trace.ExecutionStart - task.Trace.Start
                  << ", \"execution_stage\": "
                  << task.Trace.ExecutionEnd - task.Trace.ExecutionStart
                  << ", \"end_stage\": "
                  << task.Trace.End - task.Trace.ExecutionEnd
                  << "}, \"cpu\": " << task.SchedCpu << "}"
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
