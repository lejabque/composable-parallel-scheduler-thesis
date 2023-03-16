#include "../include/parallel_for.h"

#include "../include/trace.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

#define SPIN 1
#define BARRIER 2
#define RUNNING 3

namespace {
struct IterationResult {
  IterationResult(size_t count) : Tasks(count) { Start = Now(); }

  Timestamp Start;
  std::vector<Tracing::TaskInfo> Tasks;
  Timestamp End;
};
} // namespace

static thread_local Timestamp WroteTrace = 0;

static IterationResult RunWithBarrier(size_t threadNum) {
  std::atomic<size_t> reported(0);
  IterationResult result(threadNum);
  ParallelFor(0, threadNum, [&](size_t i) {
    result.Tasks[i] = Tracing::TaskInfo(i, WroteTrace);
    reported.fetch_add(1, std::memory_order_relaxed);
    // it's ok to block here because we want
    // to measure time of all threadNum threads
    while (reported.load(std::memory_order_relaxed) != threadNum) {
      CpuRelax();
    }
    result.Tasks[i].Trace.OnExecuted();
    WroteTrace = Now();
  });
  result.End = Now();
  return result;
}

static IterationResult RunWithSpin(size_t threadNum,
                                   size_t tasksPerThread = 1) {
  uint64_t spinPerIter = 100'000'000 / tasksPerThread;
  auto tasksCount = threadNum * tasksPerThread;
  IterationResult result(tasksCount);
  ParallelFor(0, tasksCount, [&](size_t i) {
    result.Tasks[i] = Tracing::TaskInfo(i, WroteTrace);
    // emulating work
    for (size_t i = 0; i < spinPerIter; ++i) {
      CpuRelax();
    }
    result.Tasks[i].Trace.OnExecuted();
    WroteTrace = Now();
  });
  return result;
}

static IterationResult RunOnce(size_t threadNum) {
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

static void PrintResults(size_t threadNum,
                         std::vector<IterationResult> &results) {
  std::cout << "{" << std::endl;
  std::cout << "\"thread_num\": " << threadNum << "," << std::endl;
  std::cout << "\"tasks_num\": " << results.size() << "," << std::endl;
  std::cout << "\"results\": [" << std::endl;
  for (size_t iter = 0; iter != results.size(); ++iter) {
    auto &&res = results[iter].Tasks;
    std::unordered_map<ThreadId, std::vector<Tracing::TaskInfo>>
        resultPerThread;
    for (size_t i = 0; i < res.size(); ++i) {
      auto task = res[i];
      resultPerThread[task.ThreadIdx].emplace_back(task);
    }
    std::cout << "  {\n";
    std::cout << "    \"start\": " << results[iter].Start << ",\n";
    std::cout << "    \"end\": " << results[iter].End << ",\n";
    std::cout << "    \"tasks\": {\n";
    size_t total = 0;
    for (auto &&[id, tasks] : resultPerThread) {
      std::cout << "        \"" << id << "\": [";
      std::sort(tasks.begin(), tasks.end());
      for (size_t i = 0; i != tasks.size(); ++i) {
        auto task = tasks[i];
        std::cout << "{\"index\": " << task.TaskIdx << ", \"trace\": {\""
                  << "prev_trace\": " << task.Trace.PreviousTrace
                  << ", \"execution_start\": " << task.Trace.ExecutionEnd
                  << ", \"execution_end\": " << task.Trace.ExecutionEnd
                  << "}, \"cpu\": " << task.SchedCpu << "}"
                  << (i == tasks.size() - 1 ? "" : ", ");
      }
      std::cout << (++total == resultPerThread.size() ? "]" : "],")
                << std::endl;
    }
    std::cout << "    }" << std::endl;
    std::cout << (iter + 1 == results.size() ? "  }" : "  },") << std::endl;
  }
  std::cout << "]" << std::endl;
  std::cout << "}" << std::endl;
}

int main(int argc, char **argv) {
  auto threadNum = GetNumThreads();
  InitParallel(threadNum);
  RunOnce(threadNum); // just for warmup

  std::vector<IterationResult> results;
  for (size_t i = 0; i < 10; ++i) {
    results.push_back(RunOnce(threadNum));
  }
  PrintResults(threadNum, results);
  return 0;
}
