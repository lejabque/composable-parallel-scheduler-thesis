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

static void RunWithBarrier(size_t threadNum, Tracing::Tracer &tracer) {
  std::atomic<size_t> reported(0);
  tracer.StartIteration(threadNum);
  ParallelFor(0, threadNum, [&](size_t i) {
    tracer.StartTask(i);
    reported.fetch_add(1, std::memory_order_relaxed);
    // it's ok to block here because we want
    // to measure time of all threadNum threads
    while (reported.load(std::memory_order_relaxed) != threadNum) {
      CpuRelax();
    }
    tracer.EndTask(i);
  });
  tracer.EndIteration();
}

static void RunWithSpin(size_t threadNum, Tracing::Tracer &tracer,
                        size_t tasksPerThread = 1) {
  uint64_t spinPerIter = 100'000'000 / tasksPerThread;
  auto tasksCount = threadNum * tasksPerThread;
  tracer.StartIteration(tasksCount);
  ParallelFor(0, tasksCount, [&](size_t i) {
    tracer.StartTask(i);
    // emulating work
    for (size_t i = 0; i < spinPerIter; ++i) {
      CpuRelax();
    }
    // TODO: fence?
    tracer.EndTask(i);
  });
  tracer.EndIteration();
}

static void RunOnce(size_t threadNum, Tracing::Tracer &tracer) {
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
  return RunWithBarrier(threadNum, tracer);
#elif SCHEDULING_MEASURE_MODE == SPIN
  return RunWithSpin(threadNum, tracer);
#elif SCHEDULING_MEASURE_MODE == MULTITASK
  return RunWithSpin(threadNum, tracer, 100);
#else
  static_assert(false, "Unsupported mode");
#endif
}

static void PrintResults(size_t threadNum,
                         const std::vector<Tracing::IterationResult> &results) {
  std::cout << "{" << std::endl;
  std::cout << "\"thread_num\": " << threadNum << "," << std::endl;
  std::cout << "\"tasks_num\": " << results.front().Tasks.size() << ","
            << std::endl;
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
                  << ", \"execution_start\": " << task.Trace.ExecutionStart
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

  Tracing::Tracer tracer;
  for (size_t i = 0; i < 10; ++i) {
    RunOnce(threadNum, tracer);
  }
  PrintResults(threadNum, tracer.GetIterations());
  return 0;
}
