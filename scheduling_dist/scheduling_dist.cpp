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
  tracer.RunIteration(threadNum, [&](size_t i) {
    reported.fetch_add(1, std::memory_order_relaxed);
    // it's ok to block here because we want
    // to measure time of all threadNum threads
    while (reported.load(std::memory_order_relaxed) != threadNum) {
      CpuRelax();
    }
  });
}

static void RunWithSpin(size_t threadNum, Tracing::Tracer &tracer,
                        size_t tasksPerThread = 1) {
  uint64_t spinPerIter = 100'000'000 / tasksPerThread;
  auto tasksCount = threadNum * tasksPerThread;
  tracer.RunIteration(tasksCount, [&](size_t i) {
    // emulating work
    for (size_t i = 0; i < spinPerIter; ++i) {
      CpuRelax();
    }
  });
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

int main(int argc, char **argv) {
  auto threadNum = GetNumThreads();
  InitParallel(threadNum);

  Tracing::Tracer tracer;
  for (size_t i = 0; i < 10; ++i) {
    RunOnce(threadNum, tracer);
  }
  std::cout << tracer.ToJson(threadNum);
  return 0;
}
