#include <benchmark/benchmark.h>

#include "../include/parallel_for.h"
#include "spmv.h"
#include <iostream>

#define RELAX 1
#define ATOMIC 2
#define DISTRIBUTED_READ 3
#define THREADLOCAL 4

static void DoSetup(const benchmark::State &state) {
  InitParallel(GetNumThreads());
}

namespace {
template <typename T> struct Aligned {
  alignas(hardware_constructive_interference_size) T value{};
};

} // namespace

template <typename F> void RunParallelFor(size_t iters, size_t tasks, F &&f) {
  for (size_t iter = 0; iter != iters; ++iter) {
    ParallelFor(0, tasks, f);
  }
}

static void BM_Spin(benchmark::State &state) {
#if SPIN_PAYLOAD == RELAX
  for (auto _ : state) {
    RunParallelFor(state.range(2), state.range(0),
                   [cnt = state.range(1)](size_t i) {
                     for (size_t i = 0; i != cnt; ++i) {
                       CpuRelax();
                     }
                   });
  }
#elif SPIN_PAYLOAD == ATOMIC
  auto atomicPtr = std::make_unique<Aligned<std::atomic<int>>>();
  for (auto _ : state) {
    RunParallelFor(state.range(2), state.range(0),
                   [p = atomicPtr.get(), cnt = state.range(1)](size_t i) {
                     volatile int y = 0;
                     for (size_t i = 0; i != cnt; ++i) {
                       y = p->value.load();
                     }
                   });
  }
#elif SPIN_PAYLOAD == DISTRIBUTED_READ
  std::vector<Aligned<std::unique_ptr<Aligned<int>>>> pointers(state.range(0));
  for (auto &p : pointers) {
    p = {std::make_unique<Aligned<int>>()};
  }
  for (auto _ : state) {
    RunParallelFor(state.range(2), state.range(0),
                   [&pointers, cnt = state.range(1)](size_t i) {
                     volatile int y = 0;
                     auto p = pointers[i].value.get();
                     for (size_t i = 0; i != cnt; ++i) {
                       y = p->value;
                     }
                   });
  }
#elif SPIN_PAYLOAD == THREADLOCAL
  for (auto _ : state) {
    RunParallelFor(state.range(2), state.range(0),
                   [cnt = state.range(1)](size_t i) {
                     static thread_local int x = 0;
                     volatile int y = 0;
                     for (size_t i = 0; i != cnt; ++i) {
                       y = x;
                     }
                   });
  }
#else
  static_assert(false, "Unsupported mode");
#endif
}

static std::string GetSpinPayload() {
#if SPIN_PAYLOAD == RELAX
  return "RELAX";
#elif SPIN_PAYLOAD == ATOMIC
  return "ATOMIC";
#elif SPIN_PAYLOAD == DISTRIBUTED_READ
  return "DISTRIBUTED_READ";
#elif SPIN_PAYLOAD == THREADLOCAL
  return "THREADLOCAL";
#else
  static_assert(false, "Unsupported mode");
#endif
}

static constexpr size_t ScaleIterations(size_t count) {
  constexpr size_t movPerRelax = 32; // TODO: tune for arm
#if SPIN_PAYLOAD == RELAX
  return count;
#else
  return count * movPerRelax;
#endif
}

BENCHMARK(BM_Spin)
    ->Name(std::string("Spin_") + GetSpinPayload() + "_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->ArgNames({"tasks", "iters", "calls"})
    ->Args({1 << 10, ScaleIterations(1 << 10), 1})         // few small tasks
    ->Args({GetNumThreads(), ScaleIterations(1 << 20), 1}) // few big tasks
    ->Args({1 << 13, ScaleIterations(1 << 13), 1}) // something in between
    ->Args({1 << 16, ScaleIterations(1 << 10), 1}) // many small tasks
    ->Args({GetNumThreads(), 1, 1024}) // multiple small parallel for runs
    ->Args({1 << 20, 1, 1})            // as in the scan benchmarks
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
