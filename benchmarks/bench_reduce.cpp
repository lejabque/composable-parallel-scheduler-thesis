#include "spmv.h"
#include <benchmark/benchmark.h>

#include "../include/parallel_for.h"

static constexpr size_t MAX_SIZE = 1 << 24;
static constexpr size_t BLOCK_SIZE = 1 << 14;
static constexpr size_t blocks = (MAX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

static void DoSetup(const benchmark::State &state) {
  InitParallel(GetNumThreads());
}

// cache data for all iterations
static auto data = SPMV::GenVector<double>(MAX_SIZE);

static void BM_ReduceBench(benchmark::State &state) {
  for (auto _ : state) {
    ParallelFor(0, blocks, [&](size_t i) {
      static thread_local double res = 0;
      double sum = 0;
      auto start = i * BLOCK_SIZE;
      auto end = std::min(start, MAX_SIZE - BLOCK_SIZE) + BLOCK_SIZE;
      for (size_t j = start; j < end; ++j) {
        sum += data[j];
      }
      res += sum;
    });
  }
}

BENCHMARK(BM_ReduceBench)
    ->Name("Reduce_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
