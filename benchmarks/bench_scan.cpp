// TODO: implement parallel scan like prefix sum, e.g.
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://moderngpu.github.io/scan.html
#include "spmv.h"
#include <benchmark/benchmark.h>

#include "parallel_for.h"

static constexpr size_t MIN_SIZE = 1 << 10;
static constexpr size_t MAX_SIZE = 1 << 27;
// TODO: custom block size?
static constexpr size_t BLOCK_SIZE = 1 << 10;

size_t GetBlockPow(size_t size) {
  size_t block_pow = 0;
  while (size > BLOCK_SIZE) {
    size >>= 1;
    block_pow++;
  }
  return block_pow;
}

static void DoScan(size_t size_pow, std::vector<double> &data) {
  auto size = data.size();
  // up-sweep phase
  for (size_t d = 0; d != size_pow; d++) {
    auto shift = (1 << (d + 1));
    auto limit = (size + shift - 1) / shift;
    ParallelFor(0, limit, [&](size_t i) {
      auto k = 0 + i * shift;
      data[k + shift - 1] += data[k + (shift >> 1) - 1];
    });
  }
  data.back() = 0;
  // down-sweep phase
  for (int64_t d = size_pow; d >= 0; d--) {
    auto shift = (1 << (d + 1));
    auto limit = (size + shift - 1) / shift;
    ParallelFor(0, limit, [&](size_t i) {
      auto k = 0 + i * shift;
      auto t = data[k + (shift >> 1) - 1];
      data[k + (shift >> 1) - 1] = data[k + shift - 1];
      data[k + shift - 1] += t;
    });
  }
}

static void BM_ScanBench(benchmark::State &state) {
  size_t size = state.range(0);
  size_t size_pow = GetBlockPow(size);
  // allocate data and result once, reuse it for all iterations
  auto generatedData = SPMV::GenVector(size);
  for (auto _ : state) {
    // TODO: is it makes sense to pause timing?
    state.PauseTiming();
    auto data = generatedData;
    state.ResumeTiming();
    DoScan(size_pow, data);
  }
}

BENCHMARK(BM_ScanBench)
    ->Name("Scan_" + GetParallelMode())
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond)
    ->Range(MIN_SIZE, MAX_SIZE);

BENCHMARK_MAIN();
