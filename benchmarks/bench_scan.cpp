#include "scan.h"
#include "spmv.h"
#include <benchmark/benchmark.h>

#include "parallel_for.h"

static constexpr size_t MIN_SIZE = 1 << 10;
static constexpr size_t MAX_SIZE = 1 << 27;
static void BM_ScanBench(benchmark::State &state) {
  size_t size = state.range(0);
  size_t size_pow = Scan::GetBlockPow(size);
  // allocate data and result once, reuse it for all iterations
  auto generatedData = SPMV::GenVector<double>(size);
  for (auto _ : state) {
    // TODO: is it makes sense to pause timing?
    state.PauseTiming();
    auto data = generatedData;
    state.ResumeTiming();
    Scan::Scan(size_pow, data);
  }
}

BENCHMARK(BM_ScanBench)
    ->Name("Scan_" + GetParallelMode())
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond)
    ->Range(MIN_SIZE, MAX_SIZE);

BENCHMARK_MAIN();
