#include "../include/benchmarks/scan.h"
#include "../include/benchmarks/spmv.h"
#include <benchmark/benchmark.h>

#include "../include/parallel_for.h"

static constexpr size_t SIZE_POW = 24;

static void DoSetup(const benchmark::State &state) {
  InitParallel(GetNumThreads());
}

static void BM_ScanBench(benchmark::State &state) {
  static auto data = SPMV::GenVector<double>(1 << SIZE_POW);
  benchmark::DoNotOptimize(data);
  for (auto _ : state) {
    Scan::Scan(state.range(0), data);
    benchmark::ClobberMemory();
  }
}


BENCHMARK(BM_ScanBench)
    ->Name("Scan_Latency_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->ArgName("SizePow")
    ->DenseRange(10, SIZE_POW)
    ->Unit(benchmark::kMicrosecond);


BENCHMARK(BM_ScanBench)
    ->Name("Scan_Throughput_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->ArgName("SizePow")
    ->DenseRange(12, SIZE_POW)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(9);


BENCHMARK_MAIN();

