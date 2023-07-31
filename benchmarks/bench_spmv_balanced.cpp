#include <benchmark/benchmark.h>

#include "../include/benchmarks/spmv.h"
#include "../include/parallel_for.h"
#include <iostream>
#include <unordered_map>

using namespace SPMV;

static void DoSetup(const benchmark::State &state) {
  InitParallel(GetNumThreads());
}

static constexpr auto width =
    std::array<size_t, 6>{1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16, 1 << 17};

static auto cachedMatrix = [] {
  std::unordered_map<size_t, SparseMatrixCSR<double>> res;
  for (auto &&w : width) {
    std::cout << "generating for " << w << std::endl;
    res[w] =
        GenSparseMatrix<double, SparseKind::BALANCED>(MATRIX_SIZE, w + (GetNumThreads() << 2) + 3, DENSITY);
    benchmark::DoNotOptimize(res[w]);
  }
  return res;
}();

static auto x = GenVector<double>(*std::prev(width.end()) + (GetNumThreads() << 2) + 3);
static std::vector<double> y(MATRIX_SIZE);

static void BM_SpmvBenchBalanced(benchmark::State &state) {
  benchmark::DoNotOptimize(x);
  benchmark::DoNotOptimize(y);
  
  auto &A = cachedMatrix.at(state.range(0));
  for (auto _ : state) {
    MultiplyMatrix(A, x, y);
    benchmark::ClobberMemory();
  }
}


BENCHMARK(BM_SpmvBenchBalanced)
    ->Name("SpmvBalanced_Latency_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->ArgName("width")
    ->RangeMultiplier(2)
    ->Range(*width.begin(), *std::prev(width.end()))
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_SpmvBenchBalanced)
    ->Name("SpmvBalanced_Throughput_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->ArgName("width")
    ->RangeMultiplier(2)
    ->Range(*width.begin(), *std::prev(width.end()))
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(9);

BENCHMARK_MAIN();

