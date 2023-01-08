#include <benchmark/benchmark.h>

#include "../include/parallel_for.h"
#include "spmv.h"
#include <iostream>

using namespace SPMV;

static void BM_SpmvBenchBalanced(benchmark::State &state) {
  InitParallel(GetNumThreads());
  auto A = GenSparseMatrix<double, SparseKind::BALANCED>(state.range(0),
                                                         state.range(1), 1e-3);
  auto x = GenVector<double>(state.range(1));
  std::cerr << "A: " << A.Dimensions.Rows << "x" << A.Dimensions.Columns
            << std::endl;
  // allocate result only once
  std::vector<double> y(A.Dimensions.Rows);
  for (auto _ : state) {
    MultiplyMatrix(A, x, y);
  }
}

static constexpr size_t MIN_SIZE = 1 << 13;
static constexpr size_t MAX_SIZE = 1 << 17;

BENCHMARK(BM_SpmvBenchBalanced)
    ->Name("SpmvBalanced_" + GetParallelMode())
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangePair(MIN_SIZE, MAX_SIZE, MIN_SIZE, MAX_SIZE);

BENCHMARK_MAIN();
