#include <benchmark/benchmark.h>

#include "../include/parallel_for.h"
#include "spmv.h"

using namespace SPMV;

static constexpr size_t MATRIX_SIZE = 1 << 16;

static void BM_SpmvBenchTriangle(benchmark::State &state) {
  InitParallel(GetNumThreads());
  auto A = GenSparseMatrix<double, SparseKind::TRIANGLE>(MATRIX_SIZE,
                                                         MATRIX_SIZE, 1e-3);
  auto x = GenVector<double>(MATRIX_SIZE);
  // allocate result only once
  std::vector<double> y(A.Dimensions.Rows);
  for (auto _ : state) {
    MultiplyMatrix(A, x, y);
  }
}

BENCHMARK(BM_SpmvBenchTriangle)
    ->Name("SpmvTriangle_" + GetParallelMode())
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
