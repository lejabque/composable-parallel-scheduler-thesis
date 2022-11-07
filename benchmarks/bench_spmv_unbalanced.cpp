#include <benchmark/benchmark.h>

#include "parallel_for.h"
#include "spmv.h"

using namespace SPMV;

static void BM_SpmvBenchUnbalanced(benchmark::State &state) {
  auto A =
      GenSparseMatrix<SparseKind::UNBALANCED>(state.range(0), state.range(1),
                                              1e-4); // TODO: density?
  auto x = GenVector(state.range(1));
  // allocate result only once
  std::vector<double> y(A.Dimensions.Rows);
  for (auto _ : state) {
    ParallelFor(0, A.Dimensions.Rows,
                [&](size_t i) { y[i] = MultiplyRow(A, x, i); });
  }
}

static constexpr size_t MIN_SIZE = 1 << 13;
static constexpr size_t MAX_SIZE = 1 << 17;

BENCHMARK(BM_SpmvBenchUnbalanced)
    ->Name("SpmvUnbalanced_" + GetParallelMode())
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond)
    ->RangePair(MIN_SIZE, MAX_SIZE, MIN_SIZE, MAX_SIZE);

BENCHMARK_MAIN();
