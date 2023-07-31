#include "../include/benchmarks/spmv.h"
#include <benchmark/benchmark.h>

#include "../include/parallel_for.h"

static const size_t MAX_SIZE = (GetNumThreads() << 19) + (GetNumThreads() << 3) + 3;
// static constexpr size_t BLOCK_SIZE = 1 << 14;
// static constexpr size_t blocks = (MAX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

static void DoSetup(const benchmark::State &state) {
  InitParallel(GetNumThreads());
}

void __attribute__((noinline,noipa)) reduceImpl(std::vector<double> &data, size_t blocks, size_t blockSize) {
  ParallelFor(0, blocks, [&](size_t i) {
    static thread_local double res = 0;
    benchmark::DoNotOptimize(res);
    double sum = 0;
    auto start = i * blockSize;
    auto end = std::min(start + blockSize, MAX_SIZE);
    for (size_t j = start; j < end; ++j) {
      sum += data[j];
    }
    res += sum;
  });
}

static void BM_ReduceBench(benchmark::State &state) {
  static auto data = SPMV::GenVector<double>(MAX_SIZE);
  benchmark::DoNotOptimize(data);
  auto blockSize = state.range(0) + GetNumThreads() + 3;
  auto blocks = (MAX_SIZE + blockSize - 1) / blockSize;
  for (auto _ : state) {
    reduceImpl(data, blocks, blockSize);
    benchmark::ClobberMemory();
  }
}


BENCHMARK(BM_ReduceBench)
    ->Name("Reduce_Latency_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->ArgName("blocksize")
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 19)
    ->Unit(benchmark::kMicrosecond);


BENCHMARK(BM_ReduceBench)
    ->Name("Reduce_Throughput_" + GetParallelMode())
    ->Setup(DoSetup)
    ->UseRealTime()
    ->MeasureProcessCPUTime()
    ->ArgName("blocksize")
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 19)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(9);


BENCHMARK_MAIN();


