#include "parallel_for.h"
#include <benchmark/benchmark.h>
#include <mutex>

static constexpr size_t MIN_SIZE = 1;
static constexpr size_t MAX_SIZE = 1 << 10;

static size_t totalGen = 1;
static thread_local size_t reportedGen = 0;

static void BM_Scheduling(benchmark::State &state) {
  std::mutex m;
  for (auto _ : state) {
    std::vector<double> times;
    auto start = std::chrono::high_resolution_clock::now();
    ParallelFor(0, state.range(0), [&](size_t i) {
      if (reportedGen < totalGen) {
        auto scheduled = std::chrono::high_resolution_clock::now();
        auto schedulingTime =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                scheduled - start);
        std::lock_guard<std::mutex> lock(m);
        times.push_back(schedulingTime.count());
        reportedGen = totalGen;
      }
    });
    state.SetIterationTime(*std::min_element(times.begin(), times.end()));
    totalGen++;
  }
}

BENCHMARK(BM_Scheduling)
    ->Name("Scheduling_" + GetParallelMode())
    ->UseManualTime()
    ->Range(MIN_SIZE, MAX_SIZE);

BENCHMARK_MAIN();
