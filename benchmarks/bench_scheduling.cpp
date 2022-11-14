#include "parallel_for.h"
#include <benchmark/benchmark.h>
#include <mutex>
#include <ratio>

static constexpr size_t MIN_SIZE = 1;
static constexpr size_t MAX_SIZE = 1 << 10;

namespace {
struct TimeReporter {
  // saves time of first report in current generation
  void ReportTime(std::chrono::system_clock::time_point before) {
    if (reportedGen < totalGen) {
      auto now = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(now - before);
      std::lock_guard<std::mutex> lock(mutex);
      times.push_back(duration);
      reportedGen = totalGen;
    }
  }

  static std::vector<std::chrono::nanoseconds> GetTimes() {
    std::lock_guard<std::mutex> lock(mutex);
    return times;
  }

  static void Reset() {
    std::lock_guard<std::mutex> lock(mutex);
    times.clear();
    totalGen++;
  }

  static inline size_t totalGen = 1;

private:
  size_t reportedGen = 0;

  // common for all threads
  static inline std::mutex mutex;
  static inline std::vector<std::chrono::nanoseconds> times; // nanoseconds
};
} // namespace

static thread_local TimeReporter timeReporter;

static void BM_Scheduling(benchmark::State &state) {
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    ParallelFor(0, state.range(0),
                [&](size_t i) { timeReporter.ReportTime(start); });
    TimeReporter::totalGen++;
    auto times = TimeReporter::GetTimes();
    std::sort(times.begin(), times.end());
    state.SetIterationTime(times.front().count() * 1e-9);
    TimeReporter::Reset();
  }
}

BENCHMARK(BM_Scheduling)
    ->Name("Scheduling_" + GetParallelMode())
    ->UseManualTime()
    ->Range(MIN_SIZE, MAX_SIZE);

BENCHMARK_MAIN();
