#include "../include/parallel_for.h"
#include <ctime>

static void RunOnce(size_t threadNum, std::vector<Timestamp> *times) {
#if defined(__x86_64__)
  asm volatile("mfence" ::: "memory");
#elif defined(__aarch64__)
  asm volatile(
      "DMB SY \n" /* Data Memory Barrier. Full runtime operation. */
      "DSB SY \n" /* Data Synchronization Barrier. Full runtime operation. */
      "ISB    \n" /* Instruction Synchronization Barrier. */
      ::
          : "memory");
#else
  static_assert(false, "Unsupported architecture");
#endif
  std::atomic<size_t> reported(0);
  auto start = Now();
  ParallelFor(0, threadNum, [&](size_t i) {
    auto now = Now();
    if (times) {
      (*times)[i] = now - start;
    }
    reported.fetch_add(1, std::memory_order_relaxed);
    // it's ok to block here because we want
    // to measure time of all threadNum threads
    while (reported.load(std::memory_order_relaxed) != threadNum) {
      CpuRelax();
    }
  });
}

static size_t PercentileIndex(double pc, size_t size) {
  return 0.5 + static_cast<double>(size - 1) * pc;
}

int main() {
  auto threadNum = GetNumThreads();
  InitParallel(threadNum);
  constexpr size_t ITERATIONS = 10000;

  for (size_t i = 0; i != 10; ++i) {
    RunOnce(threadNum, nullptr); // warmup
  }

  std::vector<std::vector<Timestamp>> results(
      ITERATIONS, std::vector<Timestamp>(threadNum));
  for (size_t i = 0; i != ITERATIONS; ++i) {
    RunOnce(threadNum, &results[i]);
  }
  std::vector<Timestamp> flat_results;
  std::vector<Timestamp> maximums;
  double sum = 0;
  double sum_max = 0;
  for (auto &&r : results) {
    auto max = r.front();
    for (auto &&v : r) {
      flat_results.push_back(v);
      sum += v;
      max = std::max(max, v);
    }
    maximums.push_back(max);
    sum_max += max;
  }
  std::sort(maximums.begin(), maximums.end());
  std::sort(flat_results.begin(), flat_results.end());
  std::cout
      << "==================================================================\n";
  std::cout << "Mode: " + GetParallelMode() + ", threads: " << threadNum
            << ", iterations: " << ITERATIONS << "\n";
  std::cout << "Average: " << sum / flat_results.size() << " (total), "
            << sum_max / maximums.size() << " (maximums) \n";
  std::cout << "Minimum: " << flat_results.front() << " (maximums) \n";
  std::cout << "Median: "
            << flat_results.at(PercentileIndex(0.5, flat_results.size()))
            << " (total), "
            << maximums.at(PercentileIndex(0.5, maximums.size()))
            << " (maximums) \n";
  std::cout << "0.95 percentile: "
            << flat_results.at(PercentileIndex(0.95, flat_results.size()))
            << " (total), "
            << maximums.at(PercentileIndex(0.95, maximums.size()))
            << " (maximums) \n";
  std::cout << "0.99 percentile: "
            << flat_results.at(PercentileIndex(0.99, flat_results.size()))
            << " (total), "
            << maximums.at(PercentileIndex(0.99, maximums.size()))
            << " (maximums) \n";
  std::cout << "Maximum: " << flat_results.back() << " (maximums) \n";
}
