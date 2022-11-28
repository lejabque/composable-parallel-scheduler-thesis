#include "parallel_for.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <ratio>
#include <thread>
#include <vector>

namespace {
struct TimeReporter {
  // saves time of first report in current generation
  // we need generations to reset times in all threads after each benchmark
  // just incrementing generation
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

int main(int argc, char **argv) {
  auto start = std::chrono::high_resolution_clock::now();
  auto threadNum = GetNumThreads();
  if (argc > 1) {
    threadNum = std::stoi(argv[1]);
  }
  // TODO: configure number of tasks to be sure that all threads are used?
  auto tasksNum = threadNum * 100;
  auto totalBenchTime = std::chrono::duration<double>(60);
  auto sleepFor = totalBenchTime * threadNum / tasksNum;
  ParallelFor(0, tasksNum, [&](size_t i) {
    timeReporter.ReportTime(start);
    // sleep for emulating work
    std::this_thread::sleep_for(sleepFor);
  });
  auto times = TimeReporter::GetTimes();
  std::sort(times.begin(), times.end());
  std::cout << "{";
  std::cout << "\"thread_num\": " << threadNum << ", ";
  std::cout << "\"used_threads\": " << times.size() << ", ";
  std::cout << "\"start_times\": [";
  for (size_t i = 0; i < times.size(); i++) {
    std::cout << times[i].count();
    if (i != times.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]}" << std::endl;
  return 0;
}