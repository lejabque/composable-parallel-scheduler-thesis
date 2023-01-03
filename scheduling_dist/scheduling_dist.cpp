#include "../include/parallel_for.h"
#include "../include/thread_logger.h"

#include <chrono>
#include <iostream>
#include <unordered_map>

#define SLEEP 1
#define BARRIER 2
#define RUNNING 3

static std::vector<ThreadLogger::ThreadId> RunOnce(size_t threadNum,
                                                   size_t tasksNum) {
  ThreadLogger logger(tasksNum);
  ParallelFor(0, tasksNum, [&](size_t i) {
    logger.Log(i);
    // spin 10ms without sleep
    auto spinStart = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - spinStart <
           std::chrono::milliseconds(10)) {
    }
  });
  return logger.GetIds();
}

static void
PrintIds(size_t threadNum,
         const std::vector<std::vector<ThreadLogger::ThreadId>> &results) {
  std::vector<std::unordered_map<ThreadLogger::ThreadId, std::vector<size_t>>>
      counters;
  for (auto &&res : results) {
    std::unordered_map<ThreadLogger::ThreadId, std::vector<size_t>> map;
    for (size_t i = 0; i < res.size(); ++i) {
      map[res[i]].push_back(i);
    }
    counters.push_back(std::move(map));
  }
  std::cout << "{" << std::endl;
  std::cout << "\"thread_num\": " << threadNum << "," << std::endl;
  std::cout << "\"tasks_num\": " << results[0].size() << "," << std::endl;
  std::cout << "\"results\": [" << std::endl;
  for (size_t i = 0; i < counters.size(); ++i) {
    std::cout << "  {" << std::endl;
    auto &&counter = counters[i];
    size_t total = 0;
    for (auto &&[threadid, tasks] : counter) {
      std::cout << "  \"" << threadid << "\": [";
      for (size_t i = 0; i < tasks.size(); ++i) {
        std::cout << tasks[i];
        if (i + 1 != tasks.size()) {
          std::cout << ", ";
        }
      }
      std::cout << "]";
      if (++total != counter.size()) {
        std::cout << ",";
      }
      std::cout << std::endl;
    }
    if (i + 1 != counters.size()) {
      std::cout << "  }," << std::endl;
    } else {
      std::cout << "  }" << std::endl;
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "}" << std::endl;
}

int main(int argc, char **argv) {
  auto threadNum = GetNumThreads();
  if (argc > 1) {
    threadNum = std::stoi(argv[1]);
  }
  RunOnce(threadNum, threadNum); // just for warmup

  std::vector<std::vector<ThreadLogger::ThreadId>> results;
  for (size_t i = 0; i < 10; ++i) {
    results.push_back(RunOnce(threadNum, 10 * threadNum));
  }
  PrintIds(threadNum, results);
  return 0;
}
