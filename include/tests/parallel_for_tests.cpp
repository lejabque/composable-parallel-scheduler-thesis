
#include "../parallel_for.h"
#include <atomic>
#include <gtest/gtest.h>

TEST(ParallelFor, Basic) {
  std::atomic<int> sum(0);
  ParallelFor(0, 100, [&](int i) { sum += i; });
  EXPECT_EQ(4950, sum);
}

TEST(ParallelFor, ConcurrentCalls) {
    std::atomic<int64_t> sum(0);
    auto maxThreads = GetNumThreads();
    std::vector<std::thread> threads;
    for (int i = 0; i < maxThreads; ++i) {
        threads.emplace_back([&]() {
            ParallelFor(0, 1024*1024, [&](int i) { sum += 1; });
        });
    }
    for (auto &t : threads) {
        t.join();
    }
    EXPECT_EQ(1024*1024*maxThreads, sum);
}

TEST(ParallelFor, MultipleCalls) {
  std::atomic<int> sum(0);
  ParallelFor(0, 100, [&](int i) { sum += i; });
  ParallelFor(0, 100, [&](int i) { sum += i; });
  EXPECT_EQ(9900, sum);
}

TEST(ParallelFor, AllThreadsUsed) {
  std::atomic<int> current(0);
  auto maxThreads = GetNumThreads();
  ParallelFor(0, maxThreads, [&](int i) {
    current++;
    while (current != maxThreads) {
      CpuRelax();
    }
  });
  EXPECT_EQ(maxThreads, current);
}

TEST(ParallelFor, Recursive) {
  std::atomic<int> sum(0);
  auto maxThreads = GetNumThreads();
  ParallelFor(0, maxThreads, [&](int i) {
    ParallelFor(0, maxThreads, [&](int j) { sum++; });
  });
  EXPECT_EQ(maxThreads * maxThreads, sum);
}

#if EIGEN_MODE == EIGEN_SIMPLE
TEST(ParallelFor, Workstealing) {
  // 1 task should blocked until all other tasks are done
  std::atomic<int> sum(0);
  auto maxThreads = GetNumThreads();
  ParallelFor(0, maxThreads * 16, [&](int i) {
    if (i == 0) {
      while (sum != maxThreads * 16 - 1) {
        CpuRelax();
      }
    } else {
      sum++;
    }
  });
}
#endif