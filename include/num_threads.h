#pragma once

#include "modes.h"
#include <cstddef>
#include <string>
#include <thread>

inline int GetNumThreads() {
  // cache result to avoid calling getenv on every call
  static int threads = []() {
    if (const char *envThreads = std::getenv("BENCH_NUM_THREADS")) {
      return std::stoi(envThreads);
    }
    // left just for compatibility
    if (const char *envThreads = std::getenv("BENCH_MAX_THREADS")) {
      return std::stoi(envThreads);
    }
#if defined(TBB_MODE)
    return tbb::info::
        default_concurrency(); // tbb::this_task_arena::max_concurrency();
#elif defined(OMP_MODE)
    return omp_get_max_threads();
#elif defined(SERIAL)
    return 1;
#elif defined(EIGEN_MODE)
    return std::thread::hardware_concurrency();
#else
    static_assert(false, "Unsupported mode");
#endif
    return 1;
  }();
  return threads;
}
