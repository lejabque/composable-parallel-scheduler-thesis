#pragma once
#include "eigen_pool.h"
#include "modes.h"
#include "num_threads.h"
#include <iostream>
#include <string>

#include <cstddef>
#include <thread>

inline int GetThreadIndex() {
#if defined(TBB_MODE)
  return tbb::this_task_arena::current_thread_index();
#elif defined(OMP_MODE)
  return omp_get_thread_num();
#elif defined(SERIAL)
  return 0;
#elif defined(EIGEN_MODE)
  return EigenPool.CurrentThreadId();
#else
  static_assert(false, "Unsupported mode");
#endif
}

using Timestamp = uint64_t;

inline Timestamp Now() {
#if defined(__x86_64__)
  return __rdtsc();
#elif defined(__aarch64__)
  Timestamp val;
  asm volatile("mrs %0, cntvct_el0" : "=r"(val));
  return val;
#else
  static_assert(false, "Unsupported architecture");
#endif
}
