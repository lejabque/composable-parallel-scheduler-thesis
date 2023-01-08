#pragma once

#include "eigen_pool.h"
#include <iostream>
#include <string>

#define STR_(x) #x
#define STR(x) STR_(x)

inline std::string GetParallelMode() {
#if defined(SERIAL)
  return "SERIAL";
#elif defined(TBB_MODE)
  return STR(TBB_MODE);
#elif defined(OMP_MODE)
  return STR(OMP_MODE);
#elif defined(EIGEN_MODE)
  return "EIGEN_MODE";
#else
  static_assert(false, "Unsupported mode");
#endif
}

#ifdef TBB_MODE
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

#ifdef OMP_MODE
#include <omp.h>
#endif

#include <cstddef>
#include <thread>

inline int GetNumThreads() {
#if defined(TBB_MODE)
  return ::tbb::info::
      default_concurrency(); // tbb::this_task_arena::max_concurrency();
#elif defined(OMP_MODE)
  return omp_get_max_threads();
#elif defined(SERIAL)
  return 1;
#elif defined(EIGEN_MODE)
  return GetEigenThreadNum();
#else
  static_assert(false, "Unsupported mode");
#endif
}

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
