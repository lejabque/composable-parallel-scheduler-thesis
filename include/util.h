#pragma once

#include <string>

#define STR_(x) #x
#define STR(x) STR_(x)

inline std::string GetParallelMode() {
#ifdef SERIAL
  return "SERIAL";
#endif
#ifdef TBB_MODE
  return STR(TBB_MODE);
#endif
#ifdef OMP_MODE
  return STR(OMP_MODE);
#endif
}

#ifdef TBB_MODE
#include "oneapi/tbb/blocked_range.h"
#include <tbb/parallel_for.h>
#endif
#ifdef OMP_MODE
#include <omp.h>
#endif
#include <cstddef>
#include <thread>

inline int GetNumThreads() {
#ifdef TBB_MODE
  return ::tbb::info::
      default_concurrency(); // tbb::this_task_arena::max_concurrency();
#endif
#ifdef OMP_MODE
  return omp_get_max_threads();
#endif
}

inline int GetThreadIndex() {
#ifdef TBB_MODE
  return tbb::this_task_arena::current_thread_index();
#endif
#ifdef OMP_MODE
  return omp_get_thread_num();
#endif
}
