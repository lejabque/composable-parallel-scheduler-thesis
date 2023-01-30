#pragma once
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
  return STR(EIGEN_MODE);
#else
  static_assert(false, "Unsupported mode");
#endif
}

#define OMP_STATIC 1
#define OMP_DYNAMIC_MONOTONIC 2
#define OMP_DYNAMIC_NONMONOTONIC 3
#define OMP_GUIDED_MONOTONIC 4
#define OMP_GUIDED_NONMONOTONIC 5
#define OMP_RUNTIME_MONOTONIC 6
#define OMP_RUNTIME_NONMONOTONIC 7

#define TBB_SIMPLE 1
#define TBB_AUTO 2
#define TBB_AFFINITY 3
#define TBB_CONST_AFFINITY 4
#define TBB_RAPID 5

#define EIGEN_SIMPLE 1
#define EIGEN_RAPID 2
#define EIGEN_TIMESPAN 3
#define EIGEN_STATIC 4

#ifdef TBB_MODE
#include <tbb/parallel_for.h>
#endif

#ifdef OMP_MODE
#include <omp.h>
#endif
