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

#define OMP_STATIC_MONOTONIC 1
#define OMP_STATIC_NONMONOTONIC 2
#define OMP_DYNAMIC_MONOTONIC 3
#define OMP_DYNAMIC_NONMONOTONIC 4
#define OMP_GUIDED_MONOTONIC 5
#define OMP_GUIDED_NONMONOTONIC 6
#define OMP_RUNTIME_MONOTONIC 7
#define OMP_RUNTIME_NONMONOTONIC 8

#define TBB_SIMPLE 1
#define TBB_AUTO 2
#define TBB_AFFINITY 3
#define TBB_CONST_AFFINITY 4

// TODO: move out some initializations from body to avoid init overhead?

template <typename Func> void ParallelFor(size_t from, size_t to, Func &&func) {
#ifdef SERIAL
  for (size_t i = from; i < to; ++i) {
    func(i);
  }
#endif

#ifdef TBB_MODE
  static tbb::task_group_context context(
      tbb::task_group_context::bound,
      tbb::task_group_context::default_traits |
          tbb::task_group_context::concurrent_wait);
#if TBB_MODE == TBB_SIMPLE
  const tbb::simple_partitioner part;
#elif TBB_MODE == TBB_AUTO
  const tbb::auto_partitioner part;
#elif TBB_MODE == TBB_AFFINITY
  // "it is important that the same affinity_partitioner object be passed to
  // loop templates to be optimized for affinity" see
  // https://spec.oneapi.io/versions/latest/elements/oneTBB/source/algorithms/partitioners/affinity_partitioner.html
  static tbb::affinity_partitioner part;
#elif TBB_MODE == TBB_CONST_AFFINITY
  tbb::affinity_partitioner part;
#else
#error Wrong PARALLEL mode
#endif
  // TODO: grain size?
  tbb::parallel_for(
      tbb::blocked_range(from, to),
      [&](const tbb::blocked_range<size_t> &range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          func(i);
        }
      },
      part, context);
#endif

#ifdef OMP_MODE
#pragma omp parallel
#if OMP_MODE == OMP_STATIC_MONOTONIC
#pragma omp for nowait schedule(monotonic : static)
#elif OMP_MODE == OMP_STATIC_NONMONOTONIC
#pragma omp for nowait schedule(nonmonotonic : static)
#elif OMP_MODE == OMP_DYNAMIC_MONOTONIC
// TODO: chunk size?
#pragma omp for nowait schedule(monotonic : dynamic)
#elif OMP_MODE == OMP_DYNAMIC_NONMONOTONIC
#pragma omp for nowait schedule(nonmonotonic : dynamic)
#elif OMP_MODE == OMP_GUIDED_MONOTONIC
#pragma omp for nowait schedule(monotonic : guided)
#elif OMP_MODE == OMP_GUIDED_NONMONOTONIC
#pragma omp for nowait schedule(nonmonotonic : guided)
#elif OMP_MODE == OMP_RUNTIME_MONOTONIC
#pragma omp for nowait schedule(monotonic : runtime)
#elif OMP_MODE == OMP_RUNTIME_NONMONOTONIC
#pragma omp for nowait schedule(nonmonotonic : runtime)
#else
#error Wrong OMP_MODE mode
#endif
  for (size_t i = from; i < to; ++i) {
    func(i);
  }
#endif
}
