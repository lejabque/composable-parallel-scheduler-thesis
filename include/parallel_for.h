#pragma once

#include "eigen_pool.h"
#include "modes.h"
#include "util.h"

#if TBB_MODE == TBB_RAPID
#include "rapid_start.h"
inline Harness::RapidStart RapidGroup;
#endif

#if EIGEN_MODE == EIGEN_RAPID
#include "rapid_start.h"
inline Harness::RapidStart<EigenPoolWrapper> RapidGroup;
#endif

inline void InitParallel(size_t threadsNum) {
#if TBB_MODE == TBB_RAPID || EIGEN_MODE == EIGEN_RAPID
  RapidGroup.init(threadsNum);
#endif
}

#ifdef EIGEN_MODE
// TODO: move to eigen header
template <typename F>
inline void EigenParallelFor(size_t from, size_t to, F &&func) {
#if EIGEN_MODE == EIGEN_SIMPLE
  size_t blocks = GetEigenThreadsNum();
  size_t blockSize = (to - from + blocks - 1) / blocks;
  Eigen::Barrier barrier(blocks - 1);
  for (size_t i = 0; i < blocks - 1; ++i) {
    size_t start = from + i * blockSize;
    size_t end = std::min(start + blockSize, to);
    EigenPool.ScheduleWithHint(
        [func, start, end, &barrier]() {
          for (size_t i = start; i < end; ++i) {
            func(i);
          }
          barrier.Notify();
        },
        i, i + 1);
  }
  // main thread
  for (size_t i = from + (blocks - 1) * blockSize; i < to; ++i) {
    func(i);
  }
  barrier.Wait();
#elif EIGEN_MODE == EIGEN_RAPID
  RapidGroup.parallel_ranges(from, to, [&func](auto from, auto to, auto part) {
    for (size_t i = from; i != to; ++i) {
      func(i);
    }
  });
#else
  static_assert(false, "Wrong EIGEN_MODE mode");
#endif
}
#endif

// TODO: move out some initializations from body to avoid init overhead?
template <typename Func> void ParallelFor(size_t from, size_t to, Func &&func) {
#if defined(SERIAL)
  for (size_t i = from; i < to; ++i) {
    func(i);
  }
#elif defined(TBB_MODE)
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
#elif TBB_MODE == TBB_RAPID
  // no partitioner
#else
  static_assert(false, "Wrong TBB_MODE mode");
#endif
  // TODO: grain size?
#if TBB_MODE == TBB_RAPID
  RapidGroup.parallel_ranges(from, to, [&](auto from, auto to, auto part) {
    for (size_t i = from; i != to; ++i) {
      func(i);
    }
  });
#else
  tbb::parallel_for(
      tbb::blocked_range(from, to),
      [&](const tbb::blocked_range<size_t> &range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          func(i);
        }
      },
      part, context);
#endif
#elif defined(OMP_MODE)
#pragma omp parallel
#if OMP_MODE == OMP_STATIC
#pragma omp for nowait schedule(static)
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
  static_assert(false, "Wrong OMP_MODE mode");
#endif
  for (size_t i = from; i < to; ++i) {
    func(i);
  }
#elif defined(EIGEN_MODE)
  EigenParallelFor(from, to, func);
#else
  static_assert(false, "Wrong mode");
#endif
}
