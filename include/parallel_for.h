#pragma once

#include "eigen_pool.h"
#include "util.h"

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

#if TBB_MODE == TBB_RAPID
#include "rapid_start.h"
inline Harness::RapidStart g_rs;
#endif
inline void InitParallel(size_t threadsNum) {
#if TBB_MODE == TBB_RAPID
  g_rs.init(threadsNum);
#endif
}

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
  g_rs.parallel_ranges(from, to, [&](auto from, auto to, auto part) {
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
  // auto blocks = GetNumThreads();
  // auto size = to - from;
  // auto blockSize = (size + blocks - 1) / blocks;
  // Eigen::Barrier barrier(blocks - 1);
  // auto executeRange = [&func, blockSize, from, to](size_t idx) {
  //   auto start = from + idx * blockSize;
  //   auto end = std::min(start + blockSize, to);
  //   for (size_t i = start; i < end; ++i) {
  //     func(i);
  //   }
  // };
  // for (size_t i = 0; i < blocks - 1; ++i) {
  //   EigenPool.Schedule([i, executeRange, &barrier]() {
  //     executeRange(i);
  //     barrier.Notify();
  //   });
  // }
  // executeRange(blocks - 1);
  Eigen::Barrier barrier(to - from);
  for (size_t i = from; i < to; ++i) {
    EigenPool.Schedule([func, i, &barrier]() {
      func(i);
      barrier.Notify();
    });
  }
  // todo: use main thread?
  barrier.Wait();
#else
  static_assert(false, "Wrong mode");
#endif
}
