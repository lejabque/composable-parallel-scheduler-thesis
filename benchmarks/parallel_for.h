#ifdef TBB_MODE
#include "oneapi/tbb/blocked_range.h"
#include <tbb/parallel_for.h>
#endif
#include <cstddef>
#include <thread>

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

inline int GetNumThreads() {
#if HAVE_TBB
  return ::tbb::info::
      default_concurrency(); // tbb::this_task_arena::max_concurrency();
#elif HAVE_OMP
  return omp_get_max_threads();
#else
  return std::thread::hardware_concurrency();
#endif
}

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
  // TODO: customize num of threads
  size_t g = GetNumThreads();
#pragma omp parallel
#if OMP_MODE == OMP_STATIC
#pragma omp for nowait schedule(static)
#elif OMP_MODE == OMP_DYNAMIC
#pragma omp for nowait schedule(dynamic, g)
#elif OMP_MODE == OMP_GUIDED
#pragma omp for nowait schedule(guided, g)
#elif OMP_MODE == OMP_RUNTIME
#pragma omp for nowait schedule(runtime)
#else
#error Wrong OMP_MODE mode
#endif
  for (size_t i = from; i < to; ++i) {
    func(i);
  }
#endif
}
