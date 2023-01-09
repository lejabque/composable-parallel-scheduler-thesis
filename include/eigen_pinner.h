#pragma once
#include "eigen_pool.h"

#if EIGEN_MODE == EIGEN_SIMPLE || EIGEN_MODE == EIGEN_TIMESPAN

struct EigenPinner {
  EigenPinner(size_t threadsNum) {
    auto pinThread = [&threadsNum](size_t i) {
      cpu_set_t *mask;
      mask = CPU_ALLOC(threadsNum);
      auto mask_size = CPU_ALLOC_SIZE(threadsNum);
      CPU_ZERO_S(mask_size, mask);
      CPU_SET_S(i, mask_size, mask);
      if (sched_setaffinity(0, mask_size, mask)) {
        std::cerr << "Error in sched_setaffinity" << std::endl;
      }
      CPU_FREE(mask);
    };
    // use ptr because we want to wait for all threads in other threads
    auto barrier = std::make_shared<Eigen::Barrier>(threadsNum - 1);
#if EIGEN_MODE == EIGEN_SIMPLE
    for (size_t i = 0; i < threadsNum - 1; ++i)
#elif EIGEN_MODE == EIGEN_TIMESPAN
    for (size_t i = 1; i < threadsNum; ++i)
#endif
    {
      EigenPool.ScheduleWithHint(
          [barrier, i, pinThread]() {
            pinThread(i);
            barrier->Notify();
            barrier->Wait();
          },
          i, i + 1);
    }

#if EIGEN_MODE == EIGEN_SIMPLE
    pinThread(threadsNum - 1);
#elif EIGEN_MODE == EIGEN_TIMESPAN
    pinThread(0);
#endif
    barrier->Wait();
  }
};

#endif
