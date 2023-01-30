#pragma once
#include "eigen_pool.h"

#if defined(EIGEN_MODE) && EIGEN_MODE != EIGEN_RAPID

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
    for (size_t i = 1; i < threadsNum; ++i) { // don't pin main thread
      EigenPool.RunOnThread(
          [barrier, i, pinThread]() {
            pinThread(i);
            barrier->Notify();
            barrier->Wait();
          },
          i);
    }
    pinThread(0);
    barrier->Wait();
  }
};

#endif
