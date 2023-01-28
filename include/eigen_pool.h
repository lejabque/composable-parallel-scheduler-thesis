#pragma once
#include "modes.h"
#include "num_threads.h"

#ifdef EIGEN_MODE

#define EIGEN_USE_THREADS
#include "../contrib/eigen/unsupported/Eigen/CXX11/Tensor"
#include "../contrib/eigen/unsupported/Eigen/CXX11/TensorSymmetry"
#include "../contrib/eigen/unsupported/Eigen/CXX11/ThreadPool"

#if EIGEN_MODE == EIGEN_TIMESPAN || EIGEN_MODE == EIGEN_SIMPLE
inline auto EigenPool = Eigen::ThreadPool(GetNumThreads(), true,
                                          true); // todo: disable spinning?
#else
inline auto EigenPool =
    Eigen::ThreadPool(GetNumThreads() - 1); // 1 for main thread
#endif

class EigenPoolWrapper {
public:
  template <typename F> void run(F &&f) {
    EigenPool.Schedule(std::forward<F>(f));
  }

  template <typename F> void run_on_thread(F &&f, size_t hint) {
    EigenPool.RunOnThread(std::forward<F>(f), hint);
  }

  void join_main_thread() { EigenPool.JoinMainThread(); }

  void wait() {
    // TODO: implement
  }
};

#endif
