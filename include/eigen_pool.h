#pragma once
#ifdef EIGEN_MODE
#define EIGEN_USE_THREADS
#include "../contrib/eigen/unsupported/Eigen/CXX11/Tensor"
#include "../contrib/eigen/unsupported/Eigen/CXX11/TensorSymmetry"
#include "../contrib/eigen/unsupported/Eigen/CXX11/ThreadPool"

#include <omp.h>
// todo: remove omp usage
size_t GetEigenThreadNum() {
#if EIGEN_MODE == EIGEN_SIMPLE
  return omp_get_max_threads();
#elif EIGEN_MODE == EIGEN_RAPID
  return omp_get_max_threads() - 1; // 1 for main
#else
  static_assert(false, "Wrong EIGEN_MODE mode");
#endif
}

inline auto EigenPool = Eigen::ThreadPool(GetEigenThreadNum());

class EigenPoolWrapper {
public:
  template <typename F> void run(F &&f) {
    EigenPool.Schedule(std::forward<F>(f));
  }

  void wait() {
    // TODO: implement
  }
};
#endif
