#pragma once
#ifdef EIGEN_MODE
#define EIGEN_USE_THREADS
#include "../contrib/eigen/unsupported/Eigen/CXX11/Tensor"
#include "../contrib/eigen/unsupported/Eigen/CXX11/TensorSymmetry"
#include "../contrib/eigen/unsupported/Eigen/CXX11/ThreadPool"

inline size_t GetEigenPoolThreadNum() {
#if EIGEN_MODE == EIGEN_SIMPLE
  return std::thread::hardware_concurrency();
#elif EIGEN_MODE == EIGEN_RAPID
  return std::thread::hardware_concurrency() - 1; // 1 for main
#else
  static_assert(false, "Wrong EIGEN_MODE mode");
#endif
}

inline auto EigenPool = Eigen::ThreadPool(GetEigenPoolThreadNum());

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
