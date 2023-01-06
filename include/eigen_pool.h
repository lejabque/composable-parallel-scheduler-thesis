#pragma once
#ifdef EIGEN_MODE
#define EIGEN_USE_THREADS
#include "../contrib/eigen/unsupported/Eigen/CXX11/Tensor"
#include "../contrib/eigen/unsupported/Eigen/CXX11/TensorSymmetry"
#include "../contrib/eigen/unsupported/Eigen/CXX11/ThreadPool"

#include <omp.h>

inline auto EigenPool = Eigen::ThreadPool(omp_get_max_threads());
#endif
