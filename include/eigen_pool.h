#pragma once
#ifdef EIGEN_MODE
#define EIGEN_USE_THREADS
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/unsupported/Eigen/CXX11/TensorSymmetry"
#include "../eigen/unsupported/Eigen/CXX11/ThreadPool"

#include <omp.h>

inline auto EigenPool = Eigen::ThreadPool(omp_get_max_threads());
#endif
