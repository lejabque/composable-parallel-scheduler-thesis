#pragma once

#include "parallel_for.h"
#include <cassert>
#include <vector>

namespace Scan {
inline size_t GetBlockPow(size_t size) {
  size_t block_pow = 0;
  while (size > 1) {
    size >>= 1;
    ++block_pow;
  }
  return block_pow;
}

// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// NB: now works only for vector with size = 2^k
template <typename T> void Scan(size_t size_pow, std::vector<T> &data) {
  auto size = data.size();
  assert(size == (1 << size_pow));
  // up-sweep phase
  for (size_t d = 0; d != size_pow; d++) {
    auto shift = (1 << (d + 1));
    auto limit = (size + shift - 1) / shift;
    ParallelFor(0, limit, [&](size_t i) {
      auto k = 0 + i * shift;
      if (k + shift - 1 < size) {
        data[k + shift - 1] += data[k + (shift >> 1) - 1];
      }
    });
  }
  data.back() = 0;
  // down-sweep phase
  for (int64_t d = size_pow; d >= 0; d--) {
    auto shift = (1 << (d + 1));
    auto limit = (size + shift - 1) / shift;
    ParallelFor(0, limit, [&](size_t i) {
      auto k = 0 + i * shift;
      auto t = data[k + (shift >> 1) - 1];
      if (k + shift - 1 < size) {
        data[k + (shift >> 1) - 1] = data[k + shift - 1];
        data[k + shift - 1] += t;
      }
    });
  }
}
} // namespace Scan
