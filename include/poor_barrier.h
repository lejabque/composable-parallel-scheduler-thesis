#pragma once

#include <atomic>
#include <cstddef>

#include "util.h"

struct SpinBarrier {
  SpinBarrier(size_t count) : remain_(count) {}

  void Notify(size_t count = 1) {
    remain_.fetch_sub(count, std::memory_order_relaxed);
  }

  void Wait() {
    while (remain_.load(std::memory_order_relaxed)) {
      CpuRelax();
    }
  }

  std::atomic<size_t> remain_;
};
