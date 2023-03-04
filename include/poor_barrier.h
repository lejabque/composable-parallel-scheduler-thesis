#pragma once

#include <atomic>
#include <cstddef>

#include "util.h"

struct SpinBarrier {
  SpinBarrier(size_t count) : remain_(count) {}

  void Notify() { remain_.fetch_sub(1, std::memory_order_relaxed); }

  void Add(size_t count) {
    remain_.fetch_add(count, std::memory_order_relaxed);
  }

  void Wait() {
    while (remain_.load(std::memory_order_relaxed)) {
      CpuRelax();
    }
  }

  std::atomic<size_t> remain_;
};
