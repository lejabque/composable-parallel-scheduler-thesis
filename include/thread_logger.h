#pragma once
#include "spin_lock.h"
#include "util.h"
#include <mutex>
#include <thread>
#include <vector>

class ThreadLogger {
public:
  using ThreadId = int;

  ThreadLogger(size_t size) : ids(size) {}

  void Log(size_t taskNum) {
    std::lock_guard<SpinLock> guard{lock};
    ids[taskNum] = GetThreadIndex();
  }

  std::vector<int> GetIds() {
    std::lock_guard<SpinLock> guard{lock};
    return ids;
  }

private:
  SpinLock lock;
  std::vector<int> ids;
};
