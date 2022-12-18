#pragma once
#include "spin_lock.h"
#include <mutex>
#include <thread>
#include <vector>

class ThreadLogger {
public:
  using ThreadId = std::thread::id;
  ThreadLogger(size_t size) : ids(size) {}

  void Log(size_t taskNum) {
    std::lock_guard<SpinLock> guard{lock};
    ids[taskNum] = std::this_thread::get_id();
  }

  std::vector<ThreadId> GetIds() {
    std::lock_guard<SpinLock> guard{lock};
    return ids;
  }

private:
  SpinLock lock;
  std::vector<ThreadId> ids;
};
