#pragma once
#include "../contrib/eigen/unsupported/Eigen/CXX11/ThreadPool"
#include "util.h"
#include <chrono>
#include <cstddef>
#include <utility>

namespace EigenPartitioner {
template <typename Scheduler, typename Func, bool DelayBalance = true>
struct Task {
  static constexpr uint64_t INIT_TIME = 100000; // todo: tune time

  // TODO: tune grain size
  Task(Scheduler &sched, size_t from, size_t to, Func func,
       size_t grainSize = 1)
      : Sched_(sched), Start_(from), End_(to), Func_(std::move(func)),
        GrainSize_(grainSize) {}

  bool IsDivisible() const { return End_ > Start_ + GrainSize_; }

  void operator()() {
    if constexpr (DelayBalance) {
      // at first we are executing job for INIT_TIME
      // and then create balancing task
      auto start = Now();
      while (Start_ < End_) {
        Func_(Start_);
        ++Start_;
        if (Now() - start > INIT_TIME) {
          // TODO: call Now() less often?
          break;
        }
      }
      // std::cerr << "Start: " << Start_ << " End: " << End_ << std::endl;
    }

    // make balancing tasks for remaining iterations
    while (IsDivisible()) {
      size_t mid = (Start_ + End_) / 2;
      // TODO: by default Eigen's Schedule push task to the current queue, maybe
      // better to push it into other thread's queue?
      // (work-stealing vs mail-boxing)
      Sched_.run(Task<Scheduler, Func, false>{Sched_, mid, End_, Func_});
      End_ = mid;
    }
    for (; Start_ < End_; ++Start_) {
      Func_(Start_);
    }
  }

private:
  Scheduler &Sched_;
  size_t Start_;
  size_t End_;
  Func Func_;
  size_t GrainSize_;
};

template <typename Sched, bool UseTimespan, typename F>
auto MakeTask(Sched &sched, size_t from, size_t to, F &&func) {
  return Task<Sched, F, UseTimespan>{sched, from, to, std::forward<F>(func)};
}

template <typename Sched, bool UseTimespan, typename F>
void ParallelFor(size_t from, size_t to, F &&func) {
  size_t blocks = GetNumThreads();
  size_t blockSize = (to - from + blocks - 1) / blocks;
  Sched sched;
  Eigen::Barrier barrier(to - from);
  for (size_t i = 0; i < blocks; ++i) {
    size_t start = from + i * blockSize;
    size_t end = std::min(start + blockSize, to);
    auto threadId = i % GetNumThreads();
    sched.run_on_thread(
        MakeTask<Sched, UseTimespan>(sched, start, end,
                                     [&func, &barrier](size_t i) {
                                       func(i);
                                       barrier.Notify();
                                     }),
        threadId);
  }
  sched.join_main_thread();
  barrier.Wait();
}

template <typename Sched, typename F>
void ParallelForTimespan(size_t from, size_t to, F &&func) {
  ParallelFor<Sched, true, F>(from, to, std::forward<F>(func));
}

template <typename Sched, typename F>
void ParallelForSimple(size_t from, size_t to, F &&func) {
  ParallelFor<Sched, false, F>(from, to, std::forward<F>(func));
}

} // namespace EigenPartitioner
