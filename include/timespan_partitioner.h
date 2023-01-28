#pragma once
#include "../contrib/eigen/unsupported/Eigen/CXX11/ThreadPool"
#include "num_threads.h"
#include "util.h"
#include <chrono>
#include <cstddef>
#include <utility>

namespace EigenPartitioner {
template <typename Scheduler, typename Func, bool DelayBalance,
          bool Initial = false>
struct Task {
  static constexpr uint64_t INIT_TIME = 100000; // todo: tune time

  // TODO: tune grain size
  Task(Scheduler &sched, size_t from, size_t to, Func func, size_t threadId,
       size_t threadCount, size_t grainSize = 1)
      : Sched_(sched), Start_(from), End_(to), Func_(std::move(func)),
        ThreadId_(threadId), ThreadCount_(threadCount), GrainSize_(grainSize) {}

  bool IsDivisible() const { return End_ > Start_ + GrainSize_; }

  void operator()() {
    if constexpr (Initial) {
      if (ThreadCount_ != 1) {
        // proportional division
        // todo: optimize it and divide in K blocks to distribute them across
        // threads as tree?
        size_t split = Start_ + (End_ - Start_) / ThreadCount_;
        Sched_.run_on_thread(
            Task<Scheduler, Func, DelayBalance, true>{
                Sched_, split, End_, Func_, ThreadId_ + 1, ThreadCount_ - 1},
            ThreadId_ + 1);
        End_ = split;
      }
    }
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
      // TODO: divide not by 2, maybe proportionally or other way
      size_t mid = (Start_ + End_) / 2;
      // TODO: by default Eigen's Schedule push task to the current queue, maybe
      // better to push it into other thread's queue?
      // (work-stealing vs mail-boxing)
      Sched_.run(Task<Scheduler, Func, false>{Sched_, mid, End_, Func_,
                                              ThreadId_, ThreadCount_});
      End_ = mid;
    }
    for (; Start_ < End_; ++Start_) {
      Func_(Start_);
    }
  }

  size_t InitThreadId() const { return ThreadId_; }

private:
  Scheduler &Sched_;
  size_t Start_;
  size_t End_;
  Func Func_;
  size_t ThreadId_;
  size_t ThreadCount_;
  size_t GrainSize_;
};

template <typename Sched, bool UseTimespan, typename F>
auto MakeInitialTask(Sched &sched, size_t from, size_t to, F &&func,
                     size_t threadId, size_t threadCount) {
  return Task<Sched, F, UseTimespan, true>{
      sched, from, to, std::forward<F>(func), threadId, threadCount};
}

template <typename Sched, bool UseTimespan, typename F>
void ParallelFor(size_t from, size_t to, F &&func) {
  Sched sched;
  Eigen::Barrier barrier(to - from);
  auto task = MakeInitialTask<Sched, UseTimespan>(
      sched, from, to,
      [&func, &barrier](size_t i) {
        func(i);
        barrier.Notify();
      },
      0, GetNumThreads());
  task();
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
