#pragma once
#include "../contrib/eigen/unsupported/Eigen/CXX11/ThreadPool"
#include "num_threads.h"
#include "util.h"
#include <array>
#include <chrono>
#include <cstddef>
#include <utility>

namespace EigenPartitioner {

struct SplitData {
  static constexpr size_t K_SPLIT = 2; // todo: tune K and fix code
  size_t ThreadId;
  size_t Height;   // height of part in tree where each layer is divided by
                   // K_SPLIT parts
  size_t PartSize; // threads to execute this part
  size_t GrainSize = 1;
};

inline uint32_t GetLog2(uint32_t value) {
  static constexpr auto table =
      std::array{0, 9,  1,  10, 13, 21, 2,  29, 11, 14, 16, 18, 22, 25, 3, 30,
                 8, 12, 20, 28, 15, 17, 24, 7,  19, 27, 23, 6,  26, 5,  4, 31};

  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return table[(((value * 0x7c4acdd) >> 27) % 32)];
}

template <typename Scheduler, typename Func, bool DelayBalance,
          bool Initial = false>
struct Task {
  static constexpr uint64_t INIT_TIME = 100000; // todo: tune time

  Task(Scheduler &sched, size_t from, size_t to, Func func, SplitData split)
      : Sched_(sched), Start_(from), End_(to), Func_(std::move(func)),
        Split_(split) {}

  bool IsDivisible() const { return Start_ + Split_.GrainSize < End_; }

  void operator()() {
    if constexpr (Initial) {
      if (Split_.PartSize != 1 && IsDivisible()) {
        // take 1/PartSize of iterations for this thread
        size_t splitFrom =
            Start_ + (End_ - Start_ + Split_.PartSize - 1) / Split_.PartSize;
        size_t splitTo = End_;
        if (splitFrom < splitTo) {
          End_ = splitFrom;
          // divide to K_SPLIT (actually 2) parts but proportionally to number
          // of threads in each part
          size_t step =
              (splitTo - splitFrom + Split_.K_SPLIT - 1) / Split_.K_SPLIT;
          // todo: can we simplify this?
          size_t leftThreads =
              std::min(static_cast<size_t>((1 << (Split_.Height - 1)) - 1),
                       Split_.PartSize - (1 << (Split_.Height - 2)));
          size_t rightThreads = Split_.PartSize - leftThreads - 1;
          // split proportionally to threads in each part
          size_t split = splitFrom + (splitTo - splitFrom) * leftThreads /
                                         (Split_.PartSize - 1);
          if (splitFrom < split) { // TODO: do we need this check?
            Sched_.run_on_thread(
                Task<Scheduler, Func, DelayBalance, true>{
                    Sched_,
                    splitFrom,
                    split,
                    Func_,
                    {2 * Split_.ThreadId + 1, GetLog2(leftThreads) + 1,
                     leftThreads, Split_.GrainSize}},
                2 * Split_.ThreadId + 1);
          }
          if (split < splitTo) {
            Sched_.run_on_thread(
                Task<Scheduler, Func, DelayBalance, true>{
                    Sched_,
                    split,
                    splitTo,
                    Func_,
                    {2 * Split_.ThreadId + 2, GetLog2(rightThreads) + 1,
                     rightThreads, Split_.GrainSize}},
                2 * Split_.ThreadId + 2);
          }
        }
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
    }

    // make balancing tasks for remaining iterations
    while (IsDivisible()) {
      // TODO: divide not by 2, maybe proportionally or other way
      size_t mid = (Start_ + End_) / 2;
      // TODO: by default Eigen's Schedule push task to the current queue, maybe
      // better to push it into other thread's queue?
      // (work-stealing vs mail-boxing)
      Sched_.run(Task<Scheduler, Func, false>{
          Sched_, mid, End_, Func_, SplitData{.GrainSize = Split_.GrainSize}});
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
  SplitData Split_;
};

template <typename Sched, bool UseTimespan, typename F>
auto MakeInitialTask(Sched &sched, size_t from, size_t to, F &&func,
                     size_t threadCount) {
  return Task<Sched, F, UseTimespan, true>{
      sched, from, to, std::forward<F>(func),
      SplitData{0, GetLog2(threadCount) + 1, threadCount}};
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
      GetNumThreads());
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
