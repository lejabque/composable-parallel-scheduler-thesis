#pragma once
#include "../contrib/eigen/unsupported/Eigen/CXX11/ThreadPool"
#include "num_threads.h"
#include "poor_barrier.h"
#include "util.h"
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <utility>

namespace EigenPartitioner {

struct Range {
  size_t From;
  size_t To;

  size_t Size() { return To - From; }
};

struct SplitData {
  static constexpr size_t K_SPLIT = 2; // todo: tune K
  Range Threads;
  size_t GrainSize = 1;
};

inline size_t CalcStep(size_t from, size_t to, size_t chunksCount) {
  return (to - from + chunksCount - 1) / chunksCount;
}

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
  static constexpr uint64_t INIT_TIME = 100000; // todo: tune time for platforms
  using StolenFlag = std::atomic<bool>;

  Task(Scheduler &sched, SpinBarrier &barrier, size_t from, size_t to,
       Func func, SplitData split)
      : Sched_(sched), Barrier_(barrier), Current_(from), End_(to),
        Func_(std::move(func)), Split_(split) {}

  Task(Scheduler &sched, SpinBarrier &barrier, size_t from, size_t to,
       Func func, SplitData split, std::shared_ptr<StolenFlag> stolen)
      : Sched_(sched), Barrier_(barrier), Current_(from), End_(to),
        Func_(std::move(func)), Split_(split), Stolen_(stolen) {}

  bool IsDivisible() const { return Current_ + Split_.GrainSize < End_; }

  void operator()() {
    if constexpr (Initial) {
      if (Split_.Threads.Size() != 1 && IsDivisible()) {
        // take 1/parts of iterations for current thread
        Range otherData{Current_ +
                            (End_ - Current_ + Split_.Threads.Size() - 1) /
                                Split_.Threads.Size(),
                        End_};
        if (otherData.From < otherData.To) {
          End_ = otherData.From;
          Range otherThreads{Split_.Threads.From + 1, Split_.Threads.To};
          size_t parts = std::min(std::min(Split_.K_SPLIT, otherThreads.Size()),
                                  otherData.Size());
          auto threadsSize = otherThreads.Size();
          auto threadStep = threadsSize / parts;
          auto increareThreadStepFor = threadsSize % parts;
          auto dataSize = otherData.Size(); // TODO: unify code with threads
          auto dataStep = dataSize / parts;
          auto increaseDataStepFor = dataSize % parts;
          for (size_t i = 0; i != parts; ++i) {
            auto threadSplit =
                std::min(otherThreads.To, otherThreads.From + threadStep +
                                              (i < increareThreadStepFor));
            auto dataSplit =
                std::min(otherData.To,
                         otherData.From + dataStep + (i < increaseDataStepFor));
            assert(otherData.From < dataSplit);
            assert(otherThreads.From < threadSplit);
            Sched_.run_on_thread(
                Task<Scheduler, Func, DelayBalance, true>{
                    Sched_, Barrier_, otherData.From, dataSplit, Func_,
                    SplitData{.Threads = {otherThreads.From, threadSplit},
                              .GrainSize = Split_.GrainSize}},
                otherThreads.From);
            otherThreads.From = threadSplit;
            otherData.From = dataSplit;
          }
          assert(otherData.From == otherData.To);
          assert(otherThreads.From == otherThreads.To ||
                 parts < Split_.K_SPLIT &&
                     otherThreads.From + (Split_.K_SPLIT - parts) ==
                         otherThreads.To);
        }
      }
    } else if (Stolen_) {
      Stolen_->store(true, std::memory_order_relaxed);
      Stolen_.reset();
    }
    if constexpr (DelayBalance) {
      // at first we are executing job for INIT_TIME
      // and then create balancing task
      auto start = Now();
      while (Current_ < End_) {
        Execute();
        // todo: call this less?
        if (Now() - start > INIT_TIME) {
          break;
        }
      }
    }

    std::shared_ptr<StolenFlag> stolen;
    size_t itersToBalance = 0;
    while (Current_ < End_) {
      // make balancing tasks for remaining iterations
      if (itersToBalance == 0 && IsDivisible()) {
        // TODO: check less times?
        itersToBalance = 128;
        if (!stolen || stolen->load(std::memory_order_relaxed)) {
          if (!stolen) {
            std::make_shared<std::atomic<bool>>(false);
          } else {
            stolen->store(false, std::memory_order_relaxed);
          }
          // TODO: maybe we need to check "depth" - number of being stolen
          // times?
          // TODO: divide not by 2, maybe proportionally or other way? maybe
          // create more than one task?
          size_t mid = (Current_ + End_) / 2;
          // eigen's scheduler will push task to the current thread queue,
          // then some other thread can steal this
          Sched_.run(Task<Scheduler, Func, false>{
              Sched_, Barrier_, mid, End_, Func_,
              SplitData{.GrainSize = Split_.GrainSize}, stolen});
          End_ = mid;
        }
      } else {
        itersToBalance--;
      }
      Execute();
    }

    Barrier_.Notify(Executed_);
  }

private:
  void Execute() {
    Func_(Current_);
    ++Executed_;
    ++Current_;
  }

  Scheduler &Sched_;
  SpinBarrier &Barrier_;
  size_t Current_;
  size_t End_;
  size_t Executed_{0};
  Func Func_;
  SplitData Split_;
  std::shared_ptr<StolenFlag> Stolen_;
};

template <typename Sched, bool UseTimespan, typename F>
auto MakeInitialTask(Sched &sched, SpinBarrier &barrier, size_t from, size_t to,
                     F func, size_t threadCount, size_t grainSize = 1) {
  return Task<Sched, F, UseTimespan, true>{
      sched,
      barrier,
      from,
      to,
      std::move(func),
      SplitData{.Threads = {0, threadCount}, .GrainSize = grainSize}};
}

template <typename Sched, bool UseTimespan, typename F>
void ParallelFor(size_t from, size_t to, F func, size_t grainSize = 1) {
  Sched sched;
  SpinBarrier barrier(to - from);
  auto task = MakeInitialTask<Sched, UseTimespan>(
      sched, barrier, from, to, std::move(func), GetNumThreads());
  task();
  sched.join_main_thread();
  barrier.Wait();
}

template <typename Sched, typename F>
void ParallelForTimespan(size_t from, size_t to, F func, size_t grainSize = 1) {
  ParallelFor<Sched, true, F>(from, to, std::move(func), grainSize);
}

template <typename Sched, typename F>
void ParallelForSimple(size_t from, size_t to, F func, size_t grainSize = 1) {
  ParallelFor<Sched, false, F>(from, to, std::move(func), grainSize);
}

template <typename Sched, typename F>
void ParallelForStatic(size_t from, size_t to, F &&func) {
  Sched sched;
  auto blocks = GetNumThreads();
  auto blockSize = (to - from + blocks - 1) / blocks;
  SpinBarrier barrier(blocks - 1);
  for (size_t i = 1; i < blocks; ++i) {
    size_t start = from + blockSize * i;
    size_t end = std::min(start + blockSize, to);
    sched.run_on_thread(
        [&func, &barrier, start, end]() {
          for (size_t i = start; i < end; ++i) {
            func(i);
          }
          barrier.Notify();
        },
        i);
  }
  for (size_t i = from; i < std::min(from + blockSize, to); ++i) {
    func(i);
  }
  sched.join_main_thread();
  barrier.Wait();
}

} // namespace EigenPartitioner
