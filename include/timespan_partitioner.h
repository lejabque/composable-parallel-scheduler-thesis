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
  size_t Depth = 0;
};

struct TaskNode {
  using NodePtr = std::shared_ptr<TaskNode>; // todo: intrusive ptr?

  TaskNode(NodePtr parent) : Parent(parent) {}

  void SpawnChild(size_t count = 1) {
    ChildWaitingSteal_.fetch_add(count, std::memory_order_relaxed);
  }

  void OnStolen() {
    Parent->ChildWaitingSteal_.fetch_sub(1, std::memory_order_relaxed);
  }

  bool AllStolen() {
    return !ChildWaitingSteal_.load(std::memory_order_relaxed);
  }

  static NodePtr MakeChildNode(NodePtr &parent) {
    parent->SpawnChild();
    return std::make_shared<TaskNode>(parent);
  }

  NodePtr Parent;

  std::atomic<size_t> ChildWaitingSteal_{0};
};

template <typename Scheduler, typename Func, bool DelayBalance,
          bool Initial = false>
struct Task {
  static constexpr uint64_t INIT_TIME = 100000; // todo: tune time for platforms
  using StolenFlag = std::atomic<bool>;

  Task(Scheduler &sched, TaskNode::NodePtr &parent, size_t from, size_t to,
       Func func, SplitData split)
      : Sched_(sched), CurrentNode_(TaskNode::MakeChildNode(parent)),
        Current_(from), End_(to), Func_(std::move(func)), Split_(split) {}

  bool IsDivisible() const { return Current_ + Split_.GrainSize < End_; }

  void DistributeWork() {
    if (Split_.Threads.Size() != 1 && IsDivisible()) {
      // take 1/parts of iterations for current thread
      Range otherData{Current_ + (End_ - Current_ + Split_.Threads.Size() - 1) /
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
                  Sched_, CurrentNode_, otherData.From, dataSplit, Func_,
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
  }

  void operator()() {
    if constexpr (Initial) {
      DistributeWork();
    } else {
      //  CurrentNode_->OnStolen();
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

    while (Current_ != End_) {
      // make balancing tasks for remaining iterations
      // TODO: check stolen? maybe not each time?
      if (IsDivisible() /*&& CurrentNode_->AllStolen()*/) {
        // TODO: maybe we need to check "depth" - number of being stolen
        // times?
        // TODO: divide not by 2, maybe proportionally or other way? maybe
        // create more than one task?
        size_t mid = (Current_ + End_) / 2;
        // eigen's scheduler will push task to the current thread queue,
        // then some other thread can steal this
        Sched_.run(Task<Scheduler, Func, DelayBalance>{
            Sched_, CurrentNode_, mid, End_, Func_,
            SplitData{.GrainSize = Split_.GrainSize,
                      .Depth = Split_.Depth + 1}});
        End_ = mid;
      } else {
        Execute();
      }
    }
    CurrentNode_.reset();
  }

private:
  void Execute() {
    Func_(Current_);
    ++Current_;
  }

  Scheduler &Sched_;
  size_t Current_;
  size_t End_;
  Func Func_;
  SplitData Split_;

  std::shared_ptr<TaskNode> CurrentNode_;
};

template <typename Sched, bool UseTimespan, typename F>
auto MakeInitialTask(Sched &sched, TaskNode::NodePtr &parent, size_t from,
                     size_t to, F func, size_t threadCount,
                     size_t grainSize = 1) {
  return Task<Sched, F, UseTimespan, true>{
      sched,
      parent,
      from,
      to,
      std::move(func),
      SplitData{.Threads = {0, threadCount}, .GrainSize = grainSize}};
}

template <typename Sched, bool UseTimespan, typename F>
void ParallelFor(size_t from, size_t to, F func, size_t grainSize = 1) {
  Sched sched;
  auto parentNode = std::make_shared<TaskNode>(nullptr);
  auto task = MakeInitialTask<Sched, UseTimespan>(
      sched, parentNode, from, to, std::move(func), GetNumThreads());
  task();
  sched.join_main_thread();
  while (!parentNode.unique()) {
    CpuRelax();
  }
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
