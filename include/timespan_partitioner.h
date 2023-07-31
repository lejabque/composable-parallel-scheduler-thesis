#pragma once
#include "intrusive_ptr.h"
#include "num_threads.h"
#include "util.h"
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace EigenPartitioner {

struct Range {
  size_t From;
  size_t To;

  size_t Size() { return To - From; }
};

struct SplitData {
  static constexpr size_t K_SPLIT = 2;
  Range Threads;
  size_t GrainSize = 1;
  size_t Depth = 0;
};

struct TaskNode : intrusive_ref_counter<TaskNode> {
  using NodePtr = IntrusivePtr<TaskNode>;

  TaskNode(NodePtr parent = nullptr) : Parent(std::move(parent)) {}

  void SpawnChild(size_t count = 1) {
    ChildWaitingSteal_.fetch_add(count, std::memory_order_relaxed);
  }

  void OnStolen() {
    Parent->ChildWaitingSteal_.fetch_sub(1, std::memory_order_relaxed);
  }

  bool AllStolen() {
    return ChildWaitingSteal_.load(std::memory_order_relaxed) == 0;
  }

  NodePtr Parent;

  std::atomic<size_t> ChildWaitingSteal_{0};
};

enum class Balance { OFF, SIMPLE, DELAYED };

enum class Initial { TRUE, FALSE };

enum class GrainSize { DEFAULT, AUTO };

template <typename Scheduler, typename Func, Balance balance,
          GrainSize grainSizeMode, Initial initial = Initial::FALSE>
struct Task {
  static inline const uint64_t INIT_TIME = [] {
  // should be calculated using timespan_tuner with EIGEN_SIMPLE
  // currently 0.99 percentile for maximums is used: 99% of iterations should
  // fit scheduling in timespan
#if defined(__x86_64__)
    if (GetNumThreads() == 48) {
      return 16500;
    }
    return 13500;
#elif defined(__aarch64__)
    return 1800;
#else
#error "Unsupported architecture"
#endif
  }();

  using StolenFlag = std::atomic<bool>;

  Task(Scheduler &sched, TaskNode::NodePtr node, size_t from, size_t to,
       Func func, SplitData split, ThreadId threadId)
      : Sched_(sched), CurrentNode_(std::move(node)), Current_(from), End_(to),
        Func_(std::move(func)), Split_(split), SupposedThread_(threadId) {}

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
        auto threadStep = otherThreads.Size() / parts;
        auto threadsMod = otherThreads.Size() % parts;
        auto dataStep = otherData.Size() / parts;
        auto dataMod = otherData.Size() % parts;
        for (size_t i = 0; i != parts; ++i) {
          auto threadSplit =
              std::min(otherThreads.To,
                       otherThreads.From + threadStep +
                           static_cast<size_t>((parts - 1 - i) < threadsMod));
          // if threads are divided equally, distribute one more task for first
          // parts of threads otherwise distribute one more task for last parts
          // of threads
          auto dataSplit = std::min(
              otherData.To,
              otherData.From + dataStep +
                  static_cast<size_t>((threadsMod == 0 ? i : (parts - 1 - i)) <
                                      dataMod));
          assert(otherData.From < dataSplit);
          assert(otherThreads.From < threadSplit);
          Sched_.run_on_thread(
              Task<Scheduler, Func, balance, grainSizeMode, Initial::TRUE>{
                  Sched_, new TaskNode(CurrentNode_), otherData.From, dataSplit,
                  Func_,
                  SplitData{.Threads = {otherThreads.From, threadSplit},
                            .GrainSize = Split_.GrainSize},
                  static_cast<ThreadId>(otherThreads.From)},
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
    if constexpr (initial == Initial::TRUE) {
      DistributeWork();
    }
    //  else if (GetThreadIndex() != SupposedThread_) {
    //   CurrentNode_->OnStolen();
    // }
    if constexpr (balance == Balance::DELAYED) {
      // at first we are executing job for INIT_TIME
      // and then create balancing task
      auto start = Now();
      while (Current_ < End_) {
        Execute();
        if (Now() - start > INIT_TIME) {
          break;
        }
        if constexpr (grainSizeMode == GrainSize::AUTO) {
          Split_.GrainSize++;
        }
      }
    }

    if constexpr (balance != Balance::OFF) {
      while (Current_ != End_ && IsDivisible()) {
        // make balancing tasks for remaining iterations
        // TODO: check stolen? maybe not each time?
        // if (CurrentNode_->AllStolen()) {
        // TODO: maybe we need to check "depth" - number of being stolen
        // times?
        size_t mid = Current_ + (End_ - Current_) / 2;
        // CurrentNode_->SpawnChild();
        // eigen's scheduler will push task to the current thread queue,
        // then some other thread can steal this
        Sched_.run(Task<Scheduler, Func, Balance::SIMPLE, GrainSize::DEFAULT>{
            Sched_, new TaskNode(CurrentNode_), mid, End_, Func_,
            SplitData{.GrainSize = Split_.GrainSize, .Depth = Split_.Depth + 1},
            GetThreadIndex()});
        End_ = mid;
      }
    }

    while (Current_ != End_) {
      Execute();
    }
    CurrentNode_.Reset();
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
  ThreadId SupposedThread_;

  IntrusivePtr<TaskNode> CurrentNode_;
};

template <typename Sched, Balance balance, GrainSize grainSizeMode, typename F>
auto MakeInitialTask(Sched &sched, TaskNode::NodePtr node, size_t from,
                     size_t to, F func, size_t threadCount,
                     size_t grainSize = 1) {
  return Task<Sched, F, balance, grainSizeMode, Initial::TRUE>{
      sched,
      std::move(node),
      from,
      to,
      std::move(func),
      SplitData{.Threads = {0, threadCount}, .GrainSize = grainSize},
      GetThreadIndex()};
}

template <typename Sched, Balance balance, GrainSize grainSizeMode, typename F>
void ParallelFor(size_t from, size_t to, F func) {
  Sched sched;
  // allocating only for top-level nodes
  TaskNode rootNode;
  IntrusivePtrAddRef(&rootNode); // avoid deletion
  Task<Sched, F, balance, grainSizeMode, Initial::TRUE> task{
      sched,
      IntrusivePtr<TaskNode>(&rootNode),
      from,
      to,
      std::move(func),
      SplitData{.Threads = {0, static_cast<size_t>(GetNumThreads())},
                .GrainSize = 1},
      GetThreadIndex()};
  task();
  sched.join_main_thread();
  while (IntrusivePtrLoadRef(&rootNode) != 1) {
    CpuRelax();
  }
}

template <typename Sched, GrainSize grainSizeMode, typename F>
void ParallelForTimespan(size_t from, size_t to, F func) {
  ParallelFor<Sched, Balance::DELAYED, grainSizeMode, F>(from, to,
                                                         std::move(func));
}

template <typename Sched, typename F>
void ParallelForSimple(size_t from, size_t to, F func) {
  ParallelFor<Sched, Balance::SIMPLE, GrainSize::DEFAULT, F>(from, to,
                                                             std::move(func));
}

template <typename Sched, typename F>
void ParallelForStatic(size_t from, size_t to, F func) {
  ParallelFor<Sched, Balance::OFF, GrainSize::DEFAULT, F>(from, to,
                                                          std::move(func));
}

} // namespace EigenPartitioner
