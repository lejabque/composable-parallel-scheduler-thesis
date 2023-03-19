#pragma once
#include "util.h"
#include <vector>

namespace Tracing {
struct TaskTrace {
  TaskTrace() = default;

  TaskTrace(Timestamp prevTrace)
      : PreviousTrace(prevTrace), ExecutionStart(Now()) {}

  void OnExecuted() { ExecutionEnd = Now(); }

  void OnWrite() { TraceWrite = Now(); }

  Timestamp PreviousTrace{};
  Timestamp Created{};
  Timestamp ExecutionStart{};
  Timestamp ExecutionEnd{};
  Timestamp TraceWrite{};
};

struct TaskInfo {
  TaskInfo() = default;

  TaskInfo(size_t taskIdx, Timestamp prevTrace) : Trace(prevTrace) {
    TaskIdx = taskIdx;
    ThreadIdx = GetThreadIndex();
    SchedCpu = sched_getcpu();
  }

  bool operator<(const TaskInfo &other) { return TaskIdx < other.TaskIdx; }

  size_t TaskIdx;
  int ThreadIdx;
  int SchedCpu;
  TaskTrace Trace;
};

struct IterationResult {
  IterationResult(size_t count) : Tasks{count}, End{} { Start = Now(); }

  void StartTask(size_t taskIdx) {
    Tasks[taskIdx] = TaskInfo(taskIdx, WroteTrace);
  }

  void EndTask(size_t taskIdx) {
    Tasks[taskIdx].Trace.OnExecuted();
    WroteTrace = Now();
  }

  static inline thread_local Timestamp WroteTrace = 0; // previous trace

  Timestamp Start;
  std::vector<Tracing::TaskInfo> Tasks;
  Timestamp End;
};

class Tracer {
public:
  Tracer() = default;

  void StartIteration(size_t tasksCount) {
    Iterations.emplace_back(tasksCount);
  }

  void EndIteration() { Iterations.back().End = Now(); }

  void StartTask(size_t taskIdx) { Iterations.back().StartTask(taskIdx); }

  void EndTask(size_t taskIdx) { Iterations.back().EndTask(taskIdx); }

  std::vector<IterationResult> GetIterations() { return Iterations; }

private:
  std::vector<IterationResult> Iterations;
};

} // namespace Tracing
