#pragma once
#include "util.h"

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

  TaskInfo(size_t taskIdx, Timestamp prevTrace)
      : Trace(prevTrace) {
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

} // namespace Tracing
