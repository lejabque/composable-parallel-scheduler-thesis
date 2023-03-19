
#include "../include/parallel_for.h"

#include "../include/trace.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

static void RunSpin(size_t threadNum, Tracing::Tracer &tracer) {
  tracer.RunIteration(threadNum, [&](size_t i) {
    // emulating small work
    for (size_t i = 0; i < 1; ++i) {
      CpuRelax();
    }
  });
}

int main(int argc, char **argv) {
  auto threadNum = GetNumThreads();
  InitParallel(threadNum);

  Tracing::Tracer tracer;
  for (size_t i = 0; i < 1024; ++i) {
    RunSpin(threadNum, tracer);
  }
  std::cout << tracer.ToJson(threadNum);
  return 0;
}
