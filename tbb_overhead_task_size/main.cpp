#include <iostream>
#include <tbb/parallel_for.h>

using Timestamp = uint64_t;

struct Result {
  Timestamp Time;
  size_t ThreadId;
};

static Timestamp GetTimestampStart() {
#if defined(__x86_64__)
  Timestamp t;
  // https://github.com/google/highwayhash/blob/master/highwayhash/tsc_timer.h
  asm volatile("lfence\n\t"
               "rdtsc\n\t"
               "shl $32, %%rdx\n\t"
               "or %%rdx, %0\n\t"
               "lfence"
               : "=a"(t)
               :
               // "memory" avoids reordering. rdx = TSC >> 32.
               // "cc" = flags modified by SHL.
               : "rdx", "memory", "cc");
  return t;
#elif defined(__aarch64__)
  Timestamp val;
  asm volatile("mrs %0, cntvct_el0" : "=r"(val));
  return val;
#else
  static_assert(false, "Unsupported architecture");
#endif
}

static Timestamp GetTimestampEnd() {
#if defined(__x86_64__)
  Timestamp t;
  asm volatile("rdtscp\n\t"
               "shl $32, %%rdx\n\t"
               "or %%rdx, %0\n\t"
               "lfence"
               : "=a"(t)
               :
               // "memory" avoids reordering. rcx = TSC_AUX. rdx = TSC >> 32.
               // "cc" = flags modified by SHL.
               : "rcx", "rdx", "memory", "cc");
  return t;
#elif defined(__aarch64__)
  Timestamp val;
  asm volatile("mrs %0, cntvct_el0" : "=r"(val));
  return val;
#else
  static_assert(false, "Unsupported architecture");
#endif
}

inline void cpu_relax() {
#if defined(__x86_64__)
  asm volatile("pause\n" : : : "memory");
#elif defined(__aarch64__)
  asm volatile("yield\n" : : : "memory");
#else
#error
#endif
}

void run(size_t threadNum, size_t spinCount, std::vector<Result> &results) {
  static tbb::task_group_context context(
      tbb::task_group_context::bound,
      tbb::task_group_context::default_traits |
          tbb::task_group_context::concurrent_wait);
  const tbb::simple_partitioner part;
  auto start = GetTimestampStart();
  tbb::parallel_for(
      static_cast<size_t>(0), threadNum,
      [&](size_t idx) {
        results[idx].Time = GetTimestampEnd() - start;
        results[idx].ThreadId = tbb::this_task_arena::current_thread_index();
        // emulating work
        for (size_t i = 0; i < spinCount; ++i) {
          cpu_relax();
        }
      },
      part, context);
}

int main(int, char *argv[]) {
  constexpr size_t threadNum = 2;
  constexpr size_t iterCount = 100;
  size_t spinCount = std::stoul(argv[1]);

  std::vector<std::vector<Result>> results(iterCount,
                                           std::vector<Result>(threadNum));
  run(threadNum, spinCount, results[0]); // warmup
  for (size_t i = 0; i < iterCount; ++i) {
    run(threadNum, spinCount, results[i]);
  }
  // write json: iter_num -> thread_id -> time
  std::cout << "{" << std::endl;
  for (size_t i = 0; i < iterCount; ++i) {
    std::cout << "  \"" << i << "\": {" << std::endl;
    for (size_t j = 0; j < threadNum; ++j) {
      std::cout << "    \"" << results[i][j].ThreadId
                << "\": " << results[i][j].Time;
      if (j + 1 == threadNum) {
        std::cout << std::endl;
      } else {
        std::cout << "," << std::endl;
      }
    }
    if (i + 1 == iterCount) {
      std::cout << "  }" << std::endl;
    } else {
      std::cout << "  }," << std::endl;
    }
  }
  std::cout << "}" << std::endl;
  return 0;
}
