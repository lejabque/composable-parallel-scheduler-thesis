#pragma once

#include <iostream>
#include <sched.h>
#include <tbb/tbb.h>

class PinningObserver : public tbb::task_scheduler_observer {
public:
  PinningObserver() { observe(true); }

  void on_scheduler_entry(bool is_worker) {
    cpu_set_t *mask;
    auto number_of_slots = tbb::this_task_arena::max_concurrency();
    mask = CPU_ALLOC(number_of_slots);
    auto mask_size = CPU_ALLOC_SIZE(number_of_slots);
    auto slot_number = tbb::this_task_arena::current_thread_index();
    CPU_ZERO_S(mask_size, mask);
    CPU_SET_S(slot_number, mask_size, mask);
    if (sched_setaffinity(0, mask_size, mask)) {
      std::cerr << "Error in sched_setaffinity" << std::endl;
    }
    CPU_FREE(mask);
  }

  ~PinningObserver() { observe(false); }
};
