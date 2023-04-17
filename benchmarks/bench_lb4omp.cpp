#include "omp.h"
#include <iostream>

int main() {
  size_t threads = omp_get_max_threads();
  std::cout << "Max threads: " << threads << std::endl;

#pragma omp parallel for schedule(runtime)
  for (size_t i = 0; i != 100000000; ++i) {
    auto thread_id = omp_get_thread_num();
    if (thread_id != 0) {
      std::cout << "Current thread: " + std::to_string(thread_id) + "\n";
    }
  }
}
