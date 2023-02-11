# benchmark
find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
  FetchContent_Declare(googlebenchmark
          GIT_REPOSITORY https://github.com/google/benchmark.git
          GIT_TAG v1.7.0) # need master for benchmark::benchmark
  FetchContent_MakeAvailable(
                  googlebenchmark)
endif()

list(APPEND BENCHMARKS bench_spmv_balanced bench_spmv_hyperbolic bench_spmv_triangle bench_reduce bench_scan)
foreach(bench IN LISTS BENCHMARKS)
    foreach(mode IN LISTS MODES)
        set(target ${bench}_${mode})
        add_target(${target} ${bench}.cpp ${mode})
        target_link_libraries(${target} benchmark::benchmark)
    endforeach()
endforeach()

if (ENABLE_TESTS)
    add_subdirectory(tests)
endif()
