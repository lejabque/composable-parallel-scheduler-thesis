# benchmark
find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
  FetchContent_Declare(googlebenchmark
          GIT_REPOSITORY https://github.com/google/benchmark.git
          GIT_TAG v1.7.0) # need master for benchmark::benchmark
  FetchContent_MakeAvailable(
                  googlebenchmark)
endif()

list(APPEND BENCHMARKS bench_spmv_balanced bench_spmv_hyperbolic bench_spmv_triangle bench_reduce bench_scan bench_mmul bench_mtranspose)
foreach(bench IN LISTS BENCHMARKS)
    foreach(mode IN LISTS MODES)
        set(target ${bench}_${mode})
        add_target(${target} ${bench}.cpp ${mode})
        target_link_libraries(${target} benchmark::benchmark)
    endforeach()
endforeach()
add_target(bench_lb4omp bench_lb4omp.cpp OMP_RUNTIME)
target_link_libraries(bench_lb4omp benchmark::benchmark)

list(APPEND SPIN_PAYLOADS RELAX ATOMIC DISTRIBUTED_READ THREADLOCAL)
foreach(payload IN LISTS SPIN_PAYLOADS)
    foreach(mode IN LISTS MODES)
        string(TOLOWER ${payload} payload_lower)
        set(target bench_spin_${payload_lower}_${mode})
        add_target(${target} bench_spin.cpp ${mode})
        target_link_libraries(${target} benchmark::benchmark)
        target_compile_definitions(${target} PRIVATE SPIN_PAYLOAD=${payload})
    endforeach()
endforeach()

if (ENABLE_TESTS)
    add_subdirectory(tests)
endif()
