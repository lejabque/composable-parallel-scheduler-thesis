list(APPEND OMP_MODES OMP_STATIC OMP_DYNAMIC OMP_GUIDED OMP_RUNTIME)
list(APPEND TBB_MODES TBB_STATIC TBB_SIMPLE TBB_AUTO TBB_AFFINITY TBB_CONST_AFFINITY)
# TODO: add rapid start proof of concept for TBB

list(APPEND BENCHMARKS bench_spmv_balanced bench_spmv_unbalanced bench_reduce)

foreach(bench IN LISTS BENCHMARKS)
    # serial
    add_executable(${bench}_serial ${bench}.cpp spmv.cpp)
    target_compile_definitions(${bench}_serial PRIVATE SERIAL=1)
    target_link_libraries(${bench}_serial benchmark::benchmark)
    # openmp
    foreach(mode IN LISTS OMP_MODES)
        add_executable(${bench}_${mode} ${bench}.cpp spmv.cpp)
        target_compile_definitions(${bench}_${mode} PRIVATE OMP_MODE=${mode})
        target_link_libraries(${bench}_${mode} benchmark::benchmark OpenMP::OpenMP_CXX)
    endforeach()
    # tbb
    foreach(mode IN LISTS TBB_MODES)
        add_executable(${bench}_${mode} ${bench}.cpp spmv.cpp)
        target_compile_definitions(${bench}_${mode} PRIVATE TBB_MODE=${mode})
        target_link_libraries(${bench}_${mode} benchmark::benchmark TBB::tbb)
    endforeach()
endforeach()
