cmake_minimum_required (VERSION 3.15)
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(composable-parallel-scheduler)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


message(STATUS "Compiler path: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

include(FetchContent)

find_package(OpenMP REQUIRED)
# todo: test that openmp is used from llvm build

# link with TBB
find_package(TBB QUIET)
if(NOT TBB_FOUND)
  FetchContent_Declare(
    onetbb
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG v2021.6.0
  )
  option(TBB_TEST OFF)
  FetchContent_MakeAvailable(onetbb)
endif()


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps=obj")

# benchmarks:
list(APPEND OMP_MODES OMP_STATIC OMP_DYNAMIC OMP_GUIDED OMP_RUNTIME)
list(APPEND TBB_MODES TBB_SIMPLE TBB_AUTO TBB_AFFINITY TBB_CONST_AFFINITY)

# openmp
foreach(mode IN LISTS OMP_MODES)
  add_executable(bench_scheduling_${mode}_${scheduling_measure_mode} bench_scheduling.cpp)
  target_compile_definitions(bench_scheduling_${mode}_${scheduling_measure_mode} PRIVATE OMP_MODE=${mode} SCHEDULING_MEASURE_MODE=${scheduling_measure_mode})
  target_link_libraries(bench_scheduling_${mode}_${scheduling_measure_mode} OpenMP::OpenMP_CXX)
endforeach()
# tbb
foreach(mode IN LISTS TBB_MODES)
  add_executable(bench_scheduling_${mode}_${scheduling_measure_mode} bench_scheduling.cpp)
  target_compile_definitions(bench_scheduling_${mode}_${scheduling_measure_mode} PRIVATE TBB_MODE=${mode} SCHEDULING_MEASURE_MODE=${scheduling_measure_mode})
  target_link_libraries(bench_scheduling_${mode}_${scheduling_measure_mode} TBB::tbb)
endforeach()

