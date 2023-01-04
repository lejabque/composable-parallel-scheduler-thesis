cmake_minimum_required (VERSION 3.15)
option(ENABLE_TESTS "Enable test" OFF)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(composable-parallel-scheduler)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # use libstdc++ instead of libc++ for compatibility with google benchmark which is compiled with libstdc++
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-aligned-allocation")
endif()

message(STATUS "Compiler path: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

include(FetchContent)

if (ENABLE_TESTS)
    # gtest
    find_package(GTest QUIET)
    if(NOT GTEST_FOUND)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.12.0
    )
    FetchContent_MakeAvailable(googletest)
    endif()
endif()

# benchmark
find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
  FetchContent_Declare(googlebenchmark
          GIT_REPOSITORY https://github.com/google/benchmark.git
          GIT_TAG v1.7.0) # need master for benchmark::benchmark
  FetchContent_MakeAvailable(
                  googlebenchmark)
endif()

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


# build options
option(USE_SANITIZERS "Enable to build with undefined,leak and address sanitizers" OFF)


# benchmarks:
list(APPEND OMP_MODES OMP_STATIC
  OMP_DYNAMIC_MONOTONIC OMP_DYNAMIC_NONMONOTONIC
  OMP_GUIDED_MONOTONIC OMP_GUIDED_NONMONOTONIC
  OMP_RUNTIME_MONOTONIC OMP_RUNTIME_NONMONOTONIC)
list(APPEND TBB_MODES TBB_SIMPLE TBB_AUTO TBB_AFFINITY TBB_CONST_AFFINITY)
# TODO: add rapid start proof of concept for TBB

list(APPEND BENCHMARKS bench_spmv_balanced bench_spmv_unbalanced bench_reduce bench_scan)
foreach(bench IN LISTS BENCHMARKS)
    # serial
    add_executable(${bench}_SERIAL ${bench}.cpp)
    target_compile_definitions(${bench}_SERIAL PRIVATE SERIAL=1)
    target_link_libraries(${bench}_SERIAL benchmark::benchmark)
    # openmp
    foreach(mode IN LISTS OMP_MODES)
        add_executable(${bench}_${mode} ${bench}.cpp)
        target_compile_definitions(${bench}_${mode} PRIVATE OMP_MODE=${mode})
        target_link_libraries(${bench}_${mode} benchmark::benchmark OpenMP::OpenMP_CXX)
    endforeach()
    # tbb
    foreach(mode IN LISTS TBB_MODES)
        add_executable(${bench}_${mode} ${bench}.cpp)
        target_compile_definitions(${bench}_${mode} PRIVATE TBB_MODE=${mode})
        target_link_libraries(${bench}_${mode} benchmark::benchmark TBB::tbb)
    endforeach()
endforeach()

if (ENABLE_TESTS)
    add_subdirectory(tests)
endif()
