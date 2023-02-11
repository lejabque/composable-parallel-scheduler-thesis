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
FetchContent_Declare(
  onetbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG v2021.6.0
)
option(TBB_TEST OFF)
FetchContent_MakeAvailable(onetbb)


# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps=obj")


# build options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined,leak -fno-sanitize-recover=all")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address,undefined,leak")


list(APPEND OMP_MODES OMP_STATIC
  OMP_DYNAMIC_MONOTONIC OMP_DYNAMIC_NONMONOTONIC
  OMP_GUIDED_MONOTONIC OMP_GUIDED_NONMONOTONIC)
list(APPEND TBB_MODES TBB_SIMPLE TBB_AUTO TBB_AFFINITY TBB_CONST_AFFINITY)
if($ENV{ENABLE_RAPID})
  list(APPEND TBB_MODES TBB_RAPID)
  list(APPEND EIGEN_MODES EIGEN_RAPID)
endif()
list(APPEND EIGEN_MODES EIGEN_SIMPLE EIGEN_TIMESPAN EIGEN_STATIC)
list(APPEND MODES ${OMP_MODES} ${TBB_MODES} ${EIGEN_MODES})

function(add_target target exec mode)
    add_executable(${target} ${exec})
    if (mode MATCHES "^SERIAL")
      target_compile_definitions(${target} PRIVATE SERIAL=1)
    elseif (mode MATCHES "^OMP")
      target_compile_definitions(${target} PRIVATE OMP_MODE=${mode})
      target_link_libraries(${target} OpenMP::OpenMP_CXX)
    elseif (mode MATCHES "^TBB")
      target_compile_definitions(${target} PRIVATE TBB_MODE=${mode})
      target_link_libraries(${target} TBB::tbb)
    elseif (mode MATCHES "^EIGEN")
      target_compile_definitions(${target} PRIVATE EIGEN_MODE=${mode})
      if (mode STREQUAL "EIGEN_RAPID")
        target_link_libraries(${target} TBB::tbb)
      endif()
    endif()
endfunction()

option(ENABLE_BENCHMARKS "Enable benchmarks" ON)
if (ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

option(ENABLE_SCHEDULING "Enable scheduling distribution bench" ON)
if (ENABLE_SCHEDULING)
  add_subdirectory(scheduling_dist)
endif()
