cmake_minimum_required (VERSION 3.15)
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
# use libstdc++ instead of libc++ for compatibility with google benchmark which is compiled with libstdc++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")

project(composable-parallel-scheduler)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


# TODO: use clang instead of default compiler?
message(STATUS "Compiler path: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

include(FetchContent)

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

add_subdirectory(benchmarks)
