cmake_minimum_required (VERSION 3.15)
project(composable-parallel-scheduler)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

# TODO: use clang instead of default compiler?
message(STATUS "Compiler path: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

include(FetchContent)

# gtest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.0
)

# benchmark
FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.7.0) # need master for benchmark::benchmark

FetchContent_MakeAvailable(
        googletest
        googlebenchmark)

find_package(OpenMP REQUIRED)

# link with TBB
FetchContent_Declare(
  onetbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG v2021.6.0
)
option(TBB_TEST OFF)

FetchContent_MakeAvailable(onetbb)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps=obj")


# build options
option(USE_SANITIZERS "Enable to build with undefined,leak and address sanitizers" OFF)

add_subdirectory(benchmarks)
