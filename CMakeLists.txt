cmake_minimum_required (VERSION 3.15)
project(composable-parallel-scheduler)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

find_package(OpenMP REQUIRED)

# link with TBB
include(FetchContent)
FetchContent_Declare(
  onetbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG v2021.6.0
)
option(TBB_TEST OFF)

FetchContent_MakeAvailable(onetbb)


# build options
option(USE_SANITIZERS "Enable to build with undefined,leak and address sanitizers" OFF)

add_subdirectory(benchmarks)
