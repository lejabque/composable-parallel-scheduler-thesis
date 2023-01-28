cmake_minimum_required (VERSION 3.15)
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(composable-parallel-scheduler)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)


message(STATUS "Compiler path: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

include(FetchContent)


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


# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps=obj")


add_executable(tbb_overhead_task_size main.cpp)
target_link_libraries(tbb_overhead_task_size TBB::tbb)
