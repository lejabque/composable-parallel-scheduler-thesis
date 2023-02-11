cmake_minimum_required (VERSION 3.15)
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(composable-parallel-scheduler)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp -fopenmp-version=50")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-aligned-allocation")
endif()


message(STATUS "Compiler path: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

include(FetchContent)

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
option(USE_SANITIZERS "Enable to build with undefined,leak and address sanitizers" OFF)


# benchmarks:
list(APPEND OMP_MODES OMP_STATIC
    OMP_DYNAMIC_MONOTONIC OMP_DYNAMIC_NONMONOTONIC
    OMP_GUIDED_MONOTONIC OMP_GUIDED_NONMONOTONIC)
list(APPEND TBB_MODES TBB_SIMPLE TBB_AUTO TBB_AFFINITY TBB_CONST_AFFINITY)
if($ENV{ENABLE_RAPID})
  list(APPEND TBB_MODES TBB_RAPID)
  list(APPEND EIGEN_MODES EIGEN_RAPID)
endif()
list(APPEND EIGEN_MODES EIGEN_SIMPLE EIGEN_TIMESPAN EIGEN_STATIC)
list(APPEND SCHEDULING_MEASURE_MODES SPIN BARRIER MULTITASK)

foreach(scheduling_measure_mode IN LISTS SCHEDULING_MEASURE_MODES)
  # openmp
  foreach(mode IN LISTS OMP_MODES)
      add_executable(scheduling_dist_${mode}_${scheduling_measure_mode} scheduling_dist.cpp)
      target_compile_definitions(scheduling_dist_${mode}_${scheduling_measure_mode} PRIVATE OMP_MODE=${mode} SCHEDULING_MEASURE_MODE=${scheduling_measure_mode})
      target_link_libraries(scheduling_dist_${mode}_${scheduling_measure_mode} OpenMP::OpenMP_CXX)
      if (USE_SANITIZERS)
        target_compile_options(scheduling_dist_${mode}_${scheduling_measure_mode} PUBLIC -fsanitize=address,undefined,leak -fno-sanitize-recover=all)
        target_link_options(scheduling_dist_${mode}_${scheduling_measure_mode} PUBLIC -fsanitize=address,undefined,leak)
      endif()
  endforeach()
  # tbb
  foreach(mode IN LISTS TBB_MODES)
      add_executable(scheduling_dist_${mode}_${scheduling_measure_mode} scheduling_dist.cpp)
      target_compile_definitions(scheduling_dist_${mode}_${scheduling_measure_mode} PRIVATE TBB_MODE=${mode} SCHEDULING_MEASURE_MODE=${scheduling_measure_mode})
      target_link_libraries(scheduling_dist_${mode}_${scheduling_measure_mode} TBB::tbb)
      if (USE_SANITIZERS)
        target_compile_options(scheduling_dist_${mode}_${scheduling_measure_mode} PUBLIC -fsanitize=address,undefined,leak -fno-sanitize-recover=all)
        target_link_options(scheduling_dist_${mode}_${scheduling_measure_mode} PUBLIC -fsanitize=address,undefined,leak)
      endif()
  endforeach()
  # eigen
  foreach(mode IN LISTS EIGEN_MODES)
      add_executable(scheduling_dist_${mode}_${scheduling_measure_mode} scheduling_dist.cpp)
      target_compile_definitions(scheduling_dist_${mode}_${scheduling_measure_mode} PRIVATE EIGEN_MODE=${mode} SCHEDULING_MEASURE_MODE=${scheduling_measure_mode})
      if (mode MATCHES "EIGEN_RAPID")
        target_link_libraries(scheduling_dist_${mode}_${scheduling_measure_mode} TBB::tbb)
      endif()
      if (USE_SANITIZERS)
        target_compile_options(scheduling_dist_${mode}_${scheduling_measure_mode} PUBLIC -fsanitize=address,undefined,leak -fno-sanitize-recover=all)
        target_link_options(scheduling_dist_${mode}_${scheduling_measure_mode} PUBLIC -fsanitize=address,undefined,leak)
      endif()
  endforeach()
endforeach()
