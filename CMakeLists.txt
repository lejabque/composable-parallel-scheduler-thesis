cmake_minimum_required (VERSION 3.15)
option(ENABLE_TESTS "Enable test" OFF)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

project(composable-parallel-scheduler)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


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
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/cmake_deps)

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


find_package(OpenMP REQUIRED)

# link with TBB
FetchContent_Declare(
  onetbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG v2021.8.0
)
option(TBB_TEST OFF)
FetchContent_MakeAvailable(onetbb)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps=obj")

# build options
option(ENABLE_SANITIZERS "Enable sanitizers" OFF)
if(ENABLE_SANITIZERS)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined,leak -fno-sanitize-recover=all -gdwarf-4")
  set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address,undefined,leak")
endif()

list(APPEND OMP_MODES OMP_STATIC
  OMP_DYNAMIC_MONOTONIC OMP_DYNAMIC_NONMONOTONIC
  OMP_GUIDED_MONOTONIC OMP_GUIDED_NONMONOTONIC)

list(APPEND TBB_MODES TBB_SIMPLE TBB_AUTO TBB_AFFINITY TBB_CONST_AFFINITY)
if($ENV{ENABLE_RAPID})
  list(APPEND TBB_MODES TBB_RAPID)
  list(APPEND EIGEN_MODES EIGEN_RAPID)
endif()
list(APPEND EIGEN_MODES EIGEN_SIMPLE EIGEN_TIMESPAN EIGEN_TIMESPAN_GRAINSIZE EIGEN_STATIC)

if ($ENV{USE_LB4OMP})
  set(OPENMP_STANDALONE_BUILD TRUE)
  set(LIBOMP_HAVE___RDTSC TRUE)
  set(LIBOMP_HAVE_X86INTRIN_H TRUE)
  FetchContent_Declare(
    lb4omp
    GIT_REPOSITORY https://github.com//unibas-dmi-hpc/LB4OMP.git
  )
  FetchContent_MakeAvailable(lb4omp)
  add_library(lb4omp SHARED IMPORTED)
  set_target_properties(lb4omp PROPERTIES IMPORTED_LOCATION ${FETCHCONTENT_BASE_DIR}/lb4omp-build/runtime/src/libomp.so)

  list(APPEND OMP_MODES OMP_RUNTIME)
endif()

list(APPEND MODES ${OMP_MODES} ${TBB_MODES} ${EIGEN_MODES})

function(add_target target exec mode)
    add_executable(${target} ${exec})
    if (mode MATCHES "^SERIAL")
      target_compile_definitions(${target} PRIVATE SERIAL=1)
    elseif (mode MATCHES "^OMP")
      target_compile_definitions(${target} PRIVATE OMP_MODE=${mode})
      if (mode STREQUAL "OMP_RUNTIME")
        target_link_libraries(${target} lb4omp)
        # tricky patch to prefer built openmp over system's
        # todo: this still doesn't work, only replacing installed openmp is working
        # add_custom_command(TARGET ${target}
        #   POST_BUILD
        #   COMMAND sudo patchelf --replace-needed "libomp.so" "${FETCHCONTENT_BASE_DIR}/lb4omp-build/runtime/src/libomp.so" $<TARGET_FILE:${target}>
        # )
        target_link_libraries(${target} OpenMP::OpenMP_CXX)
      else()
        target_link_libraries(${target} OpenMP::OpenMP_CXX)
      endif()
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

option(ENABLE_TRACE_SPIN "Enable traced spin bench" ON)
if (ENABLE_TRACE_SPIN)
  add_subdirectory(trace_spin)
endif()

option(ENABLE_TIMESPAN_TUNER "Enable timespan tuner" ON)
if (ENABLE_TIMESPAN_TUNER)
  add_subdirectory(timespan_tuner)
endif()

if (ENABLE_TESTS)
    add_subdirectory(include/tests)
endif()
