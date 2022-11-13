list(APPEND TESTS scan_tests spmv_tests)

get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(${PARENT_DIR})


foreach(test IN LISTS TESTS)
  # serial
  add_executable(${test}_SERIAL ${test}.cpp)
  target_compile_definitions(${test}_SERIAL PRIVATE SERIAL=1)
  target_link_libraries(${test}_SERIAL gtest GTest::gtest_main)
  # openmp
  foreach(mode IN LISTS OMP_MODES)
      add_executable(${test}_${mode} ${test}.cpp)
      target_compile_definitions(${test}_${mode} PRIVATE OMP_MODE=${mode})
      target_link_libraries(${test}_${mode} gtest GTest::gtest_main OpenMP::OpenMP_CXX)
  endforeach()
  # tbb
  foreach(mode IN LISTS TBB_MODES)
      add_executable(${test}_${mode} ${test}.cpp)
      target_compile_definitions(${test}_${mode} PRIVATE TBB_MODE=${mode})
      target_link_libraries(${test}_${mode} gtest GTest::gtest_main TBB::tbb)
  endforeach()
endforeach()
