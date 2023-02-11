list(APPEND TESTS scan_tests spmv_tests)

get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(${PARENT_DIR})

foreach(test IN LISTS TESTS)
  foreach(mode IN LISTS MODES)
      set(target ${test}_${mode})
      add_target(${target} ${test}.cpp ${mode})
      target_link_libraries(${target} gtest ${GTEST_MAIN_LIBRARIES})
  endforeach()
endforeach()
