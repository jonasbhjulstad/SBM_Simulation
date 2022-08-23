include(GoogleTest)

function(add_custom_test source_file)
add_executable(${source_file} "${source_file}.cpp")
target_include_directories(${source_file} PRIVATE "../include")
target_link_libraries(${source_file} PRIVATE Eigen3::Eigen GTest::gtest_main)
gtest_discover_tests(${source_file})
endfunction()

add_custom_test(feature_selection)
add_custom_test(regression)
