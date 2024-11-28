
add_custom_test(initialize)
add_custom_test(recover)
add_custom_test(state_copy)
target_link_libraries(state_copy_test PUBLIC oneDPL)