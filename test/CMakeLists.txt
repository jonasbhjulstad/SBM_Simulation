include_guard(GLOBAL)
add_subdirectory(buffer)
add_subdirectory(graph)
if(${PROJECT_NAME}_ENABLE_GRAPH_TOOL)
add_subdirectory(gt)
endif()

add_custom_target(${PROJECT_NAME}_tests ALL DEPENDS ${custom_test_targets})
message(WARNING ${custom_test_targets})