function(configure_default_target target_name)
target_link_libraries(${target_name} PUBLIC ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_EXTERNAL_PUBLIC_LIBRARIES})
endfunction()