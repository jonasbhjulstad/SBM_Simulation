function(configure_default_target target_name)
target_link_libraries(${target_name} PUBLIC ${PROJECT_NAME})
endfunction()