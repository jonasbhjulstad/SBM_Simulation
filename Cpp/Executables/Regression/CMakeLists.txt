add_executable(Single_Response Single_Response.cpp)
target_link_libraries(Single_Response PUBLIC ${DEFAULT_LIBS} ${FROLS_LIBRARIES})
target_include_directories(Single_Response PUBLIC ${FROLS_INCLUDE_DIRS})

add_executable(Multiple_Response Multiple_Response.cpp)
target_link_libraries(Multiple_Response PUBLIC ${DEFAULT_LIBS} ${FROLS_LIBRARIES})
target_include_directories(Multiple_Response PUBLIC ${FROLS_INCLUDE_DIRS})


add_executable(truncate truncate.cpp)
target_link_libraries(truncate PUBLIC ${DEFAULT_LIBS} ${FROLS_LIBRARIES})
target_include_directories(truncate PUBLIC ${FROLS_INCLUDE_DIRS})



