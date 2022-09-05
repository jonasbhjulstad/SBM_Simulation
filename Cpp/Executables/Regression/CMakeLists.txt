add_executable(Single_Feature single_feature.cpp)
target_link_libraries(Single_Feature PUBLIC ${DEFAULT_LIBS} ${FROLS_LIBRARIES})
target_include_directories(Single_Feature PRIVATE ${FROLS_INCLUDE_DIRS})

add_executable(truncate truncate.cpp)
target_link_libraries(truncate PUBLIC ${DEFAULT_LIBS} ${FROLS_LIBRARIES})
target_include_directories(truncate PRIVATE ${FROLS_INCLUDE_DIRS})



