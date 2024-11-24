add_library(regression regression.cpp)
target_include_directories(regression PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(regression PRIVATE casadi)
