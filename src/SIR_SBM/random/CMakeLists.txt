add_library(random random.cpp)
target_include_directories(random PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(random PRIVATE oneDPL)
