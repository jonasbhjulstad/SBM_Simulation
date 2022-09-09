add_library(Math STATIC Math.cpp quantiles.cpp)
target_link_libraries(Math PUBLIC Common)
target_link_libraries(Math PRIVATE DataFrame)
target_include_directories(Math PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
