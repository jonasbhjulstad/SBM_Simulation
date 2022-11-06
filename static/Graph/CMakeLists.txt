add_library(Graph INTERFACE)
target_include_directories(Graph INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")