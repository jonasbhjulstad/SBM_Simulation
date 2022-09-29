add_library(FROLS_Graph INTERFACE)
target_include_directories(FROLS_Graph INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")