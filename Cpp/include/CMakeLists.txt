add_library(Common INTERFACE)
target_include_directories(Common INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(Common INTERFACE Eigen3::Eigen)