add_library(DataFrame STATIC DataFrame.cpp DataFrame.hpp)
target_include_directories(DataFrame PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})