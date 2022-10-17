add_library(FROLS_Sycl INTERFACE)
target_include_directories(FROLS_Sycl INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
