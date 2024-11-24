add_library(sycl queue_select.cpp
	sycl_routines.cpp)
target_include_directories(sycl PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(sycl PRIVATE yaml-cpp::yaml-cpp)
