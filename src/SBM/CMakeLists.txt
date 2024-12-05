add_subdirectory(utils)
add_subdirectory(graph)
add_subdirectory(math)
add_subdirectory(random)
add_subdirectory(sycl)
custom_configure_sycl(utils)
custom_configure_sycl(random)
add_library(SBM STATIC sbm.cpp)
target_include_directories(SBM PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SBM PUBLIC utils
				graph
				math
				random
				sycl)
