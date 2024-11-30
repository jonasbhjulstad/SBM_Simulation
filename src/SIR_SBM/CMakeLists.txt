add_subdirectory(utils)
add_subdirectory(graph)
add_subdirectory(math)
add_subdirectory(random)
add_subdirectory(sycl)
custom_configure_sycl(utils)
add_library(SIR_SBM STATIC sir_sbm.cpp)
target_include_directories(SIR_SBM PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SIR_SBM PRIVATE utils
				graph
				math
				random
				sycl)
