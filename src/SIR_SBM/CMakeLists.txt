add_subdirectory(epidemiological)
add_subdirectory(utils)
add_subdirectory(simulation)
add_subdirectory(regression)
add_subdirectory(graph)
add_subdirectory(math)
add_subdirectory(random)
add_subdirectory(sycl)
custom_configure_sycl(sycl)
custom_configure_sycl(simulation)
custom_configure_sycl(epidemiological)
add_library(SIR_SBM STATIC sir_sbm.cpp)
target_include_directories(SIR_SBM PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SIR_SBM PRIVATE epidemiological
				utils
				simulation
				regression
				graph
				math
				random
				sycl)
