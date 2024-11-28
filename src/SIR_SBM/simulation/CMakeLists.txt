add_library(simulation connection_data.cpp
	sim_result.cpp
	population_data.cpp
	simulation.cpp
	sim_buffers.cpp
	sim_param.cpp)
target_include_directories(simulation PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(simulation PRIVATE sycl)
target_link_libraries(simulation PRIVATE graph)
target_link_libraries(simulation PRIVATE cppitertools::cppitertools)
target_link_libraries(simulation PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(simulation PRIVATE oneDPL)
target_link_libraries(simulation PRIVATE utils)
