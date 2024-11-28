add_library(epidemiological infection_count.cpp
	epidemiological.cpp
	infection_sampling.cpp
	population_count.cpp)
target_include_directories(epidemiological PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(epidemiological PRIVATE sycl)
target_link_libraries(epidemiological PRIVATE graph)
target_link_libraries(epidemiological PRIVATE cppitertools::cppitertools)
target_link_libraries(epidemiological PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(epidemiological PRIVATE oneDPL)
target_link_libraries(epidemiological PRIVATE utils)
