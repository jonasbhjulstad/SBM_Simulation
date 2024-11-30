add_library(graph graph.cpp
	sbm_param.cpp
	indices.cpp)
target_include_directories(graph PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(graph PRIVATE cppitertools::cppitertools)
target_link_libraries(graph PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(graph PRIVATE oneDPL)
