add_library(Graph_Types STATIC Graph_Types.cpp)
target_include_directories(Graph_Types PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})
add_custom_library(Graph)
target_link_libraries(Graph PUBLIC cppitertools math Graph_Types)
target_include_directories(Graph PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})


add_custom_library(Community_Mappings)
target_link_libraries(Community_Mappings PUBLIC cppitertools)
target_link_libraries(Community_Mappings PUBLIC Dataframe Graph_Types)
