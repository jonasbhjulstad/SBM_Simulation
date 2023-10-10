add_library(Dataframe STATIC Dataframe.cpp)
target_include_directories(Dataframe PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})
target_link_libraries(Dataframe PUBLIC Validation)
