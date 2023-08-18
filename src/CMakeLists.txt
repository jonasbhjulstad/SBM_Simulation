

add_library(Graph SHARED Graph.cpp)
target_link_libraries(Graph PRIVATE cppitertools)
target_include_directories(Graph PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})
add_custom_library(Buffer_Utils)
add_library(Regression SHARED Regression.cpp)
target_link_libraries(Regression PRIVATE ortools::ortools)
target_link_libraries(Regression PUBLIC Eigen3::Eigen)
target_include_directories(Regression PUBLIC ${SYCL_GRAPH_INCLUDE_DIR})

add_custom_library(Community_State)
add_custom_library(SIR_Infection_Sampling)
add_custom_library(Sim_Dynamics)
target_link_libraries(Sim_Dynamics PRIVATE Community_State Buffer_Utils)
add_custom_library(Simulation)
target_link_libraries(Simulation PRIVATE Buffer_Utils Graph SIR_Infection_Sampling Profiling Community_State)
add_custom_library(Profiling)
target_link_libraries(Simulation PUBLIC Graph Sim_Dynamics Profiling)
