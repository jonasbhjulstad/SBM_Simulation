
add_subdirectory(Utils)
add_subdirectory(Graph)
add_library(Sim_Types STATIC Simulation/Sim_Types.cpp)
target_include_directories(Sim_Types PUBLIC $<BUILD_INTERFACE:${SBM_SIMULATION_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>)
target_link_libraries(Sim_Types PUBLIC Complete_Graph String_Manipulation)
add_subdirectory(Database)
add_subdirectory(Regression)

add_subdirectory(Simulation)
add_subdirectory(Epidemiological)
# target_link_libraries(Simulation PUBLIC Graph Sim_Dynamics Profiling)
