
add_subdirectory(Utils)
add_subdirectory(Graph)
add_library(Sim_Types Simulation/Sim_Types.cpp)
default_configure_target(Sim_Types)
target_link_libraries(Sim_Types PUBLIC Complete_Graph String_Manipulation)
add_subdirectory(Database)
add_subdirectory(Regression)

add_subdirectory(Simulation)
add_subdirectory(Epidemiological)
# target_link_libraries(Simulation PUBLIC Graph Sim_Dynamics Profiling)
