
add_sycl_executable_nodep(parallel_SIR_Network)
target_link_libraries(parallel_SIR_Network PRIVATE -lpqxx -lpq Graph Simulation)
