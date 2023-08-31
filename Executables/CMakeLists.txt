add_sycl_executable(parallel_SIR_Network)
target_link_libraries(parallel_SIR_Network PRIVATE Graph Sim_Routines Profiling nlohmann_json::nlohmann_json)
add_sycl_executable(SBM_Regression)
target_link_libraries(SBM_Regression PRIVATE Static_RNG::Static_RNG TBB::tbb Eigen3::Eigen ortools::ortools Regression)

# add_regression_executable(SBM_Regression)
add_sycl_executable_nodep(Optimization)
target_link_libraries(Optimization PRIVATE nlohmann_json::nlohmann_json casadi)
