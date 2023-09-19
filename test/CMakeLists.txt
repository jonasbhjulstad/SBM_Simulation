add_sycl_executable(rng)
# add_sycl_executable(compute_units)
add_sycl_executable_nodep(compute_units)
add_executable(casadi_ex casadi_ex.cpp)
target_link_libraries(casadi_ex casadi)

add_sycl_executable(test_ecm_from_vcm)
add_sycl_executable(infection_sampling_test)
