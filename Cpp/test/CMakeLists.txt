include(GoogleTest)

function(add_custom_executable source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_include_directories(${source_file} PUBLIC "../include")
    target_link_libraries(${source_file} PUBLIC Eigen3::Eigen ${SUNDIALS_LIBS} ${LIBRARIES} graphlite FROLS_Graph)
endfunction()

function(add_custom_test source_file)
    add_executable(${source_file} "${source_file}.cpp")
    target_include_directories(${source_file} PUBLIC "../include")
    target_link_libraries(${source_file} PUBLIC Eigen3::Eigen ${SUNDIALS_LIBS} ${LIBRARIES} graphlite FROLS_Graph GTest::gtest_main)
    gtest_discover_tests(${source_file})
    target_compile_options(${source_file} PRIVATE "-Wfatal-errors")
endfunction()

# add_custom_executable(neighbor_test)
# #link frols_network
# target_link_libraries(neighbor_test PUBLIC FROLS_Graph FROLS_Network)
# target_include_directories(neighbor_test PUBLIC "../Executables/Generate/")

add_custom_test(Graph_Iterators)
target_include_directories(Graph_Iterators PUBLIC "../static/Graph/")
# add_custom_test(SIR_integrator)
# add_custom_test(monomials)
#add_custom_test(sir_bernoulli)
# add_custom_test(feature_transform)
