
file(GLOB SIMULATION_SRC *.cpp)

add_library(${PROJECT_NAME} ${SIMULATION_SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC -ldl -lpq Static_RNG::Static_RNG TBB::tbb)
target_compile_options(${PROJECT_NAME} PUBLIC ${SYCL_CUSTOM_FLAGS} ${DEFAULT_WARNING_FLAGS})
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${SBM_SIMULATION_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>)
target_link_options(${PROJECT_NAME} PUBLIC ${SYCL_CUSTOM_FLAGS} -fPIC)
target_link_libraries(${PROJECT_NAME} PUBLIC SBM_Database::SBM_Database SBM_Graph::SBM_Graph ${PROJECT_NAME}_Types ${PROJECT_NAME}_Utils Epidemiological)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
