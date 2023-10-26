add_default_library(SIR_Dynamics)
target_link_libraries(SIR_Dynamics PUBLIC Sim_Buffers)

add_library(SIR_Types SIR_Types.cpp)
target_include_directories(SIR_Types PUBLIC $<BUILD_INTERFACE:${SBM_SIMULATION_INCLUDE_DIR}> $<INSTALL_INTERFACE:include>)
