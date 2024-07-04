

file(GLOB SIR_SBM_core_modules ${CMAKE_CURRENT_LIST_DIR}/*.cpp)



add_library(SIR_SBM_core)
add_library(SIR_SBM_regression)
add_library(SIR_SBM_simulation)
add_library(SIR_SBM_sycl)
target_sources(SIR_SBM_core
PUBLIC FILE_SET CXX_MODULES FILES "${SIR_SBM_core_modules}"
)

file(GLOB SIR_SBM_regression_modules ${CMAKE_CURRENT_LIST_DIR}/regression/*.cpp)

target_sources(SIR_SBM_regression
PUBLIC FILE_SET CXX_MODULES FILES ${SIR_SBM_regression_modules}
)
file(GLOB SIR_SBM_simulation_modules ${CMAKE_CURRENT_LIST_DIR}/simulation/*.cpp)

target_sources(SIR_SBM_simulation
PUBLIC FILE_SET CXX_MODULES FILES ${SIR_SBM_simulation_modules}
)

file(GLOB SIR_SBM_sycl_modules ${CMAKE_CURRENT_LIST_DIR}/sycl/*.cpp)
target_sources(SIR_SBM_sycl
PUBLIC FILE_SET CXX_MODULES FILES ${SIR_SBM_sycl_modules}
)