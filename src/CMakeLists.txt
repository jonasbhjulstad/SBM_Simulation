#get all subdirs
set(subdirs "epidemiological;graph;regression;simulation;sycl;utils;random")
foreach(subdir ${subdirs})
file(GLOB ${subdir}_src ${subdir}/*.cpp)
endforeach()

add_library(SIR_SBM_core ${epidemiological_src} ${graph_src} ${simulation_src} ${sycl_src} ${utils_src} ${random_src})
custom_configure_sycl(SIR_SBM_core)
target_include_directories(SIR_SBM_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SIR_SBM_core PRIVATE ${SIR_SBM_EXTERNAL_PRIVATE_LIBRARIES})
target_link_libraries(SIR_SBM_core PUBLIC oneDPL)


add_library(SIR_SBM_regression ${regression_src})
target_link_libraries(SIR_SBM_regression PUBLIC casadi)
target_include_directories(SIR_SBM_regression PUBLIC ${PROJECT_SOURCE_DIR}/include)
