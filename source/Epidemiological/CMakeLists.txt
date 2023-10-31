file(GLOB EPIDEMIOLOGICAL_SRC "*.cpp")
add_library(Epidemiological ${EPIDEMIOLOGICAL_SRC})
default_configure_target(Epidemiological)
target_link_libraries(Epidemiological PUBLIC SBM_Graph::SBM_Graph ${PROJECT_NAME}_Utils ${PROJECT_NAME}_Types)
