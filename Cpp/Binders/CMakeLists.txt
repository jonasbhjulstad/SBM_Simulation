pybind11_add_module(FROLS FROLS.cpp)
target_link_libraries(FROLS PUBLIC OpenMP::OpenMP_CXX ${FROLS_LIBRARIES})
target_include_directories(FROLS PUBLIC ${FROLS_INCLUDE_DIRS})

