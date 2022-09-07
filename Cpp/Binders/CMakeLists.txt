pybind11_add_module(pyFROLS pyFROLS.cpp)
target_link_libraries(pyFROLS PUBLIC OpenMP::OpenMP_CXX ${FROLS_LIBRARIES} FROLS_Algorithm)
target_include_directories(pyFROLS PUBLIC ${FROLS_INCLUDE_DIRS})

