pybind11_add_module(pyFROLS pyFROLS.cpp)
target_link_libraries(pyFROLS PUBLIC OpenMP::OpenMP_CXX Features Regressors)

