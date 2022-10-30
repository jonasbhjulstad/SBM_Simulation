pybind11_add_module(FROLS_Binder pyFROLS.cpp)
target_link_libraries(FROLS_Binder PUBLIC Common Eigen3::Eigen ${ONEAPI_LIBS} Regressors FROLS_Math Features DataFrame FROLS_Eigen FROLS_Network TBB::tbb ${PYTHON_LIBRARIES})
target_include_directories(FROLS_Binder PUBLIC "../Executables/Generate" ${PYTHON_INCLUDE_DIRECTORIES})
set_target_properties(FROLS_Binder PROPERTIES OUTPUT_NAME "pyFROLS" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../Python/" SUFFIX ".so")