add_library(Features STATIC Feature_Model.cpp Polynomial.cpp)
target_link_libraries(Features PUBLIC Common)
target_link_libraries(Features PRIVATE FROLS_Eigen cppitertools::cppitertools Regressors Math casadi)
target_include_directories(Features INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")