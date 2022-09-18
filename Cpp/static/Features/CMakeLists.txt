add_library(Features STATIC Feature_Model.cpp Polynomial.cpp)
target_link_libraries(Features PUBLIC Common FROLS_Eigen cppitertools::cppitertools Regressors casadi ${DEFAULT_LIBRARIES})
target_include_directories(Features INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")