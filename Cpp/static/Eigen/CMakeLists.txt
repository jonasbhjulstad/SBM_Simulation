add_library(FROLS_Eigen STATIC FROLS_Eigen.cpp FROLS_Eigen.hpp)
target_link_libraries(FROLS_Eigen PUBLIC Common)
target_link_libraries(FROLS_Eigen PUBLIC Eigen3::Eigen DataFrame)
target_include_directories(FROLS_Eigen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})