add_library(FROLS_Algorithm STATIC FROLS_Algorithm.cpp FROLS_Algorithm.hpp)
target_include_directories(FROLS_Algorithm PUBLIC ${FROLS_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(FROLS_Algorithm PUBLIC Eigen3::Eigen FROLS_DataFrame FROLS_Features FROLS_Math cppitertools::cppitertools)

