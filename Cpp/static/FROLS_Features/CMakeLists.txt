add_library(FROLS_Features STATIC FROLS_Features.cpp FROLS_Features.hpp FROLS_Polynomial.cpp FROLS_Polynomial.hpp)
target_include_directories(FROLS_Features PUBLIC ${FROLS_INCLUDE} ${cppitertools_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(FROLS_Features PUBLIC FROLS_DataFrame FROLS_Eigen cppitertools::cppitertools)