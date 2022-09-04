add_library(FROLS_DataFrame STATIC FROLS_DataFrame.cpp FROLS_DataFrame.hpp)

add_library(FROLS_Eigen STATIC FROLS_Eigen.cpp FROLS_Eigen.hpp)
target_include_directories(FROLS_Eigen PUBLIC ${FROLS_INCLUDE})
target_link_libraries(FROLS_Eigen PUBLIC Eigen3::Eigen FROLS_DataFrame)
add_library(FROLS_Quantiles STATIC FROLS_quantiles.cpp FROLS_quantiles.hpp)
target_link_libraries(FROLS_Quantiles PRIVATE FROLS_DataFrame)

add_library(FROLS_Features STATIC FROLS_Features.cpp FROLS_Features.hpp)
target_include_directories(FROLS_Features PUBLIC ${FROLS_INCLUDE} ${cppitertools_INCLUDE_DIR})
target_link_libraries(FROLS_Features PUBLIC FROLS_DataFrame FROLS_Eigen cppitertools::cppitertools)