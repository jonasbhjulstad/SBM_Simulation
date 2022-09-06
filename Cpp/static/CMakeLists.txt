add_library(FROLS_DataFrame STATIC FROLS_DataFrame.cpp FROLS_DataFrame.hpp)
add_subdirectory(FROLS_Features)
add_library(FROLS_Math STATIC FROLS_Math.cpp FROLS_Math.hpp)
target_include_directories(FROLS_Math PUBLIC ${FROLS_INCLUDE_DIRS})
target_link_libraries(FROLS_Math PUBLIC Eigen3::Eigen)

add_library(FROLS_Eigen STATIC FROLS_Eigen.cpp FROLS_Eigen.hpp)
target_include_directories(FROLS_Eigen PUBLIC ${FROLS_INCLUDE_DIRS})
target_link_libraries(FROLS_Eigen PUBLIC Eigen3::Eigen FROLS_DataFrame)

add_library(FROLS_Algorithm STATIC FROLS_Algorithm.cpp FROLS_Algorithm.hpp)
target_include_directories(FROLS_Algorithm PUBLIC ${FROLS_INCLUDE_DIRS})
target_link_libraries(FROLS_Algorithm PUBLIC Eigen3::Eigen FROLS_DataFrame FROLS_Features cppitertools::cppitertools)

add_subdirectory(JIT)

add_library(FROLS_Quantiles STATIC FROLS_quantiles.cpp FROLS_quantiles.hpp)
target_link_libraries(FROLS_Quantiles PUBLIC FROLS_DataFrame)
