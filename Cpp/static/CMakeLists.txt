add_library(FROLS_DataFrame STATIC FROLS_DataFrame.cpp FROLS_DataFrame.hpp)
add_library(FROLS_Quantiles STATIC FROLS_quantiles.cpp FROLS_quantiles.hpp)
set(FROLS_LIBRARIES "FROLS_DataFrame;FROLS_Quantiles" CACHE INTERNAL "FROLS_LIBRARIES")