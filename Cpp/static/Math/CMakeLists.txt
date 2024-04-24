add_library(FROLS_Math STATIC FROLS_Math.cpp quantiles.cpp)
target_link_libraries(FROLS_Math PUBLIC Common DataFrame)
target_include_directories(FROLS_Math PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
