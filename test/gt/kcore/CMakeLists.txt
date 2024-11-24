add_library(kcore SHARED kcore.cpp)
graphtool_configure(kcore)
set_target_properties(kcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})