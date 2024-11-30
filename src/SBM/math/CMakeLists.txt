add_library(math combination.cpp
	occurrence.cpp
	numeric.cpp)
target_include_directories(math PUBLIC ${PROJECT_SOURCE_DIR}/include)
