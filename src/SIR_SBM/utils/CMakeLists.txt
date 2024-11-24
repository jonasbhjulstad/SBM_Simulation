add_library(utils exceptions.cpp
	csv.cpp
	ticktock.cpp)
target_include_directories(utils PUBLIC ${PROJECT_SOURCE_DIR}/include)
