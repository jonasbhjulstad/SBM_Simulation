add_library(utils exceptions.cpp
	csv.cpp
	ticktock.cpp
	${CMAKE_CURRENT_BINARY_DIR}/filepaths.cpp)
target_include_directories(utils PUBLIC ${PROJECT_SOURCE_DIR}/include)
configure_file(${CMAKE_CURRENT_LIST_DIR}/filepaths.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/filepaths.cpp @ONLY)
target_link_libraries(utils PRIVATE sycl)
target_link_libraries(utils PRIVATE epidemiological)
