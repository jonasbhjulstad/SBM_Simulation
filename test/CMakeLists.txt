

#add_custom_executable

add_executable(itertools_comb itertools_comb.cpp)
target_link_libraries(itertools_comb PUBLIC cppitertools)

add_sycl_executable(remap)