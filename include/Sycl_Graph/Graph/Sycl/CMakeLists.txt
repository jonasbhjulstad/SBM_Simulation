add_library(Graph_Sycl INTERFACE)
target_link_libraries(Graph_Sycl INTERFACE Graph_Base)
target_include_directories(Graph_Sycl INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
  $<INSTALL_INTERFACE:Sycl_Graph/Graph/Sycl>
  )