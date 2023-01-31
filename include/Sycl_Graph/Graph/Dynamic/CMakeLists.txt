add_library(Graph_Dynamic INTERFACE)
target_link_libraries(Graph_Dynamic INTERFACE Graph_Base)
target_include_directories(Graph_Dynamic INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
  $<INSTALL_INTERFACE:Sycl_Graph/Graph/Dynamic>
  )