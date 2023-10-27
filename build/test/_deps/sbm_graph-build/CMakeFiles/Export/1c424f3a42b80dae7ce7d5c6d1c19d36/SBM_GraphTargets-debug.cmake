#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SBM_Graph::SBM_Graph" for configuration "Debug"
set_property(TARGET SBM_Graph::SBM_Graph APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SBM_Graph::SBM_Graph PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/SBM_Graph-1.0/libSBM_Graph.a"
  )

list(APPEND _cmake_import_check_targets SBM_Graph::SBM_Graph )
list(APPEND _cmake_import_check_files_for_SBM_Graph::SBM_Graph "${_IMPORT_PREFIX}/lib/SBM_Graph-1.0/libSBM_Graph.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
