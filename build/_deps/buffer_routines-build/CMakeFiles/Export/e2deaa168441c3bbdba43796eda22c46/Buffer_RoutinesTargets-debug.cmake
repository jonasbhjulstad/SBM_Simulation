#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Buffer_Routines::Buffer_Routines" for configuration "Debug"
set_property(TARGET Buffer_Routines::Buffer_Routines APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(Buffer_Routines::Buffer_Routines PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/Buffer_Routines-1.0/libBuffer_Routines.a"
  )

list(APPEND _cmake_import_check_targets Buffer_Routines::Buffer_Routines )
list(APPEND _cmake_import_check_files_for_Buffer_Routines::Buffer_Routines "${_IMPORT_PREFIX}/lib/Buffer_Routines-1.0/libBuffer_Routines.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
