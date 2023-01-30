#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Static_RNG::tinymt" for configuration "Debug"
set_property(TARGET Static_RNG::tinymt APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(Static_RNG::tinymt PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libtinymt.a"
  )

list(APPEND _cmake_import_check_targets Static_RNG::tinymt )
list(APPEND _cmake_import_check_files_for_Static_RNG::tinymt "${_IMPORT_PREFIX}/lib/libtinymt.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
