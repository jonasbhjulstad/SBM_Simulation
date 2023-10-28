#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Template_Project::Template_Project" for configuration "Debug"
set_property(TARGET Template_Project::Template_Project APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(Template_Project::Template_Project PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/Template_Project-1.0/libTemplate_Project.a"
  )

list(APPEND _cmake_import_check_targets Template_Project::Template_Project )
list(APPEND _cmake_import_check_files_for_Template_Project::Template_Project "${_IMPORT_PREFIX}/lib/Template_Project-1.0/libTemplate_Project.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
