include(FindPackageHandleStandardArgs)
set(${CMAKE_FIND_PACKAGE_NAME}_CONFIG ${CMAKE_CURRENT_LIST_FILE})
find_package_handle_standard_args(Static_RNG CONFIG_MODE)

if(NOT TARGET Static_RNG::Random)
    include("${CMAKE_CURRENT_LIST_DIR}/Static_RNGTargets.cmake")
    add_library(Static_RNG::Random INTERFACE IMPORTED)
    add_library(Static_RNG::tinymt INTERFACE IMPORTED)

endif()
