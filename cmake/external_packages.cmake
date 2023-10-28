if(NOT ${${PROJECT_NAME}_EXTERNAL_PACKAGES})
set(${PROJECT_NAME}_EXTERNAL_PACKAGES ON CACHE BOOL "" FORCE)

# set(cppitertools_INSTALL_CMAKE_DIR share)
# CPMFindPackage(
#     NAME cppitertools
#     GITHUB_REPOSITORY ryanhaining/cppitertools
#     GIT_TAG master
#     OPTIONS
#     "cppitertools_INSTALL_CMAKE_DIR share"
# )

endif()