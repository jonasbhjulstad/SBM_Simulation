if ("${${PROJECT_NAME}_EXTERNAL_PACKAGES}" STREQUAL "")
include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)


# find_package(Static_RNG REQUIRED)

find_package(Boost 1.78 REQUIRED COMPONENTS date_time HINTS ${PYTHON_ENV_CMAKE_MODULE_DIR})

set(cppitertools_INSTALL_CMAKE_DIR share)
CPMAddPackage("gh:ryanhaining/cppitertools@2.1")

include(cmake/TinyOrm.cmake)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(${PROJECT_NAME}_BUILD_BINDER)
include(FindPython3)
FindPython3()
find_package(pybind11 CONFIG)
endif()

# set(${PROJECT_NAME}_CUSTOM_EXTERNAL_PACKAGES Buffer_Routines::Buffer_Routines SBM_Graph::SBM_Graph)


#spdlog
CPMFindPackage(NAME fmt
GITHUB_REPOSITORY fmtlib/fmt)
CPMFindPackage(NAME spdlog
GITHUB_REPOSITORY gabime/spdlog)

custom_submodule_add(Static_RNG)
custom_submodule_add(Dataframe)
custom_submodule_add(Buffer_Routines)
custom_submodule_add(SBM_Graph)
CPMAddPackage(NAME SBM_Database SOURCE_DIR ${SBM_Database_SOURCE_DIR} 
OPTIONS 
"${SBM_Database_EXPORT} ON"
"ENABLE_TESTING OFF"
"SBM_Database_BUILD_BINDER OFF"
"SBM_Database_TOM_ENV_SCRIPT_DIR ${PROJECT_BINARY_DIR}")

find_package(ortools REQUIRED CONFIG)

find_package(TBB REQUIRED)
CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.6.0")
# find_package(casadi REQUIRED HINTS "/home/man/mambaforge/envs/gt/lib/cmake/casadi")
set(${PROJECT_NAME}_EXTERNAL_PACKAGES ${TINY_ORM_LIBRARIES} Static_RNG::Static_RNG Dataframe::Dataframe SBM_Database::SBM_Database Buffer_Routines::Buffer_Routines SBM_Graph::SBM_Graph TBB::tbb Eigen3::Eigen cppitertools::cppitertools ortools::ortools fmt::fmt spdlog::spdlog)
endif()
