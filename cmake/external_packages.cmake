if ("${${PROJECT_NAME}_EXTERNAL_PACKAGES}" STREQUAL "")
include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)


CPMFindPackage(
    NAME Static_RNG
    GITHUB_REPOSITORY jonasbhjulstad/Static_RNG
    GIT_TAG master
)
# find_package(Static_RNG REQUIRED)

find_package(Boost 1.78 REQUIRED COMPONENTS date_time HINTS ${PYTHON_ENV_CMAKE_MODULE_DIR})

set(cppitertools_INSTALL_CMAKE_DIR share)
CPMAddPackage("gh:ryanhaining/cppitertools@2.1")
# CPMFindPackage(
#     NAME cppitertools
#     GITHUB_REPOSITORY ryanhaining/cppitertools
#     GIT_TAG master@2.1
#     OPTIONS
#     "cppitertools_INSTALL_CMAKE_DIR share"
# )
find_package(QT NAMES Qt6 Qt5 COMPONENTS Core REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core REQUIRED)

CPMAddPackage("gh:silverqx/TinyORM@0.36.3")
# find_package(TinyOrm 0.36.3 CONFIG REQUIRED)
set(TINY_ORM_LIBRARIES Qt5::Core TinyOrm::TinyOrm)

# find_package(TinyOrm CONFIG REQUIRED)
#eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

#add pqxx4
CPMFindPackage(NAME Dataframe
    GITHUB_REPOSITORY jonasbhjulstad/Dataframe
    GIT_TAG master
    OPTIONS
    "CMAKE_POSITION_INDEPENDENT_CODE ON")

CPMFindPackage(NAME Buffer_Routines
    GITHUB_REPOSITORY jonasbhjulstad/Sycl_Buffer_Routines
    GIT_TAG master)
CPMFindPackage(NAME SBM_Graph
    GITHUB_REPOSITORY jonasbhjulstad/SBM_Graph
    GIT_TAG tom)

find_package(SBM_Database_Migrations REQUIRED CONFIG)

# CPMFindPackage(NAME SBM_Database
#     GITHUB_REPOSITORY jonasbhjulstad/SBM_Database
#     GIT_TAG master
#     OPTIONS
#     #-fPIC
#     "CMAKE_POSITION_INDEPENDENT_CODE ON"
#     )

find_package(TBB REQUIRED)
CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.6.0")
# find_package(casadi REQUIRED HINTS "/home/man/mambaforge/envs/gt/lib/cmake/casadi")
set(${PROJECT_NAME}_EXTERNAL_PACKAGES ${TINY_ORM_LIBRARIES} Static_RNG::Static_RNG Dataframe::Dataframe Buffer_Routines::Buffer_Routines SBM_Graph::SBM_Graph TBB::tbb Eigen3::Eigen cppitertools::cppitertools SBM_Database_Migrations::SBM_Database_Migrations)
endif()
