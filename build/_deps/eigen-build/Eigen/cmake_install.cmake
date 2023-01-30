# Install script for directory: /home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE FILE FILES
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/Cholesky"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/CholmodSupport"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/Core"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/Dense"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/Eigen"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/Eigenvalues"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/Geometry"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/Householder"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/IterativeLinearSolvers"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/Jacobi"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/LU"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/MetisSupport"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/OrderingMethods"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/PaStiXSupport"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/PardisoSupport"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/QR"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/QtAlignedMalloc"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/SPQRSupport"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/SVD"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/Sparse"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/SparseCholesky"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/SparseCore"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/SparseLU"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/SparseQR"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/StdDeque"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/StdList"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/StdVector"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/SuperLUSupport"
    "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/UmfPackSupport"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE DIRECTORY FILES "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

