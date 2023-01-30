# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/man/Documents/Sycl_Graph/build/_deps/eigen-src"
  "/home/man/Documents/Sycl_Graph/build/_deps/eigen-build"
  "/home/man/Documents/Sycl_Graph/build/_deps/eigen-subbuild/eigen-populate-prefix"
  "/home/man/Documents/Sycl_Graph/build/_deps/eigen-subbuild/eigen-populate-prefix/tmp"
  "/home/man/Documents/Sycl_Graph/build/_deps/eigen-subbuild/eigen-populate-prefix/src/eigen-populate-stamp"
  "/home/man/Documents/Sycl_Graph/build/_deps/eigen-subbuild/eigen-populate-prefix/src"
  "/home/man/Documents/Sycl_Graph/build/_deps/eigen-subbuild/eigen-populate-prefix/src/eigen-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/man/Documents/Sycl_Graph/build/_deps/eigen-subbuild/eigen-populate-prefix/src/eigen-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/man/Documents/Sycl_Graph/build/_deps/eigen-subbuild/eigen-populate-prefix/src/eigen-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
