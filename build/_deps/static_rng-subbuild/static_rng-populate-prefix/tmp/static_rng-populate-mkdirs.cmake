# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/man/Documents/Sycl_Graph/build/_deps/static_rng-src"
  "/home/man/Documents/Sycl_Graph/build/_deps/static_rng-build"
  "/home/man/Documents/Sycl_Graph/build/_deps/static_rng-subbuild/static_rng-populate-prefix"
  "/home/man/Documents/Sycl_Graph/build/_deps/static_rng-subbuild/static_rng-populate-prefix/tmp"
  "/home/man/Documents/Sycl_Graph/build/_deps/static_rng-subbuild/static_rng-populate-prefix/src/static_rng-populate-stamp"
  "/home/man/Documents/Sycl_Graph/build/_deps/static_rng-subbuild/static_rng-populate-prefix/src"
  "/home/man/Documents/Sycl_Graph/build/_deps/static_rng-subbuild/static_rng-populate-prefix/src/static_rng-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/man/Documents/Sycl_Graph/build/_deps/static_rng-subbuild/static_rng-populate-prefix/src/static_rng-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/man/Documents/Sycl_Graph/build/_deps/static_rng-subbuild/static_rng-populate-prefix/src/static_rng-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
