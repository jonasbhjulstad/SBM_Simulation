# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/deb/Documents/SBM_Simulation/build/_deps/cppitertools-src"
  "/home/deb/Documents/SBM_Simulation/build/_deps/cppitertools-build"
  "/home/deb/Documents/SBM_Simulation/build/_deps/cppitertools-subbuild/cppitertools-populate-prefix"
  "/home/deb/Documents/SBM_Simulation/build/_deps/cppitertools-subbuild/cppitertools-populate-prefix/tmp"
  "/home/deb/Documents/SBM_Simulation/build/_deps/cppitertools-subbuild/cppitertools-populate-prefix/src/cppitertools-populate-stamp"
  "/home/deb/Documents/SBM_Simulation/build/_deps/cppitertools-subbuild/cppitertools-populate-prefix/src"
  "/home/deb/Documents/SBM_Simulation/build/_deps/cppitertools-subbuild/cppitertools-populate-prefix/src/cppitertools-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/deb/Documents/SBM_Simulation/build/_deps/cppitertools-subbuild/cppitertools-populate-prefix/src/cppitertools-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/deb/Documents/SBM_Simulation/build/_deps/cppitertools-subbuild/cppitertools-populate-prefix/src/cppitertools-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
