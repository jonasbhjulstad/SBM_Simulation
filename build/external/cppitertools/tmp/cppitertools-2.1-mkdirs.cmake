# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/deb/Documents/SBM_Simulation/build/external/cppitertools/src/cppitertools-2.1"
  "/home/deb/Documents/SBM_Simulation/build/external/cppitertools/src/cppitertools-2.1-build"
  "/home/deb/Documents/SBM_Simulation/build/external/cppitertools"
  "/home/deb/Documents/SBM_Simulation/build/external/cppitertools/tmp"
  "/home/deb/Documents/SBM_Simulation/build/external/cppitertools/src/cppitertools-2.1-stamp"
  "/home/deb/Documents/SBM_Simulation/build/external/cppitertools/src"
  "/home/deb/Documents/SBM_Simulation/build/external/cppitertools/src/cppitertools-2.1-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/deb/Documents/SBM_Simulation/build/external/cppitertools/src/cppitertools-2.1-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/deb/Documents/SBM_Simulation/build/external/cppitertools/src/cppitertools-2.1-stamp${cfgdir}") # cfgdir has leading slash
endif()
