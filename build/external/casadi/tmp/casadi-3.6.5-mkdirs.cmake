# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/deb/Documents/SBM_Simulation/build/external/casadi/src/casadi-3.6.5"
  "/home/deb/Documents/SBM_Simulation/build/external/casadi/src/casadi-3.6.5-build"
  "/home/deb/Documents/SBM_Simulation/build/external/casadi"
  "/home/deb/Documents/SBM_Simulation/build/external/casadi/tmp"
  "/home/deb/Documents/SBM_Simulation/build/external/casadi/src/casadi-3.6.5-stamp"
  "/home/deb/Documents/SBM_Simulation/build/external/casadi/src"
  "/home/deb/Documents/SBM_Simulation/build/external/casadi/src/casadi-3.6.5-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/deb/Documents/SBM_Simulation/build/external/casadi/src/casadi-3.6.5-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/deb/Documents/SBM_Simulation/build/external/casadi/src/casadi-3.6.5-stamp${cfgdir}") # cfgdir has leading slash
endif()
