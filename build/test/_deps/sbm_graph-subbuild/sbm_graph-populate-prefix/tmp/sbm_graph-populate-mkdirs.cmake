# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/deb/Documents/SBM_Graph/test/.."
  "/home/deb/Documents/SBM_Graph/build/test/_deps/sbm_graph-build"
  "/home/deb/Documents/SBM_Graph/build/test/_deps/sbm_graph-subbuild/sbm_graph-populate-prefix"
  "/home/deb/Documents/SBM_Graph/build/test/_deps/sbm_graph-subbuild/sbm_graph-populate-prefix/tmp"
  "/home/deb/Documents/SBM_Graph/build/test/_deps/sbm_graph-subbuild/sbm_graph-populate-prefix/src/sbm_graph-populate-stamp"
  "/home/deb/Documents/SBM_Graph/build/test/_deps/sbm_graph-subbuild/sbm_graph-populate-prefix/src"
  "/home/deb/Documents/SBM_Graph/build/test/_deps/sbm_graph-subbuild/sbm_graph-populate-prefix/src/sbm_graph-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/deb/Documents/SBM_Graph/build/test/_deps/sbm_graph-subbuild/sbm_graph-populate-prefix/src/sbm_graph-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/deb/Documents/SBM_Graph/build/test/_deps/sbm_graph-subbuild/sbm_graph-populate-prefix/src/sbm_graph-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
