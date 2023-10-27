# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/deb/.CPM/buffer_routines/6c9bbf61e9b2714afe7c4dfce64278d05f2787a2"
  "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-build"
  "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix"
  "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/tmp"
  "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp"
  "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src"
  "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
