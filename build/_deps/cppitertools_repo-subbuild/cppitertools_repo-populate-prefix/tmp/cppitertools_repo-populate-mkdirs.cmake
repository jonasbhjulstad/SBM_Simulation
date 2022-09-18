# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/Users/jonas/Documents/Network_Robust_MPC/build/_deps/cppitertools_repo-src"
  "C:/Users/jonas/Documents/Network_Robust_MPC/build/_deps/cppitertools_repo-build"
  "C:/Users/jonas/Documents/Network_Robust_MPC/build/_deps/cppitertools_repo-subbuild/cppitertools_repo-populate-prefix"
  "C:/Users/jonas/Documents/Network_Robust_MPC/build/_deps/cppitertools_repo-subbuild/cppitertools_repo-populate-prefix/tmp"
  "C:/Users/jonas/Documents/Network_Robust_MPC/build/_deps/cppitertools_repo-subbuild/cppitertools_repo-populate-prefix/src/cppitertools_repo-populate-stamp"
  "C:/Users/jonas/Documents/Network_Robust_MPC/build/_deps/cppitertools_repo-subbuild/cppitertools_repo-populate-prefix/src"
  "C:/Users/jonas/Documents/Network_Robust_MPC/build/_deps/cppitertools_repo-subbuild/cppitertools_repo-populate-prefix/src/cppitertools_repo-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/jonas/Documents/Network_Robust_MPC/build/_deps/cppitertools_repo-subbuild/cppitertools_repo-populate-prefix/src/cppitertools_repo-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/jonas/Documents/Network_Robust_MPC/build/_deps/cppitertools_repo-subbuild/cppitertools_repo-populate-prefix/src/cppitertools_repo-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
