# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/cmake-build-release/_deps/zipiterator_repo-src"
  "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/cmake-build-release/_deps/zipiterator_repo-build"
  "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/cmake-build-release/_deps/zipiterator_repo-subbuild/zipiterator_repo-populate-prefix"
  "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/cmake-build-release/_deps/zipiterator_repo-subbuild/zipiterator_repo-populate-prefix/tmp"
  "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/cmake-build-release/_deps/zipiterator_repo-subbuild/zipiterator_repo-populate-prefix/src/zipiterator_repo-populate-stamp"
  "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/cmake-build-release/_deps/zipiterator_repo-subbuild/zipiterator_repo-populate-prefix/src"
  "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/cmake-build-release/_deps/zipiterator_repo-subbuild/zipiterator_repo-populate-prefix/src/zipiterator_repo-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/cmake-build-release/_deps/zipiterator_repo-subbuild/zipiterator_repo-populate-prefix/src/zipiterator_repo-populate-stamp/${subDir}")
endforeach()
