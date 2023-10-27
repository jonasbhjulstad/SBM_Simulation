# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp/buffer_routines-populate-gitclone-lastrun.txt" AND EXISTS "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp/buffer_routines-populate-gitinfo.txt" AND
  "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp/buffer_routines-populate-gitclone-lastrun.txt" IS_NEWER_THAN "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp/buffer_routines-populate-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp/buffer_routines-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/deb/.CPM/buffer_routines/6c9bbf61e9b2714afe7c4dfce64278d05f2787a2"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/deb/.CPM/buffer_routines/6c9bbf61e9b2714afe7c4dfce64278d05f2787a2'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"
            clone --no-checkout --depth 1 --no-single-branch --config "advice.detachedHead=false" "https://github.com/jonasbhjulstad/Sycl_Buffer_Routines.git" "6c9bbf61e9b2714afe7c4dfce64278d05f2787a2"
    WORKING_DIRECTORY "/home/deb/.CPM/buffer_routines"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/jonasbhjulstad/Sycl_Buffer_Routines.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"
          checkout "master" --
  WORKING_DIRECTORY "/home/deb/.CPM/buffer_routines/6c9bbf61e9b2714afe7c4dfce64278d05f2787a2"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'master'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/home/deb/.CPM/buffer_routines/6c9bbf61e9b2714afe7c4dfce64278d05f2787a2"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/deb/.CPM/buffer_routines/6c9bbf61e9b2714afe7c4dfce64278d05f2787a2'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp/buffer_routines-populate-gitinfo.txt" "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp/buffer_routines-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-subbuild/buffer_routines-populate-prefix/src/buffer_routines-populate-stamp/buffer_routines-populate-gitclone-lastrun.txt'")
endif()
