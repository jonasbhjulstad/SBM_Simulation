# Install script for directory: /home/deb/.CPM/buffer_routines/6c9bbf61e9b2714afe7c4dfce64278d05f2787a2

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/deb/.CPM/buffer_routines/6c9bbf61e9b2714afe7c4dfce64278d05f2787a2/build")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Buffer_Routines_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Buffer_Routines-1.0" TYPE DIRECTORY FILES "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-build/PackageProjectInclude/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Buffer_Routines_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/Buffer_Routines-1.0" TYPE STATIC_LIBRARY FILES "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-build/libBuffer_Routines.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Buffer_Routines_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Buffer_Routines-1.0/Buffer_RoutinesTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Buffer_Routines-1.0/Buffer_RoutinesTargets.cmake"
         "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-build/CMakeFiles/Export/e2deaa168441c3bbdba43796eda22c46/Buffer_RoutinesTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Buffer_Routines-1.0/Buffer_RoutinesTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Buffer_Routines-1.0/Buffer_RoutinesTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Buffer_Routines-1.0" TYPE FILE FILES "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-build/CMakeFiles/Export/e2deaa168441c3bbdba43796eda22c46/Buffer_RoutinesTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Buffer_Routines-1.0" TYPE FILE FILES "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-build/CMakeFiles/Export/e2deaa168441c3bbdba43796eda22c46/Buffer_RoutinesTargets-debug.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Buffer_Routines_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Buffer_Routines-1.0" TYPE FILE FILES
    "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-build/Buffer_RoutinesConfigVersion.cmake"
    "/home/deb/Documents/SBM_Graph/build/test/_deps/buffer_routines-build/Buffer_RoutinesConfig.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Buffer_Routines_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Buffer_Routines-1.0" TYPE DIRECTORY FILES "/home/deb/.CPM/buffer_routines/6c9bbf61e9b2714afe7c4dfce64278d05f2787a2/include/")
endif()

