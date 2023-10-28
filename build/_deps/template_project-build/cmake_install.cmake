# Install script for directory: /home/deb/Documents/CMakeTemplateProject

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/deb/Documents/CMakeTemplateProject/build")
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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-build/test/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-build/source/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Template_Project_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Template_Project-1.0" TYPE DIRECTORY FILES "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-build/PackageProjectInclude/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Template_Project_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/Template_Project-1.0" TYPE STATIC_LIBRARY FILES "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-build/source/libTemplate_Project.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Template_Project_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Template_Project-1.0/Template_ProjectTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Template_Project-1.0/Template_ProjectTargets.cmake"
         "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-build/CMakeFiles/Export/fe8e4bb5329833d444a90fb7342a2a20/Template_ProjectTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Template_Project-1.0/Template_ProjectTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/Template_Project-1.0/Template_ProjectTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Template_Project-1.0" TYPE FILE FILES "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-build/CMakeFiles/Export/fe8e4bb5329833d444a90fb7342a2a20/Template_ProjectTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Template_Project-1.0" TYPE FILE FILES "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-build/CMakeFiles/Export/fe8e4bb5329833d444a90fb7342a2a20/Template_ProjectTargets-debug.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Template_Project_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/Template_Project-1.0" TYPE FILE FILES
    "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-build/Template_ProjectConfigVersion.cmake"
    "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-build/Template_ProjectConfig.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Template_Project_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Template_Project-1.0" TYPE DIRECTORY FILES "/home/deb/Documents/CMakeTemplateProject/include/")
endif()

