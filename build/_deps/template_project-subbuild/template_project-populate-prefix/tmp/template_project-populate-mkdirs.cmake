# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/deb/Documents/CMakeTemplateProject/test/.."
  "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-build"
  "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-subbuild/template_project-populate-prefix"
  "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-subbuild/template_project-populate-prefix/tmp"
  "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-subbuild/template_project-populate-prefix/src/template_project-populate-stamp"
  "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-subbuild/template_project-populate-prefix/src"
  "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-subbuild/template_project-populate-prefix/src/template_project-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-subbuild/template_project-populate-prefix/src/template_project-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/deb/Documents/CMakeTemplateProject/build/_deps/template_project-subbuild/template_project-populate-prefix/src/template_project-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
