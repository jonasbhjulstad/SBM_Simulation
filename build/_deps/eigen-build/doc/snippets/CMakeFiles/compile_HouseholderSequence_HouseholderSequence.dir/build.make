# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/man/Documents/Sycl_Graph

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/man/Documents/Sycl_Graph/build

# Include any dependencies generated for this target.
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/flags.make

_deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.o: _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/flags.make
_deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.o: _deps/eigen-build/doc/snippets/compile_HouseholderSequence_HouseholderSequence.cpp
_deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.o: _deps/eigen-src/doc/snippets/HouseholderSequence_HouseholderSequence.cpp
_deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.o: _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/man/Documents/Sycl_Graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.o"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets && /opt/intel/oneapi/compiler/latest/linux/bin-llvm/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.o -MF CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.o.d -o CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.o -c /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets/compile_HouseholderSequence_HouseholderSequence.cpp

_deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.i"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets && /opt/intel/oneapi/compiler/latest/linux/bin-llvm/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets/compile_HouseholderSequence_HouseholderSequence.cpp > CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.i

_deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.s"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets && /opt/intel/oneapi/compiler/latest/linux/bin-llvm/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets/compile_HouseholderSequence_HouseholderSequence.cpp -o CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.s

# Object files for target compile_HouseholderSequence_HouseholderSequence
compile_HouseholderSequence_HouseholderSequence_OBJECTS = \
"CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.o"

# External object files for target compile_HouseholderSequence_HouseholderSequence
compile_HouseholderSequence_HouseholderSequence_EXTERNAL_OBJECTS =

_deps/eigen-build/doc/snippets/compile_HouseholderSequence_HouseholderSequence: _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/compile_HouseholderSequence_HouseholderSequence.cpp.o
_deps/eigen-build/doc/snippets/compile_HouseholderSequence_HouseholderSequence: _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/build.make
_deps/eigen-build/doc/snippets/compile_HouseholderSequence_HouseholderSequence: _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/man/Documents/Sycl_Graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable compile_HouseholderSequence_HouseholderSequence"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/link.txt --verbose=$(VERBOSE)
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets && ./compile_HouseholderSequence_HouseholderSequence >/home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets/HouseholderSequence_HouseholderSequence.out

# Rule to build all files generated by this target.
_deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/build: _deps/eigen-build/doc/snippets/compile_HouseholderSequence_HouseholderSequence
.PHONY : _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/build

_deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/clean:
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets && $(CMAKE_COMMAND) -P CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/cmake_clean.cmake
.PHONY : _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/clean

_deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/depend:
	cd /home/man/Documents/Sycl_Graph/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/man/Documents/Sycl_Graph /home/man/Documents/Sycl_Graph/build/_deps/eigen-src/doc/snippets /home/man/Documents/Sycl_Graph/build /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/eigen-build/doc/snippets/CMakeFiles/compile_HouseholderSequence_HouseholderSequence.dir/depend
