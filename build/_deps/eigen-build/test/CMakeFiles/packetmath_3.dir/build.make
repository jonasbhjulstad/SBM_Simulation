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
include _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/flags.make

_deps/eigen-build/test/CMakeFiles/packetmath_3.dir/packetmath.cpp.o: _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/flags.make
_deps/eigen-build/test/CMakeFiles/packetmath_3.dir/packetmath.cpp.o: _deps/eigen-src/test/packetmath.cpp
_deps/eigen-build/test/CMakeFiles/packetmath_3.dir/packetmath.cpp.o: _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/man/Documents/Sycl_Graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/packetmath.cpp.o"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/test && /opt/intel/oneapi/compiler/latest/linux/bin-llvm/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/packetmath.cpp.o -MF CMakeFiles/packetmath_3.dir/packetmath.cpp.o.d -o CMakeFiles/packetmath_3.dir/packetmath.cpp.o -c /home/man/Documents/Sycl_Graph/build/_deps/eigen-src/test/packetmath.cpp

_deps/eigen-build/test/CMakeFiles/packetmath_3.dir/packetmath.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/packetmath_3.dir/packetmath.cpp.i"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/test && /opt/intel/oneapi/compiler/latest/linux/bin-llvm/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/man/Documents/Sycl_Graph/build/_deps/eigen-src/test/packetmath.cpp > CMakeFiles/packetmath_3.dir/packetmath.cpp.i

_deps/eigen-build/test/CMakeFiles/packetmath_3.dir/packetmath.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/packetmath_3.dir/packetmath.cpp.s"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/test && /opt/intel/oneapi/compiler/latest/linux/bin-llvm/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/man/Documents/Sycl_Graph/build/_deps/eigen-src/test/packetmath.cpp -o CMakeFiles/packetmath_3.dir/packetmath.cpp.s

# Object files for target packetmath_3
packetmath_3_OBJECTS = \
"CMakeFiles/packetmath_3.dir/packetmath.cpp.o"

# External object files for target packetmath_3
packetmath_3_EXTERNAL_OBJECTS =

_deps/eigen-build/test/packetmath_3: _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/packetmath.cpp.o
_deps/eigen-build/test/packetmath_3: _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/build.make
_deps/eigen-build/test/packetmath_3: _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/man/Documents/Sycl_Graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable packetmath_3"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/packetmath_3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/eigen-build/test/CMakeFiles/packetmath_3.dir/build: _deps/eigen-build/test/packetmath_3
.PHONY : _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/build

_deps/eigen-build/test/CMakeFiles/packetmath_3.dir/clean:
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/test && $(CMAKE_COMMAND) -P CMakeFiles/packetmath_3.dir/cmake_clean.cmake
.PHONY : _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/clean

_deps/eigen-build/test/CMakeFiles/packetmath_3.dir/depend:
	cd /home/man/Documents/Sycl_Graph/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/man/Documents/Sycl_Graph /home/man/Documents/Sycl_Graph/build/_deps/eigen-src/test /home/man/Documents/Sycl_Graph/build /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/test /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/test/CMakeFiles/packetmath_3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/eigen-build/test/CMakeFiles/packetmath_3.dir/depend
