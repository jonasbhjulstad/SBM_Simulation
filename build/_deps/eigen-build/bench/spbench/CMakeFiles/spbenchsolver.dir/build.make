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
include _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/flags.make

_deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o: _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/flags.make
_deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o: _deps/eigen-src/bench/spbench/spbenchsolver.cpp
_deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o: _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/man/Documents/Sycl_Graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/bench/spbench && /opt/intel/oneapi/compiler/latest/linux/bin-llvm/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o -MF CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o.d -o CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o -c /home/man/Documents/Sycl_Graph/build/_deps/eigen-src/bench/spbench/spbenchsolver.cpp

_deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.i"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/bench/spbench && /opt/intel/oneapi/compiler/latest/linux/bin-llvm/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/man/Documents/Sycl_Graph/build/_deps/eigen-src/bench/spbench/spbenchsolver.cpp > CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.i

_deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.s"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/bench/spbench && /opt/intel/oneapi/compiler/latest/linux/bin-llvm/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/man/Documents/Sycl_Graph/build/_deps/eigen-src/bench/spbench/spbenchsolver.cpp -o CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.s

# Object files for target spbenchsolver
spbenchsolver_OBJECTS = \
"CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o"

# External object files for target spbenchsolver
spbenchsolver_EXTERNAL_OBJECTS =

_deps/eigen-build/bench/spbench/spbenchsolver: _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/spbenchsolver.cpp.o
_deps/eigen-build/bench/spbench/spbenchsolver: _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/build.make
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libcholmod.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libamd.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libcolamd.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libcamd.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libccolamd.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libmetis.so
_deps/eigen-build/bench/spbench/spbenchsolver: _deps/eigen-build/blas/libeigen_blas_static.a
_deps/eigen-build/bench/spbench/spbenchsolver: _deps/eigen-build/lapack/libeigen_lapack_static.a
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libumfpack.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libcolamd.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libamd.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libcholmod.so
_deps/eigen-build/bench/spbench/spbenchsolver: _deps/eigen-build/blas/libeigen_blas_static.a
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libmetis.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/librt.a
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libcamd.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libccolamd.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/libumfpack.so
_deps/eigen-build/bench/spbench/spbenchsolver: /usr/lib/librt.a
_deps/eigen-build/bench/spbench/spbenchsolver: _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/man/Documents/Sycl_Graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable spbenchsolver"
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/bench/spbench && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/spbenchsolver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/build: _deps/eigen-build/bench/spbench/spbenchsolver
.PHONY : _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/build

_deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/clean:
	cd /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/bench/spbench && $(CMAKE_COMMAND) -P CMakeFiles/spbenchsolver.dir/cmake_clean.cmake
.PHONY : _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/clean

_deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/depend:
	cd /home/man/Documents/Sycl_Graph/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/man/Documents/Sycl_Graph /home/man/Documents/Sycl_Graph/build/_deps/eigen-src/bench/spbench /home/man/Documents/Sycl_Graph/build /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/bench/spbench /home/man/Documents/Sycl_Graph/build/_deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/eigen-build/bench/spbench/CMakeFiles/spbenchsolver.dir/depend
