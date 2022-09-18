# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.24

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\msys64\mingw64\bin\cmake.exe

# The command to remove a file.
RM = C:\msys64\mingw64\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\jonas\Documents\Network_Robust_MPC\Cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\jonas\Documents\Network_Robust_MPC\build

# Include any dependencies generated for this target.
include Binders/CMakeFiles/pyFROLS.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Binders/CMakeFiles/pyFROLS.dir/compiler_depend.make

# Include the progress variables for this target.
include Binders/CMakeFiles/pyFROLS.dir/progress.make

# Include the compile flags for this target's objects.
include Binders/CMakeFiles/pyFROLS.dir/flags.make

Binders/CMakeFiles/pyFROLS.dir/pyFROLS.cpp.obj: Binders/CMakeFiles/pyFROLS.dir/flags.make
Binders/CMakeFiles/pyFROLS.dir/pyFROLS.cpp.obj: Binders/CMakeFiles/pyFROLS.dir/includes_CXX.rsp
Binders/CMakeFiles/pyFROLS.dir/pyFROLS.cpp.obj: C:/Users/jonas/Documents/Network_Robust_MPC/Cpp/Binders/pyFROLS.cpp
Binders/CMakeFiles/pyFROLS.dir/pyFROLS.cpp.obj: Binders/CMakeFiles/pyFROLS.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\jonas\Documents\Network_Robust_MPC\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Binders/CMakeFiles/pyFROLS.dir/pyFROLS.cpp.obj"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\Binders && C:\msys64\mingw64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Binders/CMakeFiles/pyFROLS.dir/pyFROLS.cpp.obj -MF CMakeFiles\pyFROLS.dir\pyFROLS.cpp.obj.d -o CMakeFiles\pyFROLS.dir\pyFROLS.cpp.obj -c C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\Binders\pyFROLS.cpp

Binders/CMakeFiles/pyFROLS.dir/pyFROLS.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pyFROLS.dir/pyFROLS.cpp.i"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\Binders && C:\msys64\mingw64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\Binders\pyFROLS.cpp > CMakeFiles\pyFROLS.dir\pyFROLS.cpp.i

Binders/CMakeFiles/pyFROLS.dir/pyFROLS.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pyFROLS.dir/pyFROLS.cpp.s"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\Binders && C:\msys64\mingw64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\Binders\pyFROLS.cpp -o CMakeFiles\pyFROLS.dir\pyFROLS.cpp.s

# Object files for target pyFROLS
pyFROLS_OBJECTS = \
"CMakeFiles/pyFROLS.dir/pyFROLS.cpp.obj"

# External object files for target pyFROLS
pyFROLS_EXTERNAL_OBJECTS =

Binders/pyFROLS.cp310-win_amd64.pyd: Binders/CMakeFiles/pyFROLS.dir/pyFROLS.cpp.obj
Binders/pyFROLS.cp310-win_amd64.pyd: Binders/CMakeFiles/pyFROLS.dir/build.make
Binders/pyFROLS.cp310-win_amd64.pyd: static/libFROLS_DataFrame.a
Binders/pyFROLS.cp310-win_amd64.pyd: static/libFROLS_Quantiles.a
Binders/pyFROLS.cp310-win_amd64.pyd: static/libFROLS_Eigen.a
Binders/pyFROLS.cp310-win_amd64.pyd: static/Features/libFROLS_Features.a
Binders/pyFROLS.cp310-win_amd64.pyd: static/libFROLS_Math.a
Binders/pyFROLS.cp310-win_amd64.pyd: static/Algorithm/libFROLS_Algorithm.a
Binders/pyFROLS.cp310-win_amd64.pyd: static/Algorithm/libFROLS_Algorithm.a
Binders/pyFROLS.cp310-win_amd64.pyd: C:/Users/jonas/anaconda3/envs/Network_MPC/libs/python310.lib
Binders/pyFROLS.cp310-win_amd64.pyd: C:/msys64/mingw64/lib/libomp.dll.a
Binders/pyFROLS.cp310-win_amd64.pyd: static/Features/libFROLS_Features.a
Binders/pyFROLS.cp310-win_amd64.pyd: static/libFROLS_Eigen.a
Binders/pyFROLS.cp310-win_amd64.pyd: static/libFROLS_DataFrame.a
Binders/pyFROLS.cp310-win_amd64.pyd: static/libFROLS_Math.a
Binders/pyFROLS.cp310-win_amd64.pyd: Binders/CMakeFiles/pyFROLS.dir/linklibs.rsp
Binders/pyFROLS.cp310-win_amd64.pyd: Binders/CMakeFiles/pyFROLS.dir/objects1.rsp
Binders/pyFROLS.cp310-win_amd64.pyd: Binders/CMakeFiles/pyFROLS.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\jonas\Documents\Network_Robust_MPC\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module pyFROLS.cp310-win_amd64.pyd"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\Binders && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\pyFROLS.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Binders/CMakeFiles/pyFROLS.dir/build: Binders/pyFROLS.cp310-win_amd64.pyd
.PHONY : Binders/CMakeFiles/pyFROLS.dir/build

Binders/CMakeFiles/pyFROLS.dir/clean:
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\Binders && $(CMAKE_COMMAND) -P CMakeFiles\pyFROLS.dir\cmake_clean.cmake
.PHONY : Binders/CMakeFiles/pyFROLS.dir/clean

Binders/CMakeFiles/pyFROLS.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\jonas\Documents\Network_Robust_MPC\Cpp C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\Binders C:\Users\jonas\Documents\Network_Robust_MPC\build C:\Users\jonas\Documents\Network_Robust_MPC\build\Binders C:\Users\jonas\Documents\Network_Robust_MPC\build\Binders\CMakeFiles\pyFROLS.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : Binders/CMakeFiles/pyFROLS.dir/depend

