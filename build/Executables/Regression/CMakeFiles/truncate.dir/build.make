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
include Executables/Regression/CMakeFiles/truncate.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Executables/Regression/CMakeFiles/truncate.dir/compiler_depend.make

# Include the progress variables for this target.
include Executables/Regression/CMakeFiles/truncate.dir/progress.make

# Include the compile flags for this target's objects.
include Executables/Regression/CMakeFiles/truncate.dir/flags.make

Executables/Regression/CMakeFiles/truncate.dir/truncate.cpp.obj: Executables/Regression/CMakeFiles/truncate.dir/flags.make
Executables/Regression/CMakeFiles/truncate.dir/truncate.cpp.obj: Executables/Regression/CMakeFiles/truncate.dir/includes_CXX.rsp
Executables/Regression/CMakeFiles/truncate.dir/truncate.cpp.obj: C:/Users/jonas/Documents/Network_Robust_MPC/Cpp/Executables/Regression/truncate.cpp
Executables/Regression/CMakeFiles/truncate.dir/truncate.cpp.obj: Executables/Regression/CMakeFiles/truncate.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\jonas\Documents\Network_Robust_MPC\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Executables/Regression/CMakeFiles/truncate.dir/truncate.cpp.obj"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\Executables\Regression && C:\msys64\mingw64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Executables/Regression/CMakeFiles/truncate.dir/truncate.cpp.obj -MF CMakeFiles\truncate.dir\truncate.cpp.obj.d -o CMakeFiles\truncate.dir\truncate.cpp.obj -c C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\Executables\Regression\truncate.cpp

Executables/Regression/CMakeFiles/truncate.dir/truncate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/truncate.dir/truncate.cpp.i"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\Executables\Regression && C:\msys64\mingw64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\Executables\Regression\truncate.cpp > CMakeFiles\truncate.dir\truncate.cpp.i

Executables/Regression/CMakeFiles/truncate.dir/truncate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/truncate.dir/truncate.cpp.s"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\Executables\Regression && C:\msys64\mingw64\bin\clang++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\Executables\Regression\truncate.cpp -o CMakeFiles\truncate.dir\truncate.cpp.s

# Object files for target truncate
truncate_OBJECTS = \
"CMakeFiles/truncate.dir/truncate.cpp.obj"

# External object files for target truncate
truncate_EXTERNAL_OBJECTS =

Executables/Regression/truncate.exe: Executables/Regression/CMakeFiles/truncate.dir/truncate.cpp.obj
Executables/Regression/truncate.exe: Executables/Regression/CMakeFiles/truncate.dir/build.make
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libigraph.dll.a
Executables/Regression/truncate.exe: static/libFROLS_DataFrame.a
Executables/Regression/truncate.exe: static/libFROLS_Quantiles.a
Executables/Regression/truncate.exe: static/libFROLS_Eigen.a
Executables/Regression/truncate.exe: static/Features/libFROLS_Features.a
Executables/Regression/truncate.exe: static/libFROLS_Math.a
Executables/Regression/truncate.exe: static/Algorithm/libFROLS_Algorithm.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libsundials_cvode.dll.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libsundials_nvecserial.dll.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libarpack.dll.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libopenblas.dll.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libcxsparse.dll.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libglpk.dll.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libgmp.dll.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libopenblas.dll.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libcxsparse.dll.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libglpk.dll.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libgmp.dll.a
Executables/Regression/truncate.exe: C:/msys64/mingw64/lib/libxml2.dll.a
Executables/Regression/truncate.exe: static/Features/libFROLS_Features.a
Executables/Regression/truncate.exe: static/Algorithm/libFROLS_Algorithm.a
Executables/Regression/truncate.exe: static/libFROLS_Eigen.a
Executables/Regression/truncate.exe: static/libFROLS_DataFrame.a
Executables/Regression/truncate.exe: static/libFROLS_Math.a
Executables/Regression/truncate.exe: Executables/Regression/CMakeFiles/truncate.dir/linklibs.rsp
Executables/Regression/truncate.exe: Executables/Regression/CMakeFiles/truncate.dir/objects1.rsp
Executables/Regression/truncate.exe: Executables/Regression/CMakeFiles/truncate.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\jonas\Documents\Network_Robust_MPC\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable truncate.exe"
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\Executables\Regression && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\truncate.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Executables/Regression/CMakeFiles/truncate.dir/build: Executables/Regression/truncate.exe
.PHONY : Executables/Regression/CMakeFiles/truncate.dir/build

Executables/Regression/CMakeFiles/truncate.dir/clean:
	cd /d C:\Users\jonas\Documents\Network_Robust_MPC\build\Executables\Regression && $(CMAKE_COMMAND) -P CMakeFiles\truncate.dir\cmake_clean.cmake
.PHONY : Executables/Regression/CMakeFiles/truncate.dir/clean

Executables/Regression/CMakeFiles/truncate.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\jonas\Documents\Network_Robust_MPC\Cpp C:\Users\jonas\Documents\Network_Robust_MPC\Cpp\Executables\Regression C:\Users\jonas\Documents\Network_Robust_MPC\build C:\Users\jonas\Documents\Network_Robust_MPC\build\Executables\Regression C:\Users\jonas\Documents\Network_Robust_MPC\build\Executables\Regression\CMakeFiles\truncate.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : Executables/Regression/CMakeFiles/truncate.dir/depend

