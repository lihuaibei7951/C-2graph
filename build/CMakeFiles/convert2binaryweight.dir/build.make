# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lihuaibei/code/C-2graph

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lihuaibei/code/C-2graph/build

# Include any dependencies generated for this target.
include CMakeFiles/convert2binaryweight.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/convert2binaryweight.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/convert2binaryweight.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/convert2binaryweight.dir/flags.make

CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.o: CMakeFiles/convert2binaryweight.dir/flags.make
CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.o: /home/lihuaibei/code/C-2graph/utils/convert2binaryweight.cpp
CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.o: CMakeFiles/convert2binaryweight.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/lihuaibei/code/C-2graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.o"
	/usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.o -MF CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.o.d -o CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.o -c /home/lihuaibei/code/C-2graph/utils/convert2binaryweight.cpp

CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.i"
	/usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lihuaibei/code/C-2graph/utils/convert2binaryweight.cpp > CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.i

CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.s"
	/usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lihuaibei/code/C-2graph/utils/convert2binaryweight.cpp -o CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.s

# Object files for target convert2binaryweight
convert2binaryweight_OBJECTS = \
"CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.o"

# External object files for target convert2binaryweight
convert2binaryweight_EXTERNAL_OBJECTS =

/home/lihuaibei/code/C-2graph/bin/convert2binaryweight: CMakeFiles/convert2binaryweight.dir/utils/convert2binaryweight.cpp.o
/home/lihuaibei/code/C-2graph/bin/convert2binaryweight: CMakeFiles/convert2binaryweight.dir/build.make
/home/lihuaibei/code/C-2graph/bin/convert2binaryweight: CMakeFiles/convert2binaryweight.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/lihuaibei/code/C-2graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/lihuaibei/code/C-2graph/bin/convert2binaryweight"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convert2binaryweight.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/convert2binaryweight.dir/build: /home/lihuaibei/code/C-2graph/bin/convert2binaryweight
.PHONY : CMakeFiles/convert2binaryweight.dir/build

CMakeFiles/convert2binaryweight.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/convert2binaryweight.dir/cmake_clean.cmake
.PHONY : CMakeFiles/convert2binaryweight.dir/clean

CMakeFiles/convert2binaryweight.dir/depend:
	cd /home/lihuaibei/code/C-2graph/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lihuaibei/code/C-2graph /home/lihuaibei/code/C-2graph /home/lihuaibei/code/C-2graph/build /home/lihuaibei/code/C-2graph/build /home/lihuaibei/code/C-2graph/build/CMakeFiles/convert2binaryweight.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/convert2binaryweight.dir/depend

