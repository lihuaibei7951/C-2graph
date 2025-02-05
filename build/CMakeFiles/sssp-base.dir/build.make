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
include CMakeFiles/sssp-base.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/sssp-base.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sssp-base.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sssp-base.dir/flags.make

CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.o: CMakeFiles/sssp-base.dir/flags.make
CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.o: CMakeFiles/sssp-base.dir/includes_CUDA.rsp
CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.o: /home/lihuaibei/code/C-2graph/src/sssp/base/Main.cu
CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.o: CMakeFiles/sssp-base.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/lihuaibei/code/C-2graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.o -MF CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.o.d -x cu -c /home/lihuaibei/code/C-2graph/src/sssp/base/Main.cu -o CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.o

CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target sssp-base
sssp__base_OBJECTS = \
"CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.o"

# External object files for target sssp-base
sssp__base_EXTERNAL_OBJECTS =

/home/lihuaibei/code/C-2graph/bin/sssp-base: CMakeFiles/sssp-base.dir/src/sssp/base/Main.cu.o
/home/lihuaibei/code/C-2graph/bin/sssp-base: CMakeFiles/sssp-base.dir/build.make
/home/lihuaibei/code/C-2graph/bin/sssp-base: CMakeFiles/sssp-base.dir/linkLibs.rsp
/home/lihuaibei/code/C-2graph/bin/sssp-base: CMakeFiles/sssp-base.dir/objects1.rsp
/home/lihuaibei/code/C-2graph/bin/sssp-base: CMakeFiles/sssp-base.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/lihuaibei/code/C-2graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable /home/lihuaibei/code/C-2graph/bin/sssp-base"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sssp-base.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sssp-base.dir/build: /home/lihuaibei/code/C-2graph/bin/sssp-base
.PHONY : CMakeFiles/sssp-base.dir/build

CMakeFiles/sssp-base.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sssp-base.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sssp-base.dir/clean

CMakeFiles/sssp-base.dir/depend:
	cd /home/lihuaibei/code/C-2graph/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lihuaibei/code/C-2graph /home/lihuaibei/code/C-2graph /home/lihuaibei/code/C-2graph/build /home/lihuaibei/code/C-2graph/build /home/lihuaibei/code/C-2graph/build/CMakeFiles/sssp-base.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/sssp-base.dir/depend

