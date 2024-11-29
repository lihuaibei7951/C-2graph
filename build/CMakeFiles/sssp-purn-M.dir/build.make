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
include CMakeFiles/sssp-purn-M.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/sssp-purn-M.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sssp-purn-M.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sssp-purn-M.dir/flags.make

CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.o: CMakeFiles/sssp-purn-M.dir/flags.make
CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.o: /home/lihuaibei/code/C-2graph/src/sssp/purn-perk/SSSP.cpp
CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.o: CMakeFiles/sssp-purn-M.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/lihuaibei/code/C-2graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.o"
	/usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.o -MF CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.o.d -o CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.o -c /home/lihuaibei/code/C-2graph/src/sssp/purn-perk/SSSP.cpp

CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.i"
	/usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lihuaibei/code/C-2graph/src/sssp/purn-perk/SSSP.cpp > CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.i

CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.s"
	/usr/local/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lihuaibei/code/C-2graph/src/sssp/purn-perk/SSSP.cpp -o CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.s

CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o: CMakeFiles/sssp-purn-M.dir/flags.make
CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o: CMakeFiles/sssp-purn-M.dir/includes_CUDA.rsp
CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o: /home/lihuaibei/code/C-2graph/src/sssp/purn-perk/Main.cu
CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o: CMakeFiles/sssp-purn-M.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/lihuaibei/code/C-2graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o -MF CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o.d -x cu -rdc=true -c /home/lihuaibei/code/C-2graph/src/sssp/purn-perk/Main.cu -o CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o

CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target sssp-purn-M
sssp__purn__M_OBJECTS = \
"CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.o" \
"CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o"

# External object files for target sssp-purn-M
sssp__purn__M_EXTERNAL_OBJECTS =

CMakeFiles/sssp-purn-M.dir/cmake_device_link.o: CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.o
CMakeFiles/sssp-purn-M.dir/cmake_device_link.o: CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o
CMakeFiles/sssp-purn-M.dir/cmake_device_link.o: CMakeFiles/sssp-purn-M.dir/build.make
CMakeFiles/sssp-purn-M.dir/cmake_device_link.o: /usr/local/cuda-11.7/lib64/libcudart_static.a
CMakeFiles/sssp-purn-M.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/librt.so
CMakeFiles/sssp-purn-M.dir/cmake_device_link.o: CMakeFiles/sssp-purn-M.dir/deviceLinkLibs.rsp
CMakeFiles/sssp-purn-M.dir/cmake_device_link.o: CMakeFiles/sssp-purn-M.dir/deviceObjects1.rsp
CMakeFiles/sssp-purn-M.dir/cmake_device_link.o: CMakeFiles/sssp-purn-M.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/lihuaibei/code/C-2graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/sssp-purn-M.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sssp-purn-M.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sssp-purn-M.dir/build: CMakeFiles/sssp-purn-M.dir/cmake_device_link.o
.PHONY : CMakeFiles/sssp-purn-M.dir/build

# Object files for target sssp-purn-M
sssp__purn__M_OBJECTS = \
"CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.o" \
"CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o"

# External object files for target sssp-purn-M
sssp__purn__M_EXTERNAL_OBJECTS =

/home/lihuaibei/code/C-2graph/bin/sssp-purn-M: CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/SSSP.cpp.o
/home/lihuaibei/code/C-2graph/bin/sssp-purn-M: CMakeFiles/sssp-purn-M.dir/src/sssp/purn-perk/Main.cu.o
/home/lihuaibei/code/C-2graph/bin/sssp-purn-M: CMakeFiles/sssp-purn-M.dir/build.make
/home/lihuaibei/code/C-2graph/bin/sssp-purn-M: /usr/local/cuda-11.7/lib64/libcudart_static.a
/home/lihuaibei/code/C-2graph/bin/sssp-purn-M: /usr/lib/x86_64-linux-gnu/librt.so
/home/lihuaibei/code/C-2graph/bin/sssp-purn-M: CMakeFiles/sssp-purn-M.dir/cmake_device_link.o
/home/lihuaibei/code/C-2graph/bin/sssp-purn-M: CMakeFiles/sssp-purn-M.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/lihuaibei/code/C-2graph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable /home/lihuaibei/code/C-2graph/bin/sssp-purn-M"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sssp-purn-M.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sssp-purn-M.dir/build: /home/lihuaibei/code/C-2graph/bin/sssp-purn-M
.PHONY : CMakeFiles/sssp-purn-M.dir/build

CMakeFiles/sssp-purn-M.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sssp-purn-M.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sssp-purn-M.dir/clean

CMakeFiles/sssp-purn-M.dir/depend:
	cd /home/lihuaibei/code/C-2graph/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lihuaibei/code/C-2graph /home/lihuaibei/code/C-2graph /home/lihuaibei/code/C-2graph/build /home/lihuaibei/code/C-2graph/build /home/lihuaibei/code/C-2graph/build/CMakeFiles/sssp-purn-M.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/sssp-purn-M.dir/depend

