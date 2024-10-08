# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_SOURCE_DIR = /home/rose/ros2_ws/src/rosbag2/rosbag2_storage

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rose/ros2_ws/build/rosbag2_storage

# Include any dependencies generated for this target.
include CMakeFiles/test_storage_factory.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_storage_factory.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_storage_factory.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_storage_factory.dir/flags.make

CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.o: CMakeFiles/test_storage_factory.dir/flags.make
CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.o: /home/rose/ros2_ws/src/rosbag2/rosbag2_storage/test/rosbag2_storage/test_storage_factory.cpp
CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.o: CMakeFiles/test_storage_factory.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rose/ros2_ws/build/rosbag2_storage/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.o -MF CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.o.d -o CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.o -c /home/rose/ros2_ws/src/rosbag2/rosbag2_storage/test/rosbag2_storage/test_storage_factory.cpp

CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rose/ros2_ws/src/rosbag2/rosbag2_storage/test/rosbag2_storage/test_storage_factory.cpp > CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.i

CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rose/ros2_ws/src/rosbag2/rosbag2_storage/test/rosbag2_storage/test_storage_factory.cpp -o CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.s

# Object files for target test_storage_factory
test_storage_factory_OBJECTS = \
"CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.o"

# External object files for target test_storage_factory
test_storage_factory_EXTERNAL_OBJECTS =

test_storage_factory: CMakeFiles/test_storage_factory.dir/test/rosbag2_storage/test_storage_factory.cpp.o
test_storage_factory: CMakeFiles/test_storage_factory.dir/build.make
test_storage_factory: gmock/libgmock_main.a
test_storage_factory: gmock/libgmock.a
test_storage_factory: librosbag2_storage.so
test_storage_factory: /opt/ros/humble/lib/libament_index_cpp.so
test_storage_factory: /opt/ros/humble/lib/libclass_loader.so
test_storage_factory: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.1.0
test_storage_factory: /usr/lib/x86_64-linux-gnu/libtinyxml2.so
test_storage_factory: /opt/ros/humble/lib/librcpputils.so
test_storage_factory: /opt/ros/humble/lib/librcutils.so
test_storage_factory: /usr/lib/x86_64-linux-gnu/libyaml-cpp.so.0.7.0
test_storage_factory: CMakeFiles/test_storage_factory.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rose/ros2_ws/build/rosbag2_storage/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_storage_factory"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_storage_factory.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_storage_factory.dir/build: test_storage_factory
.PHONY : CMakeFiles/test_storage_factory.dir/build

CMakeFiles/test_storage_factory.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_storage_factory.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_storage_factory.dir/clean

CMakeFiles/test_storage_factory.dir/depend:
	cd /home/rose/ros2_ws/build/rosbag2_storage && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rose/ros2_ws/src/rosbag2/rosbag2_storage /home/rose/ros2_ws/src/rosbag2/rosbag2_storage /home/rose/ros2_ws/build/rosbag2_storage /home/rose/ros2_ws/build/rosbag2_storage /home/rose/ros2_ws/build/rosbag2_storage/CMakeFiles/test_storage_factory.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_storage_factory.dir/depend

