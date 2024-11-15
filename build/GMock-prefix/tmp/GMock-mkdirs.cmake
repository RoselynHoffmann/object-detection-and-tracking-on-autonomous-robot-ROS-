# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/usr/src/googletest/googlemock")
  file(MAKE_DIRECTORY "/usr/src/googletest/googlemock")
endif()
file(MAKE_DIRECTORY
  "/home/rose/ros_env/build/gmock"
  "/home/rose/ros_env/build/GMock-prefix"
  "/home/rose/ros_env/build/GMock-prefix/tmp"
  "/home/rose/ros_env/build/GMock-prefix/src/GMock-stamp"
  "/home/rose/ros_env/build/GMock-prefix/src"
  "/home/rose/ros_env/build/GMock-prefix/src/GMock-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/rose/ros_env/build/GMock-prefix/src/GMock-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/rose/ros_env/build/GMock-prefix/src/GMock-stamp${cfgdir}") # cfgdir has leading slash
endif()
