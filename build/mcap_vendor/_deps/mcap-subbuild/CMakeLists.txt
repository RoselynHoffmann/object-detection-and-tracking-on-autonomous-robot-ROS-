# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.22.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(mcap-populate NONE)



include(ExternalProject)
ExternalProject_Add(mcap-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/foxglove/mcap/archive/refs/tags/releases/cpp/v0.8.0.tar.gz" "URL_HASH" "SHA1=b44637791da2c9c1cec61a3ba6994f1ef63a228c"
                    SOURCE_DIR          "/home/rose/ros2_ws/build/mcap_vendor/_deps/mcap-src"
                    BINARY_DIR          "/home/rose/ros2_ws/build/mcap_vendor/_deps/mcap-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


