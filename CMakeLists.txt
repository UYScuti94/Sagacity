cmake_minimum_required(VERSION 3.16.3)

file(GLOB headers "*.hpp")
file(GLOB sources "*.cpp")

project(Sagacity)

option(BUILD_TESTS "Build the tests." OFF)

if (BUILD_TESTS)
    add_subdirectory(Tests)
endif()

add_library(Sagacity STATIC ${headers} ${sources})