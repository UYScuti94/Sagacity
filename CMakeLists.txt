cmake_minimum_required(VERSION 3.16.3)

file(GLOB headers "*.hpp")
file(GLOB sources "*.cpp")

project(Sagacity)

option(BUILD_TESTS "Build the tests." OFF)

if (BUILD_TESTS)
    add_subdirectory(Tests)
endif()

find_package(nlohmann_json REQUIRED)

add_library(Sagacity STATIC ${headers} ${sources})

target_link_directories(Sagacity PUBLIC nlohmann_json)