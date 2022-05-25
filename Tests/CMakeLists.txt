enable_testing()

file(GLOB sources "*.cpp")

project(Tests)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

add_executable(Tests ${sources})

target_link_libraries(Tests PRIVATE Sagacity ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_test(NAME Tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} COMMAND Tests)