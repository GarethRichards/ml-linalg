# CMakeList.txt : CMake project for ml-linalg, include source and define
# project specific logic here.
#

cmake_minimum_required(VERSION 3.13)
project(Neural_Tester)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

if (UNIX)
    set(CMAKE_CXX_FLAGS -pthread)
endif (UNIX)

if (WIN32)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif (WIN32)
enable_testing()

add_executable(Neural_Tester test1.cpp ../include/NeuralNet.h mnist_loader.h)
target_link_libraries(Neural_Tester GTest::GTest)

add_test(NAME Test COMMAND Neural_Tester)

target_include_directories(Neural_Tester PRIVATE "../include")
target_include_directories(Neural_Tester PRIVATE ${MDSPAN_include})
target_include_directories(Neural_Tester PRIVATE ${STDBLAS_include})

add_custom_command(TARGET Neural_Tester PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:Neural_Tester>/../data)

# TODO: Add tests and install targets if needed.