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

set(LINALG_ENABLE_KOKKOS OFF)
set(LINALG_ENABLE_CONCEPTS On)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG        stable
)
FetchContent_MakeAvailable(mdspan)
FetchContent_Declare(
  stdBLAS
  GIT_REPOSITORY https://github.com/kokkos/stdBLAS.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(stdBLAS)

add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif (UNIX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++20")
endif()

if (WIN32)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd") 
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")  
endif (WIN32)
enable_testing()

add_executable(Neural_Tester test1.cpp ../include/NeuralNet.h mnist_loader.h)
target_link_libraries(Neural_Tester GTest::GTest)

set(STDBLAS_include "${stdblas_BINARY_DIR}/include/experimental")
set(STDBLAS_include2 "${stdblas_SOURCE_DIR}/include")
set(MDSPAN_include "${mdspan_SOURCE_DIR}/include")

target_include_directories(Neural_Tester PRIVATE "../include")
target_include_directories(Neural_Tester PRIVATE ${STDBLAS_include})
target_include_directories(Neural_Tester PRIVATE ${STDBLAS_include2})
target_include_directories(Neural_Tester PRIVATE ${MDSPAN_include})

add_custom_command(TARGET Neural_Tester PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:Neural_Tester>/../data)

add_test(Neural_Tester Neural_Tester)

# TODO: Add tests and install targets if needed.