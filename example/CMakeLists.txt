# CMakeList.txt : CMake project for ml-linalg, include source and define
# project specific logic here.
#

# Add source to this project's executable.

file(GLOB sources "NeuralNet3.cpp" "mnist_loader.h" "../include/NeuralNet.h")

set(LINALG_CXX_STANDARD 23)
set(LINALG_ENABLE_KOKKOS OFF)
set(LINALG_ENABLE_CONCEPTS On)
include(FetchContent)
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

set(STDBLAS_include "${stdblas_BINARY_DIR}/include/experimental")
set(STDBLAS_include2 "${stdblas_SOURCE_DIR}/include")
set(MDSPAN_include "${mdspan_SOURCE_DIR}/include")

if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif (UNIX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++20")
endif()

add_executable(NeuralNet ${sources})
set_property(TARGET NeuralNet PROPERTY CXX_STANDARD 20)

target_include_directories(NeuralNet PRIVATE "../include")
target_include_directories(NeuralNet PRIVATE ${STDBLAS_include2})
target_include_directories(NeuralNet PRIVATE ${STDBLAS_include})
target_include_directories(NeuralNet PRIVATE ${MDSPAN_include})

add_custom_command(TARGET NeuralNet PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:NeuralNet>/../data)

# TODO: Add tests and install targets if needed.