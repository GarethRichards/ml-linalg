# CMakeList.txt : CMake project for ml-linalg, include source and define
# project specific logic here.
#

# Add source to this project's executable.

file(GLOB sources "NeuralNet3.cpp" "mnist_loader.h" "../include/NeuralNet.h")
add_executable(NeuralNet ${sources})
set_property(TARGET NeuralNet PROPERTY CXX_STANDARD 20)

target_include_directories(NeuralNet PRIVATE "../include")
target_include_directories(NeuralNet PRIVATE ${MDSPAN_include})
target_include_directories(NeuralNet PRIVATE ${STDBLAS_include})

add_custom_command(TARGET NeuralNet PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:NeuralNet>/../data)

# TODO: Add tests and install targets if needed.