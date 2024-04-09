# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(LINALG_ENABLE_BLAS On)
set(LINALG_ENABLE_CONCEPTS On)
set(LINALG_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
  stdBLAS
  GIT_REPOSITORY https://github.com/kokkos/stdBLAS.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(stdBLAS)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG        stable
)
FetchContent_MakeAvailable(mdspan)

get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

project ("ml-linalg")
if(UNIX)
	set(STDBLAS_include "${stdblas_SOURCE_DIR}/include")
	set(MDSPAN_include "${mdspan_SOURCE_DIR}/include")
endif(UNIX)
if(WIN32)
	set(MDSPAN_include "C:/Users/garet/source/mdspan-stable/include")
	set(STDBLAS_include "C:/Users/garet/source/stdBLAS-main/include")
endif(WIN32)


# Include sub-projects.
#add_subdirectory("ml-linalg")
add_subdirectory("example")
add_subdirectory("tests")

enable_testing()