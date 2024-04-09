# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

include(FetchContent)
FetchContent_Declare(
  stdBLAS
  GIT_REPOSITORY https://github.com/kokkos/stdBLAS.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(stdBLAS)

#FetchContent_Declare(
#  mdspan
#  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
#  GIT_TAG        stable
#)
#FetchContent_MakeAvailable(mdspan)

#find_package(mdspan QUIET)
#if (NOT mdspan_FOUND)
#  message(STATUS "No installed mdspan found, fetching from Github")
#  include(FetchContent)
#  FetchContent_Declare(
#    mdspan
#    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
#    GIT_TAG        stable
#  )
#  FetchContent_GetProperties(mdspan)
#  if(NOT mdspan_POPULATED)
#    FetchContent_Populate(mdspan)
#    add_subdirectory(${mdspan_SOURCE_DIR} ${mdspan_BINARY_DIR} EXCLUDE_FROM_ALL)
#  endif()
#endif()

project ("ml-linalg")
if(UNIX)
	set(MDSPAN_include "~/code/mdspan-stable/include/")	
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