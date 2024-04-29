cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(PROJECT_NAME Lab3)

project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;" CACHE STRING "" FORCE)

FILE(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.h")

# sudo apt install mpich
find_package(MPI)
if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    message("MPI_INCLUDE_PATH :${MPI_INCLUDE_PATH}")
endif (MPI_FOUND)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

add_compile_options($<$<CONFIG:Debug>: -Wall -Wextra -g>)
add_compile_options($<$<CONFIG:Release>: -Wall -Wextra -fast>)

target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX)
