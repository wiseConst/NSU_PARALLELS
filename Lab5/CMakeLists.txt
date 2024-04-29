cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(PROJECT_NAME Lab5)

project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;" CACHE STRING "" FORCE)

FILE(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.h")

# sudo apt install mpich
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(${PROJECT_NAME} ${SOURCE_FILES}
        src/ConcurrentQueue.h
)

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

add_compile_options($<$<CONFIG:Debug>: -Wall -Wextra -g>)
add_compile_options($<$<CONFIG:Release>: -Wall -Wextra -fast>)

target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)
