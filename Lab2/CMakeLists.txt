cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(Lab2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;" CACHE STRING "" FORCE)

FILE(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.h")

find_package(OpenMP REQUIRED)
if (NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found.")
endif ()


#set(ENV{OMP_PLACES} "cores")
#set(ENV{OMP_PROC_BIND} "thread")


message("OMP_PLACES: $ENV{OMP_PLACES}")
message("OMP_PROC_BIND: $ENV{OMP_PROC_BIND}")

# "close": Threads are bound to the same core as the master thread.
# "spread": Threads are spread across the available cores.
# "master": Threads are bound to the same core as the master thread, but only if the master thread is not already bound to a core.
# "thread": Each thread is bound to a single core, and the binding is not changed during the execution of the program.

add_executable(Lab2 ${SOURCE_FILES} ${DATA_FILES})

target_compile_options(Lab2 PRIVATE -Wall -Wextra -O3) # set "-g" to show debug symbols for debugger

target_link_libraries(Lab2 PUBLIC OpenMP::OpenMP_CXX)
