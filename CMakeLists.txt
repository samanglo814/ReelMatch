cmake_minimum_required(VERSION 3.16)
project(ReelMatch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized build so runtime benchmarks are meaningful
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add every new .cpp file to this list
add_executable(ReelMatch
        src/main.cpp
        src/Dataset.cpp
)

target_include_directories(ReelMatch PRIVATE include)
target_compile_options(ReelMatch PRIVATE -Wall -Wextra)