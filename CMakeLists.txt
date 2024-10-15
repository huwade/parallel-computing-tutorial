cmake_minimum_required(VERSION 3.10)

# Project name
project(ParallelCompute)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/loop_reordering.cpp
    ${PROJECT_SOURCE_DIR}/src/navie_mat_mul.cpp
)

# Create an executable
add_executable(ParallelCompute ${SOURCES})

# Link libraries (if necessary, e.g., pthread)
find_package(Threads REQUIRED)
target_link_libraries(ParallelCompute Threads::Threads)