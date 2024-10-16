cmake_minimum_required(VERSION 3.10)

# Project name
project(ParallelCompute LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the CUDA architectures (update this with your target architecture)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Example for Turing (like RTX 20xx)

# Add the include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/loop_reordering.cpp
    ${PROJECT_SOURCE_DIR}/src/navie_mat_mul.cpp
    ${PROJECT_SOURCE_DIR}/src/loop_tiling.cpp
    ${PROJECT_SOURCE_DIR}/src/matrixMul.cu
    ${PROJECT_SOURCE_DIR}/src/matrixMul_share.cu
)

# Create an executable
add_executable(ParallelCompute ${SOURCES})

# Link libraries (if necessary, e.g., pthread)
find_package(Threads REQUIRED)
target_link_libraries(ParallelCompute Threads::Threads)

# Enable CUDA support
find_package(CUDAToolkit REQUIRED)
# Add CUDA source files
set(CUDA_SOURCES
    ${PROJECT_SOURCE_DIR}/src/matrixMul.cu
    ${PROJECT_SOURCE_DIR}/src/matrixMul_share.cu
)

# Include directories required by CUDA
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Specify CUDA properties
set_target_properties(ParallelCompute PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add CUDA sources to the executable
target_sources(ParallelCompute PRIVATE ${CUDA_SOURCES})