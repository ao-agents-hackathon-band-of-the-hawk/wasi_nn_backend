cmake_minimum_required(VERSION 3.13)
project(wasi_nn_backend C CXX CUDA)

# Setup job pool for parallel builds
set(CMAKE_JOB_POOLS compile_job_pool=16 link_job_pool=8)
set(CMAKE_JOB_POOL_COMPILE compile_job_pool)
set(CMAKE_JOB_POOL_LINK link_job_pool)

# Basic settings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the default CUDA architecture
set(CUDA_ARCHITECTURE "86" CACHE STRING "CUDA architecture to build for")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif ()

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Set log level
add_compile_definitions(
  $<$<CONFIG:Debug>:NN_LOG_LEVEL=0>
  $<$<CONFIG:Release>:NN_LOG_LEVEL=2>
)



if (NOT WIN32)
  # Common flags for GCC/Clang
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wformat -Wformat-security -ffunction-sections -fdata-sections -Wno-unused-parameter -Wno-pedantic -fvisibility=hidden")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wformat -Wformat-security -Wno-unused-parameter -fvisibility=hidden")
  # Remove extra spaces
  string(REGEX REPLACE "  +" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REGEX REPLACE "  +" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
  # Add Windows specific flags if needed
endif()




# Include main build logic
include(wasi_nn)
