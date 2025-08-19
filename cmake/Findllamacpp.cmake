# Copyright (C) 2019 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)

##TODO: set stable branch

set(LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/llama.cpp")
if(EXISTS ${LLAMA_SOURCE_DIR})
  message("Use existed source code under ${LLAMA_SOURCE_DIR}")
  FetchContent_Declare(
    llamacpp
    SOURCE_DIR     ${LLAMA_SOURCE_DIR}
  )
else()
  message("download source code and store it at ${LLAMA_SOURCE_DIR}")
  FetchContent_Declare(
    llamacpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b5970
    SOURCE_DIR     ${LLAMA_SOURCE_DIR}
  )
endif()

set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
set(LLAMA_BUILD_COMMON ON)
set(LLAMA_CURL OFF)
set(GGML_CUDA ON)
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURE}")

set(CMAKE_JOB_POOLS compile_job_pool=4 )
FetchContent_MakeAvailable(llamacpp)
message(STATUS "llama.cpp available. Targets: llama, ggml common")
