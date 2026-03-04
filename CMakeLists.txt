cmake_minimum_required(VERSION 3.20)

# On macOS, set SDK and compiler before project() so standard headers (e.g. <cstdint>) are found.
if(APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE _sdk_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(CMAKE_OSX_SYSROOT "${_sdk_path}" CACHE PATH "macOS SDK path")
  set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE PATH "C compiler" FORCE)
  set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE PATH "CXX compiler" FORCE)
endif()

project(relml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE AND CMAKE_OSX_SYSROOT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CMAKE_OSX_SYSROOT}/usr/include/c++/v1" CACHE STRING "" FORCE)
endif()

include_directories(include)

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(RELML_OPENMP_FOUND 1)
endif()

# ---------------------------------------------------------------------------
# OpenBLAS — enables cblas_sgemm in Linear::forward/backward
# ---------------------------------------------------------------------------
find_package(BLAS QUIET)
if(BLAS_FOUND)
  message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
  add_compile_definitions(RELML_USE_BLAS)
else()
  message(STATUS "BLAS not found — using scalar fallback")
endif()

# ---------------------------------------------------------------------------
# CUDA (optional) — GPU training and multi-GPU
# ---------------------------------------------------------------------------
option(RELML_USE_CUDA "Build CUDA backend for GPU training" OFF)
  if(NOT CMAKE_CUDA_COMPILER)
    get_filename_component(_cuda_root "${CUDAToolkit_INCLUDE_DIR}" DIRECTORY)
    if(EXISTS "${_cuda_root}/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "${_cuda_root}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    endif()
    unset(_cuda_root)
  endif()
if(RELML_USE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING "CUDA architectures (52 = Maxwell GTX 980)")
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  message(STATUS "CUDA found: building relml_cuda")
  add_compile_definitions(RELML_CUDA)
  add_library(relml_cuda STATIC
    src/cuda/cuda_backend.cc
    src/cuda/cuda_backend_impl.cu
  )
  set_target_properties(relml_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(relml_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(relml_cuda PUBLIC CUDA::cudart CUDA::cublas)
  set_property(TARGET relml_cuda PROPERTY CUDA_ARCHITECTURES "52")
else()
  message(STATUS "CUDA not found — GPU backend disabled")
endif()

# ---------------------------------------------------------------------------
# libcurl — required by the Agent module
# ---------------------------------------------------------------------------
find_package(CURL REQUIRED)

# ---------------------------------------------------------------------------
# nlohmann/json — header-only, fetched automatically at configure time
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------
add_library(relml_database
    src/database/Column.cpp
    src/database/Table.cpp
    src/database/Database.cpp
    src/database/Typeinferrer.cpp
    src/database/FKDetector.cpp
    src/database/CSVLoader.cpp
)

add_library(relml_graph
    src/graph/HeteroGraph.cpp
    src/graph/Graphbuilder.cpp
)

add_library(relml_encoding
    src/encoding/CategoricalEncoder.cpp
    src/encoding/NumericalEncoder.cpp
    src/encoding/TimestampEncoder.cpp
    src/encoding/HeteroEncoder.cpp
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(relml_encoding PUBLIC OpenMP::OpenMP_CXX)
endif()
if(BLAS_FOUND)
  target_link_libraries(relml_encoding PUBLIC ${BLAS_LIBRARIES})
  target_include_directories(relml_encoding PRIVATE ${BLAS_INCLUDE_DIRS})
endif()

add_library(relml_gnn
    src/gnn/HeteroGraphSAGE.cpp
    src/gnn/MLPHead.cpp
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(relml_gnn PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(relml_training
    src/training/Adam.cpp
    src/training/Metrics.cpp
    src/training/TaskBuilder.cpp
    src/training/TaskParser.cpp
    src/training/TaskSpec.cpp
    src/training/Trainer.cpp
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(relml_training PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(relml_training PUBLIC nlohmann_json::nlohmann_json)

add_library(relml_agent
    src/agent/agent.cpp
    src/agent/ModelRegistry.cpp
    src/agent/RelMLSystem.cpp
)
target_include_directories(relml_agent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(relml_agent PUBLIC
    CURL::libcurl
    nlohmann_json::nlohmann_json
    relml_training
    relml_gnn
    relml_encoding
    relml_graph
    relml_database
)

# ---------------------------------------------------------------------------
# A convenience function so each example task only needs one line below.
# It links the full relml stack and adds the stdc++fs shim automatically.
# ---------------------------------------------------------------------------
function(add_relml_task target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE
        relml_training
        relml_gnn
        relml_encoding
        relml_graph
        relml_database
        stdc++fs
    )
endfunction()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_executable(test_database tests/test_database.cpp)
target_link_libraries(test_database relml_database stdc++fs)

add_executable(test_graph tests/test_graph.cpp)
target_link_libraries(test_graph relml_database relml_graph stdc++fs)

add_executable(test_encoding tests/test_encoding.cpp)
target_link_libraries(test_encoding relml_database relml_encoding stdc++fs)

add_executable(test_gnn tests/test_gnn.cpp)
target_link_libraries(test_gnn relml_database relml_graph relml_encoding relml_gnn stdc++fs)

add_executable(test_training tests/test_training.cpp)
target_link_libraries(test_training relml_training relml_gnn relml_encoding relml_graph relml_database stdc++fs)

add_executable(test_agent tests/test_agent.cpp)
target_link_libraries(test_agent relml_agent)

add_executable(test_system tests/test_system.cpp)
target_link_libraries(test_system relml_agent stdc++fs)

# ---------------------------------------------------------------------------
# Example tasks
# Each line is:  add_relml_task(<executable_name>  <source_file>)
# ---------------------------------------------------------------------------
add_relml_task(hm_churn                   src/example_tasks/hm_churn.cpp)
add_relml_task(ml1m_rating_classification src/example_tasks/ml1m_rating_classification.cpp)