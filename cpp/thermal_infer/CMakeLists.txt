cmake_minimum_required(VERSION 3.20)
project(thermal_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnxruntime CONFIG REQUIRED)

add_executable(thermal_infer_node
  src/main.cpp
)
target_link_libraries(thermal_infer_node PRIVATE onnxruntime::onnxruntime)
target_include_directories(thermal_infer_node PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
