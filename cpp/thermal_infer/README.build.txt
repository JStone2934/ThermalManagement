ONNX Runtime 实时路径（plan §7.2；TensorRT 可选，本仓库默认 ORT）。

依赖（需自行集成到工程）:
- unitree_sdk2 + CycloneDDS（宇树 SDK，通常非 vcpkg/Conan）
- onnxruntime：由 vcpkg 或 Conan 提供

--- vcpkg（推荐与 CMake 一体）---
  cd cpp
  cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
  cmake --build build

  manifest: cpp/vcpkg.json（onnxruntime CPU）。GPU 可改用端口 onnxruntime-gpu 并本机 CUDA。

--- Conan 2 ---
  cd cpp
  conan install . --output-folder=build --build=missing
  cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
  cmake --build build

布局建议: ring_buffer.*, ort_session.*, dds_thermal_node（订阅 rt/lowstate 等）。
