cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(kestrel_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Clear PyTorch's auto-detected CUDA architecture flags - we set our own
set(CMAKE_CUDA_FLAGS "")

# C++/CUDA extensions that use `torch/extension.h` (pybind11) must link against
# libtorch_python to resolve ATen/Tensor type casters at runtime.
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')"
  OUTPUT_VARIABLE TORCH_PYTHON_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_library(TORCH_PYTHON_LIBRARY
  NAMES torch_python
  PATHS "${TORCH_PYTHON_LIB_DIR}"
  NO_DEFAULT_PATH
)
if (NOT TORCH_PYTHON_LIBRARY)
  message(FATAL_ERROR "Failed to find libtorch_python in ${TORCH_PYTHON_LIB_DIR}")
endif()

# Match Torch's C++11 ABI setting.
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})

# Default to H100 (SM90a). Override with -DKESTREL_KERNELS_CUDA_ARCH=...
set(KESTREL_KERNELS_CUDA_ARCH "90a" CACHE STRING "CUDA arch (e.g. 90a, 90, 80)")

set(KESTREL_KERNELS_CUDA_FLAGS
  "-DNDEBUG"
  "-O3"
  "-lineinfo"
  "--expt-relaxed-constexpr"
  "--expt-extended-lambda"
  "-gencode=arch=compute_${KESTREL_KERNELS_CUDA_ARCH},code=sm_${KESTREL_KERNELS_CUDA_ARCH}"
)

get_filename_component(KESTREL_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(KESTREL_OPS_DIR "${KESTREL_REPO_ROOT}/kestrel/ops")

set(KESTREL_KERNELS_INCLUDE_DIRS
  "${KESTREL_OPS_DIR}"
)

function(add_kestrel_kernel module_name source_file)
  if (NOT EXISTS "${source_file}")
    message(FATAL_ERROR "Missing kernel source: ${source_file}")
  endif()

  Python_add_library(${module_name} MODULE WITH_SOABI "${source_file}")
  target_include_directories(${module_name} PRIVATE
    ${KESTREL_KERNELS_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
  )
  target_compile_definitions(${module_name} PRIVATE TORCH_EXTENSION_NAME=${module_name})
  target_compile_options(${module_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${KESTREL_KERNELS_CUDA_FLAGS}>
  )
  target_link_libraries(${module_name} PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    c10
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
  )

  install(TARGETS ${module_name} LIBRARY DESTINATION kestrel_kernels)
endfunction()

add_kestrel_kernel(activation "${KESTREL_OPS_DIR}/activation/kernel.cu")
add_kestrel_kernel(fused_linear_residual "${KESTREL_OPS_DIR}/fused_linear_residual/kernel.cu")
add_kestrel_kernel(fused_mlp "${KESTREL_OPS_DIR}/fused_mlp/kernel.cu")
add_kestrel_kernel(kv_cache_write "${KESTREL_OPS_DIR}/kv_cache_write/kernel.cu")
add_kestrel_kernel(layernorm_cuda "${KESTREL_OPS_DIR}/layernorm_cuda/kernel.cu")
add_kestrel_kernel(moe_sum "${KESTREL_OPS_DIR}/moe_sum/kernel.cu")
add_kestrel_kernel(rotary_embedding "${KESTREL_OPS_DIR}/rotary_embedding/kernel.cu")
add_kestrel_kernel(fp8_quant "${CMAKE_CURRENT_LIST_DIR}/csrc/fp8_quant/kernel.cu")
add_kestrel_kernel(tau_tail "${CMAKE_CURRENT_LIST_DIR}/csrc/tau_tail/kernel.cu")

# Precompile CuTe DSL kernels using unified precompile module
set(PRECOMPILED_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/python/kestrel_kernels/precompiled")

# Main target: precompile all kernel families
add_custom_target(precompile_cute_kernels ALL
  COMMAND ${Python_EXECUTABLE} -m scripts.precompile
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  COMMENT "Precompiling CuTe DSL kernels (topk, cute_moe, moe_align, flash_attn)"
  VERBATIM
)

# Individual targets for development/debugging
add_custom_target(precompile_topk
  COMMAND ${Python_EXECUTABLE} -m scripts.precompile --kernels topk
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  COMMENT "Precompiling CuTe DSL topk kernel"
  VERBATIM
)

add_custom_target(precompile_cute_moe
  COMMAND ${Python_EXECUTABLE} -m scripts.precompile --kernels cute_moe
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  COMMENT "Precompiling CuTe DSL cute_moe kernels"
  VERBATIM
)

add_custom_target(precompile_moe_align
  COMMAND ${Python_EXECUTABLE} -m scripts.precompile --kernels moe_align
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  COMMENT "Precompiling CuTe DSL moe_align kernels"
  VERBATIM
)

add_custom_target(precompile_flash_attn
  COMMAND ${Python_EXECUTABLE} -m scripts.precompile --kernels flash_attn
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  COMMENT "Precompiling CuTe DSL flash_attn kernels"
  VERBATIM
)

# Install precompiled kernels (.so files only)
install(
  DIRECTORY "${PRECOMPILED_OUTPUT_DIR}/"
  DESTINATION kestrel_kernels/precompiled
  FILES_MATCHING PATTERN "*.so"
)

# ============================================================================
# Python Distribution Files (allow-list)
# Only these files will be included in the wheel - NO kernel source templates
# ============================================================================

set(PYTHON_DIST_FILES
    # Core package
    python/kestrel_kernels/__init__.py
    python/kestrel_kernels/precompile.py
    python/kestrel_kernels/cutils.py
    python/kestrel_kernels/tau_tail_ops.py

    # TopK - dispatch code only (NO kernel.py template)
    python/kestrel_kernels/topk/__init__.py
    python/kestrel_kernels/topk/dispatch.py

    # MoE align - dispatch code only (NO kernel.py template)
    python/kestrel_kernels/moe_align/__init__.py
    python/kestrel_kernels/moe_align/dispatch.py

    # CuTe MoE - dispatch code only (NO cute_moe_*.py kernel templates)
    python/kestrel_kernels/cute_moe/__init__.py
    python/kestrel_kernels/cute_moe/config.py
    python/kestrel_kernels/cute_moe/dispatch.py
    python/kestrel_kernels/cute_moe/utils.py

    # Flash attention - dispatch code only (NO flash_*.py kernel templates)
    python/kestrel_kernels/flash_attn/__init__.py
    python/kestrel_kernels/flash_attn/cute/__init__.py
    python/kestrel_kernels/flash_attn/cute/interface.py
    python/kestrel_kernels/flash_attn/cute/utils.py
    python/kestrel_kernels/flash_attn/cute/cute_dsl_utils.py
    python/kestrel_kernels/flash_attn/cute/mask_definitions.py
    python/kestrel_kernels/flash_attn/cute/block_sparsity.py

    # Configs
    python/kestrel_kernels/configs/__init__.py

    # Precompiled placeholder
    python/kestrel_kernels/precompiled/__init__.py
)

# Install Python files to their correct locations
foreach(PY_FILE ${PYTHON_DIST_FILES})
    get_filename_component(PY_DIR ${PY_FILE} DIRECTORY)
    string(REPLACE "python/" "" DEST_DIR ${PY_DIR})
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PY_FILE} DESTINATION ${DEST_DIR})
endforeach()

# Install config JSON files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/kestrel_kernels/configs/
        DESTINATION kestrel_kernels/configs
        FILES_MATCHING PATTERN "*.json")
