---
name: pytorch-rocm-build
description: Build PyTorch from source on AMD ROCm platforms with Mori shmem SymmetricMemory integration. Covers Docker container setup, submodule initialization, hipification, source build, Mori backend configuration, and CI testing. Use when building PyTorch from source on MI300/MI350/MI325 AMD GPUs, modifying PyTorch Python/C++ code on ROCm, or integrating Mori shmem as the SymmetricMemory backend.
---

# PyTorch ROCm Source Build + Mori SymmetricMemory Integration

## Architecture Overview

Mori shmem is integrated into PyTorch as a SymmetricMemory backend for AMD GPUs,
supporting intra-node P2P and inter-node RDMA (IBGDA) across MLX5, BNXT, and
AINIC NICs. The integration is pure C++ — no mori Python package dependency at
runtime.

```
TORCH_SYMMMEM=MORI
    ↓
import torch → torch_hip.so → torch_mori.so → libmori_shmem.so
                                    ↓
                        static registration of MORISymmetricMemoryAllocator
                                    ↓
              PyTorch fused ops automatically use mori backend
```

### Key files

| Repository | File | Purpose |
|-----------|------|---------|
| pytorch | `torch/csrc/distributed/c10d/symm_mem/MORISymmetricMemory.cpp` | C++ backend: alloc→ShmemMalloc, rendezvous→ShmemPtrP2p, barrier→ShmemBarrierAll |
| pytorch | `caffe2/CMakeLists.txt` | Auto-detect mori from pip, build libtorch_mori.so in USE_ROCM block |
| pytorch | `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.cpp` | Accept "MORI" in TORCH_SYMMMEM env var |
| pytorch | `torch/distributed/_symmetric_memory/__init__.py` | set_backend("MORI") support |
| mori | `CMakeLists.txt` | `target_compile_definitions(spdlog PUBLIC spdlog=mori_spdlog fmt=mori_fmt)` — namespace rename to avoid ROCm librocroller symbol conflict |
| mori | `src/shmem/init.cpp` | ShmemInit re-entrancy guard |
| mori | `python/mori/cpp/__init__.py` | RTLD_NOLOAD for mixed-use safety |

### Branches

| Repo | Branch |
|------|--------|
| pytorch (fork) | `jhchouuu/pytorch` : `jiahzhou/pytorch_mori_shmem_integrate` |
| mori | `ROCm/mori` : `jiahzhou/pytorch_mori_shmem_integrate` |

## Important: All commands run inside Docker

All build and test commands MUST be executed inside a Docker container via
`sudo docker exec`. The host typically does not have ROCm build tools installed.

## Step 1: Select Docker image

| NIC Type | Docker Image |
|----------|--------------|
| mlx5 (Mellanox CX7) | `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.8.0` |
| bnxt (Broadcom Thor2) | `rocm/pytorch-private:mori_bnxt235_rocm711_ubuntu24.04_py3.12` |
| ionic (AMD AINIC) | `rocm/pytorch-private:sglang-0.5.9-rocm720-mi35x-mori-0305` |

Private images require `sudo docker login -u rocmshared`.

## Step 2: Create Docker container

```bash
DOCKER_IMAGE="rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.8.0"
CONTAINER_NAME="pytorch_rocm_build"

sudo docker run \
  --group-add video --network=host \
  --ulimit nproc=100000:100000 --pids-limit=-1 \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  -d --ipc=host --privileged -it \
  -v /home/:/home/ -v /root:/root -v /nfs:/nfs -v /mnt:/mnt \
  --name "$CONTAINER_NAME" \
  "$DOCKER_IMAGE"
```

## Step 3: Install system dependencies

```bash
sudo docker exec $CONTAINER_NAME bash -c "
  apt-get update -qq && apt-get install -y -qq \
    libopenmpi-dev openmpi-bin libpci-dev libibverbs-dev cmake ninja-build > /dev/null 2>&1
  pip install cmake ninja pyyaml typing_extensions expecttest pytest-timeout -q
  git config --global --add safe.directory '*'
  echo 'OK'
"
```

## Step 4: Install mori (prerequisite for torch_mori)

Mori must be installed before building PyTorch so cmake can find
`libmori_shmem.so` and headers.

```bash
MORI_SRC="/nfs/users/jiahzhou/mori"
sudo docker exec $CONTAINER_NAME bash -c "
  cd $MORI_SRC && rm -rf build/CMakeCache.txt build/CMakeFiles
  pip install . 2>&1 | tail -3
"
```

Verify:
```bash
sudo docker exec $CONTAINER_NAME bash -c "
  python -c 'import mori; print(\"mori OK\")'
  ls \$(python -c 'import mori,os; print(os.path.dirname(mori.__file__))')/libmori_shmem.so && echo 'lib OK'
"
```

## Step 5: Prepare PyTorch source

```bash
PT_SRC="/nfs/users/jiahzhou/pytorch"

# Copy to local disk for faster I/O (NFS is slow for builds)
sudo docker exec $CONTAINER_NAME bash -c "
  rsync -a --exclude=build/ $PT_SRC/ /tmp/pytorch/
  cd /tmp/pytorch
  git submodule update --init --recursive
  python tools/amd_build/build_amd.py  # hipify
"
```

## Step 6: Build PyTorch

```bash
sudo docker exec $CONTAINER_NAME bash -c "
  cd /tmp/pytorch
  export USE_ROCM=1 USE_CUDA=0 PYTORCH_ROCM_ARCH='gfx942'
  export USE_DISTRIBUTED=1 USE_NCCL=1 USE_KINETO=0 BUILD_TEST=0
  export USE_MKLDNN=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0
  export MAX_JOBS=32
  export PYTHONPATH=/tmp/pytorch
  python setup.py develop 2>&1 | tail -5
"
```

Build takes 8-15 minutes (incremental: 1-2 minutes for C++ changes, instant
for Python-only changes in develop mode).

### Verify Mori backend was built

```bash
sudo docker exec $CONTAINER_NAME bash -c "
  ls /tmp/pytorch/torch/lib/libtorch_mori.so && echo 'torch_mori: OK'
  ldd /tmp/pytorch/torch/lib/libtorch_mori.so | grep mori
"
```

Expected:
```
torch_mori: OK
libmori_shmem.so => /opt/venv/.../mori/libmori_shmem.so
```

### Verify import and backend registration

```bash
sudo docker exec $CONTAINER_NAME bash -c "
  TORCH_SYMMMEM=MORI PYTHONPATH=/tmp/pytorch python -c '
    import torch
    from torch._C._distributed_c10d import _SymmetricMemory
    print(\"backend:\", _SymmetricMemory.get_backend(torch.device(\"cuda:0\")))
  '
"
```

Expected: `backend: MORI`

## Step 7: Run Tests

### Quick functional test (no CI framework)

```bash
sudo docker exec $CONTAINER_NAME bash -c "
  TORCH_SYMMMEM=MORI PYTHONPATH=/tmp/pytorch \
  timeout 60 torchrun --nproc_per_node=2 \
    /nfs/users/jiahzhou/mori/examples/shmem/test_pure_cpp_backend.py 2>&1
"
```

### PyTorch CI symmetric memory tests

Download the matching test file:

```bash
sudo docker exec $CONTAINER_NAME bash -c "
  COMMIT=\$(cd /tmp/pytorch && git rev-parse HEAD)
  mkdir -p /tmp/ci_test/test/distributed
  curl -sL \"https://raw.githubusercontent.com/pytorch/pytorch/\$COMMIT/test/distributed/test_symmetric_memory.py\" \
    -o /tmp/ci_test/test/distributed/test_symmetric_memory.py
"
```

Run AsyncTP fused ops tests (all should PASS):

```bash
sudo docker exec $CONTAINER_NAME bash -c "
  cd /tmp/ci_test
  TORCH_SYMMMEM=MORI PYTHONPATH=/tmp/pytorch \
  python -m pytest test/distributed/test_symmetric_memory.py::AsyncTPTest \
    -v --timeout=60 \
    -k 'test_fused_all_gather_matmul_gather_dim or \
        test_fused_matmul_reduce_scatter_scatter_dim or \
        test_optimal_layout'
"
```

Run SymmetricMemoryTest (core API):

```bash
sudo docker exec $CONTAINER_NAME bash -c "
  cd /tmp/ci_test
  TORCH_SYMMMEM=MORI PYTHONPATH=/tmp/pytorch \
  python -m pytest test/distributed/test_symmetric_memory.py::SymmetricMemoryTest \
    -v --timeout=60 \
    -k 'test_has_multicast or test_get_signal_pad_size or \
        (test_low_contention and not symm_mem_input_True)'
"
```

### Expected CI test results

| Test | Result | Notes |
|------|--------|-------|
| `test_fused_all_gather_matmul_gather_dim_0/1/2` | PASS | |
| `test_fused_matmul_reduce_scatter_scatter_dim_0/1/2` | PASS | |
| `test_optimal_layout_dim_0/1/2` | PASS | |
| `test_has_multicast_support` | PASS | returns False for AMD |
| `test_get_signal_pad_size` | PASS | |
| `test_low_contention_all_gather_symm_mem_input_False` | PASS | |
| `test_low_contention_reduce_scatter_*_symm_mem_input_False` | PASS | |
| `test_large_alloc` | FAIL | heap size limit (set MORI_SHMEM_HEAP_SIZE=4G) |
| `test_*_symm_mem_input_True` | SKIP | needs C++ empty() alloc path |
| `test_*_scaled_matmul_*` | SKIP | FP8 not yet supported |
| `test_*_native_*` | SKIP | needs _async_input_mm |
| `test_*_multimem_*` | SKIP | needs multicast hardware |
| `test_*_timeout_*` | SKIP | needs put_signal/wait_signal impl |

## Step 8: Development workflow

### Python-only changes (instant)

Edit files in `/nfs/users/jiahzhou/pytorch/torch/distributed/_symmetric_memory/`
on the host. Changes take effect immediately in develop mode.

### C++ changes (incremental rebuild)

```bash
# Edit MORISymmetricMemory.cpp on the host, then:
sudo docker exec $CONTAINER_NAME bash -c "
  cp /nfs/users/jiahzhou/pytorch/torch/csrc/distributed/c10d/symm_mem/MORISymmetricMemory.cpp \
     /tmp/pytorch/torch/csrc/distributed/c10d/symm_mem/MORISymmetricMemory.cpp
  cd /tmp/pytorch/build && ninja -j32 torch_mori
  cd /tmp/pytorch && pip install -e . --no-build-isolation 2>&1 | tail -1
"
# Takes ~10 seconds for torch_mori only
```

### CMake changes (need reconfigure)

```bash
sudo docker exec $CONTAINER_NAME bash -c "
  cp /nfs/users/jiahzhou/pytorch/caffe2/CMakeLists.txt /tmp/pytorch/caffe2/CMakeLists.txt
  cd /tmp/pytorch && rm -f build/CMakeCache.txt
  # Then full rebuild (8-15 min)
"
```

## Key design decisions

### Why libtorch_mori.so is a separate shared library

Like NVSHMEM's `torch_nvshmem.so`, mori is isolated in its own `.so` to:
- Keep mori as an optional dependency (not found → no torch_mori, no error)
- Avoid pulling mori headers into torch_hip compilation
- Allow independent version updates

### Why static registration (not ctypes)

`libtorch_mori.so` is auto-loaded via `torch_hip → torch_mori` ELF dependency.
A static constructor registers `MORISymmetricMemoryAllocator` at import time.
This matches NVSHMEM's pattern exactly. No Python-side ctypes or manual loading
needed.

### Why spdlog namespace rename in mori

ROCm's `librocroller.so` bundles its own spdlog. When both are loaded in the
same process, two spdlog global registries coexist. Mori's spdlog code
inadvertently uses librocroller's `spdlog::details::registry`, and freeing
objects across library boundaries causes `free(): invalid size`.

Fix: one line in mori's CMakeLists.txt:
```cmake
target_compile_definitions(spdlog PUBLIC spdlog=mori_spdlog fmt=mori_fmt)
```

This renames all spdlog/fmt symbols to `mori_spdlog`/`mori_fmt`, completely
isolating them from ROCm's spdlog.

### Why RTLD_NOLOAD in mori's Python package

When users mix C++ backend (`import torch` loads `libmori_shmem.so` via
torch_hip chain) with mori Python API (`import mori.shmem` loads
`libmori_shmem.so` via pybinds), the RTLD_NOLOAD check prevents double-loading:

```python
# mori/cpp/__init__.py
RTLD_NOLOAD = 0x4
try:
    ctypes.CDLL(lib_name, mode=RTLD_NOLOAD | ctypes.RTLD_GLOBAL)
    continue  # already loaded, skip
except OSError:
    pass  # not loaded, load normally
```

## Common issues

### `free(): invalid size` when calling ShmemGetUniqueId

**Cause**: spdlog symbol conflict between mori and ROCm's librocroller.so.
**Fix**: Ensure mori is built with the `spdlog=mori_spdlog` compile definition
(branch `jiahzhou/pytorch_mori_shmem_integrate`).

### `Mori not found, not building with Mori support`

cmake cannot find `libmori_shmem.so`. Ensure mori is pip-installed in the
build environment before building PyTorch.

### `SymmetricMemory does not support device type cuda`

MORI allocator not registered. Check:
1. `libtorch_mori.so` exists in `torch/lib/`
2. `ldd torch/lib/libtorch_hip.so | grep mori` shows dependency
3. `TORCH_SYMMMEM=MORI` is set (or `set_backend("MORI")` called)

### `Could not resolve the process group registered under the name default`

`MORISymmetricMemory.cpp` falls back to group "0" if "default" is not found.
Ensure the process group is registered before calling fused ops:
```python
torch._C._distributed_c10d._register_process_group('default', dist.group.WORLD)
```

### Build takes too long

- Use local disk (`/tmp/pytorch`) instead of NFS for the build tree
- Set `MAX_JOBS=32` (or match CPU core count)
- Disable unused components: `USE_MKLDNN=0 USE_FBGEMM=0 USE_NNPACK=0 ...`
- Incremental C++ rebuild (ninja only recompiles changed files): ~10s
- Python changes: instant (develop mode)
