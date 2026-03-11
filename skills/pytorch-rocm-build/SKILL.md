---
name: pytorch-rocm-build
description: Build PyTorch from source on AMD ROCm platforms. Covers Docker container setup, submodule initialization, hipification, source build, and editable install for development. Use when building PyTorch from source on MI300/MI355 or other AMD GPUs, or when needing to modify PyTorch Python/C++ code on ROCm.
---

# PyTorch ROCm Source Build

## Important: All build commands run inside Docker

All build and test commands MUST be executed inside a Docker container via
`sudo docker exec`. The host typically does not have ROCm build tools installed.

## Step 1: Select Docker image

| NIC Type | Docker Image |
|----------|--------------|
| mlx5 (Mellanox CX7) | `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.8.0` |
| bnxt (Broadcom Thor2) | `rocm/pytorch-private:mori_bnxt_rocm7.1.1_ubuntu24.04_py3.12` |
| ionic (AMD AINIC) | `rocm/pytorch-private:sglang-0.5.8-rocm720-mi35x-mori-0216` |
| mori dev | `rocm/mori:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` |

Private images require `sudo docker login -u rocmshared`.

## Step 2: Create or start Docker container

If container already exists, start it:

```bash
CONTAINER_NAME="pytorch_rocm_build"
sudo docker start $CONTAINER_NAME 2>/dev/null
```

Otherwise, create a new container:

```bash
DOCKER_IMAGE="rocm/mori:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1"
CONTAINER_NAME="pytorch_rocm_build"
PT_SRC="/mnt/nvme1/jiahzhou/pytorch"

sudo docker run \
  --group-add video --network=host \
  --ulimit nproc=100000:100000 --pids-limit=-1 \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  -d --ipc=host --privileged -it \
  -v /home/:/home/ -v /root:/root -v /mnt:/mnt \
  --name "$CONTAINER_NAME" \
  "$DOCKER_IMAGE"
```

## Step 3: Fix git safe.directory

The source tree may be owned by a different user than the container's root:

```bash
sudo docker exec $CONTAINER_NAME bash -c \
  "git config --global --add safe.directory '*'"
```

## Step 4: Initialize submodules

Inside the container:

```bash
sudo docker exec $CONTAINER_NAME bash -c \
  "cd $PT_SRC && git submodule update --init --recursive"
```

Verify key submodules:

```bash
sudo docker exec $CONTAINER_NAME bash -c \
  "ls $PT_SRC/third_party/pybind11/CMakeLists.txt && echo 'submodules OK'"
```

If a specific submodule is missing (e.g. `psimd`):

```bash
sudo docker exec $CONTAINER_NAME bash -c \
  "cd $PT_SRC && git submodule update --init --force third_party/psimd"
```

## Step 5: Hipify source tree

**Required before every fresh build on ROCm.** This converts CUDA API calls to
HIP equivalents in the source tree:

```bash
sudo docker exec $CONTAINER_NAME bash -c \
  "cd $PT_SRC && python3 tools/amd_build/build_amd.py"
```

Expected output ends with `Successfully preprocessed all matching files.`

Note: hipify modifies files in-place. These changes should NOT be committed.
When ready to commit your actual changes, use:

```bash
git stash -u
git reset --hard <original_commit>
git stash pop
```

## Step 6: Build PyTorch

### Standard build (editable install)

```bash
sudo docker exec -d $CONTAINER_NAME bash -c \
  "cd $PT_SRC && \
   export MAX_JOBS=32 && \
   export PYTORCH_ROCM_ARCH=gfx942 && \
   export USE_MSLK=0 && \
   pip install -e . -v --no-build-isolation > /tmp/pytorch_build.log 2>&1"
```

Key environment variables:

| Variable | Value | Notes |
|----------|-------|-------|
| `MAX_JOBS` | `32` (adjust for CPU count) | Parallel compilation jobs |
| `PYTORCH_ROCM_ARCH` | `gfx942` (MI300) or `gfx950` (MI355) | Target GPU architecture |
| `USE_MSLK` | `0` | Disable mslk if submodule has issues |
| `USE_NVSHMEM` | auto | Not relevant for ROCm |

### Monitor build progress

```bash
# Check line count and latest output
sudo docker exec $CONTAINER_NAME bash -c \
  "wc -l /tmp/pytorch_build.log && tail -5 /tmp/pytorch_build.log"

# Check for compilation errors
sudo docker exec $CONTAINER_NAME bash -c \
  "grep -i 'FAILED\|fatal error' /tmp/pytorch_build.log | head -10"

# Check active build processes
sudo docker exec $CONTAINER_NAME bash -c \
  "ps aux | grep -E 'cmake|ninja|hipcc' | grep -v grep | wc -l"
```

Build typically takes 20-40 minutes with MAX_JOBS=32. Expected ~6000 compilation
steps (fewer with `USE_MSLK=0`).

### Clean rebuild

If cmake cache is stale (e.g. after changing cmake options):

```bash
sudo docker exec $CONTAINER_NAME bash -c \
  "rm -rf $PT_SRC/build"
```

Then re-run the build command.

## Step 7: Verify build

```bash
sudo docker exec $CONTAINER_NAME bash -c "python3 -c '
import torch
print(f\"PyTorch: {torch.__version__}\")
print(f\"HIP: {torch.version.hip}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(f\"Device: {torch.cuda.get_device_name(0)}\")
'"
```

Expected output:

```
PyTorch: 2.12.0a0+git<hash>
HIP: 7.1.52802
CUDA available: True
Device: AMD Instinct MI308X
```

## Step 8: Development workflow

Since the build is an **editable install** (`pip install -e .`), Python file
changes take effect immediately without rebuilding:

```bash
# Edit Python files on the host
vim $PT_SRC/torch/distributed/_symmetric_memory/__init__.py

# Test immediately in container (no rebuild needed)
sudo docker exec $CONTAINER_NAME bash -c \
  "python3 -c 'import torch; ...'"
```

C++ changes require a rebuild. For incremental C++ builds, re-run the pip
install command — ninja will only recompile changed files:

```bash
sudo docker exec -d $CONTAINER_NAME bash -c \
  "cd $PT_SRC && MAX_JOBS=32 PYTORCH_ROCM_ARCH=gfx942 USE_MSLK=0 \
   pip install -e . -v --no-build-isolation > /tmp/pytorch_rebuild.log 2>&1"
```

## Common issues

### `mslk/utils/tuning_cache_hip.cuh` not found

The mslk third-party submodule has hipify issues. Disable with `USE_MSLK=0`.

### `Did you run 'git submodule update --init --recursive'?`

Submodules not initialized. Run Step 4 inside the container.

### `detected dubious ownership in repository`

Git safe.directory not configured. Run Step 3.

### hipify modified my source files

This is expected. Use `git stash` / `git reset` before committing (see Step 5).

### Build hangs with `| tail -100`

Do NOT pipe build output through `tail`. Use file redirection instead:
`> /tmp/build.log 2>&1`.

## Running distributed tests

```bash
# 2-GPU test
sudo docker exec $CONTAINER_NAME bash -c \
  "torchrun --nproc_per_node=2 test/distributed/test_symmetric_memory.py -v"

# 4-GPU test (for one_shot_all_reduce etc.)
sudo docker exec $CONTAINER_NAME bash -c \
  "torchrun --nproc_per_node=4 test/distributed/test_symmetric_memory.py \
   SymmMemCollectiveTest -v"
```
